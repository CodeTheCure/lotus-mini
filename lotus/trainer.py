"""QLoRA training driver with live, verbose progress.

Wraps the ``mlx-lm`` LoRA trainer as a subprocess and streams its output with:
  * a tqdm progress bar (loss, tok/s, ETA) per expert
  * a watchdog heartbeat so a long model-load never looks like a stall
  * a raw byte-level reader so tqdm "\\r" updates are consumed too (a
    line-based reader would let "\\r"-only output fill the pipe buffer and
    deadlock the child).

QLoRA = a 4-bit quantized base model + LoRA adapters. Each expert is one
adapter trained on its sentiment-stratified subset.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .constants import BASE_MODEL, EXPERTS_DIR, LOGS_DIR

RE_ITER = re.compile(r"Iter\s+(\d+): Train loss ([\d.]+)")
RE_VAL = re.compile(r"Val loss ([\d.]+)")
RE_RATE = re.compile(r"(?:It/sec|Tokens/sec) ([\d.]+)")

log = logging.getLogger("lotus.trainer")


def setup_logging(verbose: bool = False, name: str = "lotus") -> logging.Logger:
    """Configure console logging with timestamps + unbuffered streams."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    return logging.getLogger(name)


@dataclass
class TrainConfig:
    iters: int = 80
    batch_size: int = 2
    grad_accum: int = 2
    max_seq_length: int = 512
    num_layers: int = 8
    learning_rate: float = 2e-5
    val_batches: int = 1
    seed: int = 42
    verbose: bool = False


def run_lora_training(cfg: TrainConfig, label: str, data_dir: str,
                      base_model: str = BASE_MODEL,
                      experts_dir: Path = EXPERTS_DIR) -> None:
    """Train one LoRA adapter (``label``) on ``data_dir`` with QLoRA."""
    adapter_path = Path(experts_dir) / label.lower()
    adapter_path.mkdir(parents=True, exist_ok=True)
    n_train = sum(1 for _ in open(Path(data_dir) / "train.jsonl"))

    log.info("=" * 70)
    log.info(">>> Expert [%s]  data=%s (%d train examples)", label, data_dir, n_train)
    log.info("    iters=%d  batch=%d  grad_accum=%d  lr=%s  seq_len=%d  layers=%d",
             cfg.iters, cfg.batch_size, cfg.grad_accum, cfg.learning_rate,
             cfg.max_seq_length, cfg.num_layers)
    log.info("    base=%s (4-bit quantized => QLoRA)", base_model)
    log.info("    Loading model + tokenizer ... (first run downloads ~3.4 GB)")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", data_dir,
        "--fine-tune-type", "lora",
        "--iters", str(cfg.iters),
        "--batch-size", str(cfg.batch_size),
        "--grad-accumulation-steps", str(cfg.grad_accum),
        "--max-seq-length", str(cfg.max_seq_length),
        "--num-layers", str(cfg.num_layers),
        "--learning-rate", str(cfg.learning_rate),
        "--mask-prompt",
        "--adapter-path", str(adapter_path),
        "--steps-per-report", "10",
        "--steps-per-eval", str(max(10, cfg.iters // 6)),
        "--val-batches", str(cfg.val_batches),
        "--seed", str(cfg.seed),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    pbar = tqdm(
        total=cfg.iters, unit="iter", desc=f"expert:{label.lower():3s}",
        ncols=100, file=sys.stderr,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            time.sleep(20)
            log.info("[heartbeat] expert [%s] still alive after %.0fs ...",
                     label, time.time() - start)

    threading.Thread(target=heartbeat, daemon=True).start()

    last_val = float("nan")
    last_iter = 0
    logfile = open(LOGS_DIR / f"{label.lower()}_train.log", "w", buffering=1)
    assert proc.stdout is not None

    def handle_piece(piece: str) -> None:
        piece = piece.rstrip("\r\n")
        if not piece.strip():
            return
        logfile.write(piece + "\n")
        nonlocal last_iter, last_val
        if (m := RE_ITER.search(piece)):
            it = int(m.group(1))
            last_iter = it
            pbar.n = it
            postfix = {"loss": m.group(2)}
            if (rm := RE_RATE.search(piece)):
                postfix["tok/s"] = rm.group(1)
            pbar.set_postfix(postfix)
            pbar.refresh()
            sys.stderr.flush()
            log.info("  iter %4d | train loss %s", it, m.group(2))
        elif (m := RE_VAL.search(piece)):
            last_val = float(m.group(1))
            log.info("  --- validation loss %s ---", m.group(1))
        elif cfg.verbose and not piece.startswith((" ", "\x1b")) and "it/s" not in piece:
            log.debug("  %s", piece[:200])

    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        buf += ch
        if ch in ("\n", "\r"):
            handle_piece(buf)
            buf = ""
    if buf:
        handle_piece(buf)
    proc.wait()
    stop.set()
    logfile.close()
    pbar.n = cfg.iters
    pbar.set_postfix({"final_loss": "n/a",
                      "val_loss": f"{last_val:.4f}" if last_val == last_val else "n/a"})
    pbar.refresh()
    pbar.close()

    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("Expert [%s] FAILED (rc=%d) after %.1f min - see %s",
                  label, proc.returncode, elapsed / 60, logfile.name)
        sys.exit(proc.returncode)

    adapters = sorted(adapter_path.glob("*.safetensors"))
    size_mb = sum(p.stat().st_size for p in adapters) / 1e6
    log.info(">>> Expert [%s] DONE in %.1f min | last iter %d | val_loss=%.4f | "
             "adapter=%s (%.1f MB, %d files)",
             label, elapsed / 60, last_iter, last_val, adapter_path, size_mb, len(adapters))
