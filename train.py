#!/usr/bin/env python3
"""QLoRA fine-tune the sentiment MoE experts.

Each expert = one LoRA adapter trained on top of the shared 4-bit quantized
Gemma 3 4B base (QLoRA). Trains adapters for the subsets produced by
`uv run prepare_data.py`. Progress is streamed live (tqdm + heartbeat).

Usage:
  uv run train.py                              # train both experts
  uv run train.py --experts POS                # train one expert
  uv run train.py --iters 120 --batch-size 4   # tune hyperparameters
  uv run train.py --verbose                    # debug-level logs
"""

import argparse
import json
import logging
import time

from lotus.constants import DATA_DIR, EXPERTS_DIR, LOGS_DIR
from lotus.trainer import TrainConfig, run_lora_training, setup_logging

log = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8,
                        help="number of transformer layers to attach LoRA to")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-batches", type=int, default=1,
                        help="batches per validation (1 = fast spot-check)")
    parser.add_argument("--experts", nargs="*", default=None,
                        help="which experts to train (default: all)")
    parser.add_argument("--verbose", action="store_true", help="debug logs")
    args = parser.parse_args()
    setup_logging(args.verbose)
    LOGS_DIR.mkdir(exist_ok=True)

    cfg = TrainConfig(
        iters=args.iters, batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length, num_layers=args.num_layers,
        learning_rate=args.learning_rate, val_batches=args.val_batches,
        verbose=args.verbose,
    )

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    total_start = time.time()
    for label, data_dir in manifest["expert_dirs"].items():
        if args.experts and label not in args.experts:
            continue
        run_lora_training(cfg, label, data_dir, experts_dir=EXPERTS_DIR)
    log.info("ALL EXPERTS TRAINED in %.1f min", (time.time() - total_start) / 60)


if __name__ == "__main__":
    main()
