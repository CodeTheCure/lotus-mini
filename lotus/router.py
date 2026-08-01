import logging
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .constants import ROUTER_DIR, ROUTER_FILES, ROUTER_HF_REPO

log = logging.getLogger("lotus.router")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def ensure_downloaded(dest_dir: Path | None = None, force: bool = False) -> Path:
    dest_dir = Path(dest_dir or ROUTER_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in ROUTER_FILES if not (dest_dir / f).exists()]
    if not missing and not force:
        log.info("sentiment router artifacts already present in %s", dest_dir)
        return dest_dir

    from huggingface_hub import hf_hub_download

    for local_name, hf_path in ROUTER_FILES.items():
        target = dest_dir / local_name
        if target.exists() and not force:
            continue
        log.info("downloading %s/%s -> %s", ROUTER_HF_REPO, hf_path, target)
        source = hf_hub_download(ROUTER_HF_REPO, hf_path)
        shutil.copy(source, target)
    log.info("sentiment router artifacts ready in %s", dest_dir)
    return dest_dir


class SentimentRouter:

    def __init__(self, model_path: Path | None = None,
                 tokenizer_path: Path | None = None):
        if model_path is None or tokenizer_path is None:
            ensure_downloaded()
        model_path = model_path or ROUTER_DIR / "model.onnx"
        tokenizer_path = tokenizer_path or ROUTER_DIR / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = tuple(s.name for s in self.session.get_inputs())

    def score(self, text: str) -> float:
        enc = self.tokenizer.encode(text)
        ids = np.array(enc.ids, dtype=np.int64)[None, :]
        mask = np.array(enc.attention_mask, dtype=np.int64)[None, :]
        (logits,) = self.session.run(
            None, {self.input_names[0]: ids, self.input_names[1]: mask}
        )
        probs = softmax(logits[0])
        return float(probs[1])

    def label(self, text: str) -> str:
        return "POS" if self.score(text) >= 0.5 else "NEG"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = SentimentRouter()
    for t in [
        "My head hurts so much, I am really worried.",
        "I feel much better now, thank you doctor!",
        "I have a mild cough and a runny nose.",
    ]:
        print(f"{router.score(t):.3f}  {router.label(t):3s}  {t}")
