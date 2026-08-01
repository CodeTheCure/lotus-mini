import json
import logging
from pathlib import Path

from mlx_lm.tuner.utils import linear_to_lora_layers

from .constants import EXPERTS_DIR

log = logging.getLogger("lotus.expert")


class ExpertLoader:

    def __init__(self, model, experts_dir: Path = EXPERTS_DIR,
                 labels: dict | None = None):
        self.model = model
        self.experts_dir = Path(experts_dir)
        self.labels = labels or {"NEG": "neg", "POS": "pos"}
        self._lora_ready = False

    def adapter_path(self, label: str) -> Path:
        return self.experts_dir / self.labels[label]

    def load(self, label: str) -> None:
        path = self.adapter_path(label)
        if not (path / "adapters.safetensors").exists():
            raise FileNotFoundError(
                f"adapter not found: {path}. Run `uv run train.py` (or the "
                f"`{label}` expert) first, or restore the experts from the repo."
            )
        if not self._lora_ready:
            config = json.loads((path / "adapter_config.json").read_text())
            linear_to_lora_layers(
                self.model, config["num_layers"], config["lora_parameters"]
            )
            self._lora_ready = True
        self.model.load_weights(str(path / "adapters.safetensors"), strict=False)
