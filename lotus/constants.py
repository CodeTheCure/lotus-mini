from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_DIR = REPO_ROOT / "router"
DATA_DIR = REPO_ROOT / "data"
EXPERTS_DIR = REPO_ROOT / "experts"
LOGS_DIR = REPO_ROOT / "logs"

BASE_MODEL = "mlx-community/gemma-3-4b-it-4bit"

EXPERT_DIRS = {"NEG": "neg", "POS": "pos"}

ROUTER_HF_REPO = "distilbert-base-uncased-finetuned-sst-2-english"
ROUTER_FILES = {
    "model.onnx": "onnx/model.onnx",
    "tokenizer.json": "onnx/tokenizer.json",
    "tokenizer_config.json": "onnx/tokenizer_config.json",
    "special_tokens_map.json": "onnx/special_tokens_map.json",
    "added_tokens.json": "onnx/added_tokens.json",
    "vocab.txt": "onnx/vocab.txt",
    "config.json": "config.json",
}

DATASET = "lavita/medical-qa-datasets"
DATASET_CONFIG = "chatdoctor_healthcaremagic"
TRAIN_FRAC = 0.95
DEFAULT_SEED = 42
