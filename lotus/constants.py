"""Central configuration for Lotus Mini."""

from pathlib import Path

# --- paths (relative to the repository root, so scripts work from anywhere) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_DIR = REPO_ROOT / "router"          # gitignored: downloaded ONNX artifacts
DATA_DIR = REPO_ROOT / "data"              # sentiment-stratified JSONL + manifest
EXPERTS_DIR = REPO_ROOT / "experts"        # trained LoRA adapters
LOGS_DIR = REPO_ROOT / "logs"              # gitignored: per-expert training logs

# --- models ---
# 4-bit quantized Gemma 3 4B (Google) in MLX format. This is the "Ollama
# gemma3:4b" weights in a format MLX can train/infer with. Downloaded from
# Hugging Face on first use (~3.4 GB, cached under ~/.cache/huggingface).
BASE_MODEL = "mlx-community/gemma-3-4b-it-4bit"

# label -> expert subdirectory name (under EXPERTS_DIR)
EXPERT_DIRS = {"NEG": "neg", "POS": "pos"}

# --- sentiment router (tiny HF classifier, ONNX, no torch) ---
ROUTER_HF_REPO = "distilbert-base-uncased-finetuned-sst-2-english"
# local file name -> path inside the HF repo
ROUTER_FILES = {
    "model.onnx": "onnx/model.onnx",
    "tokenizer.json": "onnx/tokenizer.json",
    "tokenizer_config.json": "onnx/tokenizer_config.json",
    "special_tokens_map.json": "onnx/special_tokens_map.json",
    "added_tokens.json": "onnx/added_tokens.json",
    "vocab.txt": "onnx/vocab.txt",
    "config.json": "config.json",
}

# --- fine-tuning dataset ---
DATASET = "lavita/medical-qa-datasets"
DATASET_CONFIG = "chatdoctor_healthcaremagic"
TRAIN_FRAC = 0.95
DEFAULT_SEED = 42
