"""Lotus Mini — a sentiment-routed medical Mixture-of-Experts (QLoRA) model.

A tiny sentiment classifier routes each query to one of two LoRA "expert"
adapters, all sharing a single 4-bit quantized Gemma 3 4B base model.
"""

from .constants import BASE_MODEL, EXPERT_DIRS, REPO_ROOT, ROUTER_DIR
from .router import SentimentRouter, ensure_downloaded

__version__ = "0.2.0"

__all__ = [
    "BASE_MODEL",
    "EXPERT_DIRS",
    "REPO_ROOT",
    "ROUTER_DIR",
    "SentimentRouter",
    "ensure_downloaded",
]
