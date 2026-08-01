#!/usr/bin/env python3

import argparse
import json
import logging
import math
import random

from datasets import load_dataset
from tqdm import tqdm

from lotus.chat import build_messages
from lotus.constants import (DATA_DIR, DATASET, DATASET_CONFIG, DEFAULT_SEED,
                             EXPERT_DIRS, TRAIN_FRAC)
from lotus.router import SentimentRouter
from lotus.trainer import setup_logging

log = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=12000,
                        help="total examples to sample (default 12000)")
    parser.add_argument("--max-per-expert", type=int, default=1000,
                        help="cap examples per expert for balanced training")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true", help="debug logs")
    args = parser.parse_args()
    setup_logging(args.verbose)
    random.seed(args.seed)

    log.info("Loading medical QA dataset %s (config=%s) ...", DATASET, DATASET_CONFIG)
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train")
    rows = random.sample(list(ds), min(args.total, len(ds)))
    log.info("Dataset rows: %d -> subsampled %d", len(ds), len(rows))

    router = SentimentRouter()
    log.info("Sentiment router loaded (DistilBERT SST-2, ONNX)")

    buckets = {"NEG": [], "POS": []}
    dist = {"NEG": 0, "POS": 0}
    for row in tqdm(rows, desc="classifying sentiment", unit="row",
                    ncols=100):
        label = router.label(row.get("input") or row.get("instruction") or "")
        buckets[label].append(build_messages(row))
        dist[label] += 1
    log.info("Sentiment distribution over %d examples: %s", len(rows), dist)

    for label, examples in buckets.items():
        random.shuffle(examples)
        examples = examples[: args.max_per_expert]
        log.info("  [%s] capped at %d examples for balanced experts", label, len(examples))
        n_train = int(math.ceil(len(examples) * TRAIN_FRAC))
        out_dir = DATA_DIR / EXPERT_DIRS[label]
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, split_examples in (("train", examples[:n_train]),
                                      ("valid", examples[n_train:])):
            path = out_dir / f"{split}.jsonl"
            with open(path, "w") as f:
                for ex in split_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            log.info("  [%s] wrote %d examples -> %s", label, len(split_examples), path)

    manifest = {
        "dataset": {"repo": DATASET, "config": DATASET_CONFIG},
        "router": {"repo": "distilbert-base-uncased-finetuned-sst-2-english",
                   "format": "onnx"},
        "sentiment_distribution": dist,
        "expert_dirs": {k: str(DATA_DIR / v) for k, v in EXPERT_DIRS.items()},
    }
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written to %s", DATA_DIR / "manifest.json")
    log.info("Done. %s", dist)


if __name__ == "__main__":
    main()
