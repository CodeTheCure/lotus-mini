#!/usr/bin/env python3

import argparse
import logging
import time

from lotus.router import SentimentRouter
from lotus.trainer import setup_logging

log = logging.getLogger("smoke_test")

SAMPLES = [
    "My head hurts so much, I am really worried.",
    "I feel much better now, thank you doctor!",
    "I have a mild cough and a runny nose.",
]


def test_router() -> None:
    t0 = time.time()
    router = SentimentRouter()
    log.info("router loaded in %.2fs", time.time() - t0)
    for text in SAMPLES:
        p = router.score(text)
        log.info("  P(POS)=%.3f -> expert [%s] | %s", p, "POS" if p >= 0.5 else "NEG", text)


def test_model(query: str, expert: str | None) -> None:
    from mlx_lm.utils import load

    from lotus.chat import apply_chat_template
    from lotus.constants import BASE_MODEL
    from lotus.expert import ExpertLoader

    log.info("loading base model %s (first run downloads ~3.4 GB) ...", BASE_MODEL)
    model, tokenizer = load(BASE_MODEL)
    loader = ExpertLoader(model)
    if expert is not None:
        loader.load(expert)
        log.info("forced expert [%s]", expert)
    prompt = apply_chat_template(tokenizer, query)
    t0 = time.time()
    from mlx_lm import generate
    out = generate(model, tokenizer, prompt, max_tokens=100)
    log.info("generation took %.2fs", time.time() - t0)
    print("\n" + out + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true", help="also test inference")
    parser.add_argument("--expert", choices=["NEG", "POS"], default=None)
    parser.add_argument("query", nargs="?", default="I have had a mild fever for two days.")
    args = parser.parse_args()
    setup_logging(False)

    log.info("== smoke test: sentiment router ==")
    test_router()
    if args.with_model:
        log.info("== smoke test: full MoE inference ==")
        test_model(args.query, args.expert)
    log.info("smoke test passed")


if __name__ == "__main__":
    main()
