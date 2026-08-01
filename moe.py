#!/usr/bin/env python3

import argparse
import logging
import time

from mlx_lm import generate
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from lotus.chat import apply_chat_template
from lotus.constants import BASE_MODEL
from lotus.expert import ExpertLoader
from lotus.router import SentimentRouter
from lotus.trainer import setup_logging

log = logging.getLogger("moe")

MAX_TOKENS_BY_VERBOSITY = {0: 128, 1: 256, 2: 512}


def respond(router, loader, model, tokenizer, query, verbosity=1,
            temperature=0.6, max_tokens=0, stream=True) -> str:
    t0 = time.time()
    p_pos = router.score(query)
    label = "POS" if p_pos >= 0.5 else "NEG"
    log.info("router: P(POSITIVE)=%.3f  P(NEGATIVE)=%.3f  -> expert [%s]",
             p_pos, 1 - p_pos, label)
    log.info("loading %s expert adapter ...", label)
    loader.load(label)
    log.info("adapter loaded (%.2fs)", time.time() - t0)

    if max_tokens == 0:
        max_tokens = MAX_TOKENS_BY_VERBOSITY.get(verbosity, 256)
    prompt = apply_chat_template(tokenizer, query)

    gen_t0 = time.time()
    sampler = make_sampler(temp=temperature)
    if stream:
        pieces = []
        for resp in stream_generate(model, tokenizer, prompt,
                                    max_tokens=max_tokens, sampler=sampler):
            pieces.append(resp.text)
            print(resp.text, end="", flush=True)
        out = "".join(pieces)
        print()
    else:
        out = generate(model, tokenizer, prompt, max_tokens=max_tokens,
                       sampler=sampler)
    log.info("generation took %.2fs (%d tokens)", time.time() - gen_t0,
             len(tokenizer.encode(out)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="?", help="medical question")
    parser.add_argument("--expert", choices=["NEG", "POS"],
                        help="force an expert (skip router)")
    parser.add_argument("--interactive", action="store_true", help="chat loop")
    parser.add_argument("--compare", action="store_true",
                        help="run base, NEG, and POS experts on the same query")
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2],
                        help="0=short answers, 1=normal, 2=long answers")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="debug logs")
    args = parser.parse_args()
    setup_logging(args.verbose)

    log.info("loading sentiment router ...")
    router = SentimentRouter()
    log.info("loading 4-bit base model %s ... (first run downloads ~3.4 GB)",
             BASE_MODEL)
    model, tokenizer = load(BASE_MODEL)
    log.info("base model ready")
    loader = ExpertLoader(model)

    def run(query: str, expert: str | None):
        if expert is not None:
            loader.load(expert)
            log.info(">>> [forced expert %s] %s", expert, query)
        return respond(router, loader, model, tokenizer, query,
                       args.verbosity, args.temperature, args.max_tokens,
                       not args.no_stream)

    if args.compare:
        log.info("=== BASE (no adapter) ===")
        print(generate(model, tokenizer, apply_chat_template(tokenizer, args.query),
                       max_tokens=args.max_tokens or 256,
                       sampler=make_sampler(temp=args.temperature)))
        for label in ("NEG", "POS"):
            log.info("=== EXPERT %s ===", label)
            loader.load(label)
            print(generate(model, tokenizer, apply_chat_template(tokenizer, args.query),
                           max_tokens=args.max_tokens or 256,
                           sampler=make_sampler(temp=args.temperature)))
        return

    if args.interactive:
        log.info("interactive mode - type a question, Ctrl-D to exit")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            run(q, args.expert)
    elif args.query:
        run(args.query, args.expert)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
