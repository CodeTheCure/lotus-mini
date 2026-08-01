#!/usr/bin/env python3

import argparse
import logging

from lotus.router import ensure_downloaded
from lotus.trainer import setup_logging

log = logging.getLogger("download_router")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    setup_logging(args.force)
    dest = ensure_downloaded(force=args.force)
    log.info("Router artifacts ready in %s", dest)


if __name__ == "__main__":
    main()
