#!/usr/bin/env python3
"""Publish Lotus Mini to Hugging Face Hub.

Uploads the trained LoRA adapters + sentiment router + model card to a new
(repository on the Hub, so others can pull the model and router artifacts from
HF instead of the git repo.

Prereqs:
  * an HF token: `huggingface-cli login` OR export HF_TOKEN=...
  * the router artifacts downloaded (`uv run scripts/download_router.py`)

Usage:
  uv run scripts/upload_hf.py --repo-id IndigoAtlas25/lotus-mini
  uv run scripts/upload_hf.py --repo-id IndigoAtlas25/lotus-mini --private
"""

import argparse
import logging
import os

from huggingface_hub import HfApi, create_repo, upload_folder

from lotus.constants import EXPERTS_DIR, REPO_ROOT, ROUTER_DIR
from lotus.trainer import setup_logging

log = logging.getLogger("upload_hf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="IndigoAtlas25/lotus-mini",
                        help="target HF repo id (e.g. <user>/lotus-mini)")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="create a private repo")
    parser.add_argument("--exclude-model-card", action="store_true",
                        help="skip uploading hf_model_card.md as README.md")
    args = parser.parse_args()
    setup_logging(False)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        log.error("No HF token found. Run `huggingface-cli login` or set HF_TOKEN.")
        raise SystemExit(1)

    if not (ROUTER_DIR / "model.onnx").exists():
        log.error("Router artifacts missing. Run `uv run scripts/download_router.py` first.")
        raise SystemExit(1)

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)
    log.info("Repo ready: %s", args.repo_id)

    upload_folder(
        repo_id=args.repo_id, folder_path=str(EXPERTS_DIR), path_in_repo="experts",
        token=token,
    )
    log.info("Uploaded %s -> %s/experts", EXPERTS_DIR, args.repo_id)

    upload_folder(
        repo_id=args.repo_id, folder_path=str(ROUTER_DIR), path_in_repo="router",
        token=token, ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    log.info("Uploaded %s -> %s/router", ROUTER_DIR, args.repo_id)

    if not args.exclude_model_card:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(REPO_ROOT / "hf_model_card.md"),
            path_in_repo="README.md", repo_id=args.repo_id,
        )
        log.info("Uploaded model card -> %s/README.md", args.repo_id)

    log.info("Done. View it at https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
