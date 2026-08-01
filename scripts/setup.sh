#!/usr/bin/env bash
# Lotus Mini — one-command setup.
#
#   * checks prerequisites (uv, Apple Silicon / MLX)
#   * creates the Python environment and installs dependencies
#   * downloads the sentiment-router ONNX artifacts from Hugging Face
#   * runs a quick smoke test of the router
#
# Usage:  ./scripts/setup.sh   (or:  bash scripts/setup.sh)
set -euo pipefail

BOLD="$(tput bold 2>/dev/null || printf '')"
GREEN="$(tput setaf 2 2>/dev/null || printf '')"
YELLOW="$(tput setaf 3 2>/dev/null || printf '')"
RED="$(tput setaf 1 2>/dev/null || printf '')"
RESET="$(tput sgr0 2>/dev/null || printf '')"

step()  { printf "\n${BOLD}${GREEN}==> %s${RESET}\n" "$*"; }
info()  { printf "${YELLOW}    %s${RESET}\n" "$*"; }
die()   { printf "${RED}ERROR: %s${RESET}\n" "$*" >&2; exit 1; }

# --- 0. prerequisite checks -------------------------------------------------
step "Checking prerequisites"
command -v uv >/dev/null 2>&1 || die "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
info "uv: $(uv --version)"

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  info "WARNING: this project is built on Apple Silicon (MLX). Other platforms may not run the model."
fi

# --- 1. python environment --------------------------------------------------
step "Installing Python dependencies (uv sync)"
uv sync --frozen

# --- 2. sentiment router artifacts ------------------------------------------
step "Downloading sentiment router (ONNX, ~256 MB) from Hugging Face"
uv run python scripts/download_router.py

# --- 3. smoke test ----------------------------------------------------------
step "Smoke-testing the sentiment router"
uv run python scripts/smoke_test.py

printf "\n${BOLD}${GREEN}Setup complete!${RESET}\n"
info "Chat with the model:       uv run moe.py \"my head has hurt for a week\""
info "Longer answers:            uv run moe.py --verbosity 2 \"...\""
info "Compare base vs experts:   uv run moe.py --compare \"...\""
info "Interactive chat:          uv run moe.py --interactive"
info "First inference downloads the ~3.4 GB 4-bit base model, then caches it."
