# to run this thing do:  ./scripts/setup.sh  (at least on zsh)
set -euo pipefail

BOLD="$(tput bold 2>/dev/null || printf '')"
GREEN="$(tput setaf 2 2>/dev/null || printf '')"
YELLOW="$(tput setaf 3 2>/dev/null || printf '')"
RED="$(tput setaf 1 2>/dev/null || printf '')"
RESET="$(tput sgr0 2>/dev/null || printf '')"

step()  { printf "\n${BOLD}${GREEN}==> %s${RESET}\n" "$*"; }
info()  { printf "${YELLOW}    %s${RESET}\n" "$*"; }
die()   { printf "${RED}ERROR: %s${RESET}\n" "$*" >&2; exit 1; }


step "Checking for your prerequisites  if you got them installed"
command -v uv >/dev/null 2>&1 || die "uv not found!!!! install it using: curl -LsSf https://astral.sh/uv/install.sh | sh"
info "uv: $(uv --version)"

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  info "WARNING!!!!: Lotus mini can only run for now on Apple Silicon..."
fi


step "now going to install the python dependencies (with uv sync)"
uv sync --frozen


step "Now downloading 256 MB sentiment router from HuggingFace (requires WiFi)"
uv run python scripts/download_router.py


step "Smoke-testing the sentiment router"
uv run python scripts/smoke_test.py

printf "\n${BOLD}${GREEN}Setup complete!${RESET}\n"
info "Chat with the model:       uv run moe.py \"my head has hurt for a week\""
info "Longer answers:            uv run moe.py --verbosity 2 \"...\""
info "Compare base vs experts:   uv run moe.py --compare \"...\""
info "Interactive chat:          uv run moe.py --interactive"
info "First inference downloads the ~3.4 GB 4-bit base model, then caches it."
