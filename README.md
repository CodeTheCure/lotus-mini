# Lotus Mini

A **sentiment-routed medical Mixture-of-Experts (MoE)** — a tiny sentiment
classifier routes each patient query to one of two LoRA **expert** adapters,
all sharing a single **4-bit quantized Gemma 3 4B** base model (**QLoRA**).

```
user prompt
  |--> SentimentRouter  (DistilBERT SST-2 via ONNX, no torch/GPU)
  |        P(POS) >= 0.5  -> POS expert  (grateful / reassured patients)
  |        else            -> NEG expert  (worried / anxious patients)
  v
shared 4-bit Gemma 3 4B (mlx-community/gemma-3-4b-it-4bit)
  + chosen expert's LoRA adapter
  v
medical answer
```

**Why this shape:** QLoRA keeps fine-tuning on a 16 GB Apple-Silicon laptop
(no CUDA needed), and the MoE is parameter-efficient — we never store K full
models, only K tiny LoRA adapters (~14 MB each) sharing one base. The router is
a deliberately tiny HF classifier exported to ONNX, so it adds no torch/GPU.

## Quickstart

Requires macOS with Apple Silicon and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/2028badivi/lotus-mini.git
cd lotus-mini

./scripts/setup.sh            # install deps + fetch router artifacts (~256 MB)
uv run moe.py "I've had chest pain for three days and I'm scared."
```

First inference downloads the ~3.4 GB 4-bit base model from Hugging Face once,
then caches it under `~/.cache/huggingface`.

## Usage

```bash
uv run moe.py "<question>"                        # routed by sentiment
uv run moe.py --expert NEG "just a dry cough"     # force an expert
uv run moe.py --compare "<question>"              # base vs NEG vs POS
uv run moe.py --interactive                       # chat loop
uv run moe.py --verbosity 2 "<question>"          # longer answers
uv run moe.py --verbose "<question>"              # debug logs
```

## Training from scratch

```bash
# 1. Download the medical dataset, route it into POS/NEG expert subsets
uv run prepare_data.py

# 2. QLoRA-train one LoRA adapter per sentiment expert
uv run train.py                         # both experts, ~9 min each on an M4
uv run train.py --experts POS           # or just one
uv run train.py --verbose               # debug-level logs
```

Training streams a live tqdm bar (loss, tok/s, ETA) plus a heartbeat every 20 s,
and writes full logs to `logs/<expert>_train.log`.

## Project layout

```
lotus/                core package
  constants.py        paths + model/dataset config
  router.py           SentimentRouter (ONNX) + auto-download of artifacts
  expert.py           ExpertLoader — hot-swaps LoRA adapters on the shared base
  chat.py             chat-template / dataset formatting helpers
  trainer.py          QLoRA training driver (tqdm + heartbeat streaming)
moe.py                inference CLI (sentiment-routed MoE)
prepare_data.py       dataset download + sentiment stratification
train.py              training CLI (both experts)
scripts/
  setup.sh            one-command setup (deps + router + smoke test)
  download_router.py  fetch the ONNX router artifacts from HF
  smoke_test.py       verify the setup (router-only, or full inference)
  upload_hf.py        publish adapters + router + model card to HF Hub
router/               gitignored: downloaded ONNX artifacts (see setup.sh)
data/                 sentiment-stratified JSONL expert subsets + manifest
experts/              trained LoRA adapters (neg/, pos/) — committed
logs/                 gitignored: per-expert training logs
```

## Publishing

- **GitHub:** this repo (`2028badivi/lotus-mini`).
- **Hugging Face:** the trained adapters + router + model card can be pushed to
  a new Hub repo (e.g. `IndigoAtlas25/lotus-mini`) with:

```bash
uv run scripts/download_router.py    # ensure router artifacts exist
uv run scripts/upload_hf.py --repo-id IndigoAtlas25/lotus-mini
```

See [`hf_model_card.md`](hf_model_card.md) for the model-card copy.

## Notable design details

- **Router is auto-downloaded.** The ~256 MB ONNX model is never committed to
  git; `SentimentRouter` fetches it from Hugging Face on first use, and
  `scripts/setup.sh` pre-fetches it.
- **Expert hot-swap.** All experts live on one in-memory base model; switching
  experts is a ~0.01 s adapter-weight swap, so routing costs nothing at
  inference.
- **No torch.** The router runs on `onnxruntime` (CPU) and the LLM runs on
  Apple's MLX — a full clone does not install PyTorch.
- **Deadlock-free trainer streaming.** The trainer consumes the child process's
  byte stream (including tqdm `\r` redraws) so a pipe-buffer stall can never
  freeze training silently.

## Disclaimer

Research/demo only. Trained on AI-generated medical Q&A data; **not** a
substitute for professional medical advice, diagnosis, or treatment.
