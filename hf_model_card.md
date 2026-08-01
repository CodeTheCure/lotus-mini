---
language: en
license: gemma
tags:
  - medical
  - qlora
  - lora
  - mixture-of-experts
  - sentiment-routing
  - mlx
  - gemma
  - apple-silicon
pipeline_tag: text-generation
base_model: mlx-community/gemma-3-4b-it-4bit
---

# Lotus Mini

A **sentiment-routed medical Mixture-of-Experts**: a tiny sentiment classifier
routes each patient query to one of two LoRA **expert** adapters, all sharing a
single 4-bit quantized **Gemma 3 4B** base (QLoRA).

```
user prompt
  |--> DistilBERT sentiment classifier (ONNX)
  |      P(POS) >= 0.5 -> POS expert   (grateful/reassured patients)
  |      else          -> NEG expert   (worried/anxious patients)
  v
shared 4-bit Gemma 3 4B + chosen LoRA expert
  v
medical answer
```

## Contents of this repo

| Path | What |
|------|------|
| `experts/neg` | LoRA adapter trained on negative-sentiment patient queries |
| `experts/pos` | LoRA adapter trained on positive-sentiment patient queries |
| `router/` | DistilBERT SST-2 sentiment classifier (ONNX) used as the router |

## How to use (MLX / Apple Silicon)

The adapters are trained on top of `mlx-community/gemma-3-4b-it-4bit`. Load the
base model and attach the routed adapter:

```python
from mlx_lm import load
from mlx_lm.utils import load_adapters

model, tokenizer = load("mlx-community/gemma-3-4b-it-4bit")
load_adapters(model, "lotus-mini/experts/neg")   # or .../experts/pos

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "I've had chest pain for three days and I'm scared."}],
    add_generation_prompt=True,
)
```

Or clone the source repo (has the full pipeline, setup script, and trainer):

```bash
git clone https://github.com/2028badivi/lotus-mini.git && cd lotus-mini
./scripts/setup.sh
uv run moe.py "I've had chest pain for three days and I'm scared."
```

## Training

- **Base:** `mlx-community/gemma-3-4b-it-4bit` (4-bit QLoRA base, ~3.4 GB)
- **Data:** `lavita/medical-qa-datasets` (config `chatdoctor_healthcaremagic`,
  112k doctor-patient Q&A pairs), stratified by the sentiment of the patient's
  description into POS/NEG buckets
- **Adapters:** rank-8 LoRA on 8 of 34 layers, 80 iterations each, ~14 MB each

## Disclaimer

This is a research/demo model fine-tuned from medical Q&A data.
It is **not** a substitute for professional medical advice, diagnosis, or
treatment.
