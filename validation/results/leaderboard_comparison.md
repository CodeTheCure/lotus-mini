### Leaderboard Comparison (August 2026)

**Source**: HuggingFace Open LLM Leaderboard v2 (IFEval, BBH, MATH-L5, GPQA, MUSR, MMLU-PRO)

#### Lotus-mini Custom Metrics (Our Benchmark)

| Metric | Score |
|--------|-------|
| Safety (medical) | 64.3% |
| Medical Reasoning | 85.7% |
| Empathy | 71.4% |
| Honesty | 50.0% |
| Tool Calling | 100.0% |
| Router Accuracy | 80.0% |

#### Public Models (Open LLM Leaderboard v2)

| Model | Category | IFEval | BBH | MATH-L5 | GPQA | MUSR | MMLU-PRO | Avg |
|-------|----------|--------|-----|---------|------|------|----------|-----|
| GPT-4o | frontier |  — |  — |  — |  — |  — |  — |  — |
| Claude 3.5 Sonnet | frontier |  — |  — |  — |  — |  — |  — |  — |
| Gemini 1.5 Pro | frontier |  — |  — |  — |  — |  — |  — |  — |
| Llama 3.1 70B | large-open |  — |  — |  — |  — |  — |  — |  — |
| Qwen2.5 72B | large-open |  — |  — |  — |  — |  — |  — |  — |
| DeepSeek V2.5 | large-open |  — |  — |  — |  — |  — |  — |  — |
| Llama 3.1 8B | medium-open |  — |  — |  — |  — |  — |  — |  — |
| Qwen2.5 7B | medium-open |  — |  — |  — |  — |  — |  — |  — |
| Mistral 7B | medium-open |  — |  — |  — |  — |  — |  — |  — |
| Gemma 2 4B | small-open |  — |  — |  — |  — |  — |  — |  — |
| Phi-3 Mini | small-open |  — |  — |  — |  — |  — |  — |  — |
| Llama 3.2 3B | small-open |  — |  — |  — |  — |  — |  — |  — |

#### Key Insight

Lotus-mini's **custom medical metrics** (67.9% avg) are measured against domain-specific rubrics
(safety, medical reasoning, empathy, honesty, tool calling). Public models are measured on
general-purpose benchmarks (MMLU, BBH, MATH). Direct comparison is not apples-to-apples —
our metrics are **task-specific** while theirs are **general knowledge**.
