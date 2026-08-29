#!/usr/bin/env python3
"""Benchmark eval - IFEval with more samples, BBH, MMLU-PRO only."""
import json, re, time, logging
from pathlib import Path
import mlx_lm
from datasets import load_dataset

log = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"
MODEL = "mlx-community/gemma-3-4b-it-4bit"

def extract(resp):
    m = re.search(r'\\boxed\{([^}]+)\}', resp)
    if m: return m.group(1).strip()
    if re.search(r'\bTrue\b', resp, re.I): return "True"
    if re.search(r'\bFalse\b', resp, re.I): return "False"
    m = re.search(r'\(([A-Z])\)', resp)
    if m: return m.group(1)
    for line in reversed(resp.strip().split('\n')):
        l = line.strip()
        if l in ("True","False"): return l
        m = re.search(r'\b([A-Z])\b\s*$', l)
        if m: return m.group(1)
    return resp.strip().split('\n')[-1].strip()[:5]

print("Loading model...")
model, tokenizer = mlx_lm.load(MODEL)
results = {}

# IFEval - 100 samples
print("IFEval (100 samples)...")
try:
    ds = load_dataset("google/IFEval", split="train")
    correct = total = 0
    for i, ex in enumerate(ds):
        if i >= 100: break
        prompt = ex.get("prompt","")
        if not prompt: continue
        resp = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256)
        passed = bool(resp.strip())
        kw = str(ex.get("keyphrase","")).lower()
        if "no_english" in kw: passed = not bool(re.search(r'[a-zA-Z]{3,}', resp))
        if "word_count" in kw: passed = len(resp.split()) >= 10
        if "bullet" in kw: passed = bool(re.search(r'[-*•]', resp))
        if passed: correct += 1
        total += 1
        if total % 25 == 0: print(f"  {total}/100 done, {correct/total*100:.0f}%")
    results["IFEval"] = round(correct/total*100,1) if total else 0
    print(f"  IFEval: {results['IFEval']}% ({correct}/{total})")
except Exception as e:
    print(f"  IFEval failed: {e}")
    results["IFEval"] = 0

# BBH (3 tasks, 20 samples each)
print("BBH...")
try:
    bbh_correct = bbh_total = 0
    for task in ["boolean_expressions","causal_judgement","hyperbaton"]:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        c = t = 0
        for i, ex in enumerate(ds):
            if i >= 20: break
            prompt = f"Answer the following question. Give only the answer.\n\n{ex['input']}\n\nAnswer:"
            resp = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)
            if extract(resp).lower() == ex["target"].lower(): c += 1
            t += 1
        bbh_correct += c; bbh_total += t
        print(f"    {task}: {c}/{t}")
    results["BBH"] = round(bbh_correct/bbh_total*100,1) if bbh_total else 0
    print(f"  BBH: {results['BBH']}% ({bbh_correct}/{bbh_total})")
except Exception as e:
    print(f"  BBH failed: {e}")
    results["BBH"] = 0

# MMLU-PRO (20 samples)
print("MMLU-PRO...")
try:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    labels = list("ABCDEFGHIJ")
    correct = total = 0
    for i, ex in enumerate(ds):
        if i >= 20: break
        q = ex["question"]; opts = ex["options"]
        idx = ex.get("answer_index", -1)
        try: idx = int(idx)
        except: idx = ord(ex.get("answer","A")) - ord("A")
        if idx < 0 or idx >= len(opts): continue
        choices = "\n".join(f"({labels[j]}) {o}" for j,o in enumerate(opts))
        resp = mlx_lm.generate(model, tokenizer, prompt=f"{q}\n\n{choices}\n\nAnswer:", max_tokens=50)
        if extract(resp).upper() == labels[idx]: correct += 1
        total += 1
    results["MMLU-PRO"] = round(correct/total*100,1) if total else 0
    print(f"  MMLU-PRO: {results['MMLU-PRO']}% ({correct}/{total})")
except Exception as e:
    print(f"  MMLU-PRO failed: {e}")
    results["MMLU-PRO"] = 0

avg = round(sum(results.values())/len(results),1)
print(f"\nResults: {json.dumps(results, indent=2)}")
print(f"Average: {avg}%")

out = {"model": MODEL, "benchmarks": results, "avg": avg}
with open(RESULTS_DIR / "lotus_standard_benchmarks.json","w") as f:
    json.dump(out, f, indent=2)
print(f"Saved to {RESULTS_DIR}/lotus_standard_benchmarks.json")
