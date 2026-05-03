"""
Benchmark fine-tuned LoRA router trên test set.

So sánh:
1. Rule-based router (keyword matching)
2. Learned router (TF-IDF + LogReg)
3. Fine-tuned LoRA router (qwen2.5-0.5B)
"""

import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.router_agent import classify_intent as rule_based

# ── Config ───────────────────────────────────────────────────────────────────
LORA_MODEL_DIR = Path("models/router_sft")
TEST_FILE = Path("data/router_sft_test.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

DEVICE = torch.device("mps")
INTENTS = ["order", "consultant", "faq", "ignore"]

# ── Load LoRA model ─────────────────────────────────────────────────────────
print("Loading fine-tuned LoRA model...")
tok = AutoTokenizer.from_pretrained(LORA_MODEL_DIR, trust_remote_code=True)
tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
model.eval()
print("LoRA model loaded.")


# ── Load test data ─────────────────────────────────────────────────────────
print("Loading test data...")
test_samples = []
with open(TEST_FILE) as f:
    for line in f:
        test_samples.append(json.loads(line))
print(f"Test samples: {len(test_samples)}")


# ── LoRA inference ──────────────────────────────────────────────────────────
def lora_classify(query: str) -> str:
    prompt = f"""You are an intent classifier for an F&B assistant. Classify into order/consultant/faq/ignore. Return JSON only {{"action":"..."}}. Query: {query}"""
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        data = json.loads(response.strip())
        action = data.get("action", "")
        if action in INTENTS:
            return action
    except:
        pass
    return "ignore"


# ── Benchmark ────────────────────────────────────────────────────────────────
def benchmark(name: str, classifier_fn, samples: list) -> dict:
    correct = 0
    per_intent = {i: {"correct": 0, "total": 0} for i in INTENTS}

    t0 = time.time()
    for sample in samples:
        query = sample["messages"][1]["content"]  # user message
        true_intent = sample["messages"][2]["content"]  # assistant response
        true_action = json.loads(true_intent).get("action", "")

        pred = classifier_fn(query)
        if pred == true_action:
            correct += 1
            per_intent[true_action]["correct"] += 1
        per_intent[true_action]["total"] += 1

    elapsed = time.time() - t0
    acc = correct / len(samples)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f} ({correct}/{len(samples)})")
    print(f"Total time: {elapsed:.2f}s, avg: {elapsed/len(samples)*1000:.1f}ms/query")
    print(f"\nPer-class:")
    for intent in INTENTS:
        t = per_intent[intent]["total"]
        c = per_intent[intent]["correct"]
        f1 = c / t if t > 0 else 0
        print(f"  {intent:12s}: {c:3d}/{t:3d} = {f1:.4f}")

    return {
        "accuracy": acc,
        "correct": correct,
        "total": len(samples),
        "per_intent": per_intent,
        "avg_ms": elapsed / len(samples) * 1000,
    }


# ── Run benchmarks ──────────────────────────────────────────────────────────
print(f"\nBenchmarking on {len(test_samples)} samples...")

# Rule-based
rule_results = benchmark("Rule-based Router", rule_based, test_samples)

# Learned (TF-IDF + LogReg)  
learned_correct, learned_total = 0, 0
import joblib
if Path("models/router_model.joblib").exists():
    lr_model = joblib.load("models/router_model.joblib")
    def learned_classify(q):
        pred = lr_model.predict([q])[0]
        return pred if pred in INTENTS else "ignore"
    learned_results = benchmark("Learned Router (TF-IDF + LogReg)", learned_classify, test_samples)
else:
    print("\nLearned router model not found, skipping.")
    learned_results = None

# Fine-tuned LoRA
# Use subset for speed (full test = 783 samples, may take a while)
lora_subset = test_samples[:200]
print(f"\nFine-tuned LoRA benchmark on {len(lora_subset)} samples...")
lora_results = benchmark("Fine-tuned LoRA (qwen2.5-0.5B)", lora_classify, lora_subset)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"{'Router':<35} {'Acc':>8} {'Avg ms':>8}")
print(f"{'-'*50}")
print(f"{'Rule-based':<35} {rule_results['accuracy']:>8.4f} {rule_results['avg_ms']:>7.1f}ms")
if learned_results:
    print(f"{'Learned (TF-IDF+LR)':<35} {learned_results['accuracy']:>8.4f} {learned_results['avg_ms']:>7.1f}ms")
print(f"{'Fine-tuned LoRA (subset 200)':<35} {lora_results['accuracy']:>8.4f} {lora_results['avg_ms']:>7.1f}ms")

# Save results
results = {
    "test_samples": len(test_samples),
    "lora_subset_size": len(lora_subset),
    "rule_based": rule_results,
    "learned": learned_results,
    "lora": lora_results,
}
with open("report/router_sft_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to report/router_sft_benchmark.json")
