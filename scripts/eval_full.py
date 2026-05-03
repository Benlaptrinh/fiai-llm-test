"""Full evaluation on test set - saves result to summary."""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

CHECKPOINT = "models/intent_extraction/checkpoint-550"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ = 256
SYSTEM_PROMPT = (
    "Bạn là intent extractor F&B. "
    "Trả JSON: {\"subject\":\"...\",\"action\":\"...\",\"context\":\"...\"}. "
    "action = cache key. Trả lời JSON only."
)
TEST_F = Path("data/intent_extraction/test.jsonl")
OUT_DIR = Path("models/intent_extraction")

device = torch.device("mps")

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok.pad_token = tok.eos_token

print("Loading model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True
)
base.config.pad_token_id = tok.pad_token_id

model = PeftModel.from_pretrained(base, CHECKPOINT)
model.eval()
print("Model ready!")

print("Loading test data...")
with TEST_F.open(encoding="utf-8") as f:
    test_msgs = [json.loads(line) for line in f]
print(f"Test: {len(test_msgs)} samples")

print("Evaluating...")
correct = total = 0
action_correct = subject_correct = context_correct = 0
errors = []

for i, msg in enumerate(test_msgs):
    query = msg["messages"][1]["content"]
    gt_label = json.loads(msg["messages"][2]["content"])

    text = (
        f"{tok.bos_token}"
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
        f"<|im_start|>user\n{query}<|im_end|>"
        f"<|im_start|>assistant\n"
    )
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=60, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    response = tok.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    match = re.search(r'\{[^}]+\}', response)
    if match:
        try:
            pred = json.loads(match.group())
            s_ok = pred.get("subject") == gt_label.get("subject")
            a_ok = pred.get("action") == gt_label.get("action")
            c_ok = pred.get("context") == gt_label.get("context")
            if s_ok and a_ok and c_ok:
                correct += 1
            if a_ok:
                action_correct += 1
            if s_ok:
                subject_correct += 1
            if c_ok:
                context_correct += 1
            total += 1
        except Exception as e:
            errors.append((query, response, str(e)))
            total += 1
    else:
        errors.append((query, response, "NO_JSON"))
        total += 1

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_msgs)} processed...")

n = max(total, 1)
print(f"\n=== EVAL RESULTS ===")
print(f"Full accuracy (3/3): {correct}/{n} = {100*correct/n:.1f}%")
print(f"Action accuracy: {action_correct}/{n} = {100*action_correct/n:.1f}%")
print(f"Subject accuracy: {subject_correct}/{n} = {100*subject_correct/n:.1f}%")
print(f"Context accuracy: {context_correct}/{n} = {100*context_correct/n:.1f}%")
print(f"Parse errors: {len(errors)}/{n}")

# Per-action breakdown
print(f"\n=== SAMPLE ERRORS ===")
for q, r, e in errors[:5]:
    print(f"  Q: {q}")
    print(f"  A: {repr(r[:80])}")
    print(f"  E: {e}")
    print()

# Save summary
summary = {
    "model": MODEL_NAME,
    "device": "mps",
    "dtype": "float16",
    "epochs": 2,
    "effective_batch": 8,
    "train_samples": 2200,
    "eval_samples": len(test_msgs),
    "train_time_min": 22.6,
    "lora_r": 8,
    "lora_alpha": 16,
    "max_seq_len": MAX_SEQ,
    "framework": "transformers + peft + Trainer + MPS",
    "task": "C2.1 Intent Extraction (subject + action + context)",
    "eval_loss": 0.2807,
    "results": {
        "full_accuracy": round(correct / n, 4),
        "action_accuracy": round(action_correct / n, 4),
        "subject_accuracy": round(subject_correct / n, 4),
        "context_accuracy": round(context_correct / n, 4),
        "evaluated": total,
        "parse_errors": len(errors),
    },
}
with open(OUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_DIR / 'training_summary.json'}")
print("Done!")
