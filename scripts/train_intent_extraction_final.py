"""
C2.1 Intent Extraction — Exact replica of router training pattern.

Router (A1.3) successfully trained with:
- device_map={"": device} when loading
- gradient_checkpointing=True
- BS=2, GA=4, 3 epochs
- DefaultDataCollator (no padding, already padded in map)

We replicate this exact pattern for C2.1 intent extraction.
"""

import gc
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("models/intent_extraction")
TRAIN_F = Path("data/intent_extraction/train.jsonl")
TEST_F = Path("data/intent_extraction/test.jsonl")

BS = 2
GA = 4
EPOCHS = 2
LR = 2e-4
MAX_SEQ = 256
SEED = 42

device = torch.device("mps")

SYSTEM_PROMPT = (
    "Bạn là intent extractor F&B. "
    "Trả JSON: {\"subject\":\"...\",\"action\":\"...\",\"context\":\"...\"}. "
    "action = cache key. Trả lời JSON only."
)


def load_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def format_sample(example):
    messages = example["messages"]
    user_text = messages[1]["content"]
    assistant_content = messages[2]["content"]
    return {"text": (
        f"{tok.bos_token}"
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
        f"<|im_start|>user\n{user_text}<|im_end|>"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )}


def tokenize_fn(example):
    result = tok(
        example["text"],
        max_length=MAX_SEQ,
        truncation=True,
        padding="max_length",
    )
    result["labels"] = result["input_ids"][:]
    return result


# ── Model: load directly to MPS via device_map ─────────────────────────────────
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="right")
tok.pad_token = tok.eos_token

print("Loading model (device_map)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True
)
model.config.pad_token_id = tok.pad_token_id

print("Applying LoRA...")
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
))
model.print_trainable_parameters()

# ── Data ───────────────────────────────────────────────────────────────────
print("Loading data...")
train_msgs = load_jsonl(TRAIN_F)
test_msgs = load_jsonl(TEST_F)
print(f"Train: {len(train_msgs)}, Eval: {len(test_msgs)}")

print("Formatting & tokenizing...")
train_ds = Dataset.from_list([format_sample(x) for x in train_msgs])
test_ds = Dataset.from_list([format_sample(x) for x in test_msgs])

train_tok = train_ds.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing train")
test_tok = test_ds.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing eval")

print(f"Train: {len(train_tok)}, Eval: {len(test_tok)}")

collator = DefaultDataCollator(return_tensors="pt")

# ── MPS cache cleaner callback ────────────────────────────────────────────────
class MpsCacheCleaner(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 25 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                gc.collect()


# ── Train: same as router (A1.3) ─────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    gradient_accumulation_steps=GA,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    bf16=False,
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=0.3,
    seed=SEED,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    data_collator=collator,
    callbacks=[MpsCacheCleaner()],
)

print(f"\nTraining: {EPOCHS} epochs, eff batch={BS*GA}")
print(f"System prompt tokens: ~{len(tok(SYSTEM_PROMPT)['input_ids'])}")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
trainer.save_model(str(OUT))
tok.save_pretrained(str(OUT))

# ── Eval ──────────────────────────────────────────────────────────────────────
print("\nEvaluating...")
import re as regex_re

model.eval()
correct = total = 0
action_correct = subject_correct = context_correct = 0

for i in range(len(test_msgs)):
    query = test_msgs[i]["messages"][1]["content"]
    gt_label = json.loads(test_msgs[i]["messages"][2]["content"])

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

    response = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    try:
        match = regex_re.search(r'\{[^}]+\}', response)
        if match:
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
        else:
            total += 1
    except Exception:
        total += 1

n = max(total, 1)
print(f"Full accuracy (3/3): {correct}/{n} = {100*correct/n:.1f}%")
print(f"Action accuracy: {action_correct}/{n} = {100*action_correct/n:.1f}%")
print(f"Subject accuracy: {subject_correct}/{n} = {100*subject_correct/n:.1f}%")
print(f"Context accuracy: {context_correct}/{n} = {100*context_correct/n:.1f}%")

summary = {
    "model": MODEL, "device": "mps", "dtype": "float16",
    "epochs": EPOCHS, "effective_batch": BS * GA,
    "train_samples": len(train_tok), "eval_samples": len(test_tok),
    "train_time_min": round(elapsed / 60, 1),
    "lora_r": 8, "lora_alpha": 16, "max_seq_len": MAX_SEQ,
    "framework": "transformers + peft + Trainer + MPS",
    "task": "C2.1 Intent Extraction (subject + action + context)",
    "results": {
        "full_accuracy": round(correct / n, 4),
        "action_accuracy": round(action_correct / n, 4),
        "subject_accuracy": round(subject_correct / n, 4),
        "context_accuracy": round(context_correct / n, 4),
        "evaluated": total,
    },
}
with open(OUT / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {OUT}")
print("Done!")
