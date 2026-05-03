"""
Full SFT training với FIX: system prompt rút gọn + tokenization đúng cách.

Fixes:
1. System prompt rút gọn (target < 50 tokens thay vì 144)
2. Tokenization: padding="max_length" trong map, KHÔNG dùng DataCollator pad
3. train_samples sử dụng full response thay vì system prompt trong template
"""

import json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("models/router_sft")
TRAIN_F = Path("data/router_sft_train.jsonl")
TEST_F = Path("data/router_sft_test.jsonl")

BS = 2
GA = 4
EPOCHS = 3
LR = 2e-4
MAX_SEQ = 256
SEED = 42

device = torch.device("mps")

# ── Rút gọn system prompt ──────────────────────────────────────────────────
# Trước: 144 tokens. Sau: < 50 tokens
SYSTEM_PROMPT = """Classify into: order / consultant / faq / ignore. Return JSON only {"action":"..."}."""

INTENTS = ["order", "consultant", "faq", "ignore"]

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="right")
tok.pad_token = tok.eos_token

# ── Model ──────────────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True
)
model.config.pad_token_id = tok.pad_token_id

# ── LoRA ───────────────────────────────────────────────────────────────────
print("Applying LoRA...")
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
))
model.print_trainable_parameters()

# ── Load data ────────────────────────────────────────────────────────────────
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out

print("Loading data...")
train_msgs = load_jsonl(TRAIN_F)
test_msgs = load_jsonl(TEST_F)
print(f"Train: {len(train_msgs)}, Eval: {len(test_msgs)}")

# ── Format: user msg + system prompt, assistant = label ─────────────────────────
def format_sample(example):
    """Format để training: KHÔNG include system message trong chat template.
    Đặt system prompt ngắn TRƯỚC user message nhưng KHÔNG dùng chat template.
    Assistant response = JSON label.
    """
    user_text = example["messages"][1]["content"]
    action = json.loads(example["messages"][2]["content"])["action"]
    assistant_response = f'{{"action":"{action}"}}'

    # Build: <|im_start|>system\n(short prompt)<|im_end|><|im_start|>user\n{query}<|im_end|><|im_start|>assistant\n{response}<|im_end|>
    full_text = (
        f"{tok.bos_token}"
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
        f"<|im_start|>user\n{user_text}<|im_end|>"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    )
    return {"text": full_text}

def tokenize_fn(example):
    """Tokenize, pad to MAX_SEQ, set labels = input_ids (causal LM)."""
    result = tok(
        example["text"],
        max_length=MAX_SEQ,
        truncation=True,
        padding="max_length",  # pad all to MAX_SEQ for fixed tensor shapes
    )
    result["labels"] = result["input_ids"][:]
    return result

print("Formatting & tokenizing...")
train_ds = Dataset.from_list([format_sample(x) for x in train_msgs])
test_ds = Dataset.from_list([format_sample(x) for x in test_msgs])

train_tok = train_ds.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing train")
test_tok = test_ds.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing eval")

# Verify lengths
sample_len = len(train_tok[0]["input_ids"])
max_len = max(len(train_tok[i]["input_ids"]) for i in range(min(5, len(train_tok))))
print(f"Sample length: {sample_len}, Max of first 5: {max_len}, MAX_SEQ: {MAX_SEQ}")

# Check padding
padded_count = sum(1 for i in range(len(train_tok)) if train_tok[i]["input_ids"][-1] == tok.pad_token_id)
print(f"Padded samples: {padded_count}/{len(train_tok)}")

print(f"Train tokens: {len(train_tok)}, Eval tokens: {len(test_tok)}")

# ── DataCollator: KHÔNG pad again (đã pad trong map) ──────────────────────
# Dùng custom collator hoặc DefaultDataCollator để tránh double-padding
from transformers import DefaultDataCollator
collator = DefaultDataCollator(return_tensors="pt")

# ── Train ───────────────────────────────────────────────────────────────────
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
    bf16=False, fp16=False,
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
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=test_tok,
    data_collator=collator,
)

print(f"\nTraining: {EPOCHS} epochs, effective batch={BS*GA}")
print(f"System prompt tokens: ~{len(tok(SYSTEM_PROMPT)['input_ids'])}")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

# Save
trainer.save_model(str(OUT))
tok.save_pretrained(str(OUT))

summary = {
    "model": MODEL, "device": "mps", "dtype": "float16",
    "epochs": EPOCHS, "effective_batch": BS*GA,
    "train_samples": len(train_tok), "eval_samples": len(test_tok),
    "train_time_min": round(elapsed/60, 1),
    "lora_r": 8, "lora_alpha": 16, "max_seq_len": MAX_SEQ,
    "framework": "transformers + peft + MPS",
    "system_prompt_tokens": len(tok(SYSTEM_PROMPT)["input_ids"]),
    "fix": "short system prompt + max_length padding in map",
}
with open(OUT/"training_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary: {summary}")
print("\nDone!")
