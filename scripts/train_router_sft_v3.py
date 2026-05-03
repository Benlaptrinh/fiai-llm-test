"""
Full SFT training với CHECKPOINT/RESUME support (A1.2).

Usage:
  python scripts/train_router_sft_v3.py                        # fresh start
  python scripts/train_router_sft_v3.py --resume               # auto-detect latest checkpoint
  python scripts/train_router_sft_v3.py --resume-from checkpoint-275  # specific checkpoint

A1.2: Checkpoint & Resume
- Saves: adapter weights, optimizer, scheduler, RNG, trainer_state
- Trainer handles resume via resume_from_checkpoint parameter
- Auto-detects latest checkpoint if --resume is passed
"""

import argparse
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
)

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("models/router_sft")
TRAIN_F = Path("data/router_sft_train.jsonl")
TEST_F = Path("data/router_sft_test.jsonl")

BS = 2
GA = 4
DEFAULT_EPOCHS = 3
LR = 2e-4
MAX_SEQ = 256
SEED = 42

device = torch.device("mps")

SYSTEM_PROMPT = (
    "Classify into: order / consultant / faq / ignore. "
    'Return JSON only {"action":"..."}.'
)

INTENTS = ["order", "consultant", "faq", "ignore"]


def load_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def format_sample(example):
    user_text = example["messages"][1]["content"]
    action = json.loads(example["messages"][2]["content"])["action"]
    assistant_response = f'{{"action":"{action}"}}'
    full_text = (
        f"{tok.bos_token}"
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
        f"<|im_start|>user\n{user_text}<|im_end|>"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    )
    return {"text": full_text}


def tokenize_fn(example):
    result = tok(
        example["text"],
        max_length=MAX_SEQ,
        truncation=True,
        padding="max_length",
    )
    result["labels"] = result["input_ids"][:]
    return result


# ── CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in OUTPUT_DIR")
parser.add_argument("--resume-from", type=str, default=None,
                   help="Resume from a specific checkpoint path")
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Number of epochs (default: %(default)s)")
args_cli = parser.parse_args()

# Resolve resume path
resume_path = None
if args_cli.resume_from:
    resume_path = Path(args_cli.resume_from)
elif args_cli.resume:
    if OUT.exists():
        checkpoints = sorted(OUT.glob("checkpoint-*"))
        if checkpoints:
            resume_path = checkpoints[-1]
            print(f"[A1.2] Auto-detected latest checkpoint: {resume_path}")
        else:
            print("[A1.2] Warning: --resume but no checkpoints found. Starting fresh.")
    else:
        print("[A1.2] Warning: OUTPUT_DIR does not exist. Starting fresh.")

# ── Tokenizer & Model ───────────────────────────────────────────────────
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="right")
tok.pad_token = tok.eos_token

print("Loading model...")
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

# ── Data ────────────────────────────────────────────────────────────────
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

# ── Training args ───────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
trainer_args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    gradient_accumulation_steps=GA,
    num_train_epochs=args_cli.epochs,
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
    args=trainer_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    data_collator=collator,
)

# ── Train ────────────────────────────────────────────────────────────
epochs_display = args_cli.epochs
if resume_path:
    # Load trainer state to show resume point
    state_path = resume_path / "trainer_state.json"
    if state_path.exists():
        with open(state_path) as f:
            saved = json.load(f)
        done_epochs = saved.get("epoch", 0)
        done_steps = saved.get("global_step", 0)
        remaining = args_cli.epochs - done_epochs
        print(f"\n[A1.2] Resuming from: {resume_path}")
        print(f"  Epochs done: {done_epochs:.2f}, Steps done: {done_steps}")
        print(f"  Remaining epochs: {remaining:.2f}")
        print(f"  Will train for {args_cli.epochs} total epochs (Trainer continues from checkpoint)")
    else:
        print(f"\n[A1.2] Resuming from: {resume_path}")
else:
    if args_cli.resume:
        print("\n[A1.2] Starting fresh (no checkpoints found).")
    else:
        print(f"\n[A1.2] Starting fresh training ({args_cli.epochs} epochs).")

print(f"\nTraining: {args_cli.epochs} epochs, effective batch={BS*GA}")
print(f"System prompt tokens: ~{len(tok(SYSTEM_PROMPT)['input_ids'])}")

t0 = time.time()

# A1.2: The key line — Trainer handles ALL checkpoint loading automatically
trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

# ── Save ────────────────────────────────────────────────────────────
trainer.save_model(str(OUT))
tok.save_pretrained(str(OUT))

# ── Summary ──────────────────────────────────────────────────────────
total_steps = trainer.state.global_step
summary = {
    "model": MODEL,
    "device": "mps",
    "dtype": "float16",
    "epochs": args_cli.epochs,
    "effective_batch": BS * GA,
    "train_samples": len(train_tok),
    "eval_samples": len(test_tok),
    "train_time_min": round(elapsed / 60, 1),
    "lora_r": 8,
    "lora_alpha": 16,
    "max_seq_len": MAX_SEQ,
    "framework": "transformers + peft + MPS",
    "system_prompt_tokens": len(tok(SYSTEM_PROMPT)["input_ids"]),
    "fix": "short system prompt + max_length padding in map",
    "a1_2_checkpoint_resume": True,
    "total_steps_trained": total_steps,
    "resume_from": str(resume_path) if resume_path else None,
}
with open(OUT / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary: total_steps={total_steps}, time={round(elapsed/60,1)}min")
print("\nDone!")
