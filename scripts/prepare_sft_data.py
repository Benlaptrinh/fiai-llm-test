"""
Prepare SFT training data for Router from synthetic_queries.csv.

Converts the labeled dataset into Qwen chat template format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output: data/router_sft_train.jsonl + data/router_sft_test.jsonl
"""

import csv
import json
import random
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "synthetic_queries.csv"
OUTPUT_TRAIN = DATA_DIR / "router_sft_train.jsonl"
OUTPUT_TEST = DATA_DIR / "router_sft_test.jsonl"

# Intent labels
INTENT_TO_IDX = {"order": 0, "consultant": 1, "faq": 2, "ignore": 3}
IDX_TO_INTENT = {v: k for k, v in INTENT_TO_IDX.items()}

# System prompt for router
SYSTEM_PROMPT = """You are an intent classifier for an F&B assistant (Highlands Coffee).

Classify the user query into exactly ONE of the following intents:
- order: user wants to order, buy, add, or remove menu items
- consultant: user asks for recommendations, suggestions, or preference-based advice
- faq: user asks about store information (wifi, opening hours, payment, delivery, seating, invoice, birthday)
- ignore: greeting, noise, unclear, or out-of-domain small talk

Return your answer as a JSON object with the "action" field only."""


def build_sample(text: str, intent: str) -> dict:
    """Build a single SFT sample in Qwen chat format."""
    user_msg = text.strip()
    assistant_msg = json.dumps({"action": intent})
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def load_csv(path: Path):
    """Load and parse synthetic_queries.csv."""
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row["intent"].strip()
            if intent not in INTENT_TO_IDX:
                continue
            samples.append({
                "text": row["text"].strip(),
                "intent": intent,
                "is_noise": row.get("is_noise", "").strip().lower() == "true",
                "is_hard": row.get("is_hard", "").strip().lower() == "true",
                "language": row.get("language", "vi").strip(),
            })
    return samples


def stratified_split(samples: list, test_ratio: float = 0.2, seed: int = 42):
    """Split into train/test maintaining intent distribution."""
    random.seed(seed)
    buckets = {k: [] for k in INTENT_TO_IDX}
    for s in samples:
        buckets[s["intent"]].append(s)

    train, test = [], []
    for intent, items in buckets.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def write_jsonl(samples: list, path: Path):
    """Write samples as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(build_sample(s["text"], s["intent"]), ensure_ascii=False) + "\n")


def main():
    print("Loading data...")
    samples = load_csv(INPUT_FILE)
    print(f"Total samples: {len(samples)}")

    # Intent distribution
    for intent in INTENT_TO_IDX:
        count = sum(1 for s in samples if s["intent"] == intent)
        print(f"  {intent}: {count} ({count/len(samples)*100:.1f}%)")

    # Hard samples
    hard = [s for s in samples if s["is_hard"]]
    print(f"  hard samples: {len(hard)}")

    train, test = stratified_split(samples, test_ratio=0.2)
    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    write_jsonl(train, OUTPUT_TRAIN)
    write_jsonl(test, OUTPUT_TEST)
    print(f"\nSaved:")
    print(f"  Train: {OUTPUT_TRAIN} ({len(train)} samples)")
    print(f"  Test:  {OUTPUT_TEST} ({len(test)} samples)")

    # Show examples
    print("\nSample train examples:")
    for i, s in enumerate(train[:4]):
        print(f"  [{i+1}] {s['intent']:12s} | {s['text'][:60]}")


if __name__ == "__main__":
    main()
