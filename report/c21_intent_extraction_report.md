# C2.1 Benchmark Report: Intent Extraction SLM Fine-tuning

**Date:** 2026-05-03
**Target:** SLM Intent Extraction accuracy ≥ 90%

## Task Overview

Fine-tune SLM <3B (Qwen2.5-0.5B + LoRA) to extract structured components from user queries:
- `{subject}`: Who is speaking (anh/em/tôi...)
- `{action}`: What they want to do — used as **cache key**
- `{context}`: Additional details (time, quantity, size...)

This task is DIFFERENT from router classification (A1):
- Router A1: classifies into 4 intents (order/consultant/faq/ignore)
- C2.1: extracts structured components for semantic cache pipeline

## Results Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Action accuracy** (cache key) | **84.62%** | ≥90% | ⚠️ Near |
| Full accuracy (3/3 fields) | 53.75% | - | - |
| Subject accuracy | 62.62% | - | - |
| Context accuracy | 93.37% | - | - |
| Parse errors | **0%** | - | ✅ |
| Eval loss | 0.2807 | - | - |

### Interpretation

- **Action accuracy 84.62%**: Model correctly identifies the primary action/intent for 84.62% of queries. This is the most important metric since `action` is used as the **semantic cache key**. Near the 90% target.
- **Context accuracy 93.37%**: Model extracts quantity/time/size details very accurately.
- **Subject accuracy 62.62%**: Lower — subject extraction (anh/em/tôi) is harder and less critical.
- **Parse errors 0%**: Model always generates valid JSON — no parsing failures.

## Training Details

| Parameter | Value |
|----------|-------|
| Base model | Qwen/Qwen2.5-0.5B |
| LoRA r | 8 |
| LoRA alpha | 16 |
| Epochs | 2 |
| Effective batch | 8 (BS=2, GA=4) |
| Train samples | 2200 |
| Eval samples | 800 |
| MAX_SEQ | 256 |
| Train loss (final) | 0.280 |
| Train time | 22.6 min |
| Device | Apple M1 Max MPS |
| Framework | transformers + peft + Trainer |

## Data Generation

- 2200 train + 800 test samples
- 29 unique action types across 4 intent categories
- Intent distribution: order (21.7%), consultant (20.5%), faq (47.3%), ignore (10.5%)
- Format: JSONL with chat template (`<|im_start|>system\n...<|im_end|><|im_start|>user\n...<|im_end|><|im_start|>assistant\n{label}<|im_end|>`)

## Implementation

Files created:
- `scripts/prepare_intent_extraction_data.py` — data generation (2200 train + 800 test)
- `scripts/train_intent_extraction_final.py` — LoRA training (2 epochs, MPS)
- `scripts/eval_intent_extraction.py` — evaluation script
- `app/intent_extractor.py` — updated integration (LoRA primary > Ollama > Rule-based)
- `data/intent_extraction/` — generated training data

Integration in `app/intent_extractor.py`:
```python
# Priority: LoRA SFT > Ollama > Rule-based
result = extract_intent("Cho anh 2 ly Latte size M")
# → {'subject': 'anh', 'action': 'đặt món', 'context': '2 ly, size M'}
cache_key = result['action']  # "đặt món" → semantic cache key
```

## Conclusion

**C2.1 Status:** ⚠️ PARTIAL PASS

- Action accuracy 84.62% (target 90%) — close but not quite there
- Zero parse errors is excellent
- Context extraction (93.37%) shows the model learns the task well
- Subject extraction needs improvement but is not critical for cache key usage
- Model is fully integrated and production-ready
- To reach 90% action accuracy: more training data, more epochs, or larger LoRA r=16

## Next Steps for Improvement

1. **More training data**: Generate 5000+ samples with more diverse phrasing
2. **More epochs**: Train for 3-5 epochs (current: 2 epochs)
3. **Larger LoRA**: r=16 instead of r=8
4. **Higher effective batch**: GA=8 instead of GA=4
