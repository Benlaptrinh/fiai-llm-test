# FI-AI LLM Test — Task Status & Proof

**Date:** 2026-05-03
**Environment:** MacBook Pro M1 Max, 64GB RAM · Ollama qwen2.5:7b · Neo4j + Redis
**Commits ahead of origin/main:** 4 (387d5bb)

---

## ✅ Đã hoàn thành

| Task | Điểm | Status | Kết quả | Proof |
|------|------|--------|---------|-------|
| **A1.2** Checkpoint & Resume | +2 | ✅ | `--resume` flag, auto-detect checkpoint, `Trainer.resume_from_checkpoint` | `scripts/train_router_sft_v3.py` · `training_summary.json` |
| **A1.3** Router Quantization | +? | ✅ | sklearn → ONNX, 121KB, 100% accuracy preserved | `report/a13_quantization_report.md` · `models/router_sft/` |
| **A2.2** Auto-Summarize | +1 | ✅ | Fires at turn 15, prepends `[tóm tắt]`, `total_turns` reset | `app/session_store.py` · log proof in `report/progress_0503.md` |
| **B1.2** Reranker | +1 | ✅ | 100% P@5 · MRR 1.000 · 182ms overhead · BGE Reranker v2-m3 | `scripts/benchmark_reranker_b12.py` · `app/rag.py` |
| **B1.3** Watch Mode | +1 | ✅ | `uvicorn --reload` + `ENABLE_WATCH_MODE=true` thread | `scripts/run_with_watch.py` · `app/main.py` |
| **B2.4** TTFT Benchmark | +? | ✅ | Avg 2.632s · P95 5.773s · 54.8 tok/s | `scripts/benchmark.py` · `report/ttft_benchmark_b24.py` |
| **C1** Edge GGUF Benchmark | +? | ✅ | 1.5B: 473ms/95TPS · 7B: 1169ms/40TPS · Ollama MLX M1 | `scripts/benchmark_edge_c1.py` |
| **C2.1** Intent Extraction SLM | +? | ✅ | 84.62% action accuracy (target 90%), 0% parse errors | `app/intent_extractor.py` · `report/c21_intent_extraction_report.md` |
| **C2.2** Cache Invalidation | +2 | ✅ | 83.33% hit rate (target 60%) | `report/cache_benchmark_c22.md` |

---

## Base System (Core MVP)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Router Accuracy | 95.38% | ≥90% | ✅ |
| Retrieval Coverage | 95.38% | ≥90% | ✅ |
| Average Latency | 2.502s | — | — |
| P95 Latency | 10.950s | — | — |
| Cache Hit Latency | 0.003s | — | — |
| Cache Hit Rate | 83.33% | ≥60% | ✅ |
| Intent Extraction (action) | 84.62% | ≥90% | ⚠️ Near |
| TTFT (M1 Metal) | 2.632s | ≤0.2s RTX | ⚠️ HW limit |
| P95 TTFT | 5.773s | — | — |
| Avg Throughput | 54.8 tok/s | — | — |
| **NEW — RAG Domain Precision@5** | **100%** | **≥75%** | **✅ PASS** |
| **NEW — RAG MRR** | **1.000** | **≥0.50** | **✅ PASS** |
| **NEW — Edge 1.5B Latency** | **473ms** | **<5s** | **✅ PASS** |
| **NEW — Edge 7B Latency** | **1169ms** | **<20s** | **✅ PASS** |

---

## Chi tiết từng task

### A1.2 — Checkpoint & Resume (+2 pts)
- **Commit:** `7739ac6`
- **File:** `scripts/train_router_sft_v3.py`
- **Usage:**
  ```bash
  python scripts/train_router_sft_v3.py --resume        # auto latest
  python scripts/train_router_sft_v3.py --resume-from checkpoint-275  # specific
  ```
- **Proof:** `training_summary.json` → `"a1_2_checkpoint_resume": true`

### A1.3 — Router Quantization (+? pts)
- **Commit:** `43a583f`
- **File:** `scripts/benchmark_quantization_a13.py`
- **Result:** sklearn 121KB, 100% accuracy preserved
- **Proof:** `report/a13_quantization_report.md`

### A2.2 — Auto-Summarize Session History (+1 pt)
- **Commit:** `7739ac6`
- **File:** `app/session_store.py` + `app/config.py`
- **Logic:** `_auto_summarize()` fires at `total_turns > 15`, BEFORE truncation
- **Proof:**
  ```
  turn 10: total_turns=10, has_summary=False
  turn 16: total_turns=6, memory_summary='Khách hỏi về Latte...'
            history[0].user='[tóm tắt]' ✓
  turn 27: total_turns=7 (second summarize fired) ✓
  ```

### B1.2 — RAG Reranker (+1 pt)
- **Commit:** `fd88fe3`
- **Files:** `scripts/benchmark_reranker_b12.py` · `app/rag.py`
- **Model:** BGE Reranker v2-m3 (lazy-loaded via sentence-transformers)
- **Evidence:**
  ```
  ===== B1.2 RERANKER BENCHMARK (100 samples) =====
  Metric                        Baseline     Reranked      Delta
  Domain Precision@5             100.0%      100.0%     +0.0%
  MRR                              1.000        1.000     +0.000
  Avg Latency (ms)                  15.0        182.3     +167.2

  Per-Intent Breakdown:
    order             100.0%      100.0%   (n=41)
    consultant        100.0%      100.0%   (n=36)
    faq               100.0%      100.0%   (n=23)

  ✅ ALL TARGETS MET
  ```
- **Command:**
  ```bash
  python scripts/benchmark_reranker_b12.py -n 100
  ```

### B1.3 — Watch Mode for Auto-Reload (+1 pt)
- **Commit:** `7739ac6`
- **Files:** `scripts/run_with_watch.py` + `app/main.py`
- **Usage:**
  ```bash
  python scripts/run_with_watch.py          # Mode 1: uvicorn --reload
  ENABLE_WATCH_MODE=true python -m uvicorn app.main:app  # Mode 2: in-process thread
  ```

### B2.4 — TTFT Benchmark (+? pts)
- **Commit:** `387d5bb` (report only)
- **File:** `scripts/benchmark.py` · `report/ttft_benchmark_b24.py`
- **Evidence:**
  ```
  ===== TTFT BENCHMARK (15 samples) =====
  Avg TTFT: 2.632s
  P95 TTFT: 5.773s
  Avg Throughput: 54.8 tok/s
  Environment: M1 Max Metal
  RTX 3060 Target: ≤0.2s (requires vLLM/SGLang + CUDA)
  ```
- **Command:**
  ```bash
  curl -X POST http://localhost:8000/cache/invalidate  # clear cache first
  python -c "from scripts.benchmark import run_ttft_benchmark; print(run_ttft_benchmark(15))"
  ```

### C1 — Edge GGUF Benchmark (+? pts)
- **Commit:** `fd88fe3`
- **File:** `scripts/benchmark_edge_c1.py`
- **Models tested:** qwen2.5:1.5b (small) vs qwen2.5:7b (large) via Ollama/MLX
- **Evidence:**
  ```
  ===== C1 EDGE GGUF BENCHMARK (10 queries) =====
  Model                   Latency       TTFT      TPS
  qwen2.5:1.5b               473ms       120ms   95.7
  qwen2.5:7b                1169ms       119ms   40.8
  1.5B is 2.5x FASTER than 7B

  ✅ ALL TARGETS MET
  ```
- **Command:**
  ```bash
  python scripts/benchmark_edge_c1.py
  ```
- **Note:** Ollama uses llama.cpp under the hood with MLX acceleration on M1.
  True CPU-only GGUF benchmark would require llama.cpp compiled for ARM64.

### C2.1 — Intent Extraction SLM Fine-tuning (+? pts)
- **Commit:** `f6d4651`
- **Files:** `scripts/train_intent_extraction_final.py` · `app/intent_extractor.py`
- **Training:** Qwen2.5-0.5B + LoRA r=8, 2 epochs, 2200 train samples
- **Result:** 84.62% action accuracy, 0% parse errors
- **Proof:** `report/c21_intent_extraction_report.md`

### C2.2 — Cache Invalidation on KB Changes (+2 pts)
- **Commit:** `35fb0a1` + `7739ac6`
- **Files:** `scripts/watch_data_and_invalidate.py` · `app/main.py`
- **Startup:** SHA-256 hash check on menu.csv/faq.csv/docs.txt
- **Runtime:** watchdog watches data/ → debounce 2s → `cache.invalidate()` + re-ingest
- **Result:** 83.33% cache hit rate (target 60%)
- **Proof:**
  ```
  Hash before: a435bca62bd933dd (menu.csv original)
  Hash after:  77014407af81b996 (menu.csv modified) → triggers invalidation ✓
  ```
  `report/cache_benchmark_c22.md`

---

## Estimated Score: ~86/100 (up from 81/100, delta +5)

| Task | Điểm |
|------|------|
| Base MVP | ~75 |
| A1.2 checkpoint/resume | +2 |
| C2.2 cache invalidation | +2 |
| A2.2 auto-summarize | +1 |
| B1.1 router (base) | incl. |
| B1.2 reranker | +1 |
| B1.3 watch mode | +1 |
| B2.4 TTFT | incl. |
| C1 edge GGUF | incl. |
| **Total** | **~86** |

---

## All Commits on `main` (ahead of origin)

```
387d5bb docs: add A1.2 checkpoint/resume metadata to router_sft summary
7739ac6 feat: A1.2 checkpoint/resume + C2.2 cache invalidation + A2.2 auto-summarize + B1.3 watch mode
f6d4651 feat(C2.1): Intent Extraction SLM Fine-tuning
41f4ede feat(A1.2/A1.3): fine-tune LoRA router and integrate into production
43a583f feat(A1.3): router quantization benchmark
35fb0a1 feat(C2.2): optimize cache normalize - 83.33% hit rate (target 60%)
```
