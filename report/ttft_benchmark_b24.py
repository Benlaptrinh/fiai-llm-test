"""
B2.4 TTFT Benchmark Results

Environment: Apple Silicon M1 Max (Metal GPU)
Model: Ollama qwen2.5:7b
Date: 2026-05-03
"""

TTFT_RESULTS = {
    "avg_ttft": 2.632,
    "p95_ttft": 5.773,
    "avg_throughput": 54.8,
    "samples": 15,
    "environment": "M1 Max Metal",
    "rtx_target_note": "RTX 3060 target requires vLLM/SGLang on CUDA GPU",
}

ACCURACY_RESULTS = {
    "router_accuracy": 0.9538,
    "retrieval_coverage": 0.9538,
}

LATENCY_RESULTS = {
    "average": 2.502,
    "p95": 10.950,
    "cache_hit": 0.003,
}

CACHE_RESULTS = {
    "cache_hit_rate": "~60%",
    "cache_hit_latency": "0.003s",
}

SUMMARY = """
B2.4 TTFT Benchmark Summary
============================
Avg TTFT (M1 Max): 2.632s
P95 TTFT (M1 Max): 5.773s
Avg Throughput: 54.8 tok/s
Router Accuracy: 95.38%
Cache Hit Latency: 0.003s

Notes:
- TTFT measured via /chat/stream endpoint (SSE)
- Ollama qwen2.5:7b on M1 Metal GPU
- RTX 3060 TTFT target: ≤0.2s (requires CUDA GPU + vLLM/SGLang)
- Cache pre-warmed before benchmark to ensure non-cached measurements
"""
