"""
Benchmark script for FI-AI Multi-Agent LLM prototype.

Metrics:
- Router Accuracy
- Retrieval Coverage (proxy for hit@k)
- Average Latency
- P95 Latency
- Cache Hit Latency
"""

from __future__ import annotations

import csv
import statistics
import time
from typing import Any, Dict, List, Tuple

import requests

API_URL = "http://localhost:8000/chat"
QUERY_FILE = "data/synthetic_queries.csv"


def percentile(values: List[float], p: float) -> float:
    """Compute percentile without external dependency."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    floor_idx = int(k)
    ceil_idx = min(floor_idx + 1, len(sorted_values) - 1)
    if floor_idx == ceil_idx:
        return sorted_values[floor_idx]
    return sorted_values[floor_idx] + (
        sorted_values[ceil_idx] - sorted_values[floor_idx]
    ) * (k - floor_idx)


def load_queries() -> List[Dict[str, str]]:
    """Load benchmark queries from synthetic_queries.csv."""
    rows = []
    with open(QUERY_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {"query": row["query"], "expected_intent": row["expected_intent"]}
            )
    return rows


def call_chat(
    query: str, session_id: str = "benchmark"
) -> Tuple[Dict[str, Any], float]:
    """Call chat API and measure latency."""
    start = time.time()
    response = requests.post(
        API_URL,
        json={"query": query, "session_id": session_id},
        timeout=180,
    )
    elapsed = time.time() - start
    response.raise_for_status()
    return response.json(), elapsed


def run_benchmark() -> Dict[str, float]:
    """Run benchmark on synthetic queries."""
    rows = load_queries()

    router_correct = 0
    retrieval_covered = 0
    latencies = []

    for idx, row in enumerate(rows, start=1):
        query = row["query"]
        expected = row["expected_intent"]
        result, latency = call_chat(query, session_id="benchmark")

        latencies.append(latency)
        if result["intent"] == expected:
            router_correct += 1

        # Retrieval Coverage (proxy):
        # For non-ignore intents, expect sources to be non-empty.
        # For ignore intent, expect no sources.
        if expected == "ignore":
            if not result.get("sources"):
                retrieval_covered += 1
        elif result.get("sources"):
            retrieval_covered += 1

        print(
            f"[{idx}/{len(rows)}] "
            f"expected={expected} got={result['intent']} "
            f"latency={latency:.3f}s cached={result['cached']}"
        )
        print(f"Answer: {result['answer'][:80]}")

    router_accuracy = router_correct / len(rows)
    retrieval_coverage = retrieval_covered / len(rows)
    avg_latency = statistics.mean(latencies)
    p95_latency = percentile(latencies, 95)

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Total queries: {len(rows)}")
    print(f"Router Accuracy: {router_accuracy:.4f}")
    print(f"Retrieval Coverage: {retrieval_coverage:.4f}")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"P95 Latency: {p95_latency:.4f}s")

    return {
        "router_accuracy": router_accuracy,
        "retrieval_coverage": retrieval_coverage,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
    }


def run_cache_test() -> Dict[str, Any]:
    """Measure cache-hit latency using repeated query."""
    query = "wifi tên gì vậy"

    # Warm-up request
    call_chat(query, session_id="cache-test")

    # Cached request
    result, latency = call_chat(query, session_id="cache-test")

    print("\n===== CACHE TEST =====")
    print(f"Cache Hit: {result['cached']}")
    print(f"Cache Hit Latency: {latency:.4f}s")

    return {"cache_hit": result["cached"], "cache_hit_latency": latency}


if __name__ == "__main__":
    run_benchmark()
    run_cache_test()
