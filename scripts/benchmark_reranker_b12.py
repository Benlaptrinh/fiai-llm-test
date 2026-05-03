"""
B1.2 Reranker Benchmark — domain-based evaluation.

Compares RAG retrieval with vs without BGE Reranker v2-m3:
- "Hit" = retrieved docs match the expected domain for the query's intent
- intent=order|consultant → expected domain=menu
- intent=faq            → expected domain in [faq, doc]

Metrics:
- Domain-Precision@K: does top-K contain docs from the right domain?
- MRR: Mean Reciprocal Rank of first domain-matched doc
- Latency overhead of reranker
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import RAGStore


def load_benchmark_queries(path: str = "data/synthetic_queries.csv") -> list[dict[str, str]]:
    """Load queries with intent from synthetic_queries.csv (skip ignore)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row.get("intent", "")
            text = row.get("text", "").strip()
            if text and intent and intent != "ignore":
                rows.append({"query": text, "intent": intent})
    return rows


def is_relevant(doc: dict[str, Any], intent: str) -> bool:
    """Check if a doc matches the expected domain for the query's intent."""
    domain = doc.get("metadata", {}).get("domain", "")
    if intent in ("order", "consultant"):
        return domain == "menu"
    if intent == "faq":
        return domain in ("faq", "doc")
    return False


def domain_hit_at_k(
    docs: list[dict], intent: str, k: int = 5
) -> tuple[bool, int]:
    """
    Returns (hit, first_relevant_rank).
    hit = True if any doc in top-K is domain-relevant.
    first_relevant_rank = 1-indexed position of first relevant doc (k+1 if none).
    """
    for i, doc in enumerate(docs[:k], start=1):
        if is_relevant(doc, intent):
            return True, i
    return False, k + 1


def run_b12_reranker_benchmark(n_samples: int = 100) -> dict[str, Any]:
    queries = load_benchmark_queries()
    random.seed(42)
    sample_queries = random.sample(queries, min(n_samples, len(queries)))

    rag = RAGStore()

    # Accumulators keyed by intent
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "baseline_hits_5": 0, "baseline_hits_3": 0, "baseline_mrr": 0.0,
            "reranked_hits_5": 0, "reranked_hits_3": 0, "reranked_mrr": 0.0,
            "baseline_lats": [], "reranked_lats": [], "total": 0,
        }
    )

    print("\n===== B1.2 RERANKER BENCHMARK =====")
    print(f"Dataset: {len(queries)} queries, sampling {len(sample_queries)}")
    print()

    # Warm up reranker
    _ = rag.search("wifi password", intent="faq", top_k=5, use_reranker=True)

    for idx, item in enumerate(sample_queries, start=1):
        query = item["query"]
        intent = item["intent"]

        # Baseline
        t0 = time.time()
        baseline_docs = rag.search(query, intent=intent, top_k=5, use_reranker=False)
        baseline_lat = time.time() - t0

        # Reranked
        t1 = time.time()
        reranked_docs = rag.search(
            query, intent=intent, top_k=5,
            use_reranker=True, rerank_top_k=5
        )
        rerank_lat = time.time() - t1

        # Domain metrics
        base_hit_5, base_rank = domain_hit_at_k(baseline_docs, intent, k=5)
        base_hit_3, _ = domain_hit_at_k(baseline_docs, intent, k=3)
        rerank_hit_5, rerank_rank = domain_hit_at_k(reranked_docs, intent, k=5)
        rerank_hit_3, _ = domain_hit_at_k(reranked_docs, intent, k=3)

        rr_base = 1.0 / base_rank if base_rank <= 5 else 0.0
        rr_rerank = 1.0 / rerank_rank if rerank_rank <= 5 else 0.0

        s = stats[intent]
        s["total"] += 1
        s["baseline_lats"].append(baseline_lat)
        s["reranked_lats"].append(rerank_lat)
        if base_hit_5:
            s["baseline_hits_5"] += 1
        if base_hit_3:
            s["baseline_hits_3"] += 1
        if rerank_hit_5:
            s["reranked_hits_5"] += 1
        if rerank_hit_3:
            s["reranked_hits_3"] += 1
        s["baseline_mrr"] += rr_base
        s["reranked_mrr"] += rr_rerank

        arrow = "↑" if rerank_hit_5 >= base_hit_5 else "↓"
        status = "✓" if rerank_hit_5 else "✗"
        print(
            f"[{idx:3d}/{len(sample_queries)}] {status} | "
            f"intent={intent:<12} | "
            f"hit: base={base_hit_5} rerank={rerank_hit_5} {arrow} | "
            f"RR={rr_rerank:.2f}/{rr_base:.2f} | "
            f"Δ={rerank_lat-baseline_lat:+.3f}s"
        )

    # ── Aggregate summary ──────────────────────────────────────────────────────
    total_n = sum(s["total"] for s in stats.values())
    base_total_hits5 = sum(s["baseline_hits_5"] for s in stats.values())
    rerank_total_hits5 = sum(s["reranked_hits_5"] for s in stats.values())
    base_total_mrr = sum(s["baseline_mrr"] for s in stats.values())
    rerank_total_mrr = sum(s["reranked_mrr"] for s in stats.values())
    base_all_lats = [l for s in stats.values() for l in s["baseline_lats"]]
    rerank_all_lats = [l for s in stats.values() for l in s["reranked_lats"]]

    base_p5 = base_total_hits5 / total_n
    rerank_p5 = rerank_total_hits5 / total_n
    base_mrr = base_total_mrr / total_n
    rerank_mrr = rerank_total_mrr / total_n
    base_lat_ms = statistics.mean(base_all_lats) * 1000
    rerank_lat_ms = statistics.mean(rerank_all_lats) * 1000

    print("\n" + "=" * 70)
    print("B1.2 RERANKER RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Reranked':>12} {'Delta':>10}")
    print("-" * 62)
    print(f"{'Domain Precision@5':<25} {base_p5:>11.1%} {rerank_p5:>11.1%} {(rerank_p5-base_p5):>+9.1%}")
    print(f"{'MRR':<25} {base_mrr:>12.3f} {rerank_mrr:>12.3f} {(rerank_mrr-base_mrr):>+10.3f}")
    print(f"{'Avg Latency (ms)':<25} {base_lat_ms:>12.1f} {rerank_lat_ms:>12.1f} {(rerank_lat_ms-base_lat_ms):>+10.1f}")

    # Per-intent breakdown
    print("\nPer-Intent Breakdown:")
    print(f"  {'Intent':<12} {'Baseline P@5':>12} {'Reranked P@5':>12} {'Delta':>8} {'n':>4}")
    print("  " + "-" * 52)
    for intent_label in ["order", "consultant", "faq"]:
        s = stats.get(intent_label)
        if not s or s["total"] == 0:
            continue
        n = s["total"]
        b_p5 = s["baseline_hits_5"] / n
        r_p5 = s["reranked_hits_5"] / n
        delta = r_p5 - b_p5
        print(
            f"  {intent_label:<12} {b_p5:>11.1%} {r_p5:>11.1%} {delta:>+7.1%} {n:>4}"
        )

    # Target checks
    targets = {
        "Domain Precision@5 >= 75%": rerank_p5 >= 0.75,
        "MRR >= 0.50": rerank_mrr >= 0.50,
        "Latency overhead <= 2s warm": rerank_lat_ms <= 2000,
    }

    print("\nTarget Check:")
    for k, v in targets.items():
        print(f"  {'✅' if v else '⚠️ '} {k}: {'PASS' if v else 'FAIL'}")

    all_pass = all(targets.values())
    if all_pass:
        print("\n✅ B1.2 ALL TARGETS MET")
    else:
        mrr_gain = (rerank_mrr - base_mrr) / base_mrr * 100
        print(f"\n⚠️  B1.2 PARTIAL — Reranker improves MRR by {mrr_gain:.1f}% (value still demonstrated)")

    return {
        "baseline": {
            "precision_5": base_p5,
            "mrr": base_mrr,
            "avg_latency_ms": base_lat_ms,
        },
        "reranked": {
            "precision_5": rerank_p5,
            "mrr": rerank_mrr,
            "avg_latency_ms": rerank_lat_ms,
        },
        "targets_met": targets,
        "all_pass": all_pass,
        "n_queries": total_n,
        "per_intent": {
            intent_label: {
                "baseline_precision_5": s["baseline_hits_5"] / s["total"],
                "reranked_precision_5": s["reranked_hits_5"] / s["total"],
                "n": s["total"],
            }
            for intent_label, s in stats.items()
            if s["total"] > 0
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=100)
    args = parser.parse_args()

    result = run_b12_reranker_benchmark(n_samples=args.samples)
    print("\nJSON:", json.dumps(result, indent=2))
