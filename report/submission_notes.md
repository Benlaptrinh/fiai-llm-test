# Submission Notes

## Evaluation Notes
Retrieval Coverage is used as a proxy metric for retrieval quality in this prototype.
Production evaluation should use labeled relevance data with precision@k and recall@k.

## Deployment Notes
This prototype is developed and tested on a local MacBook Pro M1 Max (64GB RAM) using Ollama for local inference.

GPU-based benchmarking on RTX 3060 is not performed in this prototype due to hardware constraints.

However, the architecture is designed to support GPU deployment by replacing only the LLM backend with vLLM/SGLang.
Router, multi-agent orchestration, RAG, session, cache, and API layers remain unchanged.

## Benchmark Environment Template
- Date: 2026-04-30
- Machine: MacBook Pro M1 Max, 64GB RAM
- Model: qwen2.5:7b
- Backend: Ollama local inference

## Final System Metrics (Real Run)

- Router Accuracy: 0.9538
- Retrieval Coverage: 0.9538
- Average Latency: 2.5019s
- P95 Latency: 10.9504s
- Cache Hit Latency: 0.0033s

## Runtime Configuration

- Graph RAG: Enabled (Neo4j)
- Cache Backend: Redis (fallback: memory)
- Concurrency Queue: max 2 LLM requests
- SLM Router: supported via qwen2.5:1.5b
