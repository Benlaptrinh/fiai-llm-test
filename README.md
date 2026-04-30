# FI-AI Multi-Agent LLM Test

## Overview
This repository contains an end-to-end MVP for a multi-agent F&B assistant:

- FastAPI serving layer
- Rule-based Router Agent
- Specialized agents: `order`, `consultant`, `faq`, `ignore`
- RAG pipeline with ChromaDB
- Local LLM inference via Ollama
- Session history + normalized cache
- Benchmark script with latency and routing metrics

## Architecture
User Query -> Router -> Specialized Agent -> RAG Retrieval -> LLM Generation -> Response

Core components:

- `app/router_agent.py`: deterministic intent routing
- `app/agents.py`: domain-specific behavior and prompting
- `app/rag.py`: ChromaDB retrieval wrapper
- `app/main.py`: API orchestration, cache, sessions
- `scripts/ingest.py`: menu/faq/docs ingestion
- `scripts/benchmark.py`: evaluation metrics

## Project Structure
```text
app/
data/
scripts/
report/
.github/workflows/
```

## Environment
Benchmark Environment:

- Date: 2026-04-30
- Machine: MacBook Pro M1 Max, 64GB RAM
- Model: `qwen2.5:7b` (Ollama)
- Backend: local inference

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama:
```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b
```

## Data Pipeline
Generate synthetic dataset:
```bash
python scripts/generate_data.py
```

Generated files:

- `data/menu.csv` (~100 rows)
- `data/faq.csv` (~30 rows)
- `data/docs.txt` (~30 chunks)
- `data/synthetic_queries.csv` (~130 rows)

Ingest into ChromaDB:
```bash
python scripts/ingest.py
```

## Run API
```bash
export OLLAMA_MODEL=qwen2.5:7b
uvicorn app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `POST /chat`

## Benchmark
Run:
```bash
python scripts/benchmark.py
```

Current sample output:

- Router Accuracy: `0.8538`
- Retrieval Coverage: `0.8538`
- Average Latency: `2.8177s`
- P95 Latency: `10.3994s`
- Cache Hit Latency: `0.0029s`

## Metric Notes
Retrieval Coverage is used as a proxy metric for retrieval quality in this prototype.
Production evaluation should use labeled relevance data with `precision@k` / `recall@k`.

## Deployment Note
This prototype runs locally on MacBook M1 Max using Ollama.
GPU-based benchmarking on RTX 3060 was not performed due to hardware constraints.

The architecture is deployment-ready for GPU serving by replacing only the LLM backend with vLLM/SGLang, while keeping router, agents, RAG, session, cache, and API layers unchanged.

## CI
GitHub Actions workflow (`.github/workflows/ci.yml`) validates:

- dependency install
- syntax compile check
- data generation pipeline

## Demo Checklist
Recommended demo flow:

1. `/health`
2. order query
3. consultant query
4. FAQ query
5. ignore query
6. repeat FAQ query (cache hit)
