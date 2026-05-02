# FI-AI Multi-Agent LLM Test
End-to-end multi-agent LLM system with RAG, local inference, and production-ready design.

## Overview
This repository contains an end-to-end MVP for a multi-agent F&B assistant:

- FastAPI serving layer
- Rule-based Router Agent
- Specialized agents: `order`, `consultant`, `faq`, `ignore`
- RAG pipeline with ChromaDB
- Local LLM inference via Ollama
- Session history + normalized cache
- Lightweight production guardrail for harmful out-of-scope content
- Benchmark script with latency and routing metrics

## Architecture
User Query -> Router -> Specialized Agent -> RAG Retrieval -> LLM Generation -> Response

**Key Insight:** Separating intent routing from generation reduces unnecessary retrieval calls and improves system interpretability.

```mermaid
flowchart LR
  U["User Query"] --> R["Router Agent (rule-based)"]
  R -->|order| OA["Order Agent"]
  R -->|consultant| CA["Consultant Agent"]
  R -->|faq| FA["FAQ Agent"]
  R -->|ignore| IA["Ignore Agent"]
  OA --> VS["RAGStore (ChromaDB)"]
  CA --> VS
  FA --> VS
  VS --> LLM["Ollama (qwen2.5:7b)"]
  IA --> RESP["FastAPI Response"]
  LLM --> RESP
  RESP --> S["SessionStore"]
  RESP --> C["SimpleCache"]
```

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

Quick curl test:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Wifi tên gì vậy?","session_id":"demo"}'
```

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

## Guardrails And Streaming Notes
- Guardrail blocks unsafe harmful prompts before routing/retrieval/generation.
- Cache uses normalized query keys to reduce duplicate LLM calls and improve latency.
- Retrieval is filtered by intent to improve relevance and reduce noise.
- The system can be extended to support token streaming via SSE in production.

## Demo Evidence
Real API responses for order, consultant, FAQ, ignore, and repeated FAQ cache-hit are saved in:

- `report/demo_chat_examples.md`
- `report/demo_chat_examples_2026-04-30.json`

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
