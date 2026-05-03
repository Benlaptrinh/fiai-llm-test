"""
Central configuration for the application.

This module defines:
- embedding model
- vector database path
- LLM provider (Ollama)
- session parameters

Designed to be easily swapped for production deployment.
"""

import os

APP_NAME = "FI-AI Multi-Agent LLM Test"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# SLM router
SLM_ROUTER_ENABLED = os.getenv("SLM_ROUTER_ENABLED", "false").lower() == "true"
SLM_ROUTER_MODEL = os.getenv("SLM_ROUTER_MODEL", "qwen2.5:1.5b")
SLM_ROUTER_TIMEOUT_SECONDS = int(os.getenv("SLM_ROUTER_TIMEOUT_SECONDS", "20"))
SLM_ROUTER_CONFIDENCE_THRESHOLD = float(
    os.getenv("SLM_ROUTER_CONFIDENCE_THRESHOLD", "0.0")
)

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8001/v1")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Graph RAG / Neo4j
ENABLE_GRAPH_RAG = os.getenv("ENABLE_GRAPH_RAG", "true").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Redis cache
ENABLE_REDIS_CACHE = os.getenv("ENABLE_REDIS_CACHE", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Semantic cache (embedding-based paraphrase matching)
# Lower threshold = more matches = higher hit rate
# Target: ≥60% cache hit rate on F&B queries
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
# A2.2: Auto-summarize when conversation exceeds this many turns
# ~15 turns ≈ 750 tokens for Vietnamese, triggers before context window fills
SESSION_SUMMARY_THRESHOLD_TOKENS = int(os.getenv("SESSION_SUMMARY_THRESHOLD_TOKENS", "15"))

MAX_CONCURRENT_LLM_REQUESTS = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "2"))
QUEUE_TIMEOUT_SECONDS = int(os.getenv("QUEUE_TIMEOUT_SECONDS", "60"))
