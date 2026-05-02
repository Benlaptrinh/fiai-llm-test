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

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8001/v1")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
