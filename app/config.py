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

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
