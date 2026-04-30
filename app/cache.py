"""
SimpleCache: reduce repeated LLM calls.

This cache reduces repeated LLM calls for identical or normalized queries.
It is intentionally lightweight for the prototype.

Production extension:
- semantic cache using embeddings
- TTL
- invalidation when menu/FAQ changes
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class SimpleCache:
    """In-memory normalized query cache."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}

    def normalize(self, text: str) -> str:
        """Normalize query text for cache key."""
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\sÀ-ỹ]", "", normalized)
        return normalized

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response by normalized query."""
        key = self.normalize(query)
        return self.store.get(key)

    def set(self, query: str, value: Dict[str, Any]) -> None:
        """Store response by normalized query."""
        key = self.normalize(query)
        self.store[key] = value
