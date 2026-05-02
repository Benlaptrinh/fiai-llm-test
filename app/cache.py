"""
Response cache layer.

Supports:
- Redis cache when available
- in-memory fallback

This keeps local development simple while enabling production-style caching.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import redis

from app.config import CACHE_TTL_SECONDS, ENABLE_REDIS_CACHE, REDIS_URL


class SimpleCache:
    """Cache with Redis backend and in-memory fallback."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None

        if ENABLE_REDIS_CACHE:
            try:
                self.redis_client = redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                )
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

    def backend_name(self) -> str:
        """Expose current active cache backend for diagnostics."""
        if self.redis_client:
            return "redis"
        return "memory"

    def normalize(self, text: str) -> str:
        """Normalize query text for cache key."""
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\sÀ-ỹ]", "", normalized)
        return normalized

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response by normalized query."""
        key = self.normalize(query)

        if self.redis_client:
            raw = self.redis_client.get(key)
            if raw:
                return json.loads(raw)
            return None

        return self.store.get(key)

    def set(self, query: str, value: Dict[str, Any]) -> None:
        """Store response by normalized query."""
        key = self.normalize(query)

        if self.redis_client:
            self.redis_client.setex(
                key,
                CACHE_TTL_SECONDS,
                json.dumps(value, ensure_ascii=False),
            )
            return

        self.store[key] = value
