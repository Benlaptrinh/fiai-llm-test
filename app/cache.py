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
        """
        Normalize query text for cache key.

        This includes:
        - lowercase
        - whitespace normalization
        - punctuation removal
        - paraphrase-lite mapping for common repeated intents
        """
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\sÀ-ỹ]", "", normalized)

        synonym_map = {
            # FAQ / wifi
            "wifi tên gì": "wifi",
            "tên wifi": "wifi",
            "cho em xin wifi": "wifi",
            "xin wifi": "wifi",
            "mật khẩu wifi": "wifi",
            "pass wifi": "wifi",
            "password wifi": "wifi",
            # Opening hours
            "mấy giờ mở cửa": "giờ mở cửa",
            "quán mở cửa mấy giờ": "giờ mở cửa",
            "mấy giờ đóng cửa": "giờ đóng cửa",
            "quán đóng cửa mấy giờ": "giờ đóng cửa",
            # Payment
            "thanh toán qr": "thanh toán",
            "thanh toán thẻ": "thanh toán",
            "trả bằng thẻ": "thanh toán",
            "quẹt thẻ": "thanh toán",
            # Recommendation
            "ít đường": "ít ngọt",
            "không ngọt": "ít ngọt",
            "bớt ngọt": "ít ngọt",
            "recommend": "gợi ý",
            "suggest": "gợi ý",
            "tư vấn": "gợi ý",
            "có gì ngon": "gợi ý",
        }

        for phrase, canonical in synonym_map.items():
            normalized = normalized.replace(phrase, canonical)

        # Canonicalize common intent buckets after phrase replacement.
        if "wifi" in normalized:
            normalized = "wifi"
        elif "giờ mở cửa" in normalized:
            normalized = "giờ mở cửa"
        elif "giờ đóng cửa" in normalized:
            normalized = "giờ đóng cửa"
        elif "thanh toán" in normalized:
            normalized = "thanh toán"
        elif "gợi ý" in normalized or "ít ngọt" in normalized:
            normalized = "gợi ý ít ngọt"

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
