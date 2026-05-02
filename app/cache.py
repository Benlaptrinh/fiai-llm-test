"""
Response cache layer with semantic (embedding-based) lookup.

Supports:
- Redis cache when available
- in-memory fallback
- Semantic similarity: uses multilingual embeddings to match
  paraphrases and semantically equivalent queries
- Similarity threshold: 0.92 (configurable via SEMANTIC_CACHE_THRESHOLD)
- Cache invalidation when knowledge base changes (manual trigger)

This keeps local development simple while enabling production-style caching.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import redis
import requests

from app.config import (
    CACHE_TTL_SECONDS,
    EMBEDDING_MODEL,
    ENABLE_REDIS_CACHE,
    REDIS_URL,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_THRESHOLD,
)


class SentenceEmbedder:
    """
    Lightweight embedder using a local sentence-transformer model.
    Caches the embedding function to avoid re-loading the model.
    Falls back gracefully if the model is unavailable.
    """

    _instance: Optional["SentenceEmbedder"] = None
    _embedding_fn = None
    _model = None

    def __new__(cls) -> "SentenceEmbedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self) -> bool:
        """Lazily load the embedding model."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            self._embedding_fn = self._model.encode
            return True
        except Exception:
            return False

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Return embeddings for a list of texts, or None on failure."""
        if not self._load():
            return None
        try:
            embeddings = self._embedding_fn(texts, convert_to_numpy=True)
            return embeddings
        except Exception:
            return None

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb = self.embed([text1, text2])
        if emb is None or len(emb) < 2:
            return 0.0
        a, b = emb[0], emb[1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Global embedder instance
_embedder: Optional[SentenceEmbedder] = None


def _get_embedder() -> SentenceEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceEmbedder()
    return _embedder


class SimpleCache:
    """
    Cache with Redis backend, in-memory fallback, and semantic similarity.

    Cache lookup strategy:
    1. Exact normalized key match (fast path)
    2. Semantic similarity check (if SEMANTIC_CACHE_ENABLED):
       - Embed query + all cached keys
       - Find best cosine similarity
       - Return if similarity >= SEMANTIC_CACHE_THRESHOLD
    """

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None
        self._semantic_cache: Dict[str, str] = {}  # text → normalized_key (in-memory only)
        self._semantic_enabled = SEMANTIC_CACHE_ENABLED

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

    def _get_all_keys(self) -> List[str]:
        """Return all cache keys from current backend."""
        if self.redis_client:
            try:
                return self.redis_client.keys("*")
            except Exception:
                return []
        return list(self.store.keys())

    def _semantic_lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Perform semantic similarity lookup across all cached keys.

        Returns the cached response if best similarity >= SEMANTIC_CACHE_THRESHOLD,
        otherwise returns None.
        """
        if not self._semantic_enabled:
            return None

        embedder = _get_embedder()
        all_keys = self._get_all_keys()
        if not all_keys:
            return None

        # Batch embed all cached keys + query for efficiency
        texts_to_embed = all_keys + [query]
        embeddings = embedder.embed(texts_to_embed)
        if embeddings is None:
            return None

        cached_embeddings = embeddings[: len(all_keys)]
        query_embedding = embeddings[len(all_keys)]

        best_similarity = 0.0
        best_key = None

        for i, key_emb in enumerate(cached_embeddings):
            norm_q = np.linalg.norm(query_embedding)
            norm_k = np.linalg.norm(key_emb)
            if norm_q == 0 or norm_k == 0:
                continue
            sim = float(np.dot(query_embedding, key_emb) / (norm_q * norm_k))
            if sim > best_similarity:
                best_similarity = sim
                best_key = all_keys[i]

        if best_similarity >= SEMANTIC_CACHE_THRESHOLD and best_key is not None:
            # Retrieve from backend
            if self.redis_client:
                raw = self.redis_client.get(best_key)
                if raw:
                    return json.loads(raw)
            else:
                return self.store.get(best_key)

        return None

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response by query.

        Tries in order:
        1. Exact normalized key match (fast)
        2. Semantic similarity match (if enabled)
        """
        # Fast path: exact normalized key
        key = self.normalize(query)

        if self.redis_client:
            try:
                raw = self.redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        else:
            cached = self.store.get(key)
            if cached:
                return cached

        # Semantic path: embedding similarity
        return self._semantic_lookup(query)

    def set(self, query: str, value: Dict[str, Any]) -> None:
        """
        Store response by normalized query.

        Also stores the raw query for semantic cache matching.
        """
        key = self.normalize(query)

        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    CACHE_TTL_SECONDS,
                    json.dumps(value, ensure_ascii=False),
                )
                # Store raw query as secondary key for semantic lookup
                # (semantic lookup uses normalized keys as corpus)
                self.redis_client.setex(
                    f"_raw:{key}",
                    CACHE_TTL_SECONDS,
                    query,
                )
                return
            except Exception:
                pass

        self.store[key] = value
        # Store raw text for semantic matching (in-memory)
        self._semantic_cache[query] = key

    def invalidate(self) -> None:
        """
        Invalidate all cache entries.

        Should be called when knowledge base (menu/FAQ) changes.
        """
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass
        self.store.clear()
        self._semantic_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics for diagnostics."""
        total_keys = len(self._get_all_keys())
        return {
            "backend": self.backend_name(),
            "total_entries": total_keys,
            "semantic_enabled": self._semantic_enabled,
            "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
        }
