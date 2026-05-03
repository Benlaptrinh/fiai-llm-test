"""
RAGStore: Retrieval-Augmented Generation layer.

This module stores and retrieves F&B knowledge using ChromaDB.
It supports:
- Intent-aware retrieval for menu, FAQ and document knowledge
- Hybrid search (vector + optional graph)
- Cross-encoder reranking (BGE Reranker v2)

Prototype:
- ChromaDB vector store
- multilingual MiniLM embeddings
- intent-aware filtering
- cross-encoder reranking

Production extension:
- Neo4j Graph RAG
- entity extraction
- graph expansion
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import CHROMA_DIR, EMBEDDING_MODEL

# Lazy import for reranker to avoid heavy import overhead
_reranker_model = None


def _get_reranker():
    """
    Lazily load BGE Cross-Encoder reranker.
    Falls back to no-op if unavailable.
    """
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            trust_remote_code=True,
        )
        return _reranker_model
    except Exception:
        _reranker_model = None
        return None


def _rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    min_score: float = 0.0,
    keyword_boost_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using BGE Reranker v2.

    Args:
        query: original user query
        documents: list of documents with 'text' key
        top_k: number of top results to return after reranking
        min_score: minimum relevance score threshold (0.0 = no filter)
        keyword_boost_weight: weight for keyword boost score (0.0 = disabled)

    Returns:
        reranked documents with 'rerank_score' added
    """
    reranker = _get_reranker()
    if reranker is None or not documents:
        return documents[:top_k]

    try:
        doc_texts = [doc.get("text", "") for doc in documents]
        scores = reranker.predict(
            [(query, doc_text) for doc_text in doc_texts],
            show_progress_bar=False,
        )

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc = dict(doc)
            base_score = float(score)
            # If keyword boost is significant, use it directly to override reranker score
            # This ensures exact name matches always surface regardless of embedding quality
            kb = doc.get("_keyword_boost", 0)
            if kb >= 1.0:
                doc["rerank_score"] = kb + base_score * 0.2
            elif keyword_boost_weight > 0 and kb > 0:
                doc["rerank_score"] = (
                    base_score * (1 - keyword_boost_weight)
                    + kb * keyword_boost_weight
                )
            else:
                doc["rerank_score"] = base_score
            scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_k]

    except Exception:
        return documents[:top_k]


class RAGStore:
    """Lightweight vector store wrapper using ChromaDB with optional reranking."""

    def __init__(self) -> None:
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name="fnb_knowledge",
            embedding_function=self.embedding_fn,
        )

    def search(
        self,
        query: str,
        intent: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
        rerank_top_k: int = 5,
        rerank_min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search relevant knowledge by query and intent.

        Args:
            query: user query
            intent: order / consultant / faq / ignore
            top_k: number of documents to retrieve from vector store (before reranking)
            use_reranker: whether to apply BGE reranker (B1.2)
            rerank_top_k: number of results to return after reranking
            rerank_min_score: minimum rerank score threshold

        Returns:
            retrieved documents with text, metadata, distance, and optionally rerank_score
        """
        where = None
        if intent in ["order", "consultant"]:
            where = {"domain": "menu"}
        elif intent == "faq":
            where = {"domain": {"$in": ["faq", "doc"]}}

        # Retrieve more candidates than final result for reranking
        fetch_k = max(top_k * 3, 10) if use_reranker else top_k

        try:
            result = self.collection.query(
                query_texts=[query], n_results=fetch_k, where=where
            )
        except Exception:
            result = self.collection.query(query_texts=[query], n_results=fetch_k)

        docs: List[Dict[str, Any]] = []
        if result and result.get("documents"):
            for i, doc in enumerate(result["documents"][0]):
                metadata = result["metadatas"][0][i]
                distance = None
                if result.get("distances"):
                    distance = result["distances"][0][i]
                docs.append({"text": doc, "metadata": metadata, "distance": distance})

        # Keyword boost: if query mentions a known menu item name, boost it
        docs = self._keyword_boost(query, docs)

        # Apply reranking if enabled
        if use_reranker and docs:
            docs = _rerank_documents(
                query=query,
                documents=docs,
                top_k=rerank_top_k,
                min_score=rerank_min_score,
                keyword_boost_weight=0.95,
            )

        return docs

    def _keyword_boost(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Boost documents where item name matches a keyword from the query.
        Prevents truncation from hiding relevant items when embedding similarity
        is slightly below threshold.
        """
        import re

        query_lower = query.lower()
        # Extract meaningful tokens (words >= 2 chars)
        query_tokens = re.findall(r"\b[\wÀ-ỹ]{2,}\b", query_lower)

        boosted = []
        for doc in docs:
            name_lower = doc["metadata"].get("name", "").lower()
            score = 0.0
            # Check for substring match first (e.g. "trà đào" in "trà đào cam sả")
            for token in query_tokens:
                if token in name_lower:
                    score += 2.0  # Substring match gets strong boost
            # If no substring match, do token-level boost
            if score == 0:
                for token in query_tokens:
                    if any(token in w for w in name_lower.split()):
                        score += 1.0
            doc = dict(doc)
            doc["_keyword_boost"] = score
            boosted.append((score, doc))

        boosted.sort(key=lambda x: (-x[0], x[1].get("distance", 999) or 999))
        return [d for _, d in boosted]

    def count(self) -> int:
        """Count all documents in vector store."""
        return self.collection.count()
