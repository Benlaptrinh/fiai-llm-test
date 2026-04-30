"""
RAGStore: Retrieval-Augmented Generation layer.

This module stores and retrieves F&B knowledge using ChromaDB.
It supports intent-aware retrieval for menu, FAQ and document knowledge.

Prototype:
- ChromaDB vector store
- multilingual MiniLM embeddings
- intent-aware filtering

Production extension:
- Neo4j Graph RAG
- entity extraction
- graph expansion
- reranking
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import CHROMA_DIR, EMBEDDING_MODEL


class RAGStore:
    """Lightweight vector store wrapper using ChromaDB."""

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
        self, query: str, intent: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search relevant knowledge by query and intent.

        Args:
            query: user query
            intent: order / consultant / faq / ignore
            top_k: number of retrieved documents

        Returns:
            retrieved documents with text, metadata and distance
        """
        where = None
        if intent in ["order", "consultant"]:
            where = {"domain": "menu"}
        elif intent == "faq":
            where = {"domain": {"$in": ["faq", "doc"]}}

        try:
            result = self.collection.query(
                query_texts=[query], n_results=top_k, where=where
            )
        except Exception:
            # Fallback for Chroma filter compatibility differences.
            result = self.collection.query(query_texts=[query], n_results=top_k)

        docs: List[Dict[str, Any]] = []
        if result and result.get("documents"):
            for i, doc in enumerate(result["documents"][0]):
                metadata = result["metadatas"][0][i]
                distance = None
                if result.get("distances"):
                    distance = result["distances"][0][i]
                docs.append({"text": doc, "metadata": metadata, "distance": distance})

        return docs

    def count(self) -> int:
        """Count all documents in vector store."""
        return self.collection.count()
