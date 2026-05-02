"""
Graph RAG lite layer using Neo4j.

This module stores structured F&B knowledge as graph nodes:
- MenuItem
- FAQ
- DocChunk

It provides lightweight graph retrieval by intent and keyword.
This is not a full graph-embedding implementation, but it demonstrates
how Graph RAG can complement vector retrieval in production.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from neo4j import GraphDatabase

from app.config import ENABLE_GRAPH_RAG, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


class GraphRAGStore:
    """Lightweight Neo4j wrapper for graph-based retrieval."""

    def __init__(self) -> None:
        self.enabled = ENABLE_GRAPH_RAG
        self.driver = None

        if self.enabled:
            try:
                self.driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                )
                self.driver.verify_connectivity()
            except Exception:
                self.enabled = False
                self.driver = None

    def close(self) -> None:
        """Close active Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def is_available(self) -> bool:
        """Return True when Neo4j connectivity is ready."""
        return self.enabled and self.driver is not None

    def clear(self) -> None:
        """Delete all graph nodes and relations."""
        if not self.is_available():
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self) -> None:
        """Create basic unique constraints for node IDs."""
        if not self.is_available():
            return

        queries = [
            "CREATE CONSTRAINT menu_item_id IF NOT EXISTS FOR (m:MenuItem) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT faq_id IF NOT EXISTS FOR (f:FAQ) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:DocChunk) REQUIRE d.id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def upsert_menu_item(self, item: Dict[str, Any]) -> None:
        """Upsert one menu item plus relations to category/taste dimensions."""
        if not self.is_available():
            return

        query = """
        MERGE (m:MenuItem {id: $id})
        SET m.name = $name,
            m.category = $category,
            m.size = $size,
            m.price = $price,
            m.caffeine = $caffeine,
            m.sweetness = $sweetness,
            m.tags = $tags,
            m.description = $description
        MERGE (c:Category {name: $category})
        MERGE (m)-[:IN_CATEGORY]->(c)
        MERGE (s:Sweetness {level: $sweetness})
        MERGE (m)-[:HAS_SWEETNESS]->(s)
        MERGE (cf:Caffeine {level: $caffeine})
        MERGE (m)-[:HAS_CAFFEINE]->(cf)
        """

        with self.driver.session() as session:
            session.run(query, **item)

    def upsert_faq(self, item: Dict[str, Any]) -> None:
        """Upsert one FAQ node."""
        if not self.is_available():
            return

        query = """
        MERGE (f:FAQ {id: $id})
        SET f.question = $question,
            f.answer = $answer
        """

        with self.driver.session() as session:
            session.run(query, **item)

    def upsert_doc_chunk(self, item: Dict[str, Any]) -> None:
        """Upsert one policy/document chunk node."""
        if not self.is_available():
            return

        query = """
        MERGE (d:DocChunk {id: $id})
        SET d.text = $text
        """

        with self.driver.session() as session:
            session.run(query, **item)

    def search(self, query_text: str, intent: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Graph retrieval by intent and keyword matching.

        Returns graph context documents compatible with vector RAG source format.
        """
        if not self.is_available():
            return []

        normalized = query_text.lower().strip()
        tokens = [t for t in re.findall(r"\w+", normalized) if len(t) >= 2]
        if not tokens:
            return []

        with self.driver.session() as session:
            if intent in ["order", "consultant"]:
                return self._search_menu(session, normalized, tokens, top_k)

            if intent == "faq":
                return self._search_faq_docs(session, normalized, tokens, top_k)

            return []

    def _search_menu(
        self, session, normalized_query: str, tokens: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (m:MenuItem)
        WHERE
            toLower(m.name) CONTAINS $q
            OR toLower(m.category) CONTAINS $q
            OR toLower(m.sweetness) CONTAINS $q
            OR toLower(m.caffeine) CONTAINS $q
            OR toLower(m.tags) CONTAINS $q
            OR toLower(m.description) CONTAINS $q
            OR ANY(token IN $tokens WHERE
                toLower(m.name) CONTAINS token
                OR toLower(m.category) CONTAINS token
                OR toLower(m.sweetness) CONTAINS token
                OR toLower(m.caffeine) CONTAINS token
                OR toLower(m.tags) CONTAINS token
                OR toLower(m.description) CONTAINS token
            )
        RETURN m
        LIMIT $top_k
        """

        result = session.run(cypher, q=normalized_query, tokens=tokens, top_k=top_k)

        docs: List[Dict[str, Any]] = []
        for record in result:
            menu_item = dict(record["m"])
            text = (
                f"[Graph] Tên món: {menu_item.get('name')}. "
                f"Danh mục: {menu_item.get('category')}. "
                f"Size: {menu_item.get('size')}. "
                f"Giá: {menu_item.get('price')} VND. "
                f"Caffeine: {menu_item.get('caffeine')}. "
                f"Sweetness: {menu_item.get('sweetness')}. "
                f"Tags: {menu_item.get('tags')}. "
                f"Mô tả: {menu_item.get('description')}."
            )
            docs.append(
                {
                    "text": text,
                    "metadata": {
                        "domain": "graph_menu",
                        "type": "MenuItem",
                        "name": menu_item.get("name"),
                        "category": menu_item.get("category"),
                        "size": menu_item.get("size"),
                        "price": menu_item.get("price"),
                    },
                    "distance": None,
                }
            )

        return docs

    def _search_faq_docs(
        self, session, normalized_query: str, tokens: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (n)
        WHERE
            (
                n:FAQ
                AND (
                    toLower(n.question) CONTAINS $q
                    OR toLower(n.answer) CONTAINS $q
                    OR ANY(token IN $tokens WHERE
                        toLower(n.question) CONTAINS token
                        OR toLower(n.answer) CONTAINS token
                    )
                )
            )
            OR
            (
                n:DocChunk
                AND (
                    toLower(n.text) CONTAINS $q
                    OR ANY(token IN $tokens WHERE toLower(n.text) CONTAINS token)
                )
            )
        RETURN n
        LIMIT $top_k
        """

        result = session.run(cypher, q=normalized_query, tokens=tokens, top_k=top_k)

        docs: List[Dict[str, Any]] = []
        for record in result:
            node = dict(record["n"])

            if "question" in node:
                text = f"[Graph] Câu hỏi: {node.get('question')}\nTrả lời: {node.get('answer')}"
                metadata = {"domain": "graph_faq", "type": "FAQ"}
            else:
                text = f"[Graph] {node.get('text')}"
                metadata = {"domain": "graph_doc", "type": "DocChunk"}

            docs.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": None,
                }
            )

        return docs
