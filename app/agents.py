"""
Specialized agents for Multi-Agent F&B assistant.

Agents:
- OrderAgent
- ConsultantAgent
- FAQAgent
- IgnoreAgent

Each agent has:
- role-specific prompt
- RAG context
- guardrails against hallucination
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.graph_rag import GraphRAGStore
from app.rag import RAGStore
from app.utils import call_llm

ORDER_ROLE = """
Bạn là Order Agent cho quán F&B.
Nhiệm vụ:
- Nhận order từ khách.
- Xác nhận món, số lượng, size nếu có.
- Chỉ dùng thông tin menu trong context.
- Không bịa món, không bịa giá.
- Nếu thiếu size hoặc số lượng, hỏi lại ngắn gọn.
"""

CONSULTANT_ROLE = """
Bạn là Consultant Agent cho quán F&B.
Nhiệm vụ:
- Tư vấn món dựa trên khẩu vị, ngân sách và context.
- Nếu khách thích ít ngọt, ưu tiên sweetness low hoặc medium.
- Nếu khách không uống caffeine, không gợi ý món caffeine high.
- Chỉ gợi ý món có trong context.
- Không bịa món hoặc giá.
"""

FAQ_ROLE = """
Bạn là FAQ Agent cho quán F&B.
Nhiệm vụ:
- Trả lời các câu hỏi chung như wifi, giờ mở cửa, thanh toán, hóa đơn.
- Chỉ trả lời dựa trên context.
- Nếu context không có thông tin, nói chưa có dữ liệu để xác nhận.
- Không bịa chính sách.
"""

IGNORE_ROLE = """
Bạn là Ignore Agent.
Nếu câu hỏi không rõ hoặc ngoài phạm vi F&B, hãy phản hồi ngắn gọn và lịch sự.
"""


def format_context(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into prompt context."""
    if not docs:
        return "Không tìm thấy context phù hợp."

    lines = []
    for idx, doc in enumerate(docs, start=1):
        lines.append(f"[{idx}] {doc['text']}")
    return "\n".join(lines)


def format_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format sources for API response."""
    sources = []
    for doc in docs:
        sources.append(
            {"metadata": doc.get("metadata", {}), "distance": doc.get("distance")}
        )
    return sources


def build_history_text(history: List[Dict[str, str]]) -> str:
    """Convert session history into text."""
    if not history:
        return "Không có lịch sử hội thoại."

    return "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history]
    )


def build_agent_prompt(
    intent: str,
    query: str,
    history: List[Dict[str, str]],
    docs: List[Dict[str, Any]],
) -> str:
    """
    Build prompt for regular and streaming endpoints.

    This keeps `/chat` and `/chat/stream` consistent with specialized behavior.
    """
    context = format_context(docs)
    history_text = build_history_text(history)

    if intent == "order":
        role = ORDER_ROLE
    elif intent == "consultant":
        role = CONSULTANT_ROLE
    elif intent == "faq":
        role = FAQ_ROLE
    else:
        role = IGNORE_ROLE

    return f"""{role}

Lịch sử hội thoại:
{history_text}

Context:
{context}

Khách nói:
{query}

Trả lời bằng tiếng Việt, tự nhiên, ngắn gọn."""


class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(self, rag: RAGStore, graph_rag: Optional[GraphRAGStore] = None):
        self.rag = rag
        self.graph_rag = graph_rag

    def retrieve(
        self, query: str, intent: str, top_k: int = 5, use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Combine graph retrieval with vector retrieval.

        Graph retrieval is optional and can be disabled by configuration.
        Uses BGE Reranker v2 for improved relevance ordering (B1.2).
        """
        graph_docs: List[Dict[str, Any]] = []
        if self.graph_rag and self.graph_rag.is_available():
            graph_docs = self.graph_rag.search(query, intent=intent, top_k=top_k)

        vector_docs = self.rag.search(
            query, intent=intent, top_k=top_k, use_reranker=use_reranker
        )
        return (graph_docs + vector_docs)[: top_k * 2]

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError


class OrderAgent(BaseAgent):
    """Agent for handling order-related queries."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.retrieve(query, intent="order", top_k=5)
        prompt = build_agent_prompt("order", query, history, docs)
        return {"answer": call_llm(prompt), "sources": format_sources(docs)}


class ConsultantAgent(BaseAgent):
    """Agent for recommendation and consulting."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.retrieve(query, intent="consultant", top_k=5)
        prompt = build_agent_prompt("consultant", query, history, docs)
        return {"answer": call_llm(prompt), "sources": format_sources(docs)}


class FAQAgent(BaseAgent):
    """Agent for FAQ and policy questions."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.retrieve(query, intent="faq", top_k=5)
        prompt = build_agent_prompt("faq", query, history, docs)
        return {"answer": call_llm(prompt), "sources": format_sources(docs)}


class IgnoreAgent(BaseAgent):
    """Agent for greeting, noise or unclear inputs."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "answer": "Dạ em đang nghe ạ. Anh/chị muốn đặt món, hỏi thông tin hay cần tư vấn món nào ạ?",
            "sources": [],
        }
