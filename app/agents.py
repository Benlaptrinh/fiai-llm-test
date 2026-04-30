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

from typing import Any, Dict, List

from app.rag import RAGStore
from app.utils import call_ollama


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
        sources.append({"metadata": doc.get("metadata", {}), "distance": doc.get("distance")})
    return sources


def build_history_text(history: List[Dict[str, str]]) -> str:
    """Convert session history into text."""
    if not history:
        return "Không có lịch sử hội thoại."

    return "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history])


class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(self, rag: RAGStore):
        self.rag = rag

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError


class OrderAgent(BaseAgent):
    """Agent for handling order-related queries."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.rag.search(query, intent="order", top_k=5)
        context = format_context(docs)
        history_text = build_history_text(history)

        prompt = f"""Bạn là Order Agent cho quán F&B.
Nhiệm vụ:
- Nhận order từ khách.
- Xác nhận món, số lượng, size nếu có.
- Chỉ dùng thông tin menu trong context.
- Không bịa món, không bịa giá.
- Nếu thiếu size hoặc số lượng, hỏi lại ngắn gọn.

Lịch sử hội thoại:
{history_text}

Menu context:
{context}

Khách nói:
{query}

Trả lời bằng tiếng Việt, thân thiện, ngắn gọn."""

        return {"answer": call_ollama(prompt), "sources": format_sources(docs)}


class ConsultantAgent(BaseAgent):
    """Agent for recommendation and consulting."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.rag.search(query, intent="consultant", top_k=5)
        context = format_context(docs)
        history_text = build_history_text(history)

        prompt = f"""Bạn là Consultant Agent cho quán F&B.
Nhiệm vụ:
- Tư vấn món dựa trên khẩu vị, ngân sách và context.
- Nếu khách thích ít ngọt, ưu tiên sweetness low hoặc medium.
- Nếu khách không uống caffeine, không gợi ý món caffeine high.
- Chỉ gợi ý món có trong context.
- Không bịa món hoặc giá.

Lịch sử hội thoại:
{history_text}

Context:
{context}

Khách nói:
{query}

Trả lời bằng tiếng Việt, tự nhiên, thân thiện."""

        return {"answer": call_ollama(prompt), "sources": format_sources(docs)}


class FAQAgent(BaseAgent):
    """Agent for FAQ and policy questions."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        docs = self.rag.search(query, intent="faq", top_k=5)
        context = format_context(docs)
        history_text = build_history_text(history)

        prompt = f"""Bạn là FAQ Agent cho quán F&B.
Nhiệm vụ:
- Trả lời các câu hỏi chung như wifi, giờ mở cửa, thanh toán, hóa đơn.
- Chỉ trả lời dựa trên context.
- Nếu context không có thông tin, nói chưa có dữ liệu để xác nhận.
- Không bịa chính sách.

Lịch sử hội thoại:
{history_text}

Context:
{context}

Khách hỏi:
{query}

Trả lời bằng tiếng Việt, ngắn gọn."""

        return {"answer": call_ollama(prompt), "sources": format_sources(docs)}


class IgnoreAgent(BaseAgent):
    """Agent for greeting, noise or unclear inputs."""

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "answer": "Dạ em đang nghe ạ. Anh/chị muốn đặt món, hỏi thông tin hay cần tư vấn món nào ạ?",
            "sources": [],
        }
