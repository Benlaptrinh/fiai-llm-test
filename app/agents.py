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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.graph_rag import GraphRAGStore
from app.rag import RAGStore
from app.utils import call_llm

if TYPE_CHECKING:
    from app.session_store import SessionStore

# Lazy session store reference (set at startup to avoid circular imports)
_session_store: Optional["SessionStore"] = None


def init_agents_session_store(store: "SessionStore") -> None:
    global _session_store
    _session_store = store


def _get_session_store() -> Optional["SessionStore"]:
    return _session_store


ORDER_ROLE = """
Bạn là Order Agent cho quán F&B.
Nhiệm vụ:
- Nhận order từ khách.
- Xác nhận món, số lượng, size nếu có.
- CHỈ DÙNG thông tin menu trong context. KHÔNG bịa món, KHÔNG bịa giá.
- Nếu thiếu size hoặc số lượng, hỏi lại ngắn gọn.

QUAN TRỌNG - Quản lý đơn hàng tích lũy:
- Hệ thống duy trì đơn hàng CHÍNH XÁC. Đơn hiện tại được hiển thị ở trên.
- Khi khách THÊM món: THÊM vào đơn hiện tại, KHÔNG thay đổi số lượng món đã có.
- Khi khách HUY món: XOA món khỏi đơn hiện tại.
- LUÔN giữ nguyên số lượng các món đã có, chỉ thêm món MỚI.
- KHÔNG được tự ý nhân đôi, tăng giảm số lượng món đã đặt.
- ĐƠN HIỆN TẠI từ hệ thống là CHÍNH XÁC - không được sửa khi trả lời.
- Khi khách hỏi "kiểm tra đơn": đọc lại ĐƠN HIỆN TẠI từ hệ thống.
- ĐẾM SỐ LƯỢNG CẨN THẬN: mỗi món chỉ liệt kê số lượng đúng như khách yêu cầu, không tự nhân đôi.
- BẮT BUỘC ghi GIA for MỌI món: "1 bánh Croissants: 25,000 VND" (KHÔNG bỏ giá).
- CHỈ thêm món khách YÊU CẦU. Không được thêm món không có trong câu hỏi của khách.
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
    session_id: Optional[str] = None,
) -> str:
    """
    Build prompt for regular and streaming endpoints.

    This keeps `/chat` and `/chat/stream` consistent with specialized behavior.
    """
    context = format_context(docs)
    history_text = build_history_text(history)

    order_summary = ""
    if intent == "order" and session_id:
        role = ORDER_ROLE
        ss = _get_session_store()
        if ss:
            order_summary = ss.format_order_summary(session_id)
            if order_summary and "chưa đặt món" not in order_summary:
                order_summary = (
                    f"\nDON HANG HIEN TAI (tu he thong, chinh xac):\n{order_summary}\n"
                )
        else:
            role = ORDER_ROLE
    elif intent == "consultant":
        role = CONSULTANT_ROLE
    elif intent == "faq":
        role = FAQ_ROLE
    else:
        role = IGNORE_ROLE

    return f"""{role}{order_summary}

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

    def answer(
        self, query: str, history: List[Dict[str, str]], session_id: str
    ) -> Dict[str, Any]:
        # Use hybrid retrieval (graph + vector) for best context
        graph_docs: List[Dict[str, Any]] = []
        if self.graph_rag and self.graph_rag.is_available():
            graph_docs = self.graph_rag.search(query, intent="order", top_k=3)
        vector_docs = self.rag.search(
            query, intent="order", top_k=10, use_reranker=True
        )
        # Merge: graph results first, then vector results, dedup by name+size
        seen, docs = set(), []
        for doc in graph_docs + vector_docs:
            key = (doc["metadata"].get("name"), doc["metadata"].get("size"))
            if key not in seen:
                seen.add(key)
                docs.append(doc)
        docs = docs[:10]
        prompt = build_agent_prompt(
            "order", query, history, docs, session_id=session_id
        )
        return {"answer": call_llm(prompt), "sources": format_sources(docs)}

    def answer_check_order(self, session_id: str) -> Dict[str, Any]:
        """
        Handle 'check order' queries - read from structured order state.
        If no order was placed, respond clearly without hallucinating items.
        """
        ss = _get_session_store()
        print(f"[DEBUG answer_check_order] session={session_id} ss={ss}")
        if ss is None:
            return {
                "answer": "He thong đang khoi dong, vui long thu lai.",
                "sources": [],
            }
        order_state = ss.get_order_state(session_id)
        print(f"[DEBUG answer_check_order] order_state={order_state}")
        if order_state["items"]:
            summary = ss.format_order_summary(session_id)
            return {
                "answer": (
                    f"Don hang hien tai cua ban:\n\n{summary}\n\n"
                    "Ban co muon them hoac thay đoi gi khong?"
                ),
                "sources": [],
            }
        return {
            "answer": (
                "Hien tai ban chua đat mon nao trong cuoc tro chuyen nay. "
                "Ban muon đat gi a?"
            ),
            "sources": [],
        }


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
            "answer": "Da em đang nghe a. Anh/chị muon đat mon, hoi thong tin hay can tu van mon nao a?",
            "sources": [],
        }
