"""
Main FastAPI application.

Request flow:
User Query
-> Router Agent
-> Specialized Agent
-> RAG retrieval
-> Local LLM generation
-> Response with sources, session and cache
"""

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.cache import SimpleCache
from app.config import APP_NAME, LLM_BACKEND
from app.agents import (
    ConsultantAgent,
    FAQAgent,
    IgnoreAgent,
    OrderAgent,
    build_agent_prompt,
    format_sources,
)
from app.guardrails import is_guardrail_blocked
from app.graph_rag import GraphRAGStore
from app.rag import RAGStore
from app.router_agent import classify_intent
from app.schemas import ChatRequest, ChatResponse
from app.session_store import SessionStore
from app.utils import call_llm, stream_ollama

app = FastAPI(title=APP_NAME)

rag_store = RAGStore()
graph_rag_store = GraphRAGStore()
session_store = SessionStore()
cache = SimpleCache()

order_agent = OrderAgent(rag_store, graph_rag_store)
consultant_agent = ConsultantAgent(rag_store, graph_rag_store)
faq_agent = FAQAgent(rag_store, graph_rag_store)
ignore_agent = IgnoreAgent(rag_store, graph_rag_store)


def hybrid_retrieve(query: str, intent: str, top_k: int = 5):
    """
    Combine Graph RAG lite with vector retrieval.

    Graph retrieval is attempted first.
    Chroma vector retrieval is used as fallback or supplement.
    """
    graph_docs = []
    if graph_rag_store.is_available():
        graph_docs = graph_rag_store.search(query, intent=intent, top_k=top_k)

    vector_docs = rag_store.search(query, intent=intent, top_k=top_k)
    combined = graph_docs + vector_docs
    return combined[: top_k * 2]


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "rag_documents": rag_store.count(),
        "graph_rag_enabled": graph_rag_store.is_available(),
        "cache_backend": cache.backend_name(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint."""
    if is_guardrail_blocked(request.query):
        return ChatResponse(
            session_id=request.session_id,
            intent="guardrail",
            answer=(
                "Xin lỗi, mình không thể hỗ trợ nội dung này. "
                "Bạn vui lòng hỏi về đặt món, tư vấn món hoặc thông tin quán."
            ),
            sources=[],
            cached=False,
        )

    cached = cache.get(request.query)
    if cached:
        return ChatResponse(
            session_id=request.session_id,
            intent=cached["intent"],
            answer=cached["answer"],
            sources=cached.get("sources", []),
            cached=True,
        )

    intent = classify_intent(request.query)["action"]
    history = session_store.get_history(request.session_id)

    if intent == "order":
        result = order_agent.answer(request.query, history)
    elif intent == "consultant":
        result = consultant_agent.answer(request.query, history)
    elif intent == "faq":
        result = faq_agent.answer(request.query, history)
    else:
        result = ignore_agent.answer(request.query, history)

    answer = result["answer"]
    sources = result.get("sources", [])

    session_store.add_turn(
        session_id=request.session_id, user=request.query, assistant=answer
    )
    cache.set(
        request.query,
        {
            "intent": intent,
            "answer": answer,
            "sources": sources,
        },
    )

    return ChatResponse(
        session_id=request.session_id,
        intent=intent,
        answer=answer,
        sources=sources,
        cached=False,
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Flow:
    - guardrail check
    - cache check
    - classify intent
    - retrieve RAG context
    - build prompt
    - stream tokens
    """
    if is_guardrail_blocked(request.query):

        def guardrail_event_stream():
            metadata = {
                "type": "metadata",
                "intent": "guardrail",
                "cached": False,
                "sources": [],
            }
            yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "token",
                        "content": (
                            "Xin lỗi, mình không thể hỗ trợ nội dung này. "
                            "Bạn vui lòng hỏi về đặt món, tư vấn món hoặc thông tin quán."
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            guardrail_event_stream(), media_type="text/event-stream"
        )

    cached = cache.get(request.query)
    if cached:

        def cached_event_stream():
            metadata = {
                "type": "metadata",
                "intent": cached["intent"],
                "cached": True,
                "sources": cached.get("sources", []),
            }
            yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {"type": "token", "content": cached["answer"]},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(cached_event_stream(), media_type="text/event-stream")

    intent = classify_intent(request.query)["action"]
    history = session_store.get_history(request.session_id)

    if intent in ["order", "consultant", "faq"]:
        docs = hybrid_retrieve(request.query, intent=intent, top_k=5)
        sources = format_sources(docs)
    else:
        docs = []
        sources = []

    if intent == "ignore":
        ignore_answer = ignore_agent.answer(request.query, history)["answer"]

        def ignore_event_stream():
            metadata = {
                "type": "metadata",
                "intent": "ignore",
                "cached": False,
                "sources": [],
            }
            yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {"type": "token", "content": ignore_answer},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            session_store.add_turn(
                session_id=request.session_id,
                user=request.query,
                assistant=ignore_answer,
            )
            cache.set(
                request.query,
                {"intent": "ignore", "answer": ignore_answer, "sources": []},
            )
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(ignore_event_stream(), media_type="text/event-stream")

    prompt = build_agent_prompt(intent, request.query, history, docs)

    def event_stream():
        metadata = {
            "type": "metadata",
            "intent": intent,
            "cached": False,
            "sources": sources,
        }
        yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"

        full_answer = ""

        if LLM_BACKEND == "openai_compat":
            # Streaming path for OpenAI-compatible backends can be mapped later.
            chunk = call_llm(prompt)
            full_answer = chunk
            yield (
                "data: "
                + json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                + "\n\n"
            )
        else:
            for token in stream_ollama(prompt):
                full_answer += token
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "token", "content": token},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

        session_store.add_turn(
            session_id=request.session_id,
            user=request.query,
            assistant=full_answer,
        )
        cache.set(
            request.query,
            {
                "intent": intent,
                "answer": full_answer,
                "sources": sources,
            },
        )
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
