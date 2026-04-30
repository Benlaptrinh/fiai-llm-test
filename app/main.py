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

from fastapi import FastAPI

from app.cache import SimpleCache
from app.config import APP_NAME
from app.agents import ConsultantAgent, FAQAgent, IgnoreAgent, OrderAgent
from app.rag import RAGStore
from app.router_agent import classify_intent
from app.schemas import ChatRequest, ChatResponse
from app.session_store import SessionStore

app = FastAPI(title=APP_NAME)

rag_store = RAGStore()
session_store = SessionStore()
cache = SimpleCache()

order_agent = OrderAgent(rag_store)
consultant_agent = ConsultantAgent(rag_store)
faq_agent = FAQAgent(rag_store)
ignore_agent = IgnoreAgent(rag_store)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "rag_documents": rag_store.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint."""
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

    session_store.add_turn(session_id=request.session_id, user=request.query, assistant=answer)
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
