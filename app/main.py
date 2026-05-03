"""
Main FastAPI application.

Request flow:
User Query
-> Router Agent
-> Specialized Agent
-> RAG retrieval
-> Local LLM generation
-> Response with sources, session and cache

Features:
- C2.1: Intent extraction (LoRA SFT) for structured query parsing
- C2.2: Auto-invalidate cache when data files (menu.csv/FAQ.csv/docs.txt) change
- B1.3: Watch mode — auto-reload when app/*.py files change
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.cache import SimpleCache
from app.config import (
    APP_NAME,
    LLM_BACKEND,
    MAX_CONCURRENT_LLM_REQUESTS,
    QUEUE_TIMEOUT_SECONDS,
    SEMANTIC_CACHE_THRESHOLD,
    SLM_ROUTER_ENABLED,
    SLM_ROUTER_MODEL,
)
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
from app.intent_extractor import extract_intent
from app.schemas import ChatRequest, ChatResponse
from app.session_store import SessionStore
from app.utils import call_llm, stream_ollama

# ── App state ──────────────────────────────────────────────────────────────
app = FastAPI(title=APP_NAME)

rag_store: RAGStore = None  # type: ignore[assignment]
graph_rag_store: GraphRAGStore = None  # type: ignore[assignment]
session_store: SessionStore = None  # type: ignore[assignment]
cache: SimpleCache = None  # type: ignore[assignment]
llm_semaphore: asyncio.Semaphore = None  # type: ignore[assignment]
order_agent = None  # type: ignore[assignment]
consultant_agent = None  # type: ignore[assignment]
faq_agent = None  # type: ignore[assignment]
ignore_agent = None  # type: ignore[assignment]

# ── C2.2: Data hash tracking for cache invalidation ─────────────────────────
DATA_FILES = [
    Path("data/menu.csv"),
    Path("data/faq.csv"),
    Path("data/docs.txt"),
]
_DATA_HASH_FILE = Path("data/.data_hash.txt")


def _compute_data_hash() -> str:
    """Compute a hash of all data files to detect changes."""
    hasher = hashlib.sha256()
    for f in DATA_FILES:
        if f.exists():
            hasher.update(f.read_bytes())
    return hasher.hexdigest()[:16]


def _load_saved_hash() -> Optional[str]:
    """Load the last saved data hash."""
    if _DATA_HASH_FILE.exists():
        return _DATA_HASH_FILE.read_text().strip()
    return None


def _save_hash(h: str) -> None:
    """Save the current data hash."""
    _DATA_HASH_FILE.write_text(h)


def _invalidate_if_data_changed() -> bool:
    """
    B1.3 / C2.2: Check if data files changed since last startup.
    If changed, invalidate cache and re-ingest RAG.
    Returns True if cache was invalidated.
    """
    current_hash = _compute_data_hash()
    saved_hash = _load_saved_hash()

    if saved_hash is None or current_hash != saved_hash:
        print(f"[C2.2] Data files changed (old={saved_hash}, new={current_hash}). Invalidating cache + RAG...")
        if cache:
            cache.invalidate()
            print(f"[C2.2] ✓ Cache invalidated ({cache.backend_name()})")
        # Re-ingest RAG
        _reingest_rag()
        _save_hash(current_hash)
        print("[C2.2] ✓ RAG re-ingested")
        return True
    else:
        print(f"[C2.2] Data files unchanged (hash={current_hash}). Cache intact.")
        return False


def _reingest_rag() -> None:
    """Re-ingest all data files into RAG store."""
    if rag_store is None:
        return
    try:
        rag_store.clear()
        # Re-ingest menu
        _reingest_menu()
        # Re-ingest FAQ
        _reingest_faq()
        # Re-ingest docs
        _reingest_docs()
    except Exception as e:
        print(f"[C2.2] RAG re-ingest error: {e}")


def _reingest_menu() -> None:
    import csv
    menu_path = Path("data/menu.csv")
    if not menu_path.exists() or rag_store is None:
        return
    with menu_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = (
                f"{row['name']} - {row['category']} - "
                f"{row['price']}VND - caffeine: {row['caffeine']} - "
                f"tags: {row['tags']} - description: {row['description']}"
            )
            rag_store.add(doc, intent="order", metadata=row)
    print("[C2.2] Re-ingested menu.csv")


def _reingest_faq() -> None:
    import csv
    faq_path = Path("data/faq.csv")
    if not faq_path.exists() or rag_store is None:
        return
    with faq_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = f"Q: {row['question']}\nA: {row['answer']}"
            rag_store.add(doc, intent="faq", metadata={"question": row["question"]})
    print("[C2.2] Re-ingested faq.csv")


def _reingest_docs() -> None:
    docs_path = Path("data/docs.txt")
    if not docs_path.exists() or rag_store is None:
        return
    content = docs_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    for chunk in chunks:
        rag_store.add(chunk, intent="faq", metadata={"source": "docs"})
    print(f"[C2.2] Re-ingested docs.txt ({len(chunks)} chunks)")


# ── B1.3: Watch mode — auto-reload on app/*.py changes ─────────────────────
_WATCHER_ACTIVE = False
_WATCHER_THREAD: Optional[threading.Thread] = None


def _start_file_watcher() -> None:
    """
    B1.3: Start a background thread that watches app/*.py files.
    On change: auto-restart the uvicorn server.

    The watcher watches for changes to app/ directory Python files.
    When a file changes, it sends SIGTERM to the current process,
    which uvicorn handles gracefully (causing a reload if --reload is used).

    Usage: Run with `python scripts/run_with_watch.py` instead of `uvicorn app.main:app`
    """
    global _WATCHER_ACTIVE

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    except ImportError:
        print("[B1.3] watchdog not installed. Run: pip install watchdog")
        return

    WATCH_DIR = Path("app")
    WATCH_FILES = list(WATCH_DIR.glob("**/*.py"))

    class AppFileHandler(FileSystemEventHandler):
        def __init__(self, watched_files: list[Path]):
            self.watched_files = set(watched_files)
            self._suppress = False
            self._debounce_time = 0.0

        def on_modified(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path not in self.watched_files:
                return
            if path.name.startswith("."):
                return

            now = time.time()
            if now - self._debounce_time < 2.0:
                return
            self._debounce_time = now

            print(f"\n[B1.3] File changed: {path.relative_to(WATCH_DIR)}")
            print("[B1.3] Triggering server reload...")

            # SIGTERM triggers uvicorn reload when --reload is active
            os.kill(os.getpid(), signal.SIGTERM)

    def run_watcher():
        handler = AppFileHandler(WATCH_FILES)
        observer = Observer()
        observer.schedule(handler, str(WATCH_DIR), recursive=True)
        observer.start()
        print(f"[B1.3] Watching {len(WATCH_FILES)} app/*.py files for changes...")
        while _WATCHER_ACTIVE:
            time.sleep(1)
        observer.stop()
        observer.join()

    _WATCHER_ACTIVE = True
    _WATCHER_THREAD = threading.Thread(target=run_watcher, daemon=True)
    _WATCHER_THREAD.start()
    print(f"[B1.3] File watcher started (PID={os.getpid()})")


def stop_file_watcher() -> None:
    """Stop the file watcher thread."""
    global _WATCHER_ACTIVE
    _WATCHER_ACTIVE = False


# ── App startup ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global rag_store, graph_rag_store, session_store, cache, llm_semaphore
    global order_agent, consultant_agent, faq_agent, ignore_agent

    rag_store = RAGStore()
    graph_rag_store = GraphRAGStore()
    session_store = SessionStore()
    cache = SimpleCache()
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

    order_agent = OrderAgent(rag_store, graph_rag_store)
    consultant_agent = ConsultantAgent(rag_store, graph_rag_store)
    faq_agent = FAQAgent(rag_store, graph_rag_store)
    ignore_agent = IgnoreAgent(rag_store, graph_rag_store)

    # C2.2: Check data files and invalidate cache if needed
    _invalidate_if_data_changed()

    # B1.3: Start file watcher for auto-reload
    if os.getenv("ENABLE_WATCH_MODE", "false").lower() == "true":
        _start_file_watcher()


@app.on_event("shutdown")
async def shutdown():
    stop_file_watcher()


# ── Helpers ────────────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, intent: str, top_k: int = 5):
    graph_docs = []
    if graph_rag_store.is_available():
        graph_docs = graph_rag_store.search(query, intent=intent, top_k=top_k)
    vector_docs = rag_store.search(query, intent=intent, top_k=top_k)
    combined = graph_docs + vector_docs
    return combined[: top_k * 2]


# ── Health & cache endpoints ───────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "rag_documents": rag_store.count(),
        "graph_rag_enabled": graph_rag_store.is_available(),
        "cache_backend": cache.backend_name(),
        "max_concurrent_llm_requests": MAX_CONCURRENT_LLM_REQUESTS,
        "queue_timeout_seconds": QUEUE_TIMEOUT_SECONDS,
        "slm_router_enabled": SLM_ROUTER_ENABLED,
        "slm_router_model": SLM_ROUTER_MODEL,
        "semantic_cache_threshold": SEMANTIC_CACHE_THRESHOLD,
    }


@app.post("/cache/invalidate")
def invalidate_cache() -> dict:
    """Manually invalidate all cache entries."""
    cache.invalidate()
    return {"status": "ok", "message": "Cache invalidated"}


@app.get("/cache/stats")
def cache_stats() -> dict:
    """Return cache statistics."""
    return cache.stats()


# ── Chat endpoints ──────────────────────────────────────────────────────────
def process_chat_request(request: ChatRequest) -> ChatResponse:
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

    routing_result = classify_intent(request.query)
    intent = routing_result["action"]

    # C2.1: Intent extraction (LoRA SFT) — extract {subject, action, context}
    query_for_agent = request.query
    if intent not in ("ignore", "guardrail"):
        extraction = extract_intent(request.query)
        if extraction.get("context"):
            query_for_agent = f"{extraction['context']} | {request.query}"
        routing_result["extraction"] = extraction

    history = session_store.get_history(request.session_id)

    if intent == "order":
        result = order_agent.answer(query_for_agent, history)
    elif intent == "consultant":
        result = consultant_agent.answer(query_for_agent, history)
    elif intent == "faq":
        result = faq_agent.answer(query_for_agent, history)
    else:
        result = ignore_agent.answer(query_for_agent, history)

    answer = result["answer"]
    sources = result.get("sources", [])

    # A2.2: Session auto-summarize (SessionStore handles internally)
    session_store.add_turn(
        session_id=request.session_id, user=request.query, assistant=answer
    )

    cache.set(
        request.query,
        {"intent": intent, "answer": answer, "sources": sources},
    )

    return ChatResponse(
        session_id=request.session_id,
        intent=intent,
        answer=answer,
        sources=sources,
        cached=False,
    )


async def run_with_llm_queue(func: Callable[..., ChatResponse], *args, **kwargs) -> ChatResponse:
    try:
        await asyncio.wait_for(
            llm_semaphore.acquire(),
            timeout=QUEUE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="Server is busy. Please retry later.",
        ) from exc

    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    finally:
        llm_semaphore.release()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_with_llm_queue(process_chat_request, request)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    if is_guardrail_blocked(request.query):
        def guardrail_event_stream():
            meta = {"type": "metadata", "intent": "guardrail", "cached": False, "sources": []}
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            yield "data: " + json.dumps({
                "type": "token",
                "content": "Xin lỗi, mình không thể hỗ trợ nội dung này. "
                           "Bạn vui lòng hỏi về đặt món, tư vấn món hoặc thông tin quán."
            }, ensure_ascii=False) + "\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(guardrail_event_stream(), media_type="text/event-stream")

    cached = cache.get(request.query)
    if cached:
        def cached_event_stream():
            meta = {"type": "metadata", "intent": cached["intent"], "cached": True,
                    "sources": cached.get("sources", [])}
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            yield "data: " + json.dumps({"type": "token", "content": cached["answer"]},
                                              ensure_ascii=False) + "\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(cached_event_stream(), media_type="text/event-stream")

    routing_result = classify_intent(request.query)
    intent = routing_result["action"]

    query_for_agent = request.query
    if intent not in ("ignore", "guardrail"):
        extraction = extract_intent(request.query)
        if extraction.get("context"):
            query_for_agent = f"{extraction['context']} | {request.query}"

    history = session_store.get_history(request.session_id)

    if intent == "ignore":
        ignore_answer = ignore_agent.answer(request.query, history)["answer"]

        def ignore_event_stream():
            meta = {"type": "metadata", "intent": "ignore", "cached": False, "sources": []}
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            yield "data: " + json.dumps({"type": "token", "content": ignore_answer},
                                              ensure_ascii=False) + "\n\n"
            session_store.add_turn(request.session_id, request.query, ignore_answer)
            cache.set(request.query, {"intent": "ignore", "answer": ignore_answer, "sources": []})
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(ignore_event_stream(), media_type="text/event-stream")

    docs = []
    if intent in ("order", "consultant", "faq"):
        docs = hybrid_retrieve(query_for_agent, intent=intent, top_k=5)
    sources = format_sources(docs)
    prompt = build_agent_prompt(intent, query_for_agent, history, docs)

    def event_stream():
        meta = {"type": "metadata", "intent": intent, "cached": False, "sources": sources}
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        full_answer = ""
        if LLM_BACKEND == "openai_compat":
            chunk = call_llm(prompt)
            full_answer = chunk
            yield "data: " + json.dumps({"type": "token", "content": chunk},
                                              ensure_ascii=False) + "\n\n"
        else:
            for token in stream_ollama(prompt):
                full_answer += token
                yield "data: " + json.dumps({"type": "token", "content": token},
                                                  ensure_ascii=False) + "\n\n"

        session_store.add_turn(request.session_id, request.query, full_answer)
        cache.set(request.query, {"intent": intent, "answer": full_answer, "sources": sources})
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
