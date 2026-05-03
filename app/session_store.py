"""
SessionStore: manage conversation history.

Features:
- keep latest N turns
- TTL-based cleanup
- lightweight in-memory storage
- Auto-summarize when history exceeds context window threshold

Auto-summarize (A2.2):
- Triggers when estimated token count > SESSION_SUMMARY_THRESHOLD
- Summarizes older turns into a compact memory block
- Keeps the last 5 recent turns intact
- Uses LLM to generate a concise summary
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from app.config import (
    MAX_HISTORY_TURNS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SESSION_SUMMARY_THRESHOLD_TOKENS,
    SESSION_TTL_SECONDS,
)

# Rough token estimate: ~4 chars per token for Vietnamese text
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token estimation for text."""
    return len(text) // CHARS_PER_TOKEN


def _build_summarization_prompt(history: List[Dict[str, str]]) -> str:
    """Build prompt for summarizing conversation history."""
    turns = []
    for i, turn in enumerate(history, 1):
        turns.append(f"Turn {i}:\nUser: {turn['user']}\nAssistant: {turn['assistant']}")

    return f"""Bạn là một trợ lý tóm tắt cuộc hội thoại.

Hãy tóm tắt ngắn gọn cuộc hội thoại sau thành 1-2 câu tiếng Việt, giữ lại:
- Các món đã đặt hoặc đang quan tâm
- Các câu hỏi FAQ đã hỏi
- Khẩu vị / sở thích của khách (nếu có)

Cuộc hội thoại:
{chr(10).join(turns)}

Tóm tắt (1-2 câu, tiếng Việt):"""


def _summarize_with_llm(history: List[Dict[str, str]]) -> Optional[str]:
    """
    Call Ollama to summarize conversation history.
    Returns None if summarization fails.
    """
    try:
        prompt = _build_summarization_prompt(history)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 100,
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        return summary if summary else None
    except Exception:
        return None


class SessionStore:
    """
    In-memory session history manager with auto-summarize support.

    Auto-summarize (A2.2) triggers when total turns exceed a window threshold
    (tracked BEFORE truncation by MAX_HISTORY_TURNS), to preserve context from
    long conversations.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation history for a session."""
        self.cleanup()
        if session_id not in self.sessions:
            return []
        self.sessions[session_id]["last_seen"] = time.time()
        return self.sessions[session_id]["history"]

    def _total_tokens(self, history: List[Dict[str, str]]) -> int:
        """Estimate total tokens in history."""
        total = 0
        for turn in history:
            total += _estimate_tokens(turn.get("user", ""))
            total += _estimate_tokens(turn.get("assistant", ""))
        return total

    def _should_summarize(self, session_id: str) -> bool:
        """
        A2.2: Trigger summarization when cumulative turns exceed threshold.
        Uses pre-truncation total_turns (not just kept history).
        Threshold: SESSION_SUMMARY_THRESHOLD_TURENS ≈ 15 turns ≈ 750 tokens for Vietnamese.
        """
        if session_id not in self.sessions:
            return False
        total_turns = self.sessions[session_id].get("total_turns", 0)
        return total_turns > SESSION_SUMMARY_THRESHOLD_TOKENS

    def _auto_summarize(self, session_id: str) -> None:
        """
        A2.2: Summarize older turns when cumulative conversation exceeds threshold.

        Strategy: keeps last MAX_HISTORY_TURNS turns + 1 summary turn intact,
        summarizes the rest into a compact memory block.
        """
        if session_id not in self.sessions:
            return

        history = self.sessions[session_id]["history"]
        # Need at least (MAX_HISTORY_TURNS + 1) turns to have older content worth summarizing
        if len(history) <= MAX_HISTORY_TURNS:
            return

        # Keep last MAX_HISTORY_TURNS turns; older ones go to summary
        recent = history[-MAX_HISTORY_TURNS:]
        older = history[:-MAX_HISTORY_TURNS]

        summary = _summarize_with_llm(older)
        if summary:
            self.sessions[session_id]["memory_summary"] = summary
            # Replace history: summary turn + last MAX_HISTORY_TURNS turns
            self.sessions[session_id]["history"] = (
                [{"user": "[tóm tắt]", "assistant": summary}] + recent
            )

        # A2.2: Reset total_turns to reflect what we kept
        self.sessions[session_id]["total_turns"] = len(
            self.sessions[session_id]["history"]
        )

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        """Add one user-assistant turn to session history."""
        now = time.time()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_seen": now,
                "memory_summary": None,
                "total_turns": 0,
            }

        self.sessions[session_id]["history"].append(
            {"user": user, "assistant": assistant}
        )
        self.sessions[session_id]["total_turns"] += 1
        self.sessions[session_id]["last_seen"] = now

        # A2.2: Summarize BEFORE truncating so older turns are still accessible
        if self._should_summarize(session_id):
            self._auto_summarize(session_id)
        else:
            # Only truncate when NOT summarizing (summarize already handles it)
            self.sessions[session_id]["history"] = (
                self.sessions[session_id]["history"][-MAX_HISTORY_TURNS:]
            )

    def get_memory_summary(self, session_id: str) -> Optional[str]:
        """Return summarized memory for a session (if any)."""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id].get("memory_summary")

    def cleanup(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            session_id
            for session_id, data in self.sessions.items()
            if now - data["last_seen"] > SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            del self.sessions[session_id]
