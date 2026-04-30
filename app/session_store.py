"""
SessionStore: manage conversation history.

Features:
- keep latest N turns
- TTL-based cleanup
- lightweight in-memory storage

Production extension:
- Redis session store
- summarization for long histories
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.config import MAX_HISTORY_TURNS, SESSION_TTL_SECONDS


class SessionStore:
    """In-memory session history manager."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation history for a session."""
        self.cleanup()
        if session_id not in self.sessions:
            return []

        self.sessions[session_id]["last_seen"] = time.time()
        return self.sessions[session_id]["history"]

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        """Add one user-assistant turn to session history."""
        now = time.time()
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "last_seen": now}

        self.sessions[session_id]["history"].append({"user": user, "assistant": assistant})
        self.sessions[session_id]["history"] = self.sessions[session_id]["history"][
            -MAX_HISTORY_TURNS:
        ]
        self.sessions[session_id]["last_seen"] = now

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
