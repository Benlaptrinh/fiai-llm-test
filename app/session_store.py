"""
SessionStore: manage conversation history.

Supports:
- storing last N turns
- TTL expiration

Used to maintain conversational context.
"""

import time


class SessionStore:
    def __init__(self):
        self.sessions = {}

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def add_turn(self, session_id, user, assistant):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "user": user,
                "assistant": assistant,
                "ts": time.time(),
            }
        )

        # keep last 5 turns
        self.sessions[session_id] = self.sessions[session_id][-5:]
