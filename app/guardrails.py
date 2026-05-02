"""
Guardrail checks for unsafe or out-of-scope harmful content.

This is a lightweight production-oriented safety gate for the prototype.
The chatbot scope is F&B ordering/consulting/FAQ, so harmful queries are blocked
before they reach router, retrieval, or generation.
"""

from __future__ import annotations

BLOCKED_KEYWORDS = [
    "cách chế bom",
    "chế bom",
    "làm bom",
    "bomb",
    "explosive",
    "hack",
    "phishing",
    "mã độc",
    "malware",
    "tấn công",
]


def is_guardrail_blocked(query: str) -> bool:
    """Return True if query matches blocked harmful patterns."""
    lowered = query.lower().strip()
    return any(keyword in lowered for keyword in BLOCKED_KEYWORDS)
