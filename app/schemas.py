"""
Pydantic schemas for API request and response.

Defines the contract between client and system.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    Incoming chat request.
    """

    query: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """
    Chat response returned to client.
    Includes intent classification and sources for transparency.
    """

    session_id: str
    intent: str
    answer: str
    sources: List[Dict[str, Any]] = []
    cached: bool = False
