"""
Intent Extraction module for structured query parsing.

Extracts structured components from user queries:
- subject: who is speaking (anh/em/...)
- action: what they want to do (đặt món, hỏi, tư vấn...)
- context: temporal/quantitative details (ngày mai, 7h, 2 ly...)

This module uses a SLM (qwen2.5:1.5b) via Ollama to extract structured
intent components. The `action` field is used as cache key; the `context`
field is passed to the appropriate agent.

For production (RTX 3060): Fine-tuned Qwen2.5-0.5B with SFT.
For M1 Max: Uses Ollama + MLX backend (qwen2.5:1.5b).
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

import requests

from app.config import OLLAMA_BASE_URL, SLM_ROUTER_MODEL

EXTRACTION_PROMPT = """Bạn là một intent extractor cho ứng dụng F&B.

Phân tích câu truy vấn và trả về JSON với 3 trường:
- "subject": chủ ngữ (anh/em/tôi/... hoặc null nếu không rõ)
- "action": hành động chính (đặt món, hỏi thông tin, tư vấn, chào hỏi, ...), dùng làm cache key
- "context": thông tin bổ sung (thời gian, số lượng, ngày tháng, ... hoặc null)

Trả về JSON theo format:
{{"subject": "...", "action": "...", "context": "..."}}

Ví dụ:
Input: "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé"
Output: {{"subject": "anh", "action": "đặt bàn", "context": "ngày mai lúc 7h, tiệc sinh nhật"}}

Input: "Em lấy 2 ly Latte size M"
Output: {{"subject": "em", "action": "đặt món", "context": "2 ly, Latte, size M"}}

Input: "Wifi tên gì vậy"
Output: {{"subject": null, "action": "hỏi thông tin", "context": "wifi"}}

Input: "có gì ngon không"
Output: {{"subject": null, "action": "tư vấn", "context": null}}

Câu truy vấn:
{query}

Trả lời (JSON only):"""


def extract_intent(query: str, timeout: int = 30) -> Dict[str, Optional[str]]:
    """
    Extract structured intent components from a query.

    Uses Ollama qwen2.5:1.5b for extraction.
    Falls back to rule-based extraction on failure.

    Args:
        query: user input text
        timeout: request timeout in seconds

    Returns:
        dict with keys: subject, action, context (values may be None)
    """
    # Try SLM extraction
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": SLM_ROUTER_MODEL,
                "prompt": EXTRACTION_PROMPT.format(query=query),
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 80,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}").strip()

        # Try to parse JSON from response
        data = json.loads(raw)
        result = {
            "subject": data.get("subject"),
            "action": data.get("action"),
            "context": data.get("context"),
        }

        # Validate result
        if result["action"] and isinstance(result["action"], str):
            return result
    except Exception:
        pass

    # Fallback: rule-based extraction
    return _rule_based_extraction(query)


def _rule_based_extraction(query: str) -> Dict[str, Optional[str]]:
    """
    Rule-based fallback for intent extraction.

    Used when the SLM is unavailable or returns invalid output.
    """
    q = query.lower().strip()

    # Subject detection
    subject = None
    if re.search(r"\b(em|anh|mình|tôi|mình)\b", q):
        if "em " in q or " em" in q:
            subject = "em"
        elif "anh " in q or " anh" in q:
            subject = "anh"
        elif "tôi" in q:
            subject = "tôi"
        else:
            subject = "mình"

    # Action detection
    action = None
    if re.search(r"\b(đặt|mua|gọi|order|lấy|thêm|mang|ship|tính)\b", q):
        action = "đặt món"
    elif re.search(r"\b(wifi|giờ|mở cửa|đóng cửa|thanh toán|giao|xuất|địa|chỗ|đậu|đặt bàn|hoàn)\b", q):
        action = "hỏi thông tin"
    elif re.search(r"\b(gợi ý|tư vấn|nên uống|recommend|suggest)\b", q):
        action = "tư vấn"
    elif re.search(r"\b(hello|hi|chào|haha|alo|ok|bye)\b", q):
        action = "chào hỏi"
    elif re.search(r"\b(xem|kiểm tra|lịch sử)\b", q):
        action = "tra cứu"
    else:
        action = "khác"

    # Context extraction: quantities, times, sizes
    context_parts = []

    qty_match = re.search(r"(\d+|một|hai|ba|bốn|năm)\s*(ly|cốc|phần|món)", q)
    if qty_match:
        context_parts.append(f"số lượng: {qty_match.group(0)}")

    size_match = re.search(r"size\s*([smlSM])", q)
    if size_match:
        context_parts.append(f"size: {size_match.group(1).upper()}")

    time_match = re.search(r"(ngày mai|chiều nay|sáng nay|tối nay|\d{1,2}h|\d{1,2}:\d{2})", q)
    if time_match:
        context_parts.append(f"thời gian: {time_match.group(0)}")

    context = ", ".join(context_parts) if context_parts else None

    return {"subject": subject, "action": action, "context": context}


def build_cache_key(extraction: Dict[str, Optional[str]]) -> str:
    """
    Build a cache key from extracted intent components.

    The action field (without context/timing) is used as cache key
    to match semantically equivalent queries with different contexts.
    """
    action = extraction.get("action") or "unknown"
    return action


def extract_with_context(extraction: Dict[str, Optional[str]]) -> str:
    """
    Build the full query context string from extraction result.

    Combines subject + action + context for agent prompt enrichment.
    """
    parts = []
    if extraction.get("subject"):
        parts.append(f"[{extraction['subject']}]")
    if extraction.get("action"):
        parts.append(extraction["action"])
    if extraction.get("context"):
        parts.append(f"({extraction['context']})")
    return " ".join(parts)
