"""
Router Agent for intent classification.

Classifies user query into exactly one of:
- order
- consultant
- faq
- ignore

This prototype uses rule-based classification for:
- low latency
- deterministic output
- stable JSON-like result

Production extension:
- fine-tuned Qwen2.5/Qwen3 small language model
- JSON-constrained decoding
"""

from __future__ import annotations

from typing import Dict

ORDER_KEYWORDS = [
    "cho",
    "lấy",
    "mua",
    "đặt",
    "order",
    "gọi",
    "thêm",
    "tính tiền",
    "1 ly",
    "2 ly",
    "một ly",
    "hai ly",
    "size",
    "món",
    "ly",
    "bánh",
]

CONSULTANT_KEYWORDS = [
    "gợi ý",
    "tư vấn",
    "nên uống",
    "có gì ngon",
    "món nào ngon",
    "ít ngọt",
    "đắng",
    "ngọt",
    "không caffeine",
    "bán chạy",
    "recommend",
    "suggest",
    "dưới",
    "hợp",
]

FAQ_KEYWORDS = [
    "wifi",
    "mở cửa",
    "đóng cửa",
    "giờ",
    "thanh toán",
    "hóa đơn",
    "giao hàng",
    "địa chỉ",
    "chỗ ngồi",
    "ổ cắm",
    "đậu xe",
    "đặt bàn",
    "size lớn",
    "hoàn tiền",
]

IGNORE_KEYWORDS = [
    "hello",
    "hi",
    "haha",
    "ừm",
    "ờ",
    "alo",
    "test",
    "hihi",
    "ok",
    "abc",
    "xyz",
    "nghe không",
]


def contains_any(text: str, keywords: list[str]) -> bool:
    """Check whether any keyword appears in text."""
    return any(keyword in text for keyword in keywords)


def classify_intent(text: str) -> Dict[str, str]:
    """
    Classify user query into 4 intents.

    Args:
        text: user input

    Returns:
        dict: {"action": intent}
    """
    query = text.lower().strip()

    if not query or len(query) <= 2:
        return {"action": "ignore"}

    if contains_any(query, FAQ_KEYWORDS):
        return {"action": "faq"}

    if contains_any(query, ORDER_KEYWORDS):
        return {"action": "order"}

    if contains_any(query, CONSULTANT_KEYWORDS):
        return {"action": "consultant"}

    if contains_any(query, IGNORE_KEYWORDS):
        return {"action": "ignore"}

    if "?" in query or query.startswith(("cho hỏi", "hỏi", "em hỏi")):
        return {"action": "faq"}

    return {"action": "ignore"}
