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

from pathlib import Path
from typing import Dict

import joblib

ROUTER_MODEL_PATH = Path("models/router_model.joblib")
_learned_router_model = None

ORDER_KEYWORDS = [
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
    "cho anh",
    "cho em",
    "cho tôi",
    "size s",
    "size m",
    "size l",
    "size",
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


def load_learned_router():
    """
    Load trained router model if available.

    If the model is missing or cannot be loaded, the system falls back
    to the deterministic rule-based router.
    """
    global _learned_router_model

    if _learned_router_model is not None:
        return _learned_router_model

    if not ROUTER_MODEL_PATH.exists():
        return None

    try:
        _learned_router_model = joblib.load(ROUTER_MODEL_PATH)
        return _learned_router_model
    except Exception:
        _learned_router_model = None
        return None


def classify_intent_learned(query: str) -> Dict[str, str]:
    """
    Classify query using trained lightweight model with confidence threshold.
    """
    model = load_learned_router()

    if model is None:
        return {}

    try:
        intent = model.predict([query])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([query])[0]
            confidence = float(max(probabilities))

        # Fall back to rule-based routing on uncertain predictions.
        if confidence is not None and confidence < 0.50:
            return {}

        if intent not in {"order", "consultant", "faq", "ignore"}:
            return {}

        result = {"action": intent, "method": "learned_router"}
        if confidence is not None:
            result["confidence"] = f"{confidence:.4f}"
        return result
    except Exception:
        return {}


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

    learned_result = classify_intent_learned(query)
    if learned_result:
        return learned_result

    if contains_any(query, FAQ_KEYWORDS):
        return {"action": "faq"}

    if contains_any(query, CONSULTANT_KEYWORDS):
        return {"action": "consultant"}

    if contains_any(query, ORDER_KEYWORDS):
        return {"action": "order"}

    if contains_any(query, IGNORE_KEYWORDS):
        return {"action": "ignore"}

    if "?" in query or query.startswith(("cho hỏi", "hỏi", "em hỏi")):
        return {"action": "faq"}

    return {"action": "ignore"}
