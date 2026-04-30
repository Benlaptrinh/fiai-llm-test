"""
Router Agent: classify user query into one of 4 intents:
- order
- consultant
- faq
- ignore

This version uses rule-based classification for:
- low latency
- deterministic JSON output

In production, can be replaced by fine-tuned SLM.
"""

from typing import Dict


def classify_intent(text: str) -> Dict[str, str]:
    """
    Classify user query into intent.

    Args:
        text (str): user input

    Returns:
        dict: {"action": intent}
    """
    q = text.lower()

    if "wifi" in q or "mở cửa" in q:
        return {"action": "faq"}

    if "cho" in q or "mua" in q:
        return {"action": "order"}

    if "gợi ý" in q or "ngon" in q:
        return {"action": "consultant"}

    return {"action": "ignore"}
