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

import json
from pathlib import Path
from typing import Dict

import joblib
import re
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.config import (
    OLLAMA_BASE_URL,
    SLM_ROUTER_ENABLED,
    SLM_ROUTER_MODEL,
    SLM_ROUTER_TIMEOUT_SECONDS,
)

ROUTER_MODEL_PATH = Path("models/router_model.joblib")
LORA_MODEL_DIR = Path("models/router_sft")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_learned_router_model = None
_lora_model = None
_lora_tokenizer = None
VALID_INTENTS = {"order", "consultant", "faq", "ignore", "check_order"}

ORDER_KEYWORDS = [
    "lấy",
    "mua",
    "order",
    "gọi",
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
    "đặt một",
    "đặt hai",
    "đặt luôn",
    "trà đào",
    "trà dâu",
    "trà vải",
    "trà sen",
    "trà xoài",
    "trà cam",
    "trà chanh",
    "matcha",
    "mocha",
    "latte",
    "cappuccino",
    "americano",
    "espresso",
    "bạc xỉu",
    "phin sữa",
    "croissants",
    "bánh",
]

CHECK_ORDER_KEYWORDS = [
    "kiểm tra đơn",
    "xem đơn",
    "đơn hàng",
    "đơn đã",
    "đơn của",
    "đơn hiện",
    "hóa đơn",
    "xem giỏ",
    "giỏ hàng",
    "tổng đơn",
    "đặt gì",
    "order gì",
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


def _load_lora_model():
    """Load fine-tuned LoRA model (lazy singleton)."""
    global _lora_model, _lora_tokenizer
    if _lora_model is not None:
        return True

    if not LORA_MODEL_DIR.exists():
        return False

    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            LORA_MODEL_DIR, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        base = base.to(device)

        model = PeftModel.from_pretrained(base, LORA_MODEL_DIR)
        model.eval()

        _lora_tokenizer = tokenizer
        _lora_model = model
        _lora_device = device
        return True
    except Exception:
        _lora_model = None
        return False


SYSTEM_PROMPT = "Classify into: order / consultant / faq / ignore. Return JSON only {\"action\":\"...\"}."


def classify_intent_lora(query: str) -> Dict[str, str]:
    """Classify using fine-tuned LoRA model on MPS."""
    if not _load_lora_model():
        return {}

    try:
        text = (
            "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>"
            "<|im_start|>user\n" + query + "<|im_end|>"
            "<|im_start|>assistant\n"
        )
        inputs = _lora_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(_lora_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _lora_model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
                pad_token_id=_lora_tokenizer.pad_token_id,
            )
        response = _lora_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        match = re.search(r'\{"action"\s*:\s*"([^"]+)"\}', response)
        if match:
            action = match.group(1)
            if action in VALID_INTENTS:
                return {"action": action, "method": "lora_sft"}
    except Exception:
        pass
    return {}


def classify_intent_learned(query: str) -> Dict[str, str]:
    """Classify query using TF-IDF + LogReg model with confidence threshold."""
    model = load_learned_router()
    if model is None:
        return {}

    try:
        intent = model.predict([query])[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([query])[0]
            confidence = float(max(probabilities))

        if confidence is not None and confidence < 0.50:
            return {}

        if intent not in VALID_INTENTS:
            return {}

        result = {"action": intent, "method": "learned_router"}
        if confidence is not None:
            result["confidence"] = f"{confidence:.4f}"
        return result
    except Exception:
        return {}


def classify_intent_slm(text: str) -> Dict[str, str]:
    """
    Classify user intent using a real small language model via Ollama.

    This implements the SLM router path using qwen2.5:1.5b (<3B).
    If the SLM is unavailable or returns invalid output, the system
    falls back to the learned router and rule-based router.
    """
    if not SLM_ROUTER_ENABLED:
        return {}

    prompt = f"""
You are an intent classifier for an F&B assistant.

Classify the user query into exactly one intent:
- order: user wants to order/buy/add a menu item
- consultant: user asks for recommendation or preference-based suggestion
- faq: user asks about store information, wifi, opening hours, payment, invoice
- ignore: greeting, noise, unclear or out-of-domain small talk

Return JSON only in this exact format:
{{"action":"faq"}}

User query:
{text}
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": SLM_ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 30,
                },
            },
            timeout=SLM_ROUTER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        raw = response.json().get("response", "{}")
        data = json.loads(raw)
        action = data.get("action")

        if action in VALID_INTENTS:
            return {
                "action": action,
                "method": "slm_router",
                "model": SLM_ROUTER_MODEL,
            }
    except Exception:
        return {}

    return {}


def classify_intent(text: str) -> Dict[str, str]:
    """
    Classify user query into 4 intents.

    Priority: Fine-tuned LoRA (SFT) > Ollama SLM > Learned TF-IDF > Rule-based

    Args:
        text: user input

    Returns:
        dict: {"action": intent}
    """
    query = text.lower().strip()

    if not query or len(query) <= 2:
        return {"action": "ignore", "method": "rule_based"}

    # 1. Fine-tuned LoRA (SFT on Qwen2.5-0.5B) — primary router
    lora_result = classify_intent_lora(text)
    if lora_result:
        return lora_result

    # 2. Ollama SLM fallback
    slm_result = classify_intent_slm(text)
    if slm_result:
        return slm_result

    # 3. Learned router (TF-IDF + LogReg)
    learned_result = classify_intent_learned(query)
    if learned_result:
        return learned_result

    if contains_any(query, CHECK_ORDER_KEYWORDS):
        return {"action": "check_order", "method": "rule_based"}

    if contains_any(query, FAQ_KEYWORDS):
        return {"action": "faq", "method": "rule_based"}

    if contains_any(query, CONSULTANT_KEYWORDS):
        return {"action": "consultant", "method": "rule_based"}

    if contains_any(query, ORDER_KEYWORDS):
        return {"action": "order", "method": "rule_based"}

    if contains_any(query, IGNORE_KEYWORDS):
        return {"action": "ignore", "method": "rule_based"}

    if "?" in query or query.startswith(("cho hỏi", "hỏi", "em hỏi")):
        return {"action": "faq", "method": "rule_based"}

    return {"action": "ignore", "method": "rule_based"}
