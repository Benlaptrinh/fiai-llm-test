"""
Intent Extraction module for structured query parsing (C2.1).

Extracts structured components from user queries:
- subject: who is speaking (anh/em/...)
- action: what they want to do (đặt món, hỏi, tư vấn...) — used as cache key
- context: temporal/quantitative details (ngày mai, 7h, 2 ly...)

Priority: LoRA SFT > Ollama > Rule-based

C2.1 benchmark:
- Action accuracy: 84.62% (target: 90%)
- Subject accuracy: 62.62%
- Context accuracy: 93.37%
- Parse errors: 0%
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.config import OLLAMA_BASE_URL, SLM_ROUTER_MODEL

# ── LoRA Model Paths ──────────────────────────────────────────────────────────
LORA_MODEL_DIR = Path("models/intent_extraction/checkpoint-550")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ = 256

# System prompt (must match training)
SYSTEM_PROMPT = (
    "Bạn là intent extractor F&B. "
    "Trả JSON: {\"subject\":\"...\",\"action\":\"...\",\"context\":\"...\"}. "
    "action = cache key. Trả lời JSON only."
)

# ── Lazy-loaded LoRA model singleton ──────────────────────────────────────────
_lora_model = None
_lora_tokenizer = None
_lora_device = None


def _load_lora_extractor() -> bool:
    """Load fine-tuned LoRA extractor. Returns True on success."""
    global _lora_model, _lora_tokenizer, _lora_device
    if _lora_model is not None:
        return True

    if not LORA_MODEL_DIR.exists():
        return False

    try:
        _lora_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _lora_tokenizer = AutoTokenizer.from_pretrained(
            LORA_MODEL_DIR, trust_remote_code=True
        )
        _lora_tokenizer.pad_token = _lora_tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        base.config.pad_token_id = _lora_tokenizer.pad_token_id
        base = base.to(_lora_device)

        _lora_model = PeftModel.from_pretrained(base, str(LORA_MODEL_DIR))
        _lora_model.eval()
        return True
    except Exception:
        _lora_model = None
        return False


# ── Ollama prompt (fallback) ──────────────────────────────────────────────────
OLLAMA_PROMPT = """Bạn là intent extractor F&B.

Trả về JSON với 3 trường:
- "subject": chủ ngữ (anh/em/tôi/... hoặc null)
- "action": hành động chính, dùng làm cache key
- "context": thông tin bổ sung (thời gian, số lượng, ... hoặc null)

Format: {"subject": "...", "action": "...", "context": "..."}

Ví dụ:
Input: "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h"
Output: {"subject": "anh", "action": "đặt bàn", "context": "ngày mai lúc 7h"}

Input: "Wifi tên gì vậy"
Output: {"subject": null, "action": "hỏi wifi", "context": "wifi"}

Input: "có gì ngon không"
Output: {"subject": null, "action": "tư vấn theo sở thích", "context": null}

Câu truy vấn:
{query}

Trả lời (JSON only):"""


# ── Main extraction function ───────────────────────────────────────────────────
def extract_intent(query: str, timeout: int = 30) -> Dict[str, Optional[str]]:
    """
    Extract structured intent components from a query.

    Priority:
    1. LoRA SFT model (C2.1 trained) — primary
    2. Ollama qwen2.5:1.5b — fallback
    3. Rule-based — last resort

    Args:
        query: user input text
        timeout: request timeout in seconds

    Returns:
        dict with keys: subject, action, context (values may be None)
    """
    # 1. Try LoRA SFT model
    if _load_lora_extractor():
        result = _extract_with_lora(query)
        if result:
            return result

    # 2. Try Ollama
    result = _extract_with_ollama(query, timeout)
    if result:
        return result

    # 3. Rule-based fallback
    return _rule_based_extraction(query)


def _extract_with_lora(query: str) -> Optional[Dict[str, Optional[str]]]:
    """Extract using fine-tuned LoRA model."""
    try:
        text = (
            f"{_lora_tokenizer.bos_token}"
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
            f"<|im_start|>user\n{query}<|im_end|>"
            f"<|im_start|>assistant\n"
        )
        inputs = _lora_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_SEQ
        )
        inputs = {k: v.to(_lora_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = _lora_model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=_lora_tokenizer.pad_token_id,
            )

        response = _lora_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            return {
                "subject": data.get("subject"),
                "action": data.get("action"),
                "context": data.get("context"),
            }
    except Exception:
        pass
    return None


def _extract_with_ollama(
    query: str, timeout: int
) -> Optional[Dict[str, Optional[str]]]:
    """Extract using Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": SLM_ROUTER_MODEL,
                "prompt": OLLAMA_PROMPT.format(query=query),
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
        data = json.loads(raw)
        result = {
            "subject": data.get("subject"),
            "action": data.get("action"),
            "context": data.get("context"),
        }
        if result.get("action") and isinstance(result["action"], str):
            return result
    except Exception:
        pass
    return None


def _rule_based_extraction(query: str) -> Dict[str, Optional[str]]:
    """Rule-based fallback for intent extraction."""
    q = query.lower().strip()

    subject = None
    if re.search(r"\b(em|anh|mình|tôi)\b", q):
        if "em " in q or " em" in q:
            subject = "em"
        elif "anh " in q or " anh" in q:
            subject = "anh"
        elif "tôi" in q:
            subject = "tôi"
        else:
            subject = "mình"

    action = None
    if re.search(r"\b(đặt|mua|gọi|order|lấy|thêm|mang|ship|tính)\b", q):
        action = "đặt món"
    elif re.search(r"\b(wifi|giờ|mở cửa|đóng cửa|thanh toán|giao|xuất|địa|chỗ|đậu|đặt bàn|hoàn)\b", q):
        action = "hỏi thông tin"
    elif re.search(r"\b(gợi ý|tư vấn|nên uống|recommend|suggest|ngon)\b", q):
        action = "tư vấn theo sở thích"
    elif re.search(r"\b(hello|hi|chào|haha|alo|ok|bye)\b", q):
        action = "chào hỏi"
    elif re.search(r"\b(xem|kiểm tra|lịch sử)\b", q):
        action = "tra cứu"
    else:
        action = "khác"

    context_parts = []
    if re.search(r"(\d+|một|hai|ba)\s*(ly|cốc|phần|món)", q):
        m = re.search(r"(\d+|một|hai|ba)\s*(ly|cốc|phần|món)", q)
        context_parts.append(f"số lượng: {m.group(0)}")
    if re.search(r"size\s*([smlSM])", q):
        m = re.search(r"size\s*([smlSM])", q)
        context_parts.append(f"size: {m.group(1).upper()}")
    if re.search(r"(ngày mai|chiều nay|sáng nay|tối nay|\d{1,2}h|\d{1,2}:\d{2})", q):
        m = re.search(r"(ngày mai|chiều nay|sáng nay|tối nay|\d{1,2}h|\d{1,2}:\d{2})", q)
        context_parts.append(f"thời gian: {m.group(0)}")

    context = ", ".join(context_parts) if context_parts else None
    return {"subject": subject, "action": action, "context": context}


def build_cache_key(extraction: Dict[str, Optional[str]]) -> str:
    """Build a cache key from extracted intent. Uses action as cache key."""
    action = extraction.get("action") or "unknown"
    return action


def extract_with_context(extraction: Dict[str, Optional[str]]) -> str:
    """Build full context string for agent prompt enrichment."""
    parts = []
    if extraction.get("subject"):
        parts.append(f"[{extraction['subject']}]")
    if extraction.get("action"):
        parts.append(extraction["action"])
    if extraction.get("context"):
        parts.append(f"({extraction['context']})")
    return " ".join(parts)
