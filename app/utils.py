"""
Utility functions for local LLM generation.

This prototype uses Ollama for local inference on Mac M1 Max.
In production, this can be replaced by vLLM or SGLang on RTX 3060.
"""

from __future__ import annotations

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def call_ollama(prompt: str) -> str:
    """
    Call local Ollama model and return generated response.

    Args:
        prompt: full prompt for the model

    Returns:
        model-generated text
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as exc:
        return (
            "Dạ hiện tại hệ thống đang gặp lỗi khi gọi local LLM. "
            f"Chi tiết kỹ thuật: {str(exc)}"
        )
