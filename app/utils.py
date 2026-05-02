"""
Utility functions for local LLM generation.

This prototype uses Ollama for local inference on Mac M1 Max.
In production, this can be replaced by vLLM or SGLang on RTX 3060.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from app.config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODEL,
)


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


def stream_ollama(prompt: str) -> Iterator[str]:
    """
    Stream tokens from local Ollama model.

    This is used by the SSE endpoint to demonstrate token-level streaming.
    In production, this can be replaced by vLLM/SGLang streaming API.
    """
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300,
                },
            },
            timeout=180,
            stream=True,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                payload = json.loads(line.decode("utf-8"))
                if payload.get("done"):
                    break

                token = payload.get("response", "")
                if token:
                    yield token

    except Exception as exc:
        yield (
            "Dạ hiện tại hệ thống đang gặp lỗi khi streaming local LLM. "
            f"Chi tiết kỹ thuật: {str(exc)}"
        )


def call_openai_compat(prompt: str) -> str:
    """
    Call OpenAI-compatible chat completion API.

    This adapter allows the same application to use vLLM/SGLang in production
    by changing environment variables only.
    """
    try:
        response = requests.post(
            f"{OPENAI_COMPAT_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_COMPAT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_COMPAT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as exc:
        return (
            "Dạ hiện tại hệ thống đang gặp lỗi khi gọi OpenAI-compatible LLM backend. "
            f"Chi tiết kỹ thuật: {str(exc)}"
        )


def call_llm(prompt: str) -> str:
    """
    Unified LLM entrypoint.

    LLM_BACKEND:
    - ollama: local development on Mac
    - openai_compat: vLLM/SGLang/OpenAI-compatible deployment
    """
    if LLM_BACKEND == "openai_compat":
        return call_openai_compat(prompt)

    return call_ollama(prompt)
