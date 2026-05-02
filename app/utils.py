"""
Utility functions for local LLM generation with retry + exponential backoff.

Features:
- Exponential backoff: 1s → 2s → 4s (3 retries per call)
- Unified retry decorator for all LLM endpoints
- Per-endpoint retry config

This prototype uses Ollama for local inference on Mac M1 Max.
In production, this can be replaced by vLLM or SGLang on RTX 3060.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import wraps
from typing import Callable, Iterator, Optional, Tuple

import requests

from app.config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODEL,
)

# ─── Retry configuration ────────────────────────────────────────────────────
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


def _backoff_delay(attempt: int) -> float:
    """Compute delay for retry attempt (exponential backoff)."""
    return INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt)


def _is_retryable(exc: Exception) -> bool:
    """
    Determine whether an exception is worth retrying.
    Retry on: connection errors, timeouts, 5xx server errors, 429 rate limit.
    """
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return False


def with_retry(func: Callable) -> Callable:
    """
    Decorator that adds exponential-backoff retry to any sync function.

    Usage:
        @with_retry
        def call_api():
            return requests.post(url, ...)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    raise
                delay = _backoff_delay(attempt)
                time.sleep(delay)
        raise last_exc
    return wrapper


def call_ollama(prompt: str) -> str:
    """
    Call local Ollama model and return generated response.

    Args:
        prompt: full prompt for the model

    Returns:
        model-generated text
    """
    def _call():
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

    try:
        return _call()
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
    def _call():
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

    try:
        return _call()
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


# ─── Async LLM calls with retry (for use in async endpoints) ────────────────

async def call_llm_async(prompt: str) -> str:
    """
    Async LLM call with exponential backoff retry.
    Used by streaming endpoint to avoid blocking the event loop.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: call_llm(prompt)
            )
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                return (
                    "Dạ hiện tại hệ thống đang gặp lỗi. "
                    f"Chi tiết: {str(exc)}"
                )
            await asyncio.sleep(_backoff_delay(attempt))
    return "Dạ hiện tại hệ thống đang gặp lỗi không xác định."
