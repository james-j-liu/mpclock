"""Provider-agnostic pairwise judge over the OpenRouter API.

Any OpenRouter model id works as the judge (config string). Default is
google/gemini-2.0-flash-001 for FedLock parity. Returns "A" or "B" (the more
hawkish excerpt) plus a confidence. To neutralise position bias, callers should
randomize which speech is A vs B (the engine does this).
"""
from __future__ import annotations

import json
import random
import re
import time

import requests

from ..config import judge_model, openrouter_key
from . import prompts

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_completion(key: str, model: str, system: str, user: str, *,
                    temperature: float = 0.0, timeout: int = 90,
                    max_attempts: int = 8) -> str:
    """POST a chat completion with rate-limit-aware retries.

    Retries on 429 / 5xx (honouring the Retry-After header when present) and on
    transient network errors, with exponential backoff + jitter. Raises the last
    error only after exhausting max_attempts, so a single hiccup never aborts a
    long run when the caller also guards the call.
    """
    payload = {"model": model, "temperature": temperature,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:  # connection/timeout
            last_err = e
            time.sleep(min(2 ** attempt + random.random(), 60))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            ra = resp.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else None
            except ValueError:
                wait = None
            if wait is None:
                wait = min(2 ** attempt + random.random(), 60)
            last_err = requests.HTTPError(f"{resp.status_code} {resp.reason}")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise last_err or RuntimeError("chat_completion exhausted retries")


class Judge:
    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_excerpt_chars: int = 9000):
        self.model = model or judge_model()
        self.temperature = temperature
        self.max_excerpt_chars = max_excerpt_chars
        self._key = openrouter_key()

    def _excerpt(self, text: str) -> str:
        if len(text) <= self.max_excerpt_chars:
            return text
        # keep head + tail (intro framing + conclusions often carry the stance)
        head = self.max_excerpt_chars * 2 // 3
        tail = self.max_excerpt_chars - head
        return text[:head] + "\n[...]\n" + text[-tail:]

    def _call(self, system: str, user: str) -> str:
        return chat_completion(self._key, self.model, system, user,
                               temperature=self.temperature)

    def compare(self, a_text: str, a_macro: str, b_text: str, b_macro: str) -> dict:
        """Return {'winner': 'A'|'B', 'confidence': float}."""
        user = prompts.build_user_prompt(
            self._excerpt(a_text), a_macro, self._excerpt(b_text), b_macro
        )
        raw = self._call(prompts.SYSTEM, user)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                w = str(obj.get("winner", "")).strip().upper()
                if w in ("A", "B"):
                    conf = float(obj.get("confidence", 0.5))
                    return {"winner": w, "confidence": max(0.0, min(1.0, conf))}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # fallback: look for a bare A/B
        up = raw.strip().upper()
        if up.startswith("A"):
            return {"winner": "A", "confidence": 0.5}
        if up.startswith("B"):
            return {"winner": "B", "confidence": 0.5}
        return {"winner": None, "confidence": 0.0}
