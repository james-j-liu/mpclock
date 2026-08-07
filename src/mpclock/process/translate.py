"""Translate non-English speeches to English before judging.

All current corpus sources already yield English text (ECB .en.html pages, the
BIS English extract, the official English speeches CSV), so this step is mostly
defensive: it detects each record's language and only calls the LLM for the rare
non-English straggler. Translated records keep their detected source language in
`orig_language` and set `translated=True`.

Translation reuses the OpenRouter chat endpoint (same provider as the judge); the
model is configurable so a cheap model can be used for this bulk, mechanical task.
"""
from __future__ import annotations

import requests
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import judge_model, openrouter_key
from ..schema import Speech

DetectorFactory.seed = 0  # deterministic detection

API_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = ("You are a professional translator. Translate the user's central-bank text into "
          "fluent English. Preserve meaning, tone, and economic terminology exactly. Output "
          "ONLY the translation, with no preamble, notes, or quotation marks.")
_CHUNK = 6000  # chars per translation call


def detect_lang(text: str) -> str:
    sample = text[:2000].strip()
    if len(sample) < 20:
        return "en"
    try:
        return detect(sample)
    except LangDetectException:
        return "en"


class Translator:
    def __init__(self, model: str | None = None):
        # default to a cheap model for bulk translation; fall back to judge model
        self.model = model or judge_model()
        self._key = openrouter_key()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call(self, text: str) -> str:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.0,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": text}]},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _translate_text(self, text: str) -> str:
        if len(text) <= _CHUNK:
            return self._call(text)
        # split on paragraph boundaries to keep chunks coherent
        parts, buf = [], ""
        for para in text.split("\n"):
            if len(buf) + len(para) + 1 > _CHUNK and buf:
                parts.append(buf)
                buf = ""
            buf += para + "\n"
        if buf.strip():
            parts.append(buf)
        return "\n".join(self._call(p) for p in parts)

    def translate(self, speech: Speech) -> Speech:
        lang = detect_lang(speech.text)
        speech.orig_language = lang
        if lang != "en":
            speech.text = self._translate_text(speech.text)
            speech.translated = True
            speech.word_count = len(speech.text.split())
        return speech

    def translate_all(self, speeches: list[Speech], concurrency: int = 8) -> list[Speech]:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        # detect first (cheap, local) so we only spin threads for real translations
        todo = []
        for s in speeches:
            s.orig_language = detect_lang(s.text)
            if s.orig_language != "en":
                todo.append(s)
        if not todo:
            return speeches
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(tqdm(ex.map(self._do, todo), total=len(todo), desc="translate"))
        return speeches

    def _do(self, s: Speech) -> Speech:
        s.text = self._translate_text(s.text)
        s.translated = True
        s.word_count = len(s.text.split())
        return s
