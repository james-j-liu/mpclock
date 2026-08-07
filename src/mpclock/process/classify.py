"""Two-stage LLM policy-relevance classifier (FedLock approach).

Stage 1 (broad): is this primarily about monetary policy / the economic outlook
                 / inflation / interest rates, vs. an off-topic talk (payments,
                 supervision, fintech, market plumbing, ceremonial)?
Stage 2 (strict): of the stage-1 survivors, is the PRIMARY PURPOSE to communicate
                  a monetary-policy *view/stance* (hawkish/dovish signal), rather
                  than describe operations or institutions neutrally?

Only documents passing both stages get is_policy=True and enter the tournament.
The Committee's own output (Monetary Policy Summary, press conference, minutes) is
policy by construction and bypasses the classifier. Monetary Policy Report sections
bypass it too, but must first look like prose: the Report's annexes include pure
projection tables and chart appendices, which carry no stance to judge.
"""
from __future__ import annotations

import json
import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import judge_model, openrouter_key
from ..schema import COUNCIL_TYPES, ST_REPORT, Speech

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_SENTENCE_RE = re.compile(r"[a-z]{3}[^.!?\n]{50,}[.!?]")


def is_prose(text: str, sample_chars: int = 6000) -> bool:
    """Cheap table detector: a projection-table annex is not judgeable text."""
    sample = text[:sample_chars]
    digits = sum(c.isdigit() for c in sample)
    letters = sum(c.isalpha() for c in sample)
    if not letters or digits / (digits + letters) > 0.15:
        return False
    return len(_SENTENCE_RE.findall(sample)) >= 3

STAGE1 = """You classify central-bank speeches. Decide if the excerpt is PRIMARILY about
monetary policy or the macroeconomic outlook: inflation, interest rates, the policy stance,
growth, employment, the economy. Talks whose main subject is payment systems, market
infrastructure, bank supervision/regulation, fintech, digital currency design, statistics,
ceremony, or institutional history are NOT policy-relevant.
Respond ONLY with JSON: {"policy": true|false}."""

STAGE2 = """You are a strict editor building a corpus of monetary-policy SIGNALS. The excerpt
already mentions the economy. Decide if its PRIMARY PURPOSE is to communicate a view or
stance on monetary policy (e.g. arguing for/against tightening or easing, assessing whether
policy is appropriately calibrated, signalling concern about inflation or growth) — as
opposed to neutrally describing mechanics, institutions, data, or history without taking a
stance. Keep only genuine stance-bearing policy communication.
Respond ONLY with JSON: {"policy": true|false}."""


class Classifier:
    def __init__(self, model: str | None = None, max_chars: int = 4000):
        self.model = model or judge_model()
        self.max_chars = max_chars
        self._key = openrouter_key()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _ask(self, system: str, content: str) -> bool:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": content}]},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return bool(json.loads(m.group(0)).get("policy"))
            except (json.JSONDecodeError, ValueError):
                pass
        return "true" in raw.lower()

    def classify(self, speech: Speech) -> bool:
        if speech.source_type == ST_REPORT:
            return is_prose(speech.text)
        if speech.source_type in COUNCIL_TYPES:
            return True
        excerpt = f"TITLE: {speech.title}\n\n{speech.text[:self.max_chars]}"
        if not self._ask(STAGE1, excerpt):
            return False
        return self._ask(STAGE2, excerpt)

    def classify_all(self, speeches: list[Speech], concurrency: int = 8) -> list[Speech]:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        def work(s: Speech):
            s.is_policy = self.classify(s)
            return s

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(tqdm(ex.map(work, speeches), total=len(speeches), desc="classify"))
        return speeches
