"""Judges backed by TypeSafe's Jev decision model (via OpenRouter).

Jev is a structured-decision model: rather than generating text it answers a typed
question and returns a probability per option. Output tokens are free, input costs
$0.042/M — about a fifth of the chat judge — so documents can be read far more
fully for less. Two primitives are used:

  choice  pairwise: which of two excerpts is the more hawkish (JevJudge)
  score   direct: a probability-weighted position on a 7-level scale (JevScorer),
          which is continuous, unlike the round numbers chat models emit

Jev's context window is 32k tokens. Two pairwise excerpts plus the instructions
must fit inside it, so each is capped at ~56k characters; that still reads the
large majority of the corpus whole. Anything longer — every Monetary Policy
Report, a few long minutes and transcripts — is sampled from its opening, middle
and close by the same function the chat judge uses, not simply cut off. A request
that still overflows (numerical text tokenises densely) is retried with the cap
shrunk rather than failed.
"""
from __future__ import annotations

import random
import time

import requests

from ..config import openrouter_key
from .direct import DirectScorer
from .openrouter import excerpt

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"

_STANCE = (
    "Hawkish = leaning toward tighter policy: concern about inflation, inflation "
    "persistence or second-round effects, a higher Bank Rate, tightening sooner or "
    "holding restrictive policy for longer, quantitative tightening, scepticism about "
    "cutting. Dovish = concern about growth, the labour market or an inflation "
    "undershoot, cuts to Bank Rate, easing sooner or faster, quantitative easing. Judge "
    "RELATIVE to conditions: urging vigilance on inflation with CPI at the 2% target is "
    "meaningfully hawkish; the same words with CPI at 10% merely state the obvious."
)
INSTRUCTIONS = (
    "Two anonymized excerpts from UK monetary policy communication are given — "
    "speeches, evidence, votes or published documents of members of the Bank of "
    "England's Monetary Policy Committee (MPC), or of the Committee itself — each with "
    "the macroeconomic context at the time it was delivered. Decide which excerpt "
    "takes the MORE HAWKISH stance RELATIVE TO its own macro context. " + _STANCE +
    " Names are replaced by tokens like [OFFICIAL]; do not guess identities."
)
CRITERIA = {"A": "Excerpt A is more hawkish relative to its macro context.",
            "B": "Excerpt B is more hawkish relative to its macro context."}

LEVELS = ["Very dovish", "Dovish", "Slightly dovish", "Neutral",
          "Slightly hawkish", "Hawkish", "Very hawkish"]
SCORE_INSTRUCTIONS = (
    "Rate the monetary-policy stance of this anonymized excerpt from UK monetary policy "
    "communication (a member of the Bank of England's Monetary Policy Committee, or the "
    "Committee itself) RELATIVE TO the macroeconomic context given. " + _STANCE
)

_MIN_CAP = 8000          # below this a shrinking retry gives up


class _Client:
    """POST to the decisions endpoint with rate-limit-aware retries; tracks spend."""

    def __init__(self, timeout: int = 120, max_attempts: int = 8):
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._key = openrouter_key()
        self.cost = 0.0      # running $ total, from the API's own usage.cost

    def post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                r = requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt + random.random(), 60))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                time.sleep(min(2 ** attempt + random.random(), 60))
                continue
            if r.status_code >= 400:
                # 400 is how an over-long request comes back; callers shrink on it
                raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
            res = r.json()
            self.cost += float(res.get("usage", {}).get("cost") or 0)
            return res
        raise last_err or RuntimeError("Jev request exhausted retries")

    def ask_shrinking(self, build, cap: int) -> dict:
        """Call build(cap) -> payload; on a 400 retry with a smaller cap."""
        while True:
            try:
                return self.post(build(cap))
            except requests.HTTPError as e:
                if str(e).startswith("400") and cap > _MIN_CAP:
                    cap = int(cap * 0.7)
                    continue
                raise


class JevJudge:
    """Pairwise judge: same compare() signature and result shape as openrouter.Judge,
    plus p_a, Jev's probability that A is the more hawkish."""

    def __init__(self, model: str = MODEL, max_excerpt_chars: int = 56000):
        self.model = model
        self.max_excerpt_chars = max_excerpt_chars
        self._client = _Client()

    @property
    def cost(self) -> float:
        return self._client.cost

    def compare(self, a_text: str, a_macro: str, b_text: str, b_macro: str,
                a_type: str = "", b_type: str = "") -> dict:
        def build(cap: int) -> dict:
            # no type is uncapped here: the 32k context is a hard limit, so the
            # minutes and the Report are sampled at the Jev cap like everything else
            return {
                "model": self.model,
                "state": {"excerpt_A": {"macro_context": a_macro, "text": excerpt(a_text, cap)},
                          "excerpt_B": {"macro_context": b_macro, "text": excerpt(b_text, cap)}},
                "questions": {"more_hawkish": {"type": "choice", "instructions": INSTRUCTIONS,
                                               "criteria": CRITERIA}},
            }

        res = self._client.ask_shrinking(build, self.max_excerpt_chars)
        ans = res["answers"]["more_hawkish"]
        w = ans.get("choice")
        p_a = float(ans.get("probabilities", {}).get("A", 0.5))
        if w not in ("A", "B"):
            return {"winner": None, "confidence": 0.0, "p_a": p_a}
        return {"winner": w, "confidence": float(ans.get("confidence", max(p_a, 1 - p_a))),
                "p_a": p_a}


class JevScorer(DirectScorer):
    """Direct score from Jev's ordered 7-level scale, mapped to the same 0-100 units
    as the pairwise ratings (Neutral = 50)."""

    def __init__(self, model: str = MODEL, max_excerpt_chars: int = 100000):
        self.model = model
        self.max_excerpt_chars = max_excerpt_chars   # one excerpt alone fits 32k tokens
        self.uncapped_types = set()
        self._client = _Client()

    @property
    def cost(self) -> float:
        return self._client.cost

    def score(self, text: str, macro: str, source_type: str = "") -> float | None:
        def build(cap: int) -> dict:
            return {"model": self.model,
                    "state": {"macro_context": macro, "excerpt": excerpt(text, cap)},
                    "questions": {"stance": {"type": "score", "instructions": SCORE_INSTRUCTIONS,
                                             "criteria": LEVELS}}}
        try:
            res = self._client.ask_shrinking(build, self.max_excerpt_chars)
        except Exception:  # noqa: BLE001 - exhausted retries: skip this one, not the run
            return None
        s = res["answers"]["stance"].get("score")
        return None if s is None else round(float(s) / (len(LEVELS) - 1) * 100, 2)
