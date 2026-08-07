"""Two-stage LLM policy-relevance classifier (FedLock approach).

Stage 1 (broad): does this speech deal with monetary policy / the economic outlook
                 / inflation / interest rates at all, vs. an off-topic talk
                 (payments, supervision, fintech, market plumbing, ceremonial)?
Stage 2 (subject): of the stage-1 survivors, is monetary policy and the economy the
                  speech's SUBJECT, rather than an aside in a talk that is really
                  about regulation, financial stability or market infrastructure?

Both stages read a sample drawn from across the speech, not its opening. A Bank of
England speech PDF opens with a title page, thanks to the host and several
paragraphs of scene-setting — judging relevance on the first few thousand
characters threw away genuine monetary-policy speeches whose argument had not
started yet.

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
from ..schema import COUNCIL_TYPES, ST_REPORT, ST_TESTIMONY, Speech

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_SENTENCE_RE = re.compile(r"[a-z]{3}[^.!?\n]{50,}[.!?]")
# academic speeches close with pages of citations; sampling the end of the file
# would otherwise hand the classifier a bibliography instead of the conclusion
_REFERENCES_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)


def _body(text: str) -> str:
    m = None
    for m in _REFERENCES_RE.finditer(text):
        pass                       # the last such heading is the real one
    if m and m.start() > 0.6 * len(text):
        return text[:m.start()]
    return text


def is_prose(text: str, sample_chars: int = 6000) -> bool:
    """Cheap table detector: a projection-table annex is not judgeable text."""
    sample = text[:sample_chars]
    digits = sum(c.isdigit() for c in sample)
    letters = sum(c.isalpha() for c in sample)
    if not letters or digits / (digits + letters) > 0.15:
        return False
    return len(_SENTENCE_RE.findall(sample)) >= 3

STAGE1 = """You classify central-bank speeches. The excerpt is sampled from across one
speech (opening, middle, end), so judge the speech as a whole, not the passage in front of
you. Decide if it engages with monetary policy or the macroeconomic outlook: inflation,
interest rates, the policy stance or its transmission, growth, the labour market, the
forecast, or the monetary-policy framework. Answer true even when the argument is
analytical, historical or methodological, as long as it bears on monetary policy or the
economy. Answer false for talks whose subject is payment systems, market infrastructure,
bank supervision or regulation, resolution, fintech, digital currency design, cyber risk,
statistics, ceremony, or institutional history.
Respond ONLY with JSON: {"policy": true|false}."""

STAGE2 = """You are building a corpus of monetary-policy communication and are checking the
previous filter. The excerpt is sampled from across one speech. Decide whether monetary
policy, inflation, or the economic outlook is the SUBJECT of this speech, rather than a
passing reference in a speech that is really about something else (financial stability,
prudential regulation, payments, market operations, climate, an institution or a career).
Keep every speech whose substance is the economy or monetary policy, INCLUDING ones that
analyse the outlook, the forecast, the transmission mechanism, the policy framework or the
policy trade-offs without stating an explicit hawkish or dovish preference — a reasoned
analysis of inflation or the rate path belongs in the corpus.
Respond ONLY with JSON: {"policy": true|false}."""


class Classifier:
    def __init__(self, model: str | None = None, max_chars: int = 7500):
        self.model = model or judge_model()
        self.max_chars = max_chars
        self._key = openrouter_key()

    def excerpt(self, speech: Speech) -> str:
        """Title plus three windows — opening, middle, close — of the speech.

        Sampling beats truncation here: the opening is host thanks and framing, the
        argument sits in the middle, and the policy conclusion sits at the end.
        """
        text = _body(speech.text)
        head = f"TITLE: {speech.title}\n\n"
        if len(text) <= self.max_chars:
            return head + text
        w = self.max_chars // 3
        mid = (len(text) - w) // 2
        return (head + text[:w]
                + "\n\n[…]\n\n" + text[mid:mid + w]
                + "\n\n[…]\n\n" + text[-w:])

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
        if speech.source_type in COUNCIL_TYPES or speech.source_type == ST_TESTIMONY:
            # Treasury Committee evidence is ingested only for the Monetary Policy
            # Report sessions, so the subject is settled before the text is read
            return True
        excerpt = self.excerpt(speech)
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
