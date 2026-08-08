"""Direct LLM hawkishness scoring (the alternative to the pairwise tournament).

FedLock's methodology page contrasts the pairwise tournament with a simpler
baseline: ask the model to rate each speech directly on a 0-100 hawkish-dovish
scale. We reproduce that here for the same corpus so the two methods can be
compared side by side on the site.

Direct scoring is one call per speech (N calls, vs ~15*N for the tournament), so
it is far cheaper but tends to cluster on round numbers and discriminates less
finely - which is exactly the limitation the pairwise method exists to overcome.

Same inputs as the judge: the anonymized excerpt + point-in-time macro context, so
the score is hawkishness *relative to conditions*, not in absolute terms.
"""
from __future__ import annotations

import json
import re

from ..config import cfg, judge_model, openrouter_key
from ..schema import Speech
from .openrouter import chat_completion, excerpt

# Scored on a signed -100..+100 scale (wider range -> finer resolution than 0-100,
# which halves the LLM's round-number clustering). The score is then mapped back to
# the same 0-100 internal units as the pairwise method via signed/2 + 50.
SYSTEM = """You are an expert analyst of UK monetary policy communication.
Rate a single anonymized excerpt from a Bank of England Monetary Policy Committee speech,
statement, or interview on a HAWKISH-DOVISH scale from -100 to +100.

Scale:
- +100 = maximally HAWKISH: strong concern about inflation/overheating, urging tighter
  policy, higher Bank Rate, faster/longer restriction, quantitative tightening.
-    0 = NEUTRAL / balanced.
- -100 = maximally DOVISH: concern about growth/employment/disinflation, urging looser
  policy, Bank Rate cuts, prolonged accommodation, quantitative easing.

CRITICAL - score RELATIVE to the macroeconomic conditions given. Urging vigilance on
inflation when CPI is at 2.0% is meaningfully hawkish; identical language when CPI is at
10% is merely stating the obvious and should score closer to neutral. Calibrate to context.

Use the full range and the precision the text warrants - distinguish fine gradations of tone.

The excerpt is anonymized (names/titles replaced with [OFFICIAL], SPEAKER:, REPORTER:);
do not guess identities - judge the text on its merits.

Respond with ONLY a JSON object: {"score": <number from -100 to 100>}. No other text."""


def build_prompt(text: str, macro: str) -> str:
    return (f"Macro context at the time: {macro}\n\n"
            f"=== EXCERPT ===\n{text}\n\n=== TASK ===\n"
            f"Rate this excerpt from -100 (most dovish) to +100 (most hawkish) relative to "
            f"its macro context. Respond with ONLY the JSON object.")


class DirectScorer:
    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_excerpt_chars: int | None = None):
        self.model = model or judge_model()
        self.temperature = temperature
        self.max_excerpt_chars = max_excerpt_chars or cfg()["judge"].get(
            "max_excerpt_chars", 9000)
        self.uncapped_types = set(cfg()["judge"].get("uncapped_types") or ())
        self._key = openrouter_key()

    def _excerpt(self, text: str, source_type: str = "") -> str:
        return excerpt(text, self.max_excerpt_chars, source_type, self.uncapped_types)

    def _call(self, user: str) -> str:
        return chat_completion(self._key, self.model, SYSTEM, user,
                               temperature=self.temperature)

    def score(self, text: str, macro: str, source_type: str = "") -> float | None:
        """Return the score on the 0-100 internal scale (same units as pairwise)."""
        try:
            raw = self._call(build_prompt(self._excerpt(text, source_type), macro))
        except Exception:
            return None  # exhausted retries -> skip this one, don't kill the run
        signed = self._parse(raw)
        if signed is None:
            return None
        return round(signed / 2 + 50, 2)   # -100..+100 -> 0..100 (pairwise units)

    @staticmethod
    def _parse(raw: str) -> float | None:
        """Parse the raw signed score in [-100, 100]."""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                v = float(json.loads(m.group(0)).get("score"))
                return max(-100.0, min(100.0, v))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if m:
            try:
                return max(-100.0, min(100.0, float(m.group(0))))
            except ValueError:
                pass
        return None

    def score_all(self, speeches: list[Speech], macro, concurrency: int = 8) -> list[Speech]:
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        def work(s: Speech):
            val = self.score(s.text_anon or s.text, macro.string(s.date), s.source_type)
            if val is not None:
                s.direct_score = round(val, 2)
            return s

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(tqdm(ex.map(work, speeches), total=len(speeches),
                      desc=f"direct[{self.model}]"))
        return speeches
