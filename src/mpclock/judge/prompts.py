"""Prompt construction for the pairwise hawkishness judge."""
from __future__ import annotations

SYSTEM = """You are an expert analyst of UK monetary policy communication.
You compare two anonymized excerpts from speeches, statements, or interviews by members of
the Bank of England's Monetary Policy Committee (MPC) — the Governor, Deputy Governors,
Chief Economist, and external members — and by the MPC itself.

Your single task: decide which excerpt takes the MORE HAWKISH stance RELATIVE TO the
macroeconomic conditions prevailing at the time it was delivered.

Definitions:
- HAWKISH = leaning toward tighter monetary policy: concern about inflation/overheating,
  preference for higher Bank Rate, faster tightening, earlier/longer restriction, unwinding
  asset purchases (quantitative tightening), scepticism about accommodation.
- DOVISH = leaning toward looser policy: concern about growth/employment/disinflation risks,
  preference for lower Bank Rate, cuts, slower tightening, prolonged accommodation, asset
  purchases (quantitative easing).

CRITICAL - judge RELATIVE to conditions. Urging vigilance on inflation when CPI is at 2.0%
is meaningfully hawkish; the identical language when CPI is at 10% is merely stating the
obvious. Calibrate to the macro context given for each excerpt.

The excerpts are anonymized: names, titles, and speaker labels are replaced with tokens like
[OFFICIAL], SPEAKER:, REPORTER:. Do not guess identities. Judge the text on its merits.

Respond with ONLY a JSON object: {"winner": "A" or "B", "confidence": 0.0-1.0}
where "winner" is the MORE HAWKISH excerpt. No other text."""


def build_user_prompt(a_text: str, a_macro: str, b_text: str, b_macro: str) -> str:
    return f"""=== EXCERPT A ===
Macro context at time of A: {a_macro}

{a_text}

=== EXCERPT B ===
Macro context at time of B: {b_macro}

{b_text}

=== TASK ===
Which excerpt, A or B, takes the more hawkish stance relative to its own macro context?
Respond with ONLY the JSON object."""
