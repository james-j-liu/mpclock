"""Five-layer anonymization, mirroring FedLock.

Goal: strip every name, title, and structural label a judge could use to
identify the speaker, while preserving concept/place names (Phillips curve,
Taylor rule, Sintra forum).

Layers (applied in order):
  1. Structural speaker labels:   "President Lagarde:"  -> "SPEAKER:"
  2. Reporter/journalist intros:   media name + outlet  -> "REPORTER:"
  3. Title + name in running text: "Governor Nagel"      -> "[OFFICIAL]"
  4. Full names / aliases:         "Christine Lagarde"   -> "[OFFICIAL]"
  5. Last-name-only mentions:      "as Draghi argued"    -> "as [OFFICIAL] argued"
"""
from __future__ import annotations

import re
import unicodedata

from .roster import EXCLUSIONS, TITLES

OFFICIAL = "[OFFICIAL]"
TITLE_ALT = "|".join(sorted((re.escape(t) for t in TITLES), key=len, reverse=True))


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _last_name(full: str) -> str:
    # crude: last whitespace-separated token, handles "de Guindos" -> "Guindos"
    parts = full.replace("-", " ").split()
    return parts[-1] if parts else full


class Anonymizer:
    def __init__(self, roster: list[str]):
        self.full_names = sorted(set(roster), key=len, reverse=True)
        # surname -> True, minus exclusions
        self.surnames = []
        for n in self.full_names:
            ln = _last_name(n)
            if len(ln) >= 3 and _strip_accents(ln).lower() not in EXCLUSIONS:
                self.surnames.append(ln)
        self.surnames = sorted(set(self.surnames), key=len, reverse=True)
        self._compile()

    def _compile(self):
        # Layer 1: speaker turn labels at line start, e.g. "President Lagarde:" or "Lagarde:"
        sur = "|".join(re.escape(s) for s in self.surnames)
        self.re_label = re.compile(
            rf"(?im)^\s*(?:(?:{TITLE_ALT})\s+)?(?:{sur})\s*[:.]",
        )
        # Layer 2: ECB Q&A question intros sometimes carry the journalist + outlet
        self.re_reporter = re.compile(
            r"(?im)^\s*(?:Question|Q)\s*(?:from|by)?\s*[^:\n]{0,60}?:",
        )
        # Layer 3: title + (any capitalized name token)
        self.re_title_name = re.compile(
            rf"(?:{TITLE_ALT})\s+[A-ZÀ-Ý][\wÀ-ÿ'’-]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ'’-]+){{0,2}}",
        )
        # Layer 4: full names
        self.re_full = [re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE)
                        for n in self.full_names]
        # Layer 5: surname-only
        self.re_surname = re.compile(rf"\b(?:{sur})\b") if self.surnames else None

    def __call__(self, text: str) -> str:
        if not text:
            return text
        t = self.re_label.sub("SPEAKER:", text)
        t = self.re_reporter.sub("REPORTER:", t)
        t = self.re_title_name.sub(OFFICIAL, t)
        for rx in self.re_full:
            t = rx.sub(OFFICIAL, t)
        if self.re_surname:
            t = self.re_surname.sub(OFFICIAL, t)
        # collapse runs like "[OFFICIAL] [OFFICIAL]"
        t = re.sub(r"(?:\[OFFICIAL\]\s*){2,}", OFFICIAL + " ", t)
        return t
