"""Five-layer anonymization, mirroring FedLock.

Goal: strip every name, title, and structural label a judge could use to
identify the speaker, while preserving concept/place names (Phillips curve,
Taylor rule, Sintra forum).

Layers (applied in order):
  1. Structural speaker labels:   "Governor Bailey:"    -> "SPEAKER:"
  2. Reporter/journalist intros:   media name + outlet  -> "REPORTER:"
  3. Title + name in running text: "Governor Nagel"      -> "[OFFICIAL]"
  4. Full names / aliases:         "Andrew Bailey"       -> "[OFFICIAL]"
  5. Last-name-only mentions:      "as Carney argued"    -> "as [OFFICIAL] argued"
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


_PARTICLES = {"de", "da", "di", "del", "van", "von", "der", "le", "la", "dos", "das"}


def _surname_forms(full: str) -> list[str]:
    """Ways a person is referred to by surname alone: the last token ("Galhau"),
    the hyphenated whole ("González-Páramo"), and for compound names the part
    before a particle ("Villeroy" in "François Villeroy de Galhau"; not the first
    name in "Luis de Guindos")."""
    forms = [_last_name(full)]
    toks = full.split()
    if toks and "-" in toks[-1]:
        forms.append(toks[-1])
    for i, t in enumerate(toks):
        if t.lower() in _PARTICLES and i >= 2:
            forms.append(toks[i - 1])
            break
    return forms


class Anonymizer:
    def __init__(self, roster: list[str]):
        self.full_names = sorted(set(roster), key=len, reverse=True)
        # surname -> True, minus exclusions
        self.surnames = []
        for n in self.full_names:
            for ln in _surname_forms(n):
                if len(ln) >= 3 and _strip_accents(ln).lower() not in EXCLUSIONS:
                    self.surnames.append(ln)
        self.surnames = sorted(set(self.surnames), key=len, reverse=True)
        # surnames that are also ordinary words ("Lane", "Cos") are excluded from
        # the bare-surname layer; catch them only in unambiguous person contexts
        self.word_surnames = sorted({ln for ln in map(_last_name, self.full_names)
                                     if _strip_accents(ln).lower() in EXCLUSIONS}, key=len, reverse=True)
        self._compile()

    def _compile(self):
        # Layer 1: speaker turn labels at line start, e.g. "Governor Bailey:" or "Bailey:"
        sur = "|".join(re.escape(s) for s in self.surnames)
        self.re_label = re.compile(
            rf"(?im)^\s*(?:(?:{TITLE_ALT})\s+)?(?:{sur})\s*[:.]",
        )
        # Layer 2: press-conference question intros sometimes carry the journalist + outlet
        self.re_reporter = re.compile(
            r"(?im)^\s*(?:Question|Q)\s*(?:from|by)?\s*[^:\n]{0,60}?:",
        )
        # Layer 3: title + (any capitalized name token), allowing middle initials
        # ("Mr Philip R Lane", "Governor Philip R. Lane")
        self.re_title_name = re.compile(
            rf"(?:{TITLE_ALT})\s+[A-ZÀ-Ý][\wÀ-ÿ'’-]+"
            rf"(?:\s+(?:[A-ZÀ-Ý]\.?(?=\s+[A-ZÀ-Ý])|[A-ZÀ-Ý][\wÀ-ÿ'’-]+)){{0,3}}",
        )
        # Layer 5b: word-like surnames only as a person: "P.R. Lane", "R Lane",
        # "Lane, P.R.", "Lane (2024)", "Lane and Milesi-Ferretti", "Lane's lecture"
        if self.word_surnames:
            w = "|".join(re.escape(s) for s in self.word_surnames)
            # …and as a cited author, which is how an academic member names
            # themselves in their own speech: "(Taylor, 2025)", "Kopecky & Taylor",
            # "Obstfeld and Taylor, Alan M", "Jordà-Schularick-Taylor"
            self.re_word_sur = re.compile(
                rf"\b(?:(?:[A-Z]\.\s?){{1,3}}|[A-Z]\s+)(?:{w})\b"
                rf"|\b(?:{w})(?=,?\s+(?:[A-Z]\.\s?)+|,\s+[A-Z][a-z]+\s+[A-Z]\b|,?\s*\(?\d{{4}}[a-z]?\b"
                rf"|\s+and\s+[A-Z][a-z]|['’]s\s|:\s|\])"
                rf"|(?<=&\s)(?:{w})\b|(?<=\band\s)(?:{w})(?=,?\s*\(?\d{{4}}|,\s+[A-Z])"
                rf"|(?<=[a-zà-ÿ]-)(?:{w})\b")
        else:
            self.re_word_sur = None
        # Layer 4: full names - one alternation (longest first) rather than one regex
        # per name: identical output, ~6x faster over the ~300-name roster
        # any whitespace between name parts: PDF-extracted text often has double
        # spaces or line breaks inside names
        alt = "|".join(r"\s+".join(map(re.escape, n.split())) for n in self.full_names)
        self.re_full = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE) if alt else None
        # Layer 5: surname-only
        self.re_surname = re.compile(rf"\b(?:{sur})\b") if self.surnames else None

    def __call__(self, text: str) -> str:
        if not text:
            return text
        t = self.re_label.sub("SPEAKER:", text)
        t = self.re_reporter.sub("REPORTER:", t)
        t = self.re_title_name.sub(OFFICIAL, t)
        if self.re_full:
            t = self.re_full.sub(OFFICIAL, t)
        if self.re_surname:
            t = self.re_surname.sub(OFFICIAL, t)
        if self.re_word_sur:
            t = self.re_word_sur.sub(OFFICIAL, t)
            # attendee lists ("[OFFICIAL]* Lane [OFFICIAL]") - next to a masked name
            # a word-like surname can only be a person
            w = "|".join(re.escape(s) for s in self.word_surnames)
            t = re.sub(rf"(?<=\[OFFICIAL\])([\s*,]+)(?:{w})\b(?=[\s*,]*\[OFFICIAL\])",
                       rf"\1{OFFICIAL}", t)
        # collapse runs like "[OFFICIAL] [OFFICIAL]"
        t = re.sub(r"(?:\[OFFICIAL\]\s*){2,}", OFFICIAL + " ", t)
        return t

    def text_of(self, s) -> str:
        """Anonymised text of a Speech, computed once and cached on the record.

        text_anon is not persisted (see schema.save_corpus), so judges call this
        rather than reading s.text_anon directly - they must never fall back to
        the raw, named text."""
        if not s.text_anon:
            s.text_anon = self(s.text)
        return s.text_anon
