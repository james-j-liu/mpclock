"""Core data model for the corpus.

A Speech is one judged unit. Individual MPC members (Governor, Deputy Governors,
Chief Economist, external members) get one record per speech. The "BoE MPC"
composite speaker (Monetary Policy Report/Summary, MPC minutes, the Governor's
monetary-policy press conference) is modelled by setting speaker == BOE_MPC on
those records, so they aggregate together in the rankings while still being scored
per-document in the tournament.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

BOE_MPC = "BoE MPC"

# Back-compat alias: much of the shared pipeline was written against the ECB name.
ECB_COUNCIL = BOE_MPC

# Document types
ST_SPEECH = "speech"            # individual MPC member speech / lecture / remarks
ST_INTERVIEW = "interview"      # media interview (incl. from news sources)
ST_STATEMENT = "mp_statement"   # Monetary Policy Summary / Governor's opening remarks
ST_QA = "mp_qa"                 # Governor's monetary-policy press conference Q&A
ST_ACCOUNT = "mp_account"       # MPC minutes
ST_REPORT = "mp_report"         # Monetary Policy / Inflation Report section, box or annex
ST_TESTIMONY = "testimony"      # one MPC member's evidence to the Treasury Committee

COUNCIL_TYPES = {ST_STATEMENT, ST_QA, ST_ACCOUNT, ST_REPORT}
MPC_TYPES = COUNCIL_TYPES


@dataclass
class Speech:
    date: str                       # ISO YYYY-MM-DD
    speaker: str                    # canonical speaker name, or ECB_COUNCIL
    title: str
    text: str                       # full text, English (post-translation)
    source_type: str                # one of the ST_* constants
    institution: str = ""           # e.g. "European Central Bank", "Deutsche Bundesbank"
    source_url: str = ""
    orig_language: str = "en"
    translated: bool = False
    word_count: int = 0
    is_policy: Optional[bool] = None      # set by classifier
    text_anon: str = ""                   # set by anonymizer

    # pairwise-tournament outputs (filled later)
    mu: Optional[float] = None
    mu_adj: Optional[float] = None
    sigma: Optional[float] = None
    n_comparisons: int = 0

    # direct-scoring outputs (alternative method: one LLM 0-100 rating per speech)
    direct_score: Optional[float] = None
    direct_adj: Optional[float] = None

    id: str = ""

    def __post_init__(self):
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.id:
            self.id = self.make_id()

    def make_id(self) -> str:
        h = hashlib.sha1(
            f"{self.date}|{self.speaker}|{self.title}|{self.source_url}".encode("utf-8")
        ).hexdigest()[:16]
        return h

    @property
    def is_council(self) -> bool:
        return self.speaker == ECB_COUNCIL or self.source_type in COUNCIL_TYPES


def keeps_text(s: "Speech") -> bool:
    """Whether a record's body text has to survive a save.

    The corpus lives in git (that is what makes the daily run incremental) and
    GitHub rejects any file over 100 MB, so text is kept where it can still be
    needed and dropped where it cannot. Everything by an MPC member or the
    Committee keeps its text whatever the classifier said, because a change to
    the classifier has to be able to re-judge a speech it previously rejected.
    Speeches by officials who never sit on the MPC keep their metadata and
    source_url only — a roster change can re-scrape them.
    """
    from .roster_mpc import is_mpc
    return is_mpc(s.speaker)


def _record(s: Speech, keep_all_text: bool = False) -> dict:
    d = asdict(s)
    d["text_anon"] = ""          # derived: recomputed per run, never stored
    if not (keep_all_text or keeps_text(s)):
        d["text"] = ""
    return d


def save_corpus(speeches: list[Speech], path: str | Path,
                keep_all_text: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in speeches:
            f.write(json.dumps(_record(s, keep_all_text), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[Speech]:
    path = Path(path)
    out: list[Speech] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Speech(**json.loads(line)))
    return out


def iter_corpus(path: str | Path) -> Iterator[Speech]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Speech(**json.loads(line))
