"""Who was on the MPC, and when — read from the minutes' attendance lists.

The roster (roster_mpc) knows who has ever sat on the Committee but not when, so
"is an MPC member" used to mean "has ever been one". That let in documents from
outside a member's time on the Committee: a former member giving evidence as an
outside commentator years later (Vickers and Tucker to the Lords in 2023), speeches
from before someone joined (Breeden's 2018 financial-stability speech), and — the
same check catches it — surname collisions and mis-dated documents.

Every set of minutes lists the members present, so each member's tenure is simply
the span of meetings they attended. A document by an individual counts only if it
falls inside that span, widened a little at the front for pre-appointment hearings
(a member faces the Treasury Committee before their first meeting) and at the end
for the speeches and evidence that follow their last one.
"""
from __future__ import annotations

import datetime as _dt
import re

from .roster_mpc import BOE_MPC, CURRENT_MPC, FORMER_MPC, canon

MPC_FIRST_MEETING = "1997-06-06"
LEAD_DAYS = 150     # appointment hearings come up to a few months before joining
TAIL_DAYS = 45      # a farewell speech or final evidence session after the last meeting

_PRESENT_RE = re.compile(r"members\s+of\s+the\s+Committee\s+were\s+present:?(.{0,900})",
                         re.I | re.S)
_BLOCK_END_RE = re.compile(r"Treasury representative|members of staff|also present|Secretariat",
                           re.I)


def derive(minutes) -> dict[str, tuple[str, str]]:
    """person -> (first meeting, last meeting) from minutes records."""
    people = {canon(n) for n in CURRENT_MPC | FORMER_MPC}
    by_surname: dict[str, set[str]] = {}
    for p in people:
        by_surname.setdefault(p.split()[-1], set()).add(p)
    seen: dict[str, list[str]] = {}
    for s in minutes:
        m = _PRESENT_RE.search(s.text)
        if not m:
            continue
        block = _BLOCK_END_RE.split(m.group(1))[0]
        for p in people:
            surname = p.split()[-1]
            # full name, or the surname alone when no one else has held it
            if re.search(rf"\b{re.escape(p)}\b", block) or (
                    len(by_surname[surname]) == 1 and re.search(rf"\b{re.escape(surname)}\b", block)):
                seen.setdefault(p, []).append(s.date)
    return {p: (min(d), max(d)) for p, d in seen.items()}


def _shift(day: str, days: int) -> str:
    return (_dt.date.fromisoformat(day) + _dt.timedelta(days=days)).isoformat()


class Tenure:
    def __init__(self, table: dict[str, tuple[str, str]]):
        self.table = table

    @classmethod
    def from_corpus(cls, corpus) -> "Tenure":
        return cls(derive(s for s in corpus if s.source_type == "mp_account"))

    def active(self, speaker: str, day: str) -> bool:
        """True if `speaker` sat on the MPC around `day` (the Committee always does)."""
        if day < MPC_FIRST_MEETING:
            return False          # nobody spoke for the MPC before it existed
        person = canon(speaker)
        if person == BOE_MPC:
            # the Committee exists from its first meeting; the Feb and May 1997
            # Inflation Reports were the Bank's, before independence
            return day >= MPC_FIRST_MEETING
        span = self.table.get(person)
        if not span:
            return False
        return _shift(span[0], -LEAD_DAYS) <= day <= _shift(span[1], TAIL_DAYS)


def for_corpus(corpus) -> Tenure:
    return Tenure.from_corpus(corpus)


# --- live roster: the Committee as the newest minutes record it -------------
_NAME_LINE_RE = re.compile(r"^([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z.'’-]*){1,3})\s*(?:,|$)")


def attendance(minutes_text: str) -> list[str]:
    """Names in a minutes' members-present block, known to the roster or not."""
    m = _PRESENT_RE.search(minutes_text)
    if not m:
        return []
    names = []
    for line in _BLOCK_END_RE.split(m.group(1))[0].splitlines():
        hit = _NAME_LINE_RE.match(line.strip().lstrip("•·▪-– "))   # March 2026 lists are bulleted
        if hit and "present" not in line.lower():
            names.append(canon(hit.group(1).strip()))
    return names


def sync_roster(corpus, path) -> dict:
    """Write roster_state.json: the current Committee is whoever the newest minutes
    list as present — which picks up a newly appointed member automatically, before
    anyone edits roster_mpc.py — and everyone else ever seen becomes former."""
    import json
    minutes = sorted((s for s in corpus if s.source_type == "mp_account"), key=lambda s: s.date)
    if not minutes:
        return {}
    current = attendance(minutes[-1].text)
    if not 5 <= len(current) <= 12:       # a misread block must not rewrite the roster
        return {}
    known = {canon(n) for n in CURRENT_MPC | FORMER_MPC}
    state = {"as_of": minutes[-1].date,
             "current": sorted(current),
             "former": sorted((known | set(derive(minutes))) - set(current)),
             "added": sorted(set(current) - known)}
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state
