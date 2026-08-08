"""MPC votes, read out of the minutes and scored in 25bp units.

The Bank publishes no machine-readable voting history, so the votes are parsed
from the minutes the corpus already holds. Wording has changed several times
since 1997, so three patterns are tried, and every parsed decision is checked
against the actual Bank Rate series — a parse that disagrees with what the rate
did is dropped rather than trusted.

Three scores are produced, because "the vote" can mean three different things:

  decision   what the Committee DID: the Bank Rate change in 25bp units
             (hold 0, +25bp +1, -25bp -1, +50bp +2).
  mean_vote  what the average member WANTED: every member's preferred change in
             the same units, averaged. A 7-2 hold with two members wanting +25bp
             scores +0.22, so dissent shows up where the decision alone is flat.
  dissent    which way the dissenters PUSHED, regardless of the outcome:
             (members preferring tighter - members preferring looser) / members.
             Independent of the decision, so it reads pressure, not action.
"""
from __future__ import annotations

import re

_NUMBER_WORDS = {"no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "all": None}   # "all members" resolved from the member count
_DIR = {"increase": 1, "increased": 1, "raise": 1, "raising": 1,
        "reduce": -1, "reduced": -1, "cut": -1, "lower": -1, "decrease": -1,
        "maintain": 0, "maintained": 0, "hold": 0, "unchanged": 0}

_WS = re.compile(r"\s+")
# the policy rate has had several names: Bank Rate today, the Bank's repo rate
# before 2006, and "interest rates" in passing throughout
_RATE = (r"(?:Bank\s+Rate|the\s+Bank(?:'s|’s)?\s+repo\s+rate|the\s+repo\s+rate|"
         r"the\s+official\s+(?:dealing\s+)?rate|interest\s+rates?)")
_AMOUNT = r"(?:by\s+)?([\d.]+)\s*(basis points?|percentage points?|bps?|%)"
_LEVEL = r"(?:\s*,?\s*to\s+([\d.]+)\s*%)"

# "voted by a majority of 8-1 to increase Bank Rate by 0.5 percentage points, to 1.75%"
_SUMMARY_RE = re.compile(
    r"voted\s+(?:unanimously\s+)?(?:by\s+a\s+majority\s+of\s+(\d+)\s*[-–—]\s*(\d+)\s+)?"
    r"to\s+(maintain|increase|reduce|raise|cut|lower)\s+" + _RATE +
    r"(?:\s+" + _AMOUNT + r")?" + _LEVEL + r"?", re.I)
# "the proposition that the repo rate should be reduced by 25 basis points to 3.50%"
_PROPOSITION_RE = re.compile(
    _RATE + r"\s+(?:should\s+be|be)\s+(maintained|increased|reduced)"
    r"(?:\s+" + _AMOUNT + r")?" + _LEVEL + r"?", re.I)
# "Three members … voted against …, preferring to increase Bank Rate by 0.25 …",
# and since 2025 the plainer "Three members voted to increase Bank Rate by 0.25 …"
_DISSENT_RE = re.compile(
    r"(\w+)\s+members?\b[^.]{0,200}?(?:prefer(?:red|ring)|voted)\s+to\s+"
    r"(increase|reduce|raise|cut|lower|maintain)\s+" + _RATE +
    r"(?:\s+" + _AMOUNT + r")?" + _LEVEL + r"?", re.I)
# "Andrew Sentance preferred to increase Bank Rate by 50 basis points"
_NAMED_DISSENT_RE = re.compile(
    r"([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]*){1,3})\s+prefer(?:red|ring)\s+to\s+"
    r"(increase|reduce|raise|cut|lower|maintain)\s+" + _RATE +
    r"(?:\s+" + _AMOUNT + r")?" + _LEVEL + r"?", re.I)
_IN_FAVOUR_RE = re.compile(r"(\w+)\s+members?\s+(?:of\s+the\s+Committee\s+)?"
                           r"(?:\([^)]*\)\s*)?voted\s+(?:in\s+favour|for\s+the\s+proposition)",
                           re.I)
# "Rachel Lomax, Charles Bean and David Blanchflower voted against, preferring to
# maintain Bank Rate at 5.5%" — the dissenters named rather than counted
_NAMED_LIST_DISSENT_RE = re.compile(
    r"([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]*){0,3}"
    r"(?:\s*(?:,|and)\s*[A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]*){0,3}){0,8})\s+"
    r"voted\s+against[^.]{0,80}?prefer(?:red|ring)\s+to\s+"
    r"(increase|reduce|raise|cut|lower|maintain)\s+" + _RATE +
    r"(?:\s+" + _AMOUNT + r")?" + _LEVEL + r"?", re.I)


def _count_names(blob: str) -> int:
    parts = [p.strip() for p in re.split(r",|\band\b", blob) if p.strip()]
    return len([p for p in parts if re.match(r"^(?:the\s+)?[A-Z]", p)])
_UNANIMOUS_RE = re.compile(r"voted\s+unanimously", re.I)
_PRESENT_RE = re.compile(r"following\s+members\s+of\s+the\s+Committee\s+were\s+present:?(.{0,700})",
                         re.I | re.S)


def _count(word: str) -> int | None:
    word = word.lower().strip()
    if word.isdigit():
        return int(word)
    return _NUMBER_WORDS.get(word)


def _bp(amount: str | None, unit: str | None) -> int:
    """Basis points from ('0.25', 'percentage points') or ('25', 'basis points')."""
    if not amount:
        return 0
    try:
        value = float(amount)
    except ValueError:
        return 0
    unit = (unit or "").lower()
    if unit.startswith(("percentage", "%")):
        return int(round(value * 100))
    return int(round(value))


def _members_present(text: str) -> int:
    m = _PRESENT_RE.search(text)
    if not m:
        return 0
    block = m.group(1)
    lines = [_WS.sub(" ", ln).strip(" ,;") for ln in block.splitlines()]
    names = [ln for ln in lines
             if 3 <= len(ln) <= 60 and re.match(r"^(?:The\s+)?[A-Z]", ln)
             and "Treasury" not in ln and "present" not in ln.lower()]
    return len(names)


def _level(raw: str | None) -> float | None:
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def parse_votes(text: str) -> dict | None:
    """{'decision_bp', 'n_members', 'prefs_bp': [...]} or None if unreadable.

    Since 2015 the same vote is stated twice — once in the Monetary Policy Summary
    at the top and once in the minutes at the end — so the two regions are read
    separately and the first complete, self-consistent reading wins. Reading them
    together counted every dissent twice.
    """
    present = _members_present(text)
    for region in (text[:4500], text[-9000:]):
        got = _parse_region(region, present)
        if got:
            return got
    return None


def _parse_region(head: str, present: int) -> dict | None:
    decision = decided_level = None
    split: tuple[int, int] | None = None      # "by a majority of 5-4"
    m = _SUMMARY_RE.search(head)
    if m:
        decision = _DIR.get(m.group(3).lower(), 0) * _bp(m.group(4), m.group(5))
        decided_level = _level(m.group(6))
        if m.group(1) and m.group(2):
            split = (int(m.group(1)), int(m.group(2)))
    if decision is None:
        m = _PROPOSITION_RE.search(head)
        if m:
            decision = _DIR.get(m.group(1).lower(), 0) * _bp(m.group(2), m.group(3))
            decided_level = _level(m.group(4))
    if decision is None:
        return None
    # the rate the meeting started from, used to price a dissent quoted as a level
    # ("preferring to maintain interest rates at 5.75%") rather than as a change
    prev_level = None if decided_level is None else decided_level - decision / 100.0

    def preferred_bp(direction: str, amount: str | None, unit: str | None,
                     level_raw: str | None) -> int:
        d = _DIR.get(direction.lower(), 0)
        if amount:
            return d * _bp(amount, unit)
        lvl = _level(level_raw)
        if lvl is not None and prev_level is not None:
            return int(round((lvl - prev_level) * 100))
        return 0 if d == 0 else d * 25      # a bare "preferred to increase" is 25bp

    in_favour = None
    m = _IN_FAVOUR_RE.search(head)
    if m:
        in_favour = _count(m.group(1))

    dissents: list[tuple[int, int]] = []          # (members, preferred bp)
    for m in _DISSENT_RE.finditer(head):
        n = _count(m.group(1))
        if n and not (split and n == split[0]):   # that count is the majority, not a dissent
            dissents.append((n, preferred_bp(m.group(2), m.group(3), m.group(4), m.group(5))))
    for m in _NAMED_LIST_DISSENT_RE.finditer(head):
        n = _count_names(m.group(1))
        if n:
            dissents.append((n, preferred_bp(m.group(2), m.group(3), m.group(4), m.group(5))))
    if not dissents:
        for m in _NAMED_DISSENT_RE.finditer(head):
            dissents.append((1, preferred_bp(m.group(2), m.group(3), m.group(4), m.group(5))))

    n_dissent = sum(n for n, _ in dissents)
    if split:
        # "a majority of 5-4" states both sides, so it settles the arithmetic; a
        # dissent count that contradicts it means the text was read wrong
        in_favour = split[0]
        if n_dissent and n_dissent != split[1]:
            return None
        if not n_dissent and split[1]:
            return None       # a stated minority whose preference we failed to read
    if _UNANIMOUS_RE.search(head) and not dissents and not split:
        n_dissent = 0
        in_favour = in_favour or present
    n_members = (in_favour or 0) + n_dissent
    n_members = n_members or present
    if not 5 <= n_members <= 12:      # the MPC is nine; anything else is a misparse
        return None

    majority = max(n_members - n_dissent, 0)
    prefs = [decision] * majority + [bp for n, bp in dissents for _ in range(n)]
    if len(prefs) != n_members:                   # counts do not add up: don't guess
        return None
    return {"decision_bp": decision, "n_members": n_members, "prefs_bp": prefs}


def scores(parsed: dict) -> dict:
    """The three vote scores, in 25bp units."""
    unit = 25.0
    prefs = parsed["prefs_bp"]
    decision = parsed["decision_bp"]
    tighter = sum(1 for p in prefs if p > decision)
    looser = sum(1 for p in prefs if p < decision)
    return {
        "decision": decision / unit,
        "mean_vote": sum(prefs) / len(prefs) / unit,
        "dissent": (tighter - looser) / len(prefs),
    }
