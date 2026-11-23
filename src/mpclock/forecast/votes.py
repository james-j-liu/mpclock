"""Per-member vote forecast for the next MPC meeting.

For each member the model predicts the CHANGE from their own previous vote
(Δ, in 25bp steps), an ordered logit over Δ ∈ {-2,-1,0,+1,+2}. Everything is signed
so that a positive value pushes towards tighter policy. Inputs, all measured over
the window from the previous meeting up to the day before this one:

  committee   prev_dev     own previous vote minus the Committee's decision
              pull         Committee's average vote minus own previous vote
  words       tone         own documents' hawkishness (tournament score) relative
                           to the member's own previous 12 months
              c_tone       Committee-wide: the average score of every document in the
                           window (members' and the Committee's own) relative to the
                           previous 12 months
              guidance     forward-guidance phrases in own words ("further increase",
                           "case for a cut", "one-for-now" …), net tighter minus looser
  data        surprise     CPI, services CPI, private regular pay and unemployment
                           outturns published in the window minus the Bank staff
                           projection for that month in the latest MPR; each member
                           weights the four by how much they talked about them
  markets     energy       % change in Brent and TTF gas (average) over the window
              fx           % change in the sterling ERI (a fall is inflationary)
              fed, ecb     change in the Fed funds target / ECB deposit rate (pp)
  pace        pace         in the current cycle, the share of same-type meetings
                           (MPR / non-MPR) at which the Bank moved, signed by the
                           cycle's direction — a quarterly cycle scores ~1 at MPR
                           meetings and ~0 in between
              moved_last   the Bank moved at the previous meeting (signed)
              mpr          this meeting comes with a Monetary Policy Report

The predicted vote is the previous vote plus Δ, folded onto the options the
Committee could plausibly choose between: from 25bp below the lowest of {last
meeting's votes, no change} to 25bp above the highest.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..roster_mpc import canon
from .inputs import load_cache

ROOT = Path(__file__).resolve().parents[3]
OWN_TYPES = {"speech", "interview", "testimony", "member_view", "press_interview"}
DELTAS = np.array([-2, -1, 0, 1, 2])

# Meetings whose three-way splits the minutes parser does not resolve. Votes in
# 25bp steps, from the published minutes.
VOTE_OVERRIDES = {
    "2021-12-15": {"Andrew Bailey": 1, "Ben Broadbent": 1, "Jon Cunliffe": 1, "Jonathan Haskel": 1,
                   "Catherine Mann": 1, "Huw Pill": 1, "Dave Ramsden": 1, "Michael Saunders": 1,
                   "Silvana Tenreyro": 0},
    "2022-09-21": {"Andrew Bailey": 2, "Ben Broadbent": 2, "Jon Cunliffe": 2, "Huw Pill": 2,
                   "Silvana Tenreyro": 2, "Jonathan Haskel": 3, "Catherine Mann": 3,
                   "Dave Ramsden": 3, "Swati Dhingra": 1},
    "2022-11-02": {"Andrew Bailey": 3, "Ben Broadbent": 3, "Jon Cunliffe": 3, "Jonathan Haskel": 3,
                   "Catherine Mann": 3, "Huw Pill": 3, "Dave Ramsden": 3, "Swati Dhingra": 2,
                   "Silvana Tenreyro": 1},
    "2025-05-07": {"Andrew Bailey": -1, "Sarah Breeden": -1, "Megan Greene": -1,
                   "Clare Lombardelli": -1, "Dave Ramsden": -1, "Swati Dhingra": -2,
                   "Alan Taylor": -2, "Catherine Mann": 0, "Huw Pill": 0},
}

# members who are Bank staff (Governor, Deputy Governors, Chief Economist) — they rarely
# vote against the Governor
INTERNAL = {"Mervyn King", "Mark Carney", "Andrew Bailey", "Charles Bean", "Charlie Bean", "Paul Tucker",
            "Spencer Dale", "Andy Haldane", "Andrew Haldane", "Ben Broadbent", "Jon Cunliffe",
            "Dave Ramsden", "Huw Pill", "Sarah Breeden", "Clare Lombardelli", "Minouche Shafik",
            "Nemat Shafik"}
_HEDGE = re.compile(r"(?i)\bconditional\b|\bgradual(ly)?\b|data[- ]dependent|not on a pre-?set path|"
                    r"meeting[- ]by[- ]meeting|in the absence of|\bunless\b|\bif the\b|\bwould depend\b|"
                    r"will depend|\bnot (yet )?(enough|sufficient) (information|evidence)|wouldn.t rule out|"
                    r"careful(ly)?\b|two-sided|\buncertain")
SIGNAL_SOURCE = "text"   # "text" or "passages": which coded vote feeds the guidance inputs
                         # (backtested: whole text 0.670 log loss, passages alone 0.680)
HISTORY = 6       # meetings of each member's voting record used, beyond the last one

FEATURE_GROUPS = {
    "committee": ["prev_dev", "pull"],
    "record": ["hist_dev", "trend"],
    "pace": ["pace", "moved_last", "mpr_x_pace"],
    # alternative codings of "pace", tested against it (scripts/test_pace.py)
    "pace_by_direction": ["pace_up", "pace_dn", "moved_last_up", "moved_last_dn", "mpr_x_pace_up", "mpr_x_pace_dn"],
    "pace_established": ["pace_est", "moved_last", "mpr_x_pace_est", "first_move", "first_x_mpr"],
    "words": ["guidance", "signal_level"],
    "words_split": ["g_rat_gap", "g_rat_lvl", "g_pol_gap", "g_pol_lvl", "g_oth_gap", "g_oth_lvl"],
    "direction": ["direction"],
    "governor": ["gov_pull"],
    "hedge": ["signal_hedged", "guidance_hedged"],
    "data": ["surprise"],          # outturns vs MPR, plus half-weighted energy-price change
    "global": ["fed", "ecb"],
    "fx": ["fx"],
    "tone": ["tone"],
    "committee_tone": ["c_tone"],
    # member-specific uncertainty: these scale the spread of the member's distribution
    # (heteroskedastic ordered logit) rather than shift it; see OrderedLogit
    "volatility": ["s:vol_own", "s:vol_rel"],
    # the opposite case on its own: members never out of step with the Committee over
    # at least six recorded meetings, free to get a narrower spread than everyone else
    "steady": ["s:steady"],
}
VOL_WINDOW = 12   # meetings of a member's record used to measure how often they change vote

# --- forward guidance lexicon --------------------------------------------------
_POLICY = re.compile(r"(?i)\b(bank rate|interest rates?|rates|policy|monetary|tighten|loosen|hike|cut)")
_TIGHT = [r"further (increase|rise|tightening|hike)s?\b", r"(more|additional) (policy )?(tightening|hikes?)",
          r"case for (a |an )?(further )?(increase|rise|raising|tightening|hike)",
          r"(need|needs|have|has) to (raise|tighten|increase|go up|rise)",
          r"rates? (may|will|would|could|might) (need|have) to (rise|go up|increase|be higher)",
          r"(raising|increasing|tighten(ing)?) (bank rate|interest rates|rates|policy)",
          r"higher (bank rate|interest rates|for longer)", r"\bhik(e|es|ing)\b",
          r"not (yet )?(sufficiently|enough) restrictive", r"behind the curve"]
_EASE = [r"further (reduction|cut|easing|loosening)s?\b", r"(cut|cutting|reduce|reducing|lower|lowering) "
         r"(bank rate|interest rates|rates|policy rate)", r"(remov|withdraw)(e|ing|al of) (policy )?restraint",
         r"case for (a |an )?(further )?(cut|reduction|easing|lower)",
         r"rates? (will|can|could|should) (fall|come down|decline)", r"downward path", r"\bless restrictive\b",
         r"too restrictive", r"(loosen|eas)(e|ing) (monetary )?policy"]
_HOLD = [r"(hold|holding|maintain|maintaining|keep|keeping) bank rate", r"one[- ]for[- ]now", r"\bpause\b",
         r"\bwait(ing)?\b", r"not on a pre-?set path", r"\bpatien(t|ce)\b", r"(sufficiently|appropriately) restrictive"]
_TIGHT_RE = [re.compile(r"(?i)" + p) for p in _TIGHT]
_EASE_RE = [re.compile(r"(?i)" + p) for p in _EASE]
_HOLD_RE = [re.compile(r"(?i)" + p) for p in _HOLD]
_SENT = re.compile(r"(?<=[.!?])\s+")

_TOPICS = {
    "cpi": re.compile(r"(?i)\b(headline inflation|CPI inflation|inflation (rose|fell|was|is now)|inflation outturn)"),
    "services": re.compile(r"(?i)services (price )?inflation|services prices|services CPI"),
    "pay": re.compile(r"(?i)\b(wage|pay growth|pay settlements?|earnings|wage growth|pay)\b"),
    "unemp": re.compile(r"(?i)\b(unemployment|labour market|slack|vacanc|employment)"),
}


def guidance_score(text: str) -> tuple[float, int]:
    """Net forward-guidance lean of a text in [-1, 1] and the number of policy
    sentences that carried a signal."""
    t = h = e = 0
    for s in _SENT.split(text or ""):
        if not _POLICY.search(s):
            continue
        tt = sum(1 for r in _TIGHT_RE if r.search(s))
        ee = sum(1 for r in _EASE_RE if r.search(s))
        hh = sum(1 for r in _HOLD_RE if r.search(s))
        t, e, h = t + tt, e + ee, h + hh
    n = t + e + h
    return ((t - e) / (n + 2.0), n)


def topic_weights(text: str) -> dict[str, float]:
    c = {k: len(r.findall(text or "")) for k, r in _TOPICS.items()}
    tot = sum(c.values())
    # everyone reacts to the data; emphasis tilts the weights, never zeroes them
    return {k: 0.5 + (2.0 * v / tot if tot else 0.5) for k, v in c.items()}


# --- data ----------------------------------------------------------------------
@dataclass
class Meeting:
    date: str
    votes: dict[str, int]            # member -> vote in 25bp steps
    decision: int
    rate_after: float


@dataclass
class Context:
    meetings: list[Meeting]
    docs: dict[str, list[tuple]]     # member -> [(date, tone, text, id, type)]
    inputs: dict
    mpr_dates: list[str]
    rate: list[tuple[str, float]]
    member_alias: dict[str, str] = field(default_factory=dict)
    all_tones: list = field(default_factory=list)   # [(date, tone)] every scored document


def _steps(bp: float) -> int:
    return int(round(bp / 25.0)) if abs(bp) >= 10 else 0


def build_context(corpus, macro: dict | None = None) -> Context:
    from ..macro.mpc_votes import member_votes_by_meeting
    macro = macro or json.loads((ROOT / "site/macro.json").read_text())
    rate = [(d, float(r)) for d, r in macro["series"]["bank_rate"]["data"]]
    mv = member_votes_by_meeting([s for s in corpus if s.source_type == "mp_account"], rate)
    meetings = []
    for d, v in mv:
        if d < "2015-01-01":
            continue
        votes = VOTE_OVERRIDES.get(d)
        if votes is None:
            if not v:
                continue
            votes = {canon(k): _steps(b) for k, b in v.items()}
        before = _rate_on(rate, _shift(d, -2))
        after = _rate_on(rate, _shift(d, 3))
        meetings.append(Meeting(d, votes, _steps((after - before) * 100), after))
    docs: dict[str, list] = {}
    for s in corpus:
        if s.source_type in OWN_TYPES and s.date >= "2014-01-01":
            tone = (s.mu - 50) / 5 if s.mu is not None else None
            docs.setdefault(canon(s.speaker), []).append((s.date, tone, s.text or "", s.id, s.source_type))
    for v in docs.values():
        v.sort(key=lambda x: x[0])
    inputs = load_cache()
    from .guidance import load_codes
    inputs["guidance"] = load_codes()
    mpr_dates = sorted(inputs.get("mpr_dates", {}).values())
    all_tones = sorted((s.date, (s.mu - 50) / 5) for s in corpus if s.mu is not None and s.date >= "2013-01-01")
    return Context(meetings, docs, inputs, mpr_dates, rate, all_tones=all_tones)


def _shift(d: str, days: int) -> str:
    return (dt.date.fromisoformat(d) + dt.timedelta(days=days)).isoformat()


def _rate_on(rate, d):
    r = rate[0][1]
    for dd, v in rate:
        if dd <= d:
            r = v
        else:
            break
    return r


# --- features ------------------------------------------------------------------
_SCALE = {"cpi": 0.2, "services": 0.3, "pay": 0.3, "unemp": 0.15}
_SIGN = {"cpi": 1, "services": 1, "pay": 1, "unemp": -1}


def _period_month(p: str) -> str:
    m = re.match(r"(\d{4})Q([1-4])", p)
    return f"{m.group(1)}-{3 * int(m.group(2)):02d}" if m else p


def data_surprises(ctx: Context, start: str, end: str) -> tuple[dict[str, float], list[dict]]:
    """Standardised surprise per series for outturns published in (start, end),
    each against the latest MPR projection published before its release."""
    from .inputs import release_date
    out = ctx.inputs.get("outturns", {})
    mpr = ctx.inputs.get("mpr", {})
    mpr_by_date = sorted(((ctx.inputs.get("mpr_dates", {}).get(t), t) for t in mpr
                          if ctx.inputs.get("mpr_dates", {}).get(t)))
    res, detail = {}, []
    for k, series in out.items():
        tot, n = 0.0, 0
        for p, actual in series.items():
            rel = release_date(k, p)
            if not (start < rel < end):
                continue
            proj = None
            for d, tag in reversed(mpr_by_date):
                if d >= rel:
                    continue
                pp = {_period_month(q): v for q, v in mpr[tag].get(k, {}).items()}
                if p in pp:
                    proj, src = pp[p], tag
                    break
            if proj is None:
                continue
            z = _SIGN[k] * (actual - proj) / _SCALE[k]
            tot, n = tot + z, n + 1
            detail.append({"series": k, "period": p, "released": rel, "actual": actual,
                           "expected": round(proj, 2), "mpr": src, "z": round(z, 2)})
        if n:
            res[k] = tot
    return res, detail


def _mkt_change(ser: dict, start: str, end: str, pct=True) -> float:
    if not ser:
        return 0.0
    keys = sorted(ser)

    def avg_before(d, n=5):
        vals = [ser[k] for k in keys if k < d][-n:]
        return sum(vals) / len(vals) if vals else None
    a, b = avg_before(start), avg_before(end)
    if a is None or b is None:
        return 0.0
    return (math.log(b / a) * 100) if pct else (b - a)


def market_changes(ctx: Context, start: str, end: str) -> dict[str, float]:
    m = ctx.inputs.get("markets", {})
    brent = _mkt_change(m.get("brent", {}), start, end)
    gas = _mkt_change(m.get("gas_ttf", {}), start, end)
    return {"brent": brent, "gas": gas, "energy": (brent + gas) / 2 / 10.0,       # per 10%
            "fx": -_mkt_change(m.get("gbp_eri", {}), start, end) / 2.0,            # per 2% fall
            "fed": _mkt_change(m.get("fed", {}), start, end, pct=False) * 4,        # per 25bp
            "ecb": _mkt_change(m.get("ecb", {}), start, end, pct=False) * 4}


def is_mpr_meeting(ctx: Context, date: str) -> bool:
    """A Monetary Policy Report meeting: within a week of a Report we hold, or in a Report
    month (Feb/May/Aug/Nov). The month rule covers Reports missing from the chart-data
    files (Aug 2017, May and Aug 2020) and future meetings, whose Reports are not out yet;
    Reports published in Jan/Apr/Jul/Oct are caught by the first rule."""
    d0 = dt.date.fromisoformat(date)
    if any(abs((d0 - dt.date.fromisoformat(x)).days) <= 7 for x in ctx.mpr_dates):
        return True
    return d0.month in (2, 5, 8, 11)


def pace_features(ctx: Context, i: int, date: str) -> dict[str, float]:
    """Revealed pace of the current cycle, from meetings before `date`."""
    zero = {"pace": 0.0, "moved_last": 0.0, "mpr_x_pace": 0.0, "mpr": 0.0, "cycle_dir": 0, "n_moves": 0}
    past = [m for m in ctx.meetings if m.date < date]
    if not past:
        return _pace_variants(zero)
    moves = [(m.date, np.sign(m.decision)) for m in past if m.decision != 0]
    if not moves:
        return _pace_variants({**zero, "mpr": float(is_mpr_meeting(ctx, date))})
    cdir = moves[-1][1]
    # the cycle began at the first move in this direction after the last opposite move,
    # or after a gap of more than 400 days between moves (moves between scheduled
    # meetings, like the March 2020 emergency cuts, are not in the meeting record, so a
    # long pause also ends a cycle)
    start = moves[-1][0]
    n_moves = 0
    nxt = None
    for d, s in reversed(moves):
        if s != cdir or (nxt and (dt.date.fromisoformat(nxt) - dt.date.fromisoformat(d)).days > 400):
            break
        start, nxt = d, d
        n_moves += 1
    # a cycle whose last move is over a year old is dormant
    active = (dt.date.fromisoformat(date) - dt.date.fromisoformat(moves[-1][0])).days < 400
    this_mpr = is_mpr_meeting(ctx, date)
    same = [m for m in past if m.date >= start and is_mpr_meeting(ctx, m.date) == this_mpr]
    moved = sum(1 for m in same if np.sign(m.decision) == cdir)
    pace = (moved + 0.5) / (len(same) + 1.0) if active else 0.0
    last = past[-1]
    return _pace_variants({"pace": float(cdir * pace), "moved_last": float(np.sign(last.decision)),
                           "mpr_x_pace": float(cdir * pace * (1 if this_mpr else -1)) if active else 0.0,
                           "mpr": float(this_mpr), "cycle_dir": int(cdir) if active else 0,
                           "n_moves": n_moves if active else 0})


def _pace_variants(f: dict) -> dict:
    """Alternative codings of the pace inputs, for testing (see FEATURE_GROUPS):
    split by the cycle's direction, and gated until the cycle has a revealed cadence
    (two or more moves), with a separate input for 'one move so far'."""
    up, dn = f["cycle_dir"] > 0, f["cycle_dir"] < 0
    est = f["n_moves"] >= 2
    first = float(f["cycle_dir"]) if f["n_moves"] == 1 else 0.0
    sgn_mpr = 1.0 if f["mpr"] else -1.0
    return {**f,
            "pace_up": f["pace"] * up, "pace_dn": f["pace"] * dn,
            "mpr_x_pace_up": f["mpr_x_pace"] * up, "mpr_x_pace_dn": f["mpr_x_pace"] * dn,
            "moved_last_up": max(f["moved_last"], 0.0), "moved_last_dn": min(f["moved_last"], 0.0),
            "pace_est": f["pace"] * est, "mpr_x_pace_est": f["mpr_x_pace"] * est,
            "first_move": first, "first_x_mpr": first * sgn_mpr}


# Document weights. Tested against the backtest: removing the floor so off-topic texts
# (a lecture on independence) count for less made it slightly worse (0.6665 -> 0.6676);
# counting the member's own vote rationale double made it slightly better (-> 0.6662).
FOCUS_FLOOR = 0.25      # minimum weight of a document however little it is about policy
FOCUS_POWER = 1.0       # >1 would make low-focus documents count for even less
RATIONALE_WEIGHT = 2.0  # the member's own vote rationale in the minutes counts double
HALF_LIFE = 14          # days: a document's weight halves every HALF_LIFE days before the meeting
                        # (chosen in the backtest from 7/14/30/60; None = all words count equally)
FOCUS_SPLIT = 0.5       # documents at least this much about policy are "policy-focused"


def _focus_weight(f) -> float:
    return FOCUS_FLOOR + (f or 0.0) ** FOCUS_POWER


def member_window(ctx: Context, member: str, start: str, end: str, prev_vote: int = 0):
    """A member's own words between two meetings: tone, coded guidance, emphasis."""
    docs = [d for d in ctx.docs.get(member, []) if start <= d[0] < end]
    prior = [d for d in ctx.docs.get(member, []) if _shift(start, -365) <= d[0] < start and d[1] is not None]
    tones = [d[1] for d in docs if d[1] is not None]
    tone = (np.mean(tones) - (np.mean([p[1] for p in prior]) if prior else np.mean(tones))) if tones else 0.0
    text = "\n".join(d[2] for d in docs)
    # coded guidance: the vote each document signals. Recent words count more: each
    # document's weight halves every HALF_LIFE days before the meeting (None = equal).
    codes = ctx.inputs.get("guidance", {})
    coded = [(d, codes[d[3]]) for d in docs if d[3] in codes]

    def decay(d):
        if HALF_LIFE is None:
            return 1.0
        age = (dt.date.fromisoformat(end) - dt.date.fromisoformat(d[0])).days
        return 0.5 ** (max(age, 0) / HALF_LIFE)

    def avg(items, wfun):
        if not items:
            return None
        w = np.array([wfun(d, c) for d, c in items])
        return float((w * np.array([c["ev"] for _, c in items])).sum() / w.sum())

    gap, has, signal, direction = 0.0, 0.0, None, 0.0
    comp = {}
    if coded:
        # combined signal: weighted by focus and recency, the vote rationale double
        def wcomb(d, c):
            return (_focus_weight(c.get("focus")) * decay(d)
                    * (RATIONALE_WEIGHT if d[4] == "member_view" else 1.0))
        signal = avg(coded, wcomb)
        gap, has = signal - prev_vote, 1.0
        w = np.array([wcomb(d, c) for d, c in coded])
        dv = np.array([c.get("dir_ev", 0.0) for _, c in coded])
        direction = float((w * dv).sum() / w.sum())
    # the same split by kind of document, so the model can estimate how much each counts
    rat = [(d, c) for d, c in coded if d[4] == "member_view"]
    pol = [(d, c) for d, c in coded if d[4] != "member_view" and (c.get("focus") or 0) >= FOCUS_SPLIT]
    oth = [(d, c) for d, c in coded if d[4] != "member_view" and (c.get("focus") or 0) < FOCUS_SPLIT]
    for key, items in (("rat", rat), ("pol", pol), ("oth", oth)):
        sg = avg(items, lambda d, c: decay(d))
        comp[key] = sg
    g_regex, _ = guidance_score(text)
    # how conditional is the guidance: hedging phrases per policy sentence in the window
    from .guidance import policy_passages
    pas = " ".join(policy_passages(d[2], 20000) for d in docs if d[4] != "member_view")
    n_pol = max(1, len(re.findall(r"(?i)bank rate|interest rates?|tighten|restrictive|further increase|cut", pas)))
    hedge = float(min(1.0, len(_HEDGE.findall(pas)) / n_pol)) if pas else 0.0
    return {"hedge": hedge, "comp": comp,
            "tone": float(tone), "guidance": float(gap), "has_guidance": has, "signal": signal,
            "direction": direction,
            "guidance_regex": float(g_regex), "n_docs": len(docs), "topic_w": topic_weights(text),
            "doc_dates": [d[0] for d in docs],
            "docs": [{"date": d[0], "type": d[4], "id": d[3],
                      "signal": codes.get(d[3], {}).get("ev"), "p": codes.get(d[3], {}).get("p"),
                      "dir": codes.get(d[3], {}).get("dir")}
                     for d in docs]}


def committee_tone(ctx: Context, start: str, end: str) -> float:
    """Committee-wide shift in hawkishness: the average tournament score of every scored
    document (members' speeches, interviews, evidence and vote rationales, and the
    Committee's minutes, Report and press conferences) in the window, minus the
    average over the 12 months before it."""
    now = [t for d, t in ctx.all_tones if start <= d < end]
    before = [t for d, t in ctx.all_tones if _shift(start, -365) <= d < start]
    if not now or not before:
        return 0.0
    return float(np.mean(now) - np.mean(before))


def features_for(ctx: Context, prev: Meeting, date: str, members: list[str],
                 words_since: str | None = None, freeze_data: bool = False) -> list[dict]:
    """Rows for each member at a meeting on `date` (a past meeting or a future one).
    words_since: read members' words from this date instead of `prev` (used to carry
    their latest words forward to a meeting after a hypothetical one); freeze_data:
    assume no data or market news (for meetings beyond the next)."""
    start, end = prev.date, date
    surp, detail = ({}, []) if freeze_data else data_surprises(ctx, start, end)
    mk = ({k: 0.0 for k in ("brent", "gas", "energy", "fx", "fed", "ecb")} if freeze_data
          else market_changes(ctx, start, end))
    pace = pace_features(ctx, 0, date)
    mean_prev = np.mean(list(prev.votes.values()))
    # the meetings before `prev`, newest first: a member's record over several meetings,
    # not just the last vote
    i_prev = next((i for i, m in enumerate(ctx.meetings) if m.date == prev.date), len(ctx.meetings) - 1)
    earlier = ctx.meetings[max(0, i_prev - HISTORY):i_prev][::-1]
    from ..corpus.boe_interviews import governor_on
    try:
        gov = governor_on(date)
    except StopIteration:
        gov = None
    g_prev = prev.votes.get(gov, prev.decision)
    w_start = words_since or start
    gw = member_window(ctx, gov, w_start, end, g_prev) if gov else None
    gov_signal = (gw["signal"] if gw and gw["signal"] is not None else g_prev)
    c_tone = committee_tone(ctx, w_start, end)
    recent = ctx.meetings[max(0, i_prev - VOL_WINDOW):i_prev + 1]
    rows = []
    for p in members:
        pv = prev.votes.get(p, prev.decision)
        # how often the member has changed vote (own), and changed it differently from the
        # Committee's decision (idiosyncratic), over their last VOL_WINDOW meetings
        pairs = [(a, b) for a, b in zip(recent, recent[1:]) if p in a.votes and p in b.votes]
        vol_own = float(np.mean([b.votes[p] != a.votes[p] for a, b in pairs])) if pairs else 0.0
        vol_rel = float(np.mean([(b.votes[p] - a.votes[p]) != (b.decision - a.decision) for a, b in pairs])) if pairs else 0.0
        hist = [(m.votes[p], m.decision) for m in earlier if p in m.votes]
        # persistent lean: how far the member sat from the Committee's decision over the
        # earlier meetings (excluding the last one, which prev_dev already carries),
        # more recent meetings counting more
        if hist:
            wts = np.array([0.8 ** k for k in range(len(hist))])
            hist_dev = float((wts * np.array([v - d for v, d in hist])).sum() / wts.sum())
            dissents = sum(1 for v, d in hist if v != d) / len(hist)
            trend = float(pv - hist[0][0])          # did the member move at the last meeting?
            streak = next((k for k, (v, _) in enumerate(hist) if v != pv), len(hist))
        else:
            hist_dev, dissents, trend, streak = 0.0, 0.0, 0.0, 0
        w = member_window(ctx, p, w_start, end, pv)
        if surp:
            tw = w["topic_w"]
            personal = sum(tw[k] * v for k, v in surp.items()) / sum(tw[k] for k in surp)
        else:
            personal = 0.0
        rows.append({
            "_hist": [{"date": m.date, "vote": m.votes[p], "decision": m.decision}
                      for m in earlier if p in m.votes],
            "member": p, "prev_vote": pv, "prev_decision": prev.decision,
            "hist_dev": hist_dev, "trend": trend, "dissent_rate": dissents,
            "streak": float(min(streak, HISTORY)) / HISTORY * np.sign(pv - prev.decision or pv or 0),
            "prev_dev": pv - prev.decision, "pull": mean_prev - pv,
            "vol_own": vol_own, "vol_rel": vol_rel, "n_record": float(len(pairs)),
            "steady": float(len(pairs) >= 6 and vol_rel == 0.0),
            "tone": w["tone"], "c_tone": c_tone, "guidance": w["guidance"], "n_docs": w["n_docs"],
            # energy prices are data too: a 10% rise counts as half a standard surprise
            "surprise": float(np.clip(personal + 0.5 * np.clip(mk["energy"], -4, 4), -5, 5)),
            "surprise_data": float(np.clip(personal, -4, 4)),
            "energy": float(np.clip(mk["energy"], -4, 4)), "fx": float(np.clip(mk["fx"], -4, 4)),
            "fed": mk["fed"], "ecb": mk["ecb"],
            **{k: pace[k] for k in ("pace", "moved_last", "mpr_x_pace", "pace_up", "pace_dn", "mpr_x_pace_up",
                                    "mpr_x_pace_dn", "moved_last_up", "moved_last_dn", "pace_est",
                                    "mpr_x_pace_est", "first_move", "first_x_mpr")},
            "cycle_dir": pace["cycle_dir"], "n_moves": pace["n_moves"],
            "has_guidance": w["has_guidance"],
            # split by kind of document: change from last vote (gap) and level, each 0 if none
            **{f"g_{k}_gap": float(0.0 if w["comp"].get(k) is None else w["comp"][k] - pv) for k in ("rat", "pol", "oth")},
            **{f"g_{k}_lvl": float(0.0 if w["comp"].get(k) is None else w["comp"][k]) for k in ("rat", "pol", "oth")},
            # the vote the member's own words point to, as a level (not just the change
            # from their last vote): "more tightening needed" from someone who just voted
            # to hike is news against the drift back to hold after a move
            "signal_level": float(w["has_guidance"] * (w["guidance"] + pv)),
            "internal": float(p in INTERNAL and p != gov),
            # internal members lean towards the vote the Governor's own words point to
            "gov_pull": float((gov_signal - pv) if (p in INTERNAL and p != gov) else 0.0),
            # hedged guidance: the same signal, scaled by how conditional the wording was
            "signal_hedged": float(w["has_guidance"] * (w["guidance"] + pv) * w["hedge"]),
            "guidance_hedged": float(w["guidance"] * w["hedge"]),
            "hedge": w["hedge"],
            # which way the member's next move leans over the next few meetings (+1 up, -1 down)
            "direction": float(w["direction"]),
            "_mpr": pace["mpr"], "_surprises": surp, "_detail": detail, "_markets": mk,
            "_doc_dates": w["doc_dates"], "_docs": w["docs"], "_signal": w["signal"],
        })
    return rows


def panel(ctx: Context, since: str = "2021-01-01") -> list[dict]:
    rows = []
    for prev, cur in zip(ctx.meetings, ctx.meetings[1:]):
        if cur.date < since:
            continue
        for r in features_for(ctx, prev, cur.date, list(cur.votes)):
            r["date"] = cur.date
            r["y_vote"] = cur.votes[r["member"]]
            r["y"] = int(np.clip(r["y_vote"] - r["prev_vote"], -2, 2))
            r["decision"] = cur.decision
            r["options"] = options_for(prev)
            rows.append(r)
    return rows


def options_for(prev: Meeting) -> list[int]:
    vs = list(prev.votes.values()) + [0]
    return list(range(min(vs) - 1, max(vs) + 2))


# --- ordered logit ---------------------------------------------------------------
class OrderedLogit:
    """Ordered logit on Δ. Features named "s:<name>" do not shift the distribution:
    they scale its spread, member by member (heteroskedastic ordered logit):
    P(Δ <= k) = σ((c_k - x·b) / exp(z·g)), so a member with a more volatile record can
    be given a wider distribution."""
    def __init__(self, feats: list[str], l2: float = 1.0):
        self.feats, self.l2 = feats, l2
        self.loc = [f for f in feats if not f.startswith("s:")]
        self.scale = [f[2:] for f in feats if f.startswith("s:")]

    def _X(self, rows, cols=None):
        cols = self.loc if cols is None else cols
        return np.array([[r[f] for f in cols] for r in rows], dtype=float) if cols else \
            np.zeros((len(rows), 0))

    @staticmethod
    def _probs(eta, cuts, sd=None):
        # P(y <= k) = σ((c_k - η) / sd); categories DELTAS
        sd = np.ones(len(eta)) if sd is None else sd
        cdf = 1 / (1 + np.exp(-(cuts[None, :] - eta[:, None]) / sd[:, None]))
        cdf = np.hstack([np.zeros((len(eta), 1)), cdf, np.ones((len(eta), 1))])
        return np.clip(np.diff(cdf, axis=1), 1e-9, 1)

    def fit(self, rows):
        X, Z = self._X(rows), self._X(rows, self.scale)
        self.z_mu = Z.mean(0) if Z.shape[1] else np.zeros(0)
        Z = Z - self.z_mu
        y = np.array([r["y"] for r in rows]) - DELTAS[0]
        k, m = X.shape[1], Z.shape[1]

        def unpack(th):
            b = th[:k]
            g = th[k:k + m]
            cuts = np.cumsum(np.concatenate([[th[k + m]], np.exp(th[k + m + 1:])]))
            return b, g, cuts

        def nll(th):
            b, g, cuts = unpack(th)
            p = self._probs(X @ b, cuts, np.exp(Z @ g))
            return -np.log(p[np.arange(len(y)), y]).sum() + self.l2 * (b @ b + g @ g)
        th0 = np.concatenate([np.zeros(k + m), [-4.0], np.log([2.0, 4.0, 2.0])])
        res = minimize(nll, th0, method="L-BFGS-B")
        self.b, self.g, self.cuts = unpack(res.x)
        return self

    def spread(self, rows):
        Z = self._X(rows, self.scale)
        return np.exp((Z - self.z_mu) @ self.g) if Z.shape[1] else np.ones(len(rows))

    def predict(self, rows):
        return self._probs(self._X(rows) @ self.b, self.cuts, self.spread(rows))


def vote_distribution(prob_delta: np.ndarray, prev_vote: int, options: list[int]) -> dict[int, float]:
    """Δ probabilities -> probabilities over the allowed vote options (folding
    anything outside the range onto its edge)."""
    out = {o: 0.0 for o in options}
    for d, p in zip(DELTAS, prob_delta):
        v = int(np.clip(prev_vote + d, options[0], options[-1]))
        out[v] += float(p)
    return out


CORRELATION = 0.0   # members' common "news" factor in the Committee simulation; set from
                    # the backtest by scripts/vote_forecast.py (calibrate_correlation)


def simulate_votes(member_probs: list[dict[int, float]], n_sims: int = 20000, seed: int = 0,
                   rho: float | None = None) -> tuple[list[int], np.ndarray]:
    """Joint draws of every member's vote. Each member keeps exactly their own
    probabilities (a Gaussian copula); rho is how strongly they move together — a shared
    factor pushes everyone hawkish or dovish at once, as common news does."""
    from scipy.stats import norm
    rho = CORRELATION if rho is None else rho
    rng = np.random.default_rng(seed)
    opts = sorted({o for mp in member_probs for o in mp})
    common = rng.standard_normal(n_sims)
    draws = []
    for mp in member_probs:
        pr = np.array([mp.get(o, 0.0) for o in opts], float)
        cdf = np.cumsum(pr / pr.sum())
        z = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.standard_normal(n_sims)
        u = norm.cdf(z)
        draws.append(np.array(opts)[np.minimum(np.searchsorted(cdf, u), len(opts) - 1)])
    return opts, np.vstack(draws)


def decision_distribution(member_probs: list[dict[int, float]], n_sims: int = 20000,
                          seed: int = 0, rho: float | None = None) -> dict[int, float]:
    """The Committee's decision as the median member's vote, with members' votes
    correlated through a common factor (rho; 0 = independent)."""
    opts, draws = simulate_votes(member_probs, n_sims, seed, rho)
    med = np.median(draws, axis=0)
    return {o: float(np.mean(med == o)) for o in opts}
