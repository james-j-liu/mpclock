"""Per-member vote forecast for the next MPC meeting.

For each member the model predicts the CHANGE from their own previous vote
(Δ, in 25bp steps), an ordered logit over Δ ∈ {-2,-1,0,+1,+2}. Everything is signed
so that a positive value pushes towards tighter policy. Inputs, all measured over
the window from the previous meeting up to the day before this one:

  committee   prev_dev     own previous vote minus the Committee's decision
              pull         Committee's average vote minus own previous vote
  words       tone         own documents' hawkishness (tournament score) relative
                           to the member's own previous 12 months
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

SIGNAL_SOURCE = "text"   # "text" or "passages": which coded vote feeds the guidance inputs
                         # (backtested: whole text 0.670 log loss, passages alone 0.680)
HISTORY = 6       # meetings of each member's voting record used, beyond the last one

FEATURE_GROUPS = {
    "committee": ["prev_dev", "pull"],
    "record": ["hist_dev", "trend"],
    "pace": ["pace", "moved_last", "mpr_x_pace"],
    "words": ["guidance", "signal_level"],
    "direction": ["direction"],
    "data": ["surprise"],          # outturns vs MPR, plus half-weighted energy-price change
    "global": ["fed", "ecb"],
    "fx": ["fx"],
    "tone": ["tone"],
}

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
    return Context(meetings, docs, inputs, mpr_dates, rate)


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


def pace_features(ctx: Context, i: int, date: str) -> dict[str, float]:
    """Revealed pace of the current cycle, from meetings before `date`."""
    past = [m for m in ctx.meetings if m.date < date]
    if not past:
        return {"pace": 0.0, "moved_last": 0.0, "mpr_x_pace": 0.0, "mpr": 0.0, "cycle_dir": 0}
    moves = [(m.date, np.sign(m.decision)) for m in past if m.decision != 0]
    if not moves:
        return {"pace": 0.0, "moved_last": 0.0, "mpr_x_pace": 0.0, "mpr": 0.0, "cycle_dir": 0}
    cdir = moves[-1][1]
    # the cycle began at the first move in this direction after the last opposite move
    start = moves[-1][0]
    for d, s in reversed(moves):
        if s != cdir:
            break
        start = d
    # a cycle whose last move is over a year old is dormant
    active = (dt.date.fromisoformat(date) - dt.date.fromisoformat(moves[-1][0])).days < 400
    is_mpr = lambda d: any(abs((dt.date.fromisoformat(d) - dt.date.fromisoformat(x)).days) <= 2
                           for x in ctx.mpr_dates)
    this_mpr = is_mpr(date)
    same = [m for m in past if m.date >= start and is_mpr(m.date) == this_mpr]
    moved = sum(1 for m in same if np.sign(m.decision) == cdir)
    pace = (moved + 0.5) / (len(same) + 1.0) if active else 0.0
    last = past[-1]
    return {"pace": float(cdir * pace), "moved_last": float(np.sign(last.decision)),
            "mpr_x_pace": float(cdir * pace * (1 if this_mpr else -1)) if active else 0.0,
            "mpr": float(this_mpr), "cycle_dir": int(cdir) if active else 0}


def member_window(ctx: Context, member: str, start: str, end: str, prev_vote: int = 0):
    """A member's own words between two meetings: tone, coded guidance, emphasis."""
    docs = [d for d in ctx.docs.get(member, []) if start <= d[0] < end]
    prior = [d for d in ctx.docs.get(member, []) if _shift(start, -365) <= d[0] < start and d[1] is not None]
    tones = [d[1] for d in docs if d[1] is not None]
    tone = (np.mean(tones) - (np.mean([p[1] for p in prior]) if prior else np.mean(tones))) if tones else 0.0
    text = "\n".join(d[2] for d in docs)
    # coded guidance: the vote each document signals, weighted by how much it is
    # about the policy outlook and by recency (the latest word counts double)
    codes = ctx.inputs.get("guidance", {})
    coded = [(d, codes[d[3]]) for d in docs if d[3] in codes]
    gap, has, signal, direction = 0.0, 0.0, None, 0.0
    if coded:
        w = np.array([(0.25 + (c.get("focus") or 0.0)) * (2.0 if i == len(coded) - 1 else 1.0)
                      for i, (_, c) in enumerate(coded)])
        # the vote signalled by the policy passages where they were coded separately
        # (a long speech's guidance is often a short section at the end), else the text
        ev = np.array([c.get("ev_pass", c["ev"]) if SIGNAL_SOURCE == "passages" else c["ev"]
                       for _, c in coded])
        signal = float((w * ev).sum() / w.sum())
        gap, has = signal - prev_vote, 1.0
        dv = np.array([c.get("dir_ev_pass", c.get("dir_ev", 0.0)) for _, c in coded])
        direction = float((w * dv).sum() / w.sum())
    g_regex, _ = guidance_score(text)
    return {"tone": float(tone), "guidance": float(gap), "has_guidance": has, "signal": signal,
            "direction": direction,
            "guidance_regex": float(g_regex), "n_docs": len(docs), "topic_w": topic_weights(text),
            "doc_dates": [d[0] for d in docs],
            "docs": [{"date": d[0], "type": d[4], "id": d[3],
                      "signal": codes.get(d[3], {}).get("ev"), "p": codes.get(d[3], {}).get("p"),
                      "dir": codes.get(d[3], {}).get("dir")}
                     for d in docs]}


def features_for(ctx: Context, prev: Meeting, date: str, members: list[str]) -> list[dict]:
    """Rows for each member at a meeting on `date` (a past meeting or a future one)."""
    start, end = prev.date, date
    surp, detail = data_surprises(ctx, start, end)
    mk = market_changes(ctx, start, end)
    pace = pace_features(ctx, 0, date)
    mean_prev = np.mean(list(prev.votes.values()))
    # the meetings before `prev`, newest first: a member's record over several meetings,
    # not just the last vote
    i_prev = next((i for i, m in enumerate(ctx.meetings) if m.date == prev.date), len(ctx.meetings) - 1)
    earlier = ctx.meetings[max(0, i_prev - HISTORY):i_prev][::-1]
    rows = []
    for p in members:
        pv = prev.votes.get(p, prev.decision)
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
        w = member_window(ctx, p, start, end, pv)
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
            "tone": w["tone"], "guidance": w["guidance"], "n_docs": w["n_docs"],
            # energy prices are data too: a 10% rise counts as half a standard surprise
            "surprise": float(np.clip(personal + 0.5 * np.clip(mk["energy"], -4, 4), -5, 5)),
            "surprise_data": float(np.clip(personal, -4, 4)),
            "energy": float(np.clip(mk["energy"], -4, 4)), "fx": float(np.clip(mk["fx"], -4, 4)),
            "fed": mk["fed"], "ecb": mk["ecb"],
            "pace": pace["pace"], "moved_last": pace["moved_last"], "mpr_x_pace": pace["mpr_x_pace"],
            "has_guidance": w["has_guidance"],
            # the vote the member's own words point to, as a level (not just the change
            # from their last vote): "more tightening needed" from someone who just voted
            # to hike is news against the drift back to hold after a move
            "signal_level": float(w["has_guidance"] * (w["guidance"] + pv)),
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
    def __init__(self, feats: list[str], l2: float = 1.0):
        self.feats, self.l2 = feats, l2

    def _X(self, rows):
        return np.array([[r[f] for f in self.feats] for r in rows], dtype=float) if self.feats else \
            np.zeros((len(rows), 0))

    @staticmethod
    def _probs(eta, cuts):
        # P(y <= k) = σ(c_k - η); categories DELTAS
        cdf = 1 / (1 + np.exp(-(cuts[None, :] - eta[:, None])))
        cdf = np.hstack([np.zeros((len(eta), 1)), cdf, np.ones((len(eta), 1))])
        return np.clip(np.diff(cdf, axis=1), 1e-9, 1)

    def fit(self, rows):
        X = self._X(rows)
        y = np.array([r["y"] for r in rows]) - DELTAS[0]
        k = X.shape[1]

        def unpack(th):
            b = th[:k]
            c0 = th[k]
            cuts = np.cumsum(np.concatenate([[c0], np.exp(th[k + 1:])]))
            return b, cuts

        def nll(th):
            b, cuts = unpack(th)
            p = self._probs(X @ b, cuts)
            return -np.log(p[np.arange(len(y)), y]).sum() + self.l2 * (b @ b)
        th0 = np.concatenate([np.zeros(k), [-4.0], np.log([2.0, 4.0, 2.0])])
        res = minimize(nll, th0, method="L-BFGS-B")
        self.b, self.cuts = unpack(res.x)
        return self

    def predict(self, rows):
        return self._probs(self._X(rows) @ self.b, self.cuts)


def vote_distribution(prob_delta: np.ndarray, prev_vote: int, options: list[int]) -> dict[int, float]:
    """Δ probabilities -> probabilities over the allowed vote options (folding
    anything outside the range onto its edge)."""
    out = {o: 0.0 for o in options}
    for d, p in zip(DELTAS, prob_delta):
        v = int(np.clip(prev_vote + d, options[0], options[-1]))
        out[v] += float(p)
    return out


def decision_distribution(member_probs: list[dict[int, float]], n_sims: int = 20000,
                          seed: int = 0) -> dict[int, float]:
    """The Committee's decision as the median member's vote, simulating members
    independently. Independence understates how often members move together, so
    these odds are a little too confident in the modal outcome."""
    rng = np.random.default_rng(seed)
    opts = sorted(member_probs[0])
    draws = []
    for mp in member_probs:
        pr = np.array([mp[o] for o in opts])
        draws.append(rng.choice(opts, size=n_sims, p=pr / pr.sum()))
    med = np.median(np.vstack(draws), axis=0)
    return {o: float(np.mean(med == o)) for o in opts}
