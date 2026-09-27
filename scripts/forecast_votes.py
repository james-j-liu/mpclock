"""Forecast each MPC member's vote at the next meeting — and say how good that is.

For every member at every meeting with named votes, predict whether they vote to
HIKE (or, separately, to CUT) at the following meeting from what was known before
it: their own vote at the last meeting, how many colleagues voted that way at the
last meeting, and their tone since then — their own documents between the two
meetings (speeches, interviews, evidence, and from Nov 2025 their published vote
rationale), both absolute and relative to colleagues who spoke in the same window.

Skill is measured out of sample: each year is predicted by a model fitted on the
other years only, and scored by Brier score and log loss against a model that
knows everything except the tone — so the gain from tone is isolated. Probabilities
are also checked for calibration.

    python scripts/forecast_votes.py            # backtest, hikes and cuts
    python scripts/forecast_votes.py --member "Clare Lombardelli"   # next-meeting odds
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from mpclock.corpus.boe_interviews import governor_on
from mpclock.macro.mpc_votes import checked_member_votes
from mpclock.roster_mpc import canon
from mpclock.schema import load_corpus
from mpclock.tenure import attendance

OWN = {"speech", "interview", "testimony", "member_view"}
BASE = ["prev_same", "n_same_prev"]
TONE = ["tone", "tone_rel", "has_tone"]


def load():
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    docs = pd.DataFrame([{"p": canon(s.speaker), "d": pd.Timestamp(s.date), "m": (s.mu - 50) / 5}
                         for s in corpus if s.source_type in OWN and s.mu is not None])
    rate = pd.DataFrame(json.loads((ROOT / "site/macro.json").read_text())["series"]
                        ["bank_rate"]["data"], columns=["d", "r"])
    rate["d"] = pd.to_datetime(rate["d"])
    meetings = []
    for s in sorted((s for s in corpus if s.source_type == "mp_account"), key=lambda s: s.date):
        before = rate[rate.d < pd.Timestamp(s.date)]
        v = checked_member_votes(s.text, float(before.r.iloc[-1]) if len(before) else None,
                                 attendance(s.text), governor_on(s.date))
        meetings.append((pd.Timestamp(s.date), v))
    return docs, meetings


def tone_window(docs, start, end):
    """Each member's mean tone from `start` (inclusive: a vote rationale is published
    with the decision) up to `end` (exclusive), and relative to others in the window."""
    w = docs[(docs.d >= start) & (docs.d < end)]
    own = w.groupby("p")["m"].mean()
    return own, (own - own.mean()) if len(own) > 1 else own * 0


def panel(docs, meetings, side: int) -> pd.DataFrame:
    """One row per member at meeting t+1, features from meeting t and the gap between."""
    rows = []
    for (d0, v0), (d1, v1) in zip(meetings, meetings[1:]):
        if not v0 or not v1:
            continue
        own, rel = tone_window(docs, d0, d1)
        n_same = sum(1 for b in v0.values() if np.sign(b) == side)
        for p, b1 in v1.items():
            if p not in v0:
                continue
            rows.append({"meeting": d1, "year": d1.year, "member": p,
                         "y": int(np.sign(b1) == side),
                         "prev_same": int(np.sign(v0[p]) == side),
                         "n_same_prev": n_same / len(v0),
                         "tone": own.get(p, 0.0) * side,
                         "tone_rel": rel.get(p, 0.0) * side,
                         "has_tone": int(p in own)})
    return pd.DataFrame(rows)


def backtest(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for yr in sorted(df.year.unique()):
        tr, te = df.year != yr, df.year == yr
        if df.loc[tr, "y"].nunique() < 2:
            pred[te.values] = df.loc[tr, "y"].mean()
            continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(df.loc[tr, feats], df.loc[tr, "y"])
        pred[te.values] = m.predict_proba(df.loc[te, feats])[:, 1]
    return pred


def report(df: pd.DataFrame, label: str) -> None:
    base, full = backtest(df, BASE), backtest(df, BASE + TONE)
    y = df.y.values
    print(f"\n=== {label}: {len(df)} member-votes, {y.sum()} {label.split()[0].lower()} votes "
          f"({y.mean():.1%})")
    for name, p in (("last vote + colleagues", base), ("... + own tone", full)):
        print(f"  {name:24} Brier {brier_score_loss(y, p):.4f}  log loss {log_loss(y, p):.4f}")
    # the cases that matter: members who did NOT vote this way last time
    sw = df.prev_same.values == 0
    print(f"  switches only (not voting this way last time): n={sw.sum()}, "
          f"{y[sw].sum()} switched")
    for name, p in (("last vote + colleagues", base), ("... + own tone", full)):
        print(f"    {name:24} Brier {brier_score_loss(y[sw], p[sw]):.4f}  "
              f"log loss {log_loss(y[sw], p[sw], labels=[0, 1]):.4f}")
    bins = pd.cut(full[sw], [0, .02, .05, .1, .2, .4, 1.0])
    cal = pd.DataFrame({"bin": bins, "y": y[sw], "p": full[sw]}).groupby("bin", observed=True)
    print("    calibration on switches (predicted -> actual):")
    for b, g in cal:
        print(f"      {str(b):14} n={len(g):4d}  predicted {g.p.mean():6.1%}  actual {g.y.mean():6.1%}")


def forecast(docs, meetings, member: str) -> None:
    today = pd.Timestamp.today().normalize()
    last_d, last_v = [m for m in meetings if m[1]][-1]
    own, rel = tone_window(docs, last_d, today + pd.Timedelta(days=1))
    print(f"\nlast meeting with named votes: {last_d:%Y-%m-%d}; tone window {last_d:%d %b} - today")
    print("  documents in the window:",
          ", ".join(f"{p.split()[-1]} {own[p]:+.2f}" for p in own.index) or "none")
    for side, label in ((1, "hike"), (-1, "cut")):
        df = panel(docs, meetings, side)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(df[BASE + TONE], df.y)
        n_same = sum(1 for b in last_v.values() if np.sign(b) == side) / len(last_v)
        x = pd.DataFrame([{"prev_same": int(np.sign(last_v.get(member, 0)) == side),
                           "n_same_prev": n_same,
                           "tone": own.get(member, 0.0) * side,
                           "tone_rel": rel.get(member, 0.0) * side,
                           "has_tone": int(member in own)}])
        base = LogisticRegression(C=1.0, max_iter=1000).fit(df[BASE], df.y)
        print(f"  P({member} votes to {label} next meeting) = "
              f"{model.predict_proba(x[BASE + TONE])[0, 1]:.1%}   "
              f"(without tone: {base.predict_proba(x[BASE])[0, 1]:.1%})")
        coefs = dict(zip(BASE + TONE, model.coef_[0]))
        print("    coefficients: " + ", ".join(f"{k} {v:+.2f}" for k, v in coefs.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    docs, meetings = load()
    if args.member:
        forecast(docs, meetings, args.member)
        return
    report(panel(docs, meetings, 1), "HIKE votes")
    report(panel(docs, meetings, -1), "CUT votes")


if __name__ == "__main__":
    main()
