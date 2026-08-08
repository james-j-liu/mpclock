"""Does the hawkishness index predict how the MPC votes?

Builds a meeting-level panel and asks three questions of each vote score:

  1. does hawkish talk BEFORE a meeting line up with that meeting's vote?
  2. does it predict the NEXT meeting's vote?
  3. does it still predict the next vote once you know the current one?

(3) is the one that matters. Votes are persistent — a hold is usually followed by
a hold — so a raw correlation with the next vote mostly measures that persistence.
The regression puts the current vote on the right-hand side, so the hawkishness
coefficient answers "does what they say tell you anything you did not already know
from what they just did?".

The index for a meeting is built only from documents published BEFORE it, and only
from documents the Committee itself did not produce (speeches, interviews,
Treasury Committee evidence). Including the meeting's own minutes or Report would
make the exercise circular.

Usage:  python scripts/vote_analysis.py [--window 60] [--min-docs 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

# what the Committee itself publishes — excluded from the index for its own meeting
COMMITTEE_TYPES = {"mp_account", "mp_report", "mp_statement", "mp_qa", "member_view"}
VOTE_SERIES = ["vote_score"]


def load_panel(window: int, min_docs: int, individual: bool) -> pd.DataFrame:
    data = json.loads((ROOT / "site" / "data.json").read_text(encoding="utf-8"))
    macro = json.loads((ROOT / "site" / "macro.json").read_text(encoding="utf-8"))["series"]

    docs = pd.DataFrame([{"d": pd.Timestamp(s["d"]), "m": s["m"], "st": s["st"],
                          "a": s["a"]} for s in data["speeches"] if s["m"] is not None])
    if individual:
        docs = docs[~docs["st"].isin(COMMITTEE_TYPES)]
    docs = docs.sort_values("d")

    rows = []
    for key in VOTE_SERIES:
        if key not in macro:
            continue
        for day, value in macro[key]["data"]:
            rows.append({"date": pd.Timestamp(day), key: value})
    votes = (pd.DataFrame(rows).groupby("date").first().sort_index()
             if rows else pd.DataFrame())

    idx, n_docs = [], []
    for day in votes.index:
        w = docs[(docs["d"] < day) & (docs["d"] >= day - pd.Timedelta(days=window))]
        idx.append(w["m"].mean() if len(w) else np.nan)
        n_docs.append(len(w))
    votes["hawk"] = idx
    votes["n_docs"] = n_docs
    votes = votes[votes["n_docs"] >= min_docs]
    # display units: the site's -10..+10 scale
    votes["hawk"] = (votes["hawk"] - 50) / 5
    return votes


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> list[tuple[str, float, float, float]]:
    """OLS with a constant; returns (name, coef, t, p) per regressor."""
    import scipy.stats as st

    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    out = []
    for i, name in enumerate(["const"] + names):
        t = beta[i] / se[i] if se[i] else np.nan
        out.append((name, beta[i], t, 2 * (1 - st.t.cdf(abs(t), dof))))
    return out


def alternatives(p: pd.DataFrame, key: str) -> None:
    """Specifications where a signal is likelier to show than in the level test.

    Talk is a level and a vote is a decision, so the change in the index may carry
    what its level does not; and a shift in stance may take more than one meeting
    to reach a vote, so horizons out to three meetings are tried too.
    """
    d = p[[key, "hawk"]].dropna().copy()
    d["dhawk"] = d["hawk"].diff()
    d["dvote"] = d[key].diff()
    print(f"  --- alternative readings of {key} ---")
    for h in (1, 2, 3):
        d[f"next{h}"] = d[key].shift(-h)
        d[f"dnext{h}"] = d[key].shift(-h) - d[key]
        sub = d[["hawk", "dhawk", f"next{h}", f"dnext{h}", key]].dropna()
        if len(sub) < 20:
            continue
        c_lvl = sub["hawk"].corr(sub[f"dnext{h}"])
        c_chg = sub["dhawk"].corr(sub[f"dnext{h}"])
        res = ols(sub[f"dnext{h}"].values,
                  np.column_stack([sub["hawk"].values, sub["dhawk"].values]),
                  ["hawk", "dhawk"])
        ps = {n: pv for n, _, _, pv in res}
        print(f"    {h} meeting(s) ahead (n={len(sub)}): "
              f"corr(hawk, vote change) {c_lvl:+.3f}  corr(Δhawk, vote change) {c_chg:+.3f}"
              f"   | regression p: hawk {ps['hawk']:.3f}, Δhawk {ps['dhawk']:.3f}")
    # direction-only test: when the next meeting moves, does the index call the way?
    moved = d[(d[f"next1"] - d[key]).abs() > 1e-9].dropna(subset=["hawk"])
    if len(moved) >= 20:
        agree = np.sign(moved[f"next1"] - moved[key]) == np.sign(moved["hawk"])
        print(f"    direction of the next move, when there is one (n={len(moved)}): "
              f"index called it {100*agree.mean():.0f}% of the time")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60,
                    help="days of documents before a meeting that make its index")
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--all-docs", action="store_true",
                    help="include the Committee's own output in the index (circular)")
    args = ap.parse_args()

    p = load_panel(args.window, args.min_docs, individual=not args.all_docs)
    print(f"Meetings with an index: {len(p)}  ({p.index.min().date()} .. {p.index.max().date()})")
    print(f"Index = mean hawkishness of "
          f"{'all documents' if args.all_docs else 'members-only documents'} in the "
          f"{args.window} days before each meeting (min {args.min_docs} documents)\n")

    for key in VOTE_SERIES:
        if key not in p:
            continue
        d = p[[key, "hawk"]].dropna().copy()
        d["next"] = d[key].shift(-1)
        both = d.dropna()
        if len(both) < 20:
            print(f"{key}: too few meetings ({len(both)}) to say anything\n")
            continue
        same = d[[key, "hawk"]].corr().iloc[0, 1]
        nxt = both[["next", "hawk"]].corr().iloc[0, 1]
        persist = both[[key, "next"]].corr().iloc[0, 1]
        print(f"=== {key}  (n={len(both)} meeting pairs, {both.index.min().date()}..{both.index.max().date()})")
        print(f"  corr(hawk, this vote)      {same:+.3f}")
        print(f"  corr(hawk, next vote)      {nxt:+.3f}")
        print(f"  corr(this vote, next vote) {persist:+.3f}   <- persistence to beat")
        res = ols(both["next"].values,
                  np.column_stack([both[key].values, both["hawk"].values]),
                  ["this_vote", "hawk"])
        print("  next_vote ~ this_vote + hawk")
        for name, coef, t, pv in res:
            star = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
            print(f"     {name:10} {coef:+8.4f}  t={t:+6.2f}  p={pv:.4f} {star}")
        print()
        alternatives(p, key)


if __name__ == "__main__":
    main()
