"""Does the hawkishness index lead UK rates — the 10-year gilt yield and Bank Rate?

ECBLock found its index led the 10-year Bund-area yield by 6-8 months. This runs the
same test for the UK, with one precaution: two trending series correlate at any lag,
so the cross-correlation is shown in levels AND in monthly changes, and the Granger
tests run on changes (stationary), in both directions. A lead only in levels would
be a shared trend, not information.

Index: monthly mean of raw pairwise ratings (display units), 3-month smoothed as in
ECBLock. --members-only drops the Committee's own documents (minutes, Report, press
conference), since those are published alongside the decisions themselves.

    python scripts/gilt_leadlag.py [--members-only] [--since 1998-01]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

COMMITTEE = {"mp_account", "mp_report", "mp_statement", "mp_qa", "member_view"}


def monthly(series: dict, key: str) -> pd.Series:
    s = pd.Series({pd.Timestamp(d): v for d, v in series[key]["data"]})
    return s.sort_index().resample("ME").last().ffill()


def ccf(x: pd.Series, y: pd.Series, lags=range(-12, 13)) -> dict[int, float]:
    """corr(x_t, y_{t+k}): positive k = x leads y by k months."""
    return {k: x.corr(y.shift(-k)) for k in lags}


def granger_p(cause: pd.Series, effect: pd.Series, maxlag: int = 12) -> dict[int, float]:
    df = pd.concat([effect, cause], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = grangercausalitytests(df.values, maxlag=maxlag, verbose=False)
    return {lag: r[0]["ssr_ftest"][1] for lag, r in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members-only", action="store_true")
    ap.add_argument("--since", default="1998-01")
    args = ap.parse_args()

    data = json.loads((ROOT / "site/data.json").read_text(encoding="utf-8"))["speeches"]
    macro = json.loads((ROOT / "site/macro.json").read_text(encoding="utf-8"))["series"]
    docs = pd.DataFrame([{"d": pd.Timestamp(s["d"]), "m": (s["m"] - 50) / 5}
                         for s in data if s["m"] is not None
                         and not (args.members_only and s["st"] in COMMITTEE)])
    idx = docs.set_index("d")["m"].resample("ME").mean().interpolate(limit=3)
    idx = idx.rolling(3, min_periods=1).mean()

    rates = {"10Y gilt": monthly(macro, "gilt_10y"), "Bank Rate": monthly(macro, "bank_rate")}
    start = pd.Timestamp(args.since)
    idx = idx[idx.index >= start]
    print(f"index: {len(idx)} months {idx.index.min():%Y-%m}..{idx.index.max():%Y-%m} "
          f"({'members only' if args.members_only else 'all documents'})")
    print(f"ADF p-values (levels -> changes): index {adfuller(idx.dropna())[1]:.3f} -> "
          f"{adfuller(idx.diff().dropna())[1]:.3f}")

    for name, r in rates.items():
        r = r[r.index >= start].reindex(idx.index).ffill()
        lvl, chg = ccf(idx, r), ccf(idx.diff(), r.diff())
        best_l = max(lvl, key=lambda k: lvl[k]); best_c = max(chg, key=lambda k: abs(chg[k]))
        print(f"\n=== {name}  (ADF levels p={adfuller(r.dropna())[1]:.3f})")
        print(f"  cross-correlation, levels : peak r={lvl[best_l]:+.2f} at k={best_l:+d} months "
              f"(k>0: index leads) | k=0 {lvl[0]:+.2f}, k=+6 {lvl[6]:+.2f}, k=-6 {lvl[-6]:+.2f}")
        print(f"  cross-correlation, changes: peak r={chg[best_c]:+.2f} at k={best_c:+d} months "
              f"| k=0 {chg[0]:+.2f}, k=+3 {chg[3]:+.2f}, k=+6 {chg[6]:+.2f}")
        fwd = granger_p(idx.diff(), r.diff())
        back = granger_p(r.diff(), idx.diff())
        f_best = min(fwd, key=fwd.get); b_best = min(back, key=back.get)
        print(f"  Granger on changes, index -> {name}: min p={fwd[f_best]:.4f} at lag {f_best} "
              f"(p at 3/6/12: {fwd[3]:.3f}/{fwd[6]:.3f}/{fwd[12]:.3f})")
        print(f"  Granger on changes, {name} -> index: min p={back[b_best]:.4f} at lag {b_best} "
              f"(p at 3/6/12: {back[3]:.3f}/{back[6]:.3f}/{back[12]:.3f})")


if __name__ == "__main__":
    main()
