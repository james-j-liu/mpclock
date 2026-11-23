"""Choose the vote model's inputs and recency half-life — with a significance check.

Forward selection: starting from "each member repeats their last vote", input groups
are offered one at a time in a fixed order. A group is kept only if it lowers the
out-of-sample log loss by more than chance: a paired bootstrap over meetings (members
at the same meeting move together, so the meeting is the unit resampled, 2000 draws)
must put the 95% interval of the improvement wholly below zero.

Out-of-sample means nested leave-one-year-out: each year is predicted by a model
fitted on the other years, and the recency half-life of members' words (7/14/30/60
days; recent words always count more) is itself chosen inside those other years, never
by looking at the year being predicted.

    python scripts/select_vote_model.py     # writes data/processed/vote_model.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mpclock.forecast import votes as V
from mpclock.schema import load_corpus

OUT = ROOT / "data/processed/vote_model.json"
HALF_LIVES = [7, 14, 30, 60]
G = V.FEATURE_GROUPS
# Core structure, chosen when the model was designed (not in response to individual
# forecasts): kept as a bundle if the bundle beats "repeat your last vote" significantly.
CORE = [("committee", G["committee"]), ("voting record", G["record"]), ("pace", G["pace"]),
        ("own words", G["words_split"]), ("data & energy", G["data"])]
# Add-ons proposed later: each must clear the significance test on its own.
ADDONS = [("hedged guidance", G["hedge"]), ("votes with Governor", G["governor"]),
          ("Fed & ECB", G["global"]), ("sterling", G["fx"]), ("speech tone", G["tone"]),
          ("Committee-wide tone shift", G["committee_tone"]),
          ("member-specific volatility (spread)", G["volatility"]),
          ("steady members (narrower spread)", G["steady"])]
WORD_FEATS = set(G["words"]) | set(G["words_split"]) | set(G["hedge"]) | set(G["governor"])


def row_loss(rows, P):
    y = np.array([r["y"] for r in rows]) - V.DELTAS[0]
    return -np.log(P[np.arange(len(y)), y])


def loyo(rows, feats, l2=1.0, years=None):
    years = years or sorted({r["date"][:4] for r in rows})
    P = np.zeros((len(rows), len(V.DELTAS)))
    for y in years:
        tr = [r for r in rows if r["date"][:4] != y]
        idx = [i for i, r in enumerate(rows) if r["date"][:4] == y]
        if idx:
            P[idx] = V.OrderedLogit(feats, l2).fit(tr).predict([rows[i] for i in idx])
    return P


def nested(panels, feats):
    """Outer leave-one-year-out; the half-life chosen on the training years only.
    Returns (predictions, {year: half-life chosen})."""
    uses_words = bool(WORD_FEATS & set(feats))
    hl_grid = HALF_LIVES if uses_words else [HALF_LIVES[1]]
    ref = panels[hl_grid[0]]
    years = sorted({r["date"][:4] for r in ref})
    P = np.zeros((len(ref), len(V.DELTAS)))
    chosen = {}
    for y in years:
        best = None
        for h in hl_grid:
            rows = panels[h]
            tr = [r for r in rows if r["date"][:4] != y]
            if len(hl_grid) > 1:
                inner = loyo(tr, feats)
                score = row_loss(tr, inner).mean()
            else:
                score = 0.0
            if best is None or score < best[0]:
                best = (score, h)
        h = best[1]
        chosen[y] = h
        rows = panels[h]
        tr = [r for r in rows if r["date"][:4] != y]
        idx = [i for i, r in enumerate(rows) if r["date"][:4] == y]
        P[idx] = V.OrderedLogit(feats).fit(tr).predict([rows[i] for i in idx])
    return P, chosen


def bootstrap(rows, loss_new, loss_old, n=2000, seed=0):
    """Mean per-vote change in log loss (new - old), with a 95% interval from resampling
    meetings."""
    meet = np.array([r["date"] for r in rows])
    ids = sorted(set(meet))
    d = loss_new - loss_old
    per = np.array([d[meet == m].sum() for m in ids])
    cnt = np.array([(meet == m).sum() for m in ids])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        k = rng.integers(0, len(ids), len(ids))
        stats.append(per[k].sum() / cnt[k].sum())
    return float(d.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    warnings.filterwarnings("ignore")
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    ctx = V.build_context(corpus)
    panels = {}
    for h in HALF_LIVES + [None]:
        V.HALF_LIFE = h
        panels[h] = V.panel(ctx, "2015-06-01")
    ref = panels[HALF_LIVES[0]]
    n_meet = len({r["date"] for r in ref})
    print(f"{len(ref)} member-votes at {n_meet} meetings")
    base_loss = row_loss(ref, loyo(ref, []))

    core = [f for _, g in CORE for f in g]
    Pc, hl = nested(panels, core)
    core_loss = row_loss(ref, Pc)
    m, lo, hi = bootstrap(ref, core_loss, base_loss)
    print(f"core vs last vote: {core_loss.mean():.4f} vs {base_loss.mean():.4f}, change {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    report = {"baseline_log_loss": round(float(base_loss.mean()), 4),
              "core": {"log_loss": round(float(core_loss.mean()), 4), "change": round(m, 4),
                       "ci95": [round(lo, 4), round(hi, 4)], "significant": hi < 0,
                       "half_life_by_year": hl, "groups": []}}
    if not hi < 0:
        print("core is not significantly better than the baseline: using the baseline")
        core, Pc, core_loss = [], loyo(ref, []), base_loss
    # each core group's contribution, for the record
    for name, g in CORE:
        rest = [f for f in core if f not in g]
        P, _ = nested(panels, rest)
        mm, l2, h2 = bootstrap(ref, core_loss, row_loss(ref, P))
        report["core"]["groups"].append({"group": name, "contribution": round(mm, 4),
                                         "ci95": [round(l2, 4), round(h2, 4)], "significant_alone": h2 < 0})
        print(f"  core {name:14} contribution {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")
    # words: estimated split vs one hand-weighted signal, for the record
    alt = [f for f in core if f not in G["words_split"]] + G["words"]
    Pa, _ = nested(panels, alt)
    mm, l2, h2 = bootstrap(ref, core_loss, row_loss(ref, Pa))
    report["words_split_vs_combined"] = {"change": round(mm, 4), "ci95": [round(l2, 4), round(h2, 4)]}
    print(f"  split words vs one weighted signal: {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")

    cur, cur_loss = list(core), core_loss
    report["addons"] = []
    for name, g in ADDONS:
        P, _ = nested(panels, cur + g)
        nl = row_loss(ref, P)
        mm, l2, h2 = bootstrap(ref, nl, cur_loss)
        keep = h2 < 0
        report["addons"].append({"group": name, "log_loss": round(float(nl.mean()), 4), "change": round(mm, 4),
                                 "ci95": [round(l2, 4), round(h2, 4)], "kept": keep})
        print(f"  add-on {'KEEP' if keep else 'drop'} {name:20} {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")
        if keep:
            cur, cur_loss = cur + g, nl

    scores = {h: float(row_loss(panels[h], loyo(panels[h], cur)).mean()) for h in HALF_LIVES}
    live_hl = min(scores, key=scores.get)
    flat = float(row_loss(panels[None], loyo(panels[None], cur)).mean())
    out = {"features": cur, "half_life": live_hl, "half_life_scores": scores, "flat_words_log_loss": flat,
           "nested_log_loss": round(float(cur_loss.mean()), 4), "report": report, "meetings": n_meet,
           "selected": dt.date.today().isoformat(),
           "method": "core structure kept as a bundle if significantly better than 'repeat your last vote'; "
                     "each later add-on kept only if it significantly lowers log loss. Nested leave-one-year-out "
                     "(the recency half-life chosen on training years only); meeting-level paired bootstrap, "
                     "2000 draws, 95% interval."}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"selected {cur}; half-life {live_hl} days {scores}; equal weights {flat:.4f}")


if __name__ == "__main__":
    main()
