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
# offered in this order; "words" is offered in two forms and the better one competes
ORDER = [("committee", G["committee"]), ("voting record", G["record"]), ("pace", G["pace"]),
         ("own words (one weighted signal)", G["words"]),
         ("own words (split: rationale / policy-focused / other)", G["words_split"]),
         ("data & energy", G["data"]), ("hedged guidance", G["hedge"]),
         ("votes with Governor", G["governor"]), ("Fed & ECB", G["global"]),
         ("sterling", G["fx"]), ("speech tone", G["tone"])]
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
        print(f"panel built, half-life {h}")
    ref = panels[HALF_LIVES[0]]
    print(f"{len(ref)} member-votes at {len({r['date'] for r in ref})} meetings")

    cur, cur_P, cur_hl = [], loyo(ref, []), {}
    cur_loss = row_loss(ref, cur_P)
    steps = [{"group": "last vote only", "kept": True, "log_loss": round(float(cur_loss.mean()), 4)}]
    words_done = False
    for name, grp in ORDER:
        if name.startswith("own words") and words_done:
            continue
        if name.startswith("own words"):
            # offer both forms; the better one is the candidate
            cands = [(n2, g2) for n2, g2 in ORDER if n2.startswith("own words")]
            res = []
            for n2, g2 in cands:
                P, hl = nested(panels, cur + g2)
                res.append((row_loss(ref, P).mean(), n2, g2, P, hl))
            res.sort(key=lambda x: x[0])
            _, name, grp, P, hl = res[0]
            words_done = True
            other = res[1]
            print(f"  own words: split {'beats' if 'split' in name else 'loses to'} one weighted signal "
                  f"({res[0][0]:.4f} vs {other[0]:.4f})")
        else:
            P, hl = nested(panels, cur + grp)
        new_loss = row_loss(ref, P)
        m, lo, hi = bootstrap(ref, new_loss, cur_loss)
        keep = hi < 0
        steps.append({"group": name, "features": grp, "kept": keep, "log_loss": round(float(new_loss.mean()), 4),
                      "change": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "half_life_by_year": hl if WORD_FEATS & set(cur + grp) else None})
        print(f"  {'KEEP' if keep else 'drop'} {name:52} log loss {new_loss.mean():.4f}  "
              f"change {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        if keep:
            cur, cur_P, cur_loss, cur_hl = cur + grp, P, new_loss, hl

    # the live half-life: chosen by leave-one-year-out on all years
    if WORD_FEATS & set(cur):
        scores = {h: float(row_loss(panels[h], loyo(panels[h], cur)).mean()) for h in HALF_LIVES}
        live_hl = min(scores, key=scores.get)
    else:
        scores, live_hl = {}, HALF_LIVES[1]
    flat = float(row_loss(panels[None], loyo(panels[None], cur)).mean()) if WORD_FEATS & set(cur) else None
    print(f"half-life scores {scores} -> {live_hl} days; all words equal (for reference): {flat}")
    out = {"features": cur, "half_life": live_hl, "half_life_scores": scores, "flat_words_log_loss": flat,
           "steps": steps, "nested_log_loss": round(float(cur_loss.mean()), 4),
           "selected": dt.date.today().isoformat(),
           "method": "forward selection; nested leave-one-year-out; meeting-level paired bootstrap, "
                     "keep if the 95% interval of the change in log loss is below zero"}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"selected: {cur}  half-life {live_hl}  -> {OUT}")


if __name__ == "__main__":
    main()
