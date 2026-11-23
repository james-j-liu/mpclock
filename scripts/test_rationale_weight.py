"""Vote rationales (the individual boxes in the minutes, published since November 2025):
their own weight, or the weight of a policy-focused speech?

  split   (before) boxes have their own inputs (g_rat_gap, g_rat_lvl), whose weight is
          learned from the boxes alone: when 2026 is predicted, from the two 2025 meetings
  pooled  (now) boxes count as policy-focused documents, sharing the speeches' weight, learned
          from ten years of speeches

Pooling removes two inputs, so it is preferred unless the split version is significantly
better (the same asymmetry as for add-ons: extra inputs must earn their place). Nested
leave-one-year-out; meeting-level paired bootstrap, 2000 draws, 95% interval; reported
overall and for the meetings since boxes began.

    python scripts/test_rationale_weight.py   # -> data/processed/rationale_test.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import select_vote_model as S
from mpclock.forecast import votes as V
from mpclock.schema import load_corpus

OUT = ROOT / "data/processed/rationale_test.json"
BOXES_FROM = "2025-12-01"   # first meeting whose window contains a box (Nov 2025 minutes)


def run(ctx, pooled, feats):
    V.RATIONALE_AS_POLICY = pooled
    panels = {}
    for h in S.HALF_LIVES:
        V.HALF_LIFE = h
        panels[h] = V.panel(ctx, "2015-06-01")
    ref = panels[S.HALF_LIVES[0]]
    P, _ = S.nested(panels, feats)
    V.HALF_LIFE = 60
    mdl = V.OrderedLogit(feats).fit(panels[60])
    return ref, S.row_loss(ref, P), P, {f: round(float(b), 3) for f, b in zip(mdl.loc, mdl.b) if f.startswith("g_")}


def main():
    warnings.filterwarnings("ignore")
    ctx = V.build_context(load_corpus(ROOT / "data/processed/corpus.jsonl"))
    pooled_feats = [f for f in json.loads((ROOT / "data/processed/vote_model.json").read_text())["features"]
                    if f not in ("g_rat_gap", "g_rat_lvl")]
    i = pooled_feats.index("g_pol_gap")
    sel = pooled_feats[:i] + ["g_rat_gap", "g_rat_lvl"] + pooled_feats[i:]
    ref, ls, Ps, cs = run(ctx, False, sel)
    _, lp, Pp, cp = run(ctx, True, pooled_feats)
    V.RATIONALE_AS_POLICY = True
    era = np.array([r["date"] >= BOXES_FROM for r in ref])
    m, lo, hi = S.bootstrap(ref, ls, lp)                      # split minus pooled
    sub = [r for r, e in zip(ref, era) if e]
    m2, lo2, hi2 = S.bootstrap(sub, ls[era], lp[era])
    out = {"split_log_loss": round(float(ls.mean()), 4), "pooled_log_loss": round(float(lp.mean()), 4),
           "split_minus_pooled": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "split_significantly_better": hi < 0,
           "since_boxes": {"n": int(era.sum()), "split": round(float(ls[era].mean()), 4),
                           "pooled": round(float(lp[era].mean()), 4), "split_minus_pooled": round(m2, 4),
                           "ci95": [round(lo2, 4), round(hi2, 4)]},
           "coefficients": {"split": cs, "pooled": cp}}
    # Bailey, December 2025
    for lab, P in (("split", Ps), ("pooled", Pp)):
        for i, r in enumerate(ref):
            if r["date"] == "2025-12-17" and r["member"] == "Andrew Bailey":
                vd = V.vote_distribution(P[i], r["prev_vote"], r["options"])
                out.setdefault("bailey_dec_2025_p_cut", {})[lab] = round(vd.get(-1, 0.0), 3)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
