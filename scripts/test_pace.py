"""Test alternative codings of the cycle-pace inputs against the current one.

  current       pace, moved_last, mpr_x_pace (one set for hiking and cutting cycles)
  by direction  the same three inputs, separately for hiking and cutting cycles
  established   pace and mpr_x_pace only once the cycle has two or more moves; after a
                single move, a separate 'first move' input (and its MPR-meeting version)
                lets the model learn what follows a first move instead of reading a
                cadence into one data point
  both          by direction is not combined with established (too few first-move
                meetings per direction)

Every other input is the selected model's. Nested leave-one-year-out as in
select_vote_model.py; meeting-level paired bootstrap, 2000 draws, 95% interval. The
current coding is kept unless an alternative is significantly better.

    python scripts/test_pace.py   # -> data/processed/pace_test.json
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

OUT = ROOT / "data/processed/pace_test.json"
G = V.FEATURE_GROUPS


def main():
    warnings.filterwarnings("ignore")
    ctx = V.build_context(load_corpus(ROOT / "data/processed/corpus.jsonl"))
    panels = {}
    for h in S.HALF_LIVES:
        V.HALF_LIFE = h
        panels[h] = V.panel(ctx, "2015-06-01")
    ref = panels[S.HALF_LIVES[0]]
    sel = json.loads((ROOT / "data/processed/vote_model.json").read_text())["features"]
    rest = [f for f in sel if f not in G["pace"]]
    base = rest + G["pace"]
    P0, _ = S.nested(panels, base)
    l0 = S.row_loss(ref, P0)
    cyc = np.array([r["cycle_dir"] for r in ref])
    nm = np.array([r["n_moves"] for r in ref])
    out = {"n": len(ref), "current_log_loss": round(float(l0.mean()), 4), "alternatives": [],
           "rows": {"hiking": int((cyc > 0).sum()), "cutting": int((cyc < 0).sum()), "no cycle": int((cyc == 0).sum()),
                    "after one move": int((nm == 1).sum())}}
    # how well the current coding does in each kind of cycle
    out["current_by_cycle"] = {k: round(float(l0[m].mean()), 4) for k, m in
                               (("hiking", cyc > 0), ("cutting", cyc < 0), ("no cycle", cyc == 0),
                                ("after one move", nm == 1), ("established (2+ moves)", nm >= 2))}
    V.HALF_LIFE = 30
    rows30 = panels[30]
    for name, g in (("by direction", G["pace_by_direction"]), ("established", G["pace_established"])):
        P1, _ = S.nested(panels, rest + g)
        l1 = S.row_loss(ref, P1)
        m, lo, hi = S.bootstrap(ref, l1, l0)
        mdl = V.OrderedLogit(rest + g).fit(rows30)
        coefs = {f: round(float(b), 3) for f, b in zip(mdl.loc, mdl.b) if f in g}
        by = {k: round(float(l1[mk].mean()), 4) for k, mk in
              (("hiking", cyc > 0), ("cutting", cyc < 0), ("after one move", nm == 1), ("established (2+ moves)", nm >= 2))}
        out["alternatives"].append({"coding": name, "log_loss": round(float(l1.mean()), 4), "change": round(m, 4),
                                    "ci95": [round(lo, 4), round(hi, 4)], "better": hi < 0, "coefficients": coefs,
                                    "by_cycle": by})
        print(f"{name:14} {l1.mean():.4f} vs {l0.mean():.4f}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {coefs}  {by}")
    mdl = V.OrderedLogit(base).fit(rows30)
    out["current_coefficients"] = {f: round(float(b), 3) for f, b in zip(mdl.loc, mdl.b) if f in G["pace"]}
    print("current", out["current_log_loss"], out["current_coefficients"], out["current_by_cycle"], out["rows"])
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
