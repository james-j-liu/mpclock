"""Compare decision models for reading members' documents (the 'vote signalled' codes).

Arms (codes in data/processed/guidance_ab/, made by scripts/code_guidance_ab.py):
  jev113_dated     the live codes: Jev 1.13, shown the date and the member's name
  jev113_dated_oldboxes   the same, with vote rationales as read before the 'now' note
  jev113_blind     Jev 1.13, blind: no date, names -> [OFFICIAL], years -> [YEAR]
  jev114_blind     Jev 1.14, blind
  decider12_blind  Perplexity Decider V1.2, blind
  ensemble_blind   the average of Jev 1.14 and Decider V1.2 (no extra calls)

Each arm's codes go through the same selected vote model (data/processed/vote_model.json),
nested leave-one-year-out. The reference is jev113_blind: an arm is adopted only if it is
significantly better (meeting-level paired bootstrap, 2000 draws, 95% interval below zero).
jev113_dated vs jev113_blind measures how much the live backtest gains from the model
being able to place the document in time (which may include recalling what happened).

    python scripts/test_guidance_models.py   # -> data/processed/guidance_ab/comparison.json
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
from mpclock.forecast.guidance import load_codes
from mpclock.schema import load_corpus

AB = ROOT / "data/processed/guidance_ab"
REF = "jev113_blind"


def ensemble(a: dict, b: dict) -> dict:
    out = {}
    for k in a.keys() & b.keys():
        x, y = a[k], b[k]
        r = dict(x)
        for key in ("ev", "dir_ev"):
            if x.get(key) is not None and y.get(key) is not None:
                r[key] = (x[key] + y[key]) / 2
        for key in ("p", "dir"):
            if x.get(key) and y.get(key):
                r[key] = {o: (x[key].get(o, 0) + y[key].get(o, 0)) / 2 for o in x[key]}
        if x.get("focus") is not None and y.get("focus") is not None:
            r["focus"] = (x["focus"] + y["focus"]) / 2
        out[k] = r
    return out


def main():
    warnings.filterwarnings("ignore")
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    ctx = V.build_context(corpus)
    feats = json.loads((ROOT / "data/processed/vote_model.json").read_text())["features"]
    arms = {"jev113_dated": load_codes()}
    # the same live codes with the vote rationales as read before the 'now' note (v2)
    arms["jev113_dated_oldboxes"] = {k: ({**v, **{f: v["prev"][f] for f in ("p", "ev", "dir", "dir_ev")}}
                                         if v.get("type") == "member_view" and v.get("prev") else v)
                                     for k, v in arms["jev113_dated"].items()}
    for a in ("jev113_blind", "jev114_blind", "decider12_blind"):
        f = AB / f"{a}.json"
        if f.exists():
            arms[a] = load_codes(f)
    if "jev114_blind" in arms and "decider12_blind" in arms:
        arms["ensemble_blind"] = ensemble(arms["jev114_blind"], arms["decider12_blind"])
    losses, ref_rows = {}, None
    for name, codes in arms.items():
        ctx.inputs["guidance"] = codes
        panels = {}
        for h in S.HALF_LIVES:
            V.HALF_LIFE = h
            panels[h] = V.panel(ctx, "2015-06-01")
        ref_rows = panels[S.HALF_LIVES[0]]
        P, _ = S.nested(panels, feats)
        losses[name] = S.row_loss(ref_rows, P)
        print(f"{name:16} n_codes={len(codes):5}  log loss {losses[name].mean():.4f}", flush=True)
    out = {"reference": REF, "arms": {}}
    base = losses.get(REF)
    recent = np.array([r["date"] >= "2025-01-01" for r in ref_rows])
    for name, l in losses.items():
        row = {"log_loss": round(float(l.mean()), 4), "log_loss_since_2025": round(float(l[recent].mean()), 4),
               "n_codes": len(arms[name])}
        if base is not None and name != REF:
            m, lo, hi = S.bootstrap(ref_rows, l, base)
            row.update(change_vs_reference=round(m, 4), ci95=[round(lo, 4), round(hi, 4)], significantly_better=hi < 0)
        out["arms"][name] = row
        print(name, row)
    (AB / "comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # one case: Bailey's November 2025 box under each arm
    for name, codes in arms.items():
        for k, c in codes.items():
            if c.get("date") == "2025-11-05" and "Bailey" in (c.get("speaker") or "") and c.get("type") == "member_view":
                print(f"  Bailey Nov 2025 box, {name}: next vote {c.get('p')}, direction {c.get('dir')}")


if __name__ == "__main__":
    main()
