"""Should a low-confidence Jev verdict count as a win, or as a draw?

Jev returns a calibrated probability with every verdict, and most verdicts are
close calls — the tournament deliberately pairs documents with similar ratings.
Recording a 0.55 call as a full win feeds TrueSkill near-coin-flips as if they
were decisive. TrueSkill supports draws natively, so a verdict below a confidence
threshold can be recorded as a draw instead.

This decides between the rules by split-half reliability, using only the log (no
API calls): the comparisons are split at random into two halves, every document is
rated from each half separately, and the two sets of ratings are correlated. A rule
that throws away signal lowers that correlation; one that removes noise raises it.
Several random splits are averaged.

    python scripts/draw_experiment.py [--log data/processed/tournament_log.next.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import trueskill

MU, SIGMA = 50.0, 8.333


def rate(records, ids, draw_below: float | None):
    draws = 0
    if draw_below is not None:
        draws = sum(1 for r in records if r.get("confidence", 1) < draw_below)
    share = min(max(draws / max(1, len(records)), 1e-6), 0.9)
    env = trueskill.TrueSkill(mu=MU, sigma=SIGMA, beta=SIGMA / 2, tau=SIGMA / 100,
                              draw_probability=share if draw_below is not None else 0.0)
    r = {i: env.create_rating() for i in ids}
    for rec in records:
        a, b, w = rec["a"], rec["b"], rec["winner"]
        if draw_below is not None and rec.get("confidence", 1) < draw_below:
            r[a], r[b] = env.rate_1vs1(r[a], r[b], drawn=True)
        elif w == "A":
            r[a], r[b] = env.rate_1vs1(r[a], r[b])
        else:
            r[b], r[a] = env.rate_1vs1(r[b], r[a])
    return {i: r[i].mu for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "data/processed/tournament_log.next.jsonl"))
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("winner") in ("A", "B")]
    ids = sorted({r["a"] for r in recs} | {r["b"] for r in recs})
    conf = np.array([r.get("confidence", 1) for r in recs])
    print(f"{len(recs)} comparisons over {len(ids)} documents; "
          f"median confidence {np.median(conf):.2f}")

    rules = [("every verdict a win (current)", None)] + \
            [(f"draw below {t:.2f}", t) for t in (0.55, 0.60, 0.65, 0.70)]
    print(f"\n{'rule':32} {'draw share':>10} {'split-half r':>13}  (mean of {args.splits} splits)")
    for name, t in rules:
        rs = []
        for seed in range(args.splits):
            rng = random.Random(seed)
            half = [rng.random() < 0.5 for _ in recs]
            a = rate([r for r, h in zip(recs, half) if h], ids, t)
            b = rate([r for r, h in zip(recs, half) if not h], ids, t)
            rs.append(np.corrcoef([a[i] for i in ids], [b[i] for i in ids])[0, 1])
        share = 0.0 if t is None else float((conf < t).mean())
        print(f"{name:32} {share:10.0%} {np.mean(rs):13.3f}")


if __name__ == "__main__":
    main()
