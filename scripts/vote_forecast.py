"""Backtest the per-member vote model and forecast the next meeting.

    python scripts/vote_forecast.py                # backtest report + site/forecast.json
    python scripts/vote_forecast.py --no-backtest  # forecast only (daily run)

Backtest: every meeting since 2021 is predicted by a model fitted on the other
years only (leave-one-year-out), so no meeting's own outcome informs its forecast.
Feature groups are added one at a time to show what each contributes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
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

from mpclock.forecast import votes as V
from mpclock.schema import load_corpus

LADDER = [
    ("last vote only", []),
    ("+ committee", V.FEATURE_GROUPS["committee"]),
    ("+ pace", V.FEATURE_GROUPS["committee"] + V.FEATURE_GROUPS["pace"]),
    ("+ words", V.FEATURE_GROUPS["committee"] + V.FEATURE_GROUPS["pace"] + V.FEATURE_GROUPS["words"]),
    ("+ data surprises", V.FEATURE_GROUPS["committee"] + V.FEATURE_GROUPS["pace"] + V.FEATURE_GROUPS["words"]
     + V.FEATURE_GROUPS["data"]),
    ("+ energy & FX", V.FEATURE_GROUPS["committee"] + V.FEATURE_GROUPS["pace"] + V.FEATURE_GROUPS["words"]
     + V.FEATURE_GROUPS["data"] + V.FEATURE_GROUPS["markets"]),
    ("+ Fed & ECB", V.FEATURE_GROUPS["committee"] + V.FEATURE_GROUPS["pace"] + V.FEATURE_GROUPS["words"]
     + V.FEATURE_GROUPS["data"] + V.FEATURE_GROUPS["markets"] + V.FEATURE_GROUPS["global"]),
]


def cv_predict(rows, feats, l2=1.0):
    years = sorted({r["date"][:4] for r in rows})
    out = [None] * len(rows)
    for y in years:
        tr = [r for r in rows if r["date"][:4] != y]
        idx = [i for i, r in enumerate(rows) if r["date"][:4] == y]
        m = V.OrderedLogit(feats, l2).fit(tr)
        P = m.predict([rows[i] for i in idx])
        for j, i in enumerate(idx):
            out[i] = P[j]
    return np.array(out)


def evaluate(rows, P):
    """Member-level and decision-level accuracy of Δ-probabilities P."""
    y = np.array([r["y"] for r in rows]) - V.DELTAS[0]
    ll = -np.mean(np.log(P[np.arange(len(y)), y]))
    onehot = np.eye(len(V.DELTAS))[y]
    brier = np.mean(np.sum((P - onehot) ** 2, axis=1))
    vote_probs = [V.vote_distribution(P[i], r["prev_vote"], r["options"]) for i, r in enumerate(rows)]
    hit = np.mean([max(vp, key=vp.get) == r["y_vote"] for vp, r in zip(vote_probs, rows)])
    p_actual = np.array([vp.get(r["y_vote"], 0.0) for vp, r in zip(vote_probs, rows)])
    sw = np.array([r["y"] != 0 for r in rows])
    hit_sw = np.mean([max(vp, key=vp.get) == r["y_vote"] for vp, r, s in zip(vote_probs, rows, sw) if s])
    # decisions
    by_meet = {}
    for vp, r in zip(vote_probs, rows):
        by_meet.setdefault(r["date"], []).append((vp, r))
    dec_hit, dec_p, dec_rows = [], [], []
    for d, lst in sorted(by_meet.items()):
        dist = V.decision_distribution([vp for vp, _ in lst], n_sims=4000)
        actual = lst[0][1]["decision"]
        dec_hit.append(max(dist, key=dist.get) == actual)
        dec_p.append(dist.get(actual, 0.0))
        dec_rows.append({"date": d, "actual": actual, "predicted": max(dist, key=dist.get),
                         "p_actual": round(dist.get(actual, 0.0), 3),
                         "dist": {str(k): round(v, 3) for k, v in dist.items() if v >= 0.005}})
    return {"n": len(rows), "log_loss": ll, "brier": brier, "vote_hit": hit, "p_actual": p_actual.mean(),
            "n_switch": int(sw.sum()), "switch_hit": hit_sw, "meetings": len(by_meet),
            "decision_hit": float(np.mean(dec_hit)), "decision_p": float(np.mean(dec_p)),
            "decisions": dec_rows, "vote_probs": vote_probs}


def next_meeting(corpus, last_date: str) -> str:
    """'The minutes of the Committee meeting ending on 16 December will be published …'"""
    mins = sorted([s for s in corpus if s.source_type == "mp_account"], key=lambda s: s.date)
    t = mins[-1].text or ""
    m = re.search(r"meeting\s+ending\s+on\s+(\d{1,2})\s+([A-Z][a-z]+)(?:\s+(\d{4}))?\s+will\s+be\s+published", t)
    if m:
        y = int(m.group(3) or last_date[:4])
        d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)} {y}", "%d %B %Y").date()
        if d.isoformat() <= last_date:
            d = d.replace(year=y + 1)
        return d.isoformat()
    return (dt.date.fromisoformat(last_date) + dt.timedelta(days=42)).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--out", default=str(ROOT / "site/forecast.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    ctx = V.build_context(corpus)
    rows = V.panel(ctx, args.since)
    full = LADDER[-1][1]
    chosen = LADDER[-2][1]          # default model; replaced below by the best CV ladder step

    report = {}
    if not args.no_backtest:
        print(f"backtest: {len(rows)} member-votes at {len({r['date'] for r in rows})} meetings "
              f"since {args.since}, leave-one-year-out")
        print(f"  {'model':20} {'logloss':>8} {'Brier':>7} {'vote hit':>9} {'P(actual)':>9} "
              f"{'switch hit':>11} {'decision hit':>13} {'P(decision)':>11}")
        best = None
        ladder_out = []
        for name, feats in LADDER:
            ev = evaluate(rows, cv_predict(rows, feats))
            print(f"  {name:20} {ev['log_loss']:8.4f} {ev['brier']:7.4f} {ev['vote_hit']:9.1%} "
                  f"{ev['p_actual']:9.1%} {ev['switch_hit']:11.1%} {ev['decision_hit']:13.1%} "
                  f"{ev['decision_p']:11.1%}")
            ladder_out.append({"model": name, **{k: round(float(v), 4) for k, v in ev.items()
                                                  if k in ("log_loss", "brier", "vote_hit", "p_actual",
                                                           "switch_hit", "decision_hit", "decision_p")},
                               "n": ev["n"], "n_switch": ev["n_switch"], "meetings": ev["meetings"]})
            if best is None or ev["log_loss"] < best[1]["log_loss"] - 1e-4:
                best = (name, ev, feats)
        chosen = best[2]
        print(f"chosen model (lowest out-of-sample log loss): {best[0]}")
        report = {"ladder": ladder_out, "chosen": best[0], "decisions": best[1]["decisions"],
                  "since": args.since, "n": len(rows)}
        # per-meeting detail for the site
        P = cv_predict(rows, chosen)
        ev = evaluate(rows, P)
        report["member_calls"] = [
            {"date": r["date"], "member": r["member"], "actual": r["y_vote"],
             "p_actual": round(vp.get(r["y_vote"], 0.0), 3),
             "predicted": max(vp, key=vp.get)}
            for vp, r in zip(ev["vote_probs"], rows)]
        # calibration of the probability given to "this member changes their vote"
        p_move = np.array([1 - P[i][2] for i in range(len(rows))])
        moved = np.array([r["y"] != 0 for r in rows])
        cal = []
        for lo, hi in ((0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, .7), (.7, 1.01)):
            m = (p_move >= lo) & (p_move < hi)
            if m.sum():
                cal.append({"bin": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                            "predicted": round(float(p_move[m].mean()), 3),
                            "actual": round(float(moved[m].mean()), 3)})
        report["calibration_move"] = cal
        print("  calibration, P(member changes vote):")
        for c in cal:
            print(f"    {c['bin']:9} n={c['n']:4d} predicted {c['predicted']:6.1%} actual {c['actual']:6.1%}")

    model = V.OrderedLogit(chosen).fit(rows)
    coefs = dict(zip(chosen, map(float, model.b)))

    # ---- live forecast -----------------------------------------------------------
    last = ctx.meetings[-1]
    nxt = next_meeting(corpus, last.date)
    asof = min(dt.date.today().isoformat(), nxt)
    asof_end = (dt.date.fromisoformat(asof) + dt.timedelta(days=1)).isoformat()
    frows = V.features_for(ctx, last, asof_end, list(last.votes))
    P = model.predict(frows)
    opts = V.options_for(last)
    members = []
    for r, p in zip(frows, P):
        vp = V.vote_distribution(p, r["prev_vote"], opts)
        contrib = {f: round(coefs[f] * r[f], 3) for f in chosen}
        members.append({"member": r["member"], "prev_vote": r["prev_vote"],
                        "probs": {str(k): round(v, 4) for k, v in vp.items()},
                        "expected_bp": round(25 * sum(k * v for k, v in vp.items()), 1),
                        "features": {f: round(float(r[f]), 3) for f in V.FEATURE_GROUPS["committee"]
                                     + V.FEATURE_GROUPS["words"] + V.FEATURE_GROUPS["data"]
                                     + V.FEATURE_GROUPS["markets"] + V.FEATURE_GROUPS["global"]
                                     + V.FEATURE_GROUPS["pace"]},
                        "contributions": contrib, "n_docs": r["n_docs"], "doc_dates": r["_doc_dates"]})
    dist = V.decision_distribution([{int(k): v for k, v in m["probs"].items()} for m in members])
    out = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "last_meeting": {"date": last.date, "decision_bp": last.decision * 25, "bank_rate": last.rate_after,
                         "votes": {k: v * 25 for k, v in last.votes.items()}},
        "next_meeting": nxt, "as_of": asof, "options_bp": [o * 25 for o in opts],
        "members": sorted(members, key=lambda m: -m["expected_bp"]),
        "decision": {str(k * 25): round(v, 4) for k, v in dist.items()},
        "inputs": {"surprises": frows[0]["_surprises"], "surprise_detail": frows[0]["_detail"],
                   "markets": {k: round(v, 3) for k, v in frows[0]["_markets"].items()},
                   "mpr_meeting": bool(frows[0]["_mpr"]), "pace": frows[0]["pace"],
                   "moved_last": frows[0]["moved_last"]},
        "model": {"features": chosen, "coefficients": {k: round(v, 3) for k, v in coefs.items()},
                  "cutpoints": [round(float(c), 3) for c in model.cuts]},
        "backtest": report,
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nnext meeting {nxt} (as of {asof}); wrote {args.out}")
    print("  decision: " + ", ".join(f"{int(k):+d}bp {v:.0%}" for k, v in out["decision"].items() if v >= .005))
    for m in out["members"]:
        print(f"  {m['member']:20} last {m['prev_vote'] * 25:+4d}bp  " +
              "  ".join(f"{int(k) * 25:+d}:{v:5.1%}" for k, v in m["probs"].items()))


if __name__ == "__main__":
    main()
