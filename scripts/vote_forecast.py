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

_G = V.FEATURE_GROUPS
LADDER = [
    ("last vote only", []),
    ("+ committee", _G["committee"]),
    ("+ voting record", _G["committee"] + _G["record"]),
    ("+ pace", _G["committee"] + _G["record"] + _G["pace"]),
    ("+ own guidance", _G["committee"] + _G["record"] + _G["pace"] + _G["words"]),
    ("+ data & energy", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"]),
    ("+ hedged guidance", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"] + _G["hedge"]),
    ("+ votes with Governor", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"]
     + _G["hedge"] + _G["governor"]),
    ("+ Fed & ECB", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"] + _G["hedge"]
     + _G["governor"] + _G["global"]),
    ("+ sterling", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"] + _G["hedge"]
     + _G["governor"] + _G["global"] + _G["fx"]),
    ("+ speech tone", _G["committee"] + _G["record"] + _G["pace"] + _G["words"] + _G["data"] + _G["hedge"]
     + _G["governor"] + _G["global"] + _G["fx"] + _G["tone"]),
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


def upcoming_dates(after: str) -> list[str]:
    """Meeting end dates (the day before each announcement) after `after`, from the
    Bank's published schedule; falls back to ~6-weekly spacing."""
    import requests
    try:
        html = requests.get("https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        import html as _html
        txt = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", html)).replace("\xa0", " "))
        out = []
        for blk in re.finditer(r"(20\d\d) confirmed dates(.*?)(?=20\d\d confirmed dates|$)", txt):
            y = int(blk.group(1))
            for m in re.finditer(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday) (\d{1,2}) ([A-Z][a-z]+)", blk.group(2)):
                d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)} {y}", "%d %B %Y").date() - dt.timedelta(days=1)
                if d.isoformat() > after:
                    out.append(d.isoformat())
        if out:
            return sorted(set(out))
    except Exception:  # noqa: BLE001
        pass
    return [(dt.date.fromisoformat(after) + dt.timedelta(days=42 * k)).isoformat() for k in (1, 2)]


def second_meeting(ctx, model, chosen, last, members_out, dec_dist, nxt):
    """Each member's vote at the meeting after next, conditional on the two most likely
    outcomes of the next one. Members' December votes are drawn given that December
    outcome; their latest words are carried forward; no new data or market news is
    assumed (it cannot be known yet). February is a Report meeting, so the pace inputs
    differ from December's."""
    dates = [d for d in upcoming_dates(nxt) if d > nxt]
    if not dates:
        return None
    feb = dates[0]
    rng = np.random.default_rng(1)
    names = [m["member"] for m in members_out]
    opts = sorted(int(k) for k in members_out[0]["probs"])
    n = 40000
    draws = np.vstack([rng.choice(opts, size=n, p=np.array([m["probs"][str(o)] for o in opts]) /
                                  sum(m["probs"][str(o)] for o in opts)) for m in members_out])
    dec = np.median(draws, axis=0)
    top = sorted(dec_dist.items(), key=lambda kv: -kv[1])[:2]
    scen = []
    for s, p_s in top:
        sel = draws[:, dec == s]
        if sel.shape[1] < 50:
            continue
        mean_vote = float(sel.mean())
        mem = {}
        for i, nm in enumerate(names):
            vals, cnt = np.unique(sel[i], return_counts=True)
            mix = {}
            for v, c in zip(vals, cnt):
                w = c / sel.shape[1]
                if w < 0.02:
                    continue
                votes = {k: int(round(np.median(sel[j]))) for j, k in enumerate(names)}
                votes[nm] = int(v)
                syn = V.Meeting(nxt, votes, int(s), last.rate_after + 0.25 * s)
                ctx.meetings.append(syn)
                try:
                    r = V.features_for(ctx, syn, feb, [nm], words_since=last.date, freeze_data=True)[0]
                finally:
                    ctx.meetings.pop()
                r["pull"] = mean_vote - int(v)
                pr = model.predict([r])[0]
                fo = list(range(min(opts) - (1 if s <= 0 else 0), max(opts) + 1))
                for k2, q in V.vote_distribution(pr, int(v), fo).items():
                    mix[k2] = mix.get(k2, 0.0) + w * q
            tot = sum(mix.values()) or 1.0
            mem[nm] = {str(k): round(q / tot, 4) for k, q in sorted(mix.items())}
        fdist = V.decision_distribution([{int(k): v for k, v in mem[nm].items()} for nm in names])
        scen.append({"december_bp": int(s) * 25, "p_december": round(float(p_s), 4),
                     "members": mem, "decision": {str(int(k) * 25): round(v, 4) for k, v in fdist.items()}})
    return {"date": feb, "scenarios": scen,
            "note": "Conditional on each December outcome; members' latest words carried forward; "
                    "no new data or market moves assumed; February is an MPR meeting."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="first refresh ONS/market/MPR inputs and code new member documents (daily run)")
    ap.add_argument("--since", default="2015-06-01")
    ap.add_argument("--recent", default=None, help="also report meetings from this date (default: last 5 years)")
    ap.add_argument("--out", default=str(ROOT / "site/forecast.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    if args.refresh:
        from mpclock.forecast import inputs
        try:
            inputs.refresh(corpus)
        except Exception as e:  # noqa: BLE001 - a data outage must not stop the forecast
            print(f"[warn] input refresh failed: {e}")
        try:
            from mpclock.forecast.guidance import code_documents, member_documents
            rate = [(d, float(r)) for d, r in
                    json.loads((ROOT / "site/macro.json").read_text())["series"]["bank_rate"]["data"]]
            code_documents(member_documents(corpus), rate)
        except Exception as e:  # noqa: BLE001 - no key / API down: forecast on existing codes
            print(f"[warn] guidance coding skipped: {e}")
    # inputs and recency half-life come from scripts/select_vote_model.py, which keeps the
    # core structure only if it beats "repeat your last vote" significantly and each later
    # add-on only if it clears the same test on its own
    sel_path = ROOT / "data/processed/vote_model.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else None
    if selection:
        V.HALF_LIFE = selection.get("half_life", V.HALF_LIFE)
    ctx = V.build_context(corpus)
    rows = V.panel(ctx, args.since)
    chosen = selection["features"] if selection else LADDER[6][1]
    ladder = [("last vote only", []), ("selected model", chosen)]

    report = {}
    if not args.no_backtest:
        print(f"backtest: {len(rows)} member-votes at {len({r['date'] for r in rows})} meetings "
              f"since {args.since}, leave-one-year-out")
        print(f"  {'model':20} {'logloss':>8} {'Brier':>7} {'vote hit':>9} {'P(actual)':>9} "
              f"{'switch hit':>11} {'decision hit':>13} {'P(decision)':>11}")
        best = None
        ladder_out = []
        for name, feats in ladder:
            ev = evaluate(rows, cv_predict(rows, feats))
            print(f"  {name:20} {ev['log_loss']:8.4f} {ev['brier']:7.4f} {ev['vote_hit']:9.1%} "
                  f"{ev['p_actual']:9.1%} {ev['switch_hit']:11.1%} {ev['decision_hit']:13.1%} "
                  f"{ev['decision_p']:11.1%}")
            ladder_out.append({"model": name, **{k: round(float(v), 4) for k, v in ev.items()
                                                  if k in ("log_loss", "brier", "vote_hit", "p_actual",
                                                           "switch_hit", "decision_hit", "decision_p")},
                               "n": ev["n"], "n_switch": ev["n_switch"], "meetings": ev["meetings"]})
            best = (name, ev, feats)
        print(f"selected inputs: {chosen}; half-life {V.HALF_LIFE} days")
        report = {"ladder": ladder_out, "chosen": "selected model", "decisions": best[1]["decisions"],
                  "since": args.since, "n": len(rows), "selection": selection}
        # per-meeting detail for the site
        P = cv_predict(rows, chosen)
        ev = evaluate(rows, P)
        recent = args.recent or (dt.date.today() - dt.timedelta(days=5 * 365 + 1)).isoformat()
        idx = [i for i, r in enumerate(rows) if r["date"] >= recent]
        evr = evaluate([rows[i] for i in idx], P[idx])
        report["recent"] = {"since": recent, **{k: round(float(evr[k]), 4) for k in
                            ("log_loss", "brier", "vote_hit", "p_actual", "switch_hit", "decision_hit", "decision_p")},
                            "meetings": evr["meetings"], "n": evr["n"], "n_switch": evr["n_switch"]}
        print(f"  last five years (since {recent}): {evr['n']} votes, member votes called {evr['vote_hit']:.1%}, "
              f"vote changes called {evr['switch_hit']:.1%} of {evr['n_switch']}, decisions called "
              f"{evr['decision_hit']:.1%} of {evr['meetings']}, avg P(actual decision) {evr['decision_p']:.1%}")
        # the same window under the naive "everyone repeats their last vote" model
        evn = evaluate([rows[i] for i in idx], cv_predict(rows, [])[idx])
        report["recent"]["naive_decision_hit"] = round(float(evn["decision_hit"]), 4)
        report["recent"]["naive_vote_hit"] = round(float(evn["vote_hit"]), 4)
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
                        "features": {f: round(float(r[f]), 3) for g in V.FEATURE_GROUPS.values() for f in g},
                        "surprise_data": round(r["surprise_data"], 3), "energy": round(r["energy"], 3),
                        "contributions": contrib, "n_docs": r["n_docs"], "doc_dates": r["_doc_dates"],
                        "docs": [{k: d.get(k) for k in ("date", "type", "signal", "dir")} for d in r["_docs"]],
                        "signal": r["_signal"],
                        "record": r["_hist"]})
    dist = V.decision_distribution([{int(k): v for k, v in m["probs"].items()} for m in members])
    second = second_meeting(ctx, model, chosen, last, members, dist, nxt)
    out = {
        "second_meeting": second,
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
    from mpclock.forecast.data_vs_mpr import build as build_dvm
    (ROOT / "site/data_vs_mpr.json").write_text(json.dumps(build_dvm(ctx.inputs)), encoding="utf-8")
    reg = ROOT / "data/processed/press_interviews.json"      # the Press interviews tab reads this
    (ROOT / "site/press_interviews.json").write_text(reg.read_text(encoding="utf-8") if reg.exists() else "[]",
                                                     encoding="utf-8")
    print(f"\nnext meeting {nxt} (as of {asof}); wrote {args.out}")
    print("  decision: " + ", ".join(f"{int(k):+d}bp {v:.0%}" for k, v in out["decision"].items() if v >= .005))
    for m in out["members"]:
        print(f"  {m['member']:20} last {m['prev_vote'] * 25:+4d}bp  " +
              "  ".join(f"{int(k) * 25:+d}:{v:5.1%}" for k, v in m["probs"].items()))


if __name__ == "__main__":
    main()
