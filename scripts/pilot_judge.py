"""Pilot a scoring-model switch: re-judge a random sample of logged comparisons with
another decision model and compare with the logged verdicts (and the cost).

    PAIRWISE_MODEL=nace-ai/drex-max-finance python scripts/pilot_judge.py --n 200
Writes data/processed/judge_pilot.json. Needs OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mpclock.config import PROCESSED
from mpclock.judge.factory import make_pairwise_judge
from mpclock.macro.uk_macro import MacroContext
from mpclock.process.anonymize import Anonymizer
from mpclock.process.roster import build_roster
from mpclock.schema import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    corpus = load_corpus(PROCESSED / "corpus.jsonl")
    by_id = {s.id: s for s in corpus}
    log = [json.loads(l) for l in (PROCESSED / "tournament_log.jsonl").open(encoding="utf-8") if l.strip()]
    log = [r for r in log if r["a"] in by_id and r["b"] in by_id and r.get("winner") in ("A", "B") and "p_a" in r]
    sample = random.Random(args.seed).sample(log, min(args.n, len(log)))
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    macro = MacroContext()
    judge = make_pairwise_judge()
    print(f"pilot: {len(sample)} logged comparisons re-judged by {judge.model}")

    def one(r):
        a, b = by_id[r["a"]], by_id[r["b"]]
        res = judge.compare(anon.text_of(a), macro.string(a.date), anon.text_of(b), macro.string(b.date),
                            a_type=a.source_type, b_type=b.source_type)
        return r, res

    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fu in as_completed([ex.submit(one, r) for r in sample]):
            try:
                r, res = fu.result()
            except Exception as e:  # noqa: BLE001
                print("  [skip]", str(e)[:120])
                continue
            out.append({"a": r["a"], "b": r["b"], "old_winner": r["winner"], "old_p_a": r["p_a"],
                        "old_conf": r.get("confidence"), "new_winner": res.get("winner"),
                        "new_p_a": res.get("p_a"), "new_conf": res.get("confidence")})
    agree = np.mean([o["old_winner"] == o["new_winner"] for o in out])
    confident = [o for o in out if (o["old_conf"] or 0) >= 0.6 and (o["new_conf"] or 0) >= 0.6]
    agree_conf = np.mean([o["old_winner"] == o["new_winner"] for o in confident]) if confident else None
    corr = float(np.corrcoef([o["old_p_a"] for o in out], [o["new_p_a"] for o in out])[0, 1])
    cost = float(getattr(judge, "cost", 0.0))
    summ = {"model": judge.model, "n": len(out), "agree": round(float(agree), 3),
            "agree_both_confident": None if agree_conf is None else round(float(agree_conf), 3),
            "n_both_confident": len(confident), "corr_p_a": round(corr, 3),
            "new_draw_share": round(float(np.mean([(o["new_conf"] or 0) < 0.6 for o in out])), 3),
            "old_draw_share": round(float(np.mean([(o["old_conf"] or 0) < 0.6 for o in out])), 3),
            "cost": round(cost, 4), "cost_per_comparison": round(cost / max(len(out), 1), 6),
            "logged_comparisons": len(log)}
    print(json.dumps(summ, indent=1))
    (PROCESSED / "judge_pilot.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
