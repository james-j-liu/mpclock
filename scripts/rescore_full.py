"""Full re-score of the pool with the configured scoring models (e.g. a judge switch).

Resumable in chunks, so each call fits a time limit and a killed run loses nothing:
  1. pairwise  -> a fresh log (tournament_log.next.jsonl) until ~appearances*N/2
  2. direct    -> scores kept in direct.next.json until every pool document is done
  3. --finalize swaps the new log in (the old one kept as tournament_log.prev.jsonl),
     writes the new ratings and direct scores into the corpus, rebuilds data.json.
Nothing touches the live log or the corpus scores until --finalize, so the site and
the daily job carry on with the old ratings while this runs.

    python scripts/rescore_full.py --chunk 4000      # repeat until it says "done"
    python scripts/rescore_full.py --finalize
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mpclock.config import PROCESSED, cfg
from mpclock.judge.factory import make_direct_scorer, make_pairwise_judge
from mpclock.macro.uk_macro import MacroContext
from mpclock.output.build_data import era_adjust, write_data_json
from mpclock.process.anonymize import Anonymizer
from mpclock.process.roster import build_roster
from mpclock.roster_mpc import is_mpc
from mpclock.tenure import for_corpus
from mpclock.schema import load_corpus, save_corpus
from mpclock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"
LOG, NEXT_LOG, PREV_LOG = (PROCESSED / f for f in
                           ("tournament_log.jsonl", "tournament_log.next.jsonl",
                            "tournament_log.prev.jsonl"))
NEXT_DIRECT = PROCESSED / "direct.next.json"
SINCE = "1997-01-01"


def _n_logged() -> int:
    return sum(1 for _ in NEXT_LOG.open(encoding="utf-8")) if NEXT_LOG.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int,
                    default=cfg()["tournament"]["target_appearances_per_speech"])
    ap.add_argument("--chunk", type=int, default=4000, help="max new API calls this run")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(CORPUS)
    today = datetime.date.today().isoformat()
    tenure = for_corpus(corpus)      # members only while they sat on the Committee
    pool = [s for s in corpus if s.is_policy and is_mpc(s.speaker) and SINCE <= s.date <= today
            and tenure.active(s.speaker, s.date)]
    target = args.appearances * len(pool) // 2
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    macro = MacroContext()
    direct = json.loads(NEXT_DIRECT.read_text()) if NEXT_DIRECT.exists() else {}

    if not args.finalize:
        have = _n_logged()
        print(f"pool {len(pool)} | pairwise {have}/{target} | direct {len(direct)}/{len(pool)}",
              flush=True)
        if have < target:
            judge = make_pairwise_judge()
            run_tournament(pool, judge, appearances_per_speech=args.appearances, macro=macro,
                           log_path=NEXT_LOG, resume=True, anonymizer=anon,
                           concurrency=args.concurrency, min_total=target, max_new=args.chunk)
            print(f"pairwise now {_n_logged()}/{target}  "
                  f"(${getattr(judge, 'cost', 0):.3f} this chunk)", flush=True)
            return
        todo = [s for s in pool if s.id not in direct][:args.chunk]
        if todo:
            scorer = make_direct_scorer()
            for s in todo:
                s.direct_score = None
            scorer.score_all(todo, macro, concurrency=args.concurrency, anonymizer=anon)
            direct.update({s.id: s.direct_score for s in todo if s.direct_score is not None})
            NEXT_DIRECT.write_text(json.dumps(direct))
            print(f"direct now {len(direct)}/{len(pool)}  "
                  f"(${getattr(scorer, 'cost', 0):.3f} this chunk)", flush=True)
            if len(direct) < len(pool):
                return
        print("done - run with --finalize")
        return

    # ---- finalize: swap in the new log and scores ----
    missing = [s.id for s in pool if s.id not in direct]
    if _n_logged() < target * 0.95 or len(missing) > len(pool) * 0.01:
        sys.exit(f"not complete: pairwise {_n_logged()}/{target}, direct missing {len(missing)}")
    if LOG.exists():
        shutil.copyfile(LOG, PREV_LOG)
    shutil.move(NEXT_LOG, LOG)
    # replay the new log for the final ratings (max_new=0: no API calls)
    replay_only = type("Replay", (), {"model": "replay"})()
    run_tournament(pool, replay_only, appearances_per_speech=args.appearances, macro=macro,
                   log_path=LOG, resume=True, anonymizer=anon, max_new=0)
    for s in pool:
        s.direct_score = direct.get(s.id)
    save_corpus(corpus, CORPUS)
    NEXT_DIRECT.unlink()
    era_adjust(pool)
    meta = write_data_json(pool, ROOT / "site" / "data.json")
    print(f"finalized: {meta['n_speeches']} documents, pairwise={meta['n_pairwise']} "
          f"direct={meta['n_direct']}")


if __name__ == "__main__":
    main()
