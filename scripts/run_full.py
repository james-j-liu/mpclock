"""Full run: classified corpus -> anonymize -> pairwise tournament + direct
scoring -> data.json.

Loads data/processed/corpus.jsonl, keeps policy-relevant records within the date
window, anonymizes, then runs BOTH scoring methods on the same pool:
  - the pairwise TrueSkill tournament (mu / mu_adj), and
  - direct 0-100 LLM scoring (direct_score / direct_adj), FedLock's cheaper baseline.
Both are era-adjusted and written to the site's data.json so the site can offer
them as alternative views.

Usage:
  python scripts/run_full.py --since 2010-05-28 --until 2026-05-28
  python scripts/run_full.py --since 2010-05-28 --appearances 30 --concurrency 20
  python scripts/run_full.py --since 2010-05-28 --dry-run     # mocks, no API
  python scripts/run_full.py --since 2010-05-28 --no-direct   # pairwise only
  python scripts/run_full.py --since 2010-05-28 --no-pairwise # direct only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# accented GC names (Vujčić, Šimkus, ...) crash the default Windows cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from mpclock.config import PROCESSED
from mpclock.macro.uk_macro import MacroContext
from mpclock.output.build_data import era_adjust, write_data_json
from mpclock.process.anonymize import Anonymizer
from mpclock.process.roster import build_roster
from mpclock.roster_mpc import is_mpc
from mpclock.schema import load_corpus
from mpclock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"


class MockDirectScorer:
    """Deterministic 0-100 stand-in for --dry-run (no API).

    Reuses the same crude hawk/dove lexicon as MockJudge and squashes the net
    count into 0-100 so the wiring + era adjustment can be validated for free.
    """
    model = "mock"
    HAWK = ["inflation", "price stability", "tighten", "restrictive", "vigilant",
            "overheating", "hike", "raise rates", "upside risk", "anchor"]
    DOVE = ["growth", "unemployment", "accommodation", "stimulus", "downside",
            "support", "cut", "ease", "recovery", "slack", "purchase"]

    def score(self, text: str, macro: str) -> float:
        tl = text.lower()
        net = sum(tl.count(w) for w in self.HAWK) - sum(tl.count(w) for w in self.DOVE)
        return max(0.0, min(100.0, 50.0 + 2.5 * net))

    def score_all(self, speeches, macro, concurrency: int = 8):
        for s in speeches:
            s.direct_score = round(self.score(s.text_anon or s.text, macro.string(s.date)), 2)
        return speeches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO date lower bound (inclusive)")
    ap.add_argument("--until", default=None, help="ISO date upper bound (inclusive)")
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-pairwise", action="store_true", help="skip the tournament")
    ap.add_argument("--no-direct", action="store_true", help="skip direct 0-100 scoring")
    ap.add_argument("--direct-concurrency", type=int, default=8)
    ap.add_argument("--include-non-mpc", action="store_true",
                    help="keep non-GC speakers (NCB deputies etc.); default excludes them")
    ap.add_argument("--resume", action="store_true",
                    help="replay the tournament log and continue (don't re-pay completed comparisons)")
    ap.add_argument("--path", default=str(CORPUS))
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--log", default="data/processed/tournament_log.jsonl",
                    help="comparison log (point a --dry-run at a scratch file so mock "
                         "outcomes never land in the real one)")
    args = ap.parse_args()

    corpus = load_corpus(args.path)
    pool = [s for s in corpus if s.is_policy]
    if args.since:
        pool = [s for s in pool if s.date >= args.since]
    if args.until:
        pool = [s for s in pool if s.date <= args.until]

    # Restrict the scoring pool to actual MPC members (Governor, Deputy Governors,
    # Chief Economist, external members) plus the "BoE MPC" composite. The BoE also
    # publishes many speeches by prudential-regulation (PRA), financial-stability
    # (FPC) and markets officials who never vote on Bank Rate; scoring them would
    # distort the TrueSkill ratings and the era means.
    if not args.include_non_mpc:
        before = len(pool)
        pool = [s for s in pool if is_mpc(s.speaker)]
        print(f"MPC filter: kept {len(pool)} / {before} records "
              f"({before - len(pool)} non-MPC records removed)")

    print(f"Tournament pool: {len(pool)} policy-relevant records"
          + (f" since {args.since}" if args.since else ""))
    if not pool:
        sys.exit("No policy-relevant records in window. Run classify_corpus.py first.")

    # anonymize using a roster drawn from the FULL corpus (every recognisable name)
    roster = build_roster([s.speaker for s in corpus])
    anon = Anonymizer(roster)
    for s in pool:
        if not s.text_anon:
            s.text_anon = anon(s.text)

    macro = MacroContext()

    # --- Pairwise TrueSkill tournament ---
    if not args.no_pairwise:
        if args.dry_run:
            from run_poc import MockJudge
            judge = MockJudge()
            print("DRY RUN: MockJudge (no API calls)")
        else:
            from mpclock.judge.openrouter import Judge
            judge = Judge(model=args.model)
            print(f"Judge model: {judge.model}")
        run_tournament(pool, judge, appearances_per_speech=args.appearances,
                       macro=macro, seed=args.seed, concurrency=args.concurrency,
                       resume=args.resume, full_run=True, log_path=args.log)
        # persist pairwise results immediately so a later crash can't lose them
        era_adjust(pool)
        write_data_json(pool, args.out)
        print(f"Pairwise done, checkpointed to {args.out}")

    # --- Direct 0-100 scoring (FedLock's cheaper baseline) ---
    if not args.no_direct:
        if args.dry_run:
            scorer = MockDirectScorer()
            print("DRY RUN: MockDirectScorer (no API calls)")
        else:
            from mpclock.judge.direct import DirectScorer
            scorer = DirectScorer(model=args.model)
            print(f"Direct scorer model: {scorer.model}")
        scorer.score_all(pool, macro, concurrency=args.direct_concurrency)

    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"Wrote {args.out}: {meta['n_speeches']} speeches, {meta['n_speakers']} speakers, "
          f"pairwise={meta['n_pairwise']} direct={meta['n_direct']} "
          f"({meta['date_min']}..{meta['date_max']})")

    if not args.no_pairwise:
        ranked = sorted([s for s in pool if s.mu_adj is not None], key=lambda s: -s.mu_adj)
        print("\n[PAIRWISE] Most hawkish (era-adjusted):")
        for s in ranked[:10]:
            print(f"  {s.mu_adj:5.1f}  {s.date}  {s.speaker[:24]:24}  {s.title[:46]}")
        print("[PAIRWISE] Most dovish:")
        for s in ranked[-10:]:
            print(f"  {s.mu_adj:5.1f}  {s.date}  {s.speaker[:24]:24}  {s.title[:46]}")

    if not args.no_direct:
        dranked = sorted([s for s in pool if s.direct_adj is not None], key=lambda s: -s.direct_adj)
        print("\n[DIRECT] Most hawkish (era-adjusted):")
        for s in dranked[:10]:
            print(f"  {s.direct_adj:5.1f} (raw {s.direct_score:5.1f})  {s.date}  "
                  f"{s.speaker[:24]:24}  {s.title[:42]}")
        print("[DIRECT] Most dovish:")
        for s in dranked[-10:]:
            print(f"  {s.direct_adj:5.1f} (raw {s.direct_score:5.1f})  {s.date}  "
                  f"{s.speaker[:24]:24}  {s.title[:42]}")


if __name__ == "__main__":
    main()
