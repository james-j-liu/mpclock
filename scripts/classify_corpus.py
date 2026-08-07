"""Classify the corpus for monetary-policy relevance, in place.

Loads data/processed/corpus.jsonl, runs the two-stage classifier (composite records
auto-pass), writes is_policy back to the same file, and reports how many survive
plus the resulting tournament cost estimate.

Classification is checkpointed every --chunk records and is resumable: records that
already carry an is_policy verdict are skipped, so a killed run continues where it
left off rather than re-paying. --mpc-only restricts to the MPC scoring pool (the
only records that ever enter the tournament), which is faster and cheaper.

Usage:
  python scripts/classify_corpus.py --mpc-only
  python scripts/classify_corpus.py --concurrency 16
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpclock.config import PROCESSED
from mpclock.process.classify import Classifier
from mpclock.roster_mpc import is_mpc
from mpclock.schema import COUNCIL_TYPES, load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"
COST_PER_COMPARISON = 0.0005  # ~gemini-2.5-flash-lite ($0.10/M in, $0.40/M out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=150, help="checkpoint every N records")
    ap.add_argument("--mpc-only", action="store_true",
                    help="only classify MPC-roster speakers (the scoring pool)")
    ap.add_argument("--since", default=None,
                    help="only classify records with date >= this ISO date")
    ap.add_argument("--path", default=str(CORPUS))
    args = ap.parse_args()

    corpus = load_corpus(args.path)
    print(f"Loaded {len(corpus)} records")

    window = corpus
    if args.since:
        window = [s for s in window if s.date >= args.since]
    if args.mpc_only:
        window = [s for s in window if is_mpc(s.speaker)]
    print(f"Scope: {len(window)} records ({'MPC-only, ' if args.mpc_only else ''}"
          f"{'since '+args.since if args.since else 'all dates'})")

    clf = Classifier(model=args.model)
    print(f"Classifier model: {clf.model}")

    # resume: only classify records without a verdict yet; checkpoint per chunk
    todo = [s for s in window if s.is_policy is None]
    print(f"To classify: {len(todo)} (skipping {len(window)-len(todo)} already done)")
    for i in range(0, len(todo), args.chunk):
        clf.classify_all(todo[i:i + args.chunk], concurrency=args.concurrency)
        save_corpus(corpus, args.path)   # checkpoint so a kill never loses progress
        print(f"  checkpoint: {min(i+args.chunk, len(todo))}/{len(todo)} classified & saved")

    save_corpus(corpus, args.path)

    policy = [s for s in window if s.is_policy]
    council = [s for s in window if s.source_type in COUNCIL_TYPES]
    by_type = collections.Counter(s.source_type for s in policy)
    n = len(policy)
    comparisons = args.appearances // 2 * n  # 15*N at 30 appearances
    print(f"\nPolicy-relevant: {n}/{len(window)} ({100*n/len(window):.1f}%)")
    print(f"  council (auto-pass): {len(council)}")
    print(f"  by source type: {dict(by_type)}")
    print(f"\nTournament at {args.appearances} appearances -> ~{comparisons:,} comparisons")
    print(f"Estimated tournament cost: ~${comparisons * COST_PER_COMPARISON:,.0f}")


if __name__ == "__main__":
    main()
