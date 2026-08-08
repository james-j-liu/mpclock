"""Ingest the MPC composite and the BIS cross-check into an existing corpus.

The speech scrape is expensive (1,500 PDFs) and its records already carry
classifier verdicts, so this adds only the newer sources on top of
data/processed/corpus.jsonl and leaves everything already there untouched.

Usage:
  python scripts/add_sources.py                 # MPC composite + BIS backfill
  python scripts/add_sources.py --only mpc      # just the MPC composite
  python scripts/add_sources.py --no-split      # one record per report PDF
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from mpclock.config import PROCESSED
from mpclock.corpus import assemble, bis_boe, boe_mpc, tsc_evidence
from mpclock.roster_mpc import canon, is_mpc
from mpclock.schema import (COUNCIL_TYPES, ST_ACCOUNT, ST_MEMBER_VIEW, ST_REPORT,
                            load_corpus, save_corpus)

CORPUS = PROCESSED / "corpus.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="mpc | tsc | bis (default: all three)")
    ap.add_argument("--split", action="store_true",
                    help="score every report section/box/annex separately "
                         "(default follows config.yaml corpus.split_report_sections)")
    ap.add_argument("--refresh-reports", action="store_true",
                    help="delete every existing Report record first, so a change in "
                         "how a round is cut (sections vs. whole + annexes) takes effect")
    ap.add_argument("--refresh-minutes", action="store_true",
                    help="delete every existing minutes / member-view record first, so a "
                         "change in how the minutes are cut takes effect")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--path", default=str(CORPUS))
    args = ap.parse_args()

    corpus = load_corpus(args.path)
    print(f"Existing corpus: {len(corpus)} records")
    incoming = []

    # A refresh rebuilds records from source, so any scores they carry would be
    # lost. Ratings are keyed on the record id (date, speaker, title, url), which
    # a re-cut of the same document keeps, so they are stashed and put back.
    scores = {s.id: (s.mu, s.mu_adj, s.sigma, s.n_comparisons,
                     s.direct_score, s.direct_adj, s.is_policy) for s in corpus}

    if args.refresh_reports:
        before = len(corpus)
        corpus = [s for s in corpus if s.source_type != ST_REPORT]
        print(f"Dropped {before - len(corpus)} existing Report records")

    if args.refresh_minutes:
        before = len(corpus)
        corpus = [s for s in corpus
                  if s.source_type not in (ST_ACCOUNT, ST_MEMBER_VIEW)]
        print(f"Dropped {before - len(corpus)} existing minutes / member-view records")

    if args.only in ("", "mpc"):
        print("\nBoE MPC composite (minutes / reports / press conferences)...")
        incoming += boe_mpc.load(use_cache=not args.no_cache, start_year=1997,
                                 concurrency=args.concurrency,
                                 split_sections_=args.split or None)

    if args.only in ("", "tsc"):
        print("\nTreasury Committee evidence (per MPC member)...")
        incoming += tsc_evidence.load(use_cache=not args.no_cache)

    if args.only in ("", "bis"):
        print("\nBIS cross-check...")
        site = [s for s in corpus if s.institution == "Bank of England"]
        bis = bis_boe.load(use_cache=not args.no_cache)
        incoming += bis_boe.crosscheck(site, bis)

    merged = assemble.merge(corpus, incoming)
    restored = 0
    for s in merged:
        if s.mu is None and s.id in scores:
            (s.mu, s.mu_adj, s.sigma, s.n_comparisons,
             s.direct_score, s.direct_adj, s.is_policy) = scores[s.id]
            restored += 1
    if restored:
        print(f"Restored scores for {restored} refreshed records")
    save_corpus(merged, args.path)

    by_type = collections.Counter(s.source_type for s in merged)
    pool = [s for s in merged if is_mpc(s.speaker)]
    council = [s for s in merged if s.source_type in COUNCIL_TYPES]
    print(f"\nSaved {len(merged)} records to {args.path}")
    print("By source type:", dict(by_type))
    print(f"MPC scoring pool (roster + composite): {len(pool)}")
    print(f"  of which MPC composite documents: {len(council)}")
    print(f"Distinct MPC speakers: {len({canon(s.speaker) for s in pool})}")
    print(f"Unclassified records in pool: {sum(1 for s in pool if s.is_policy is None)}")


if __name__ == "__main__":
    main()
