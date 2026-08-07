"""Assemble the MPCLock corpus from the Bank of England.

Unlike the euro-area build (which stitched together four institutions), the Bank
of England is a single institution with a clean, complete speech archive reachable
via its sitemap, so the Bank's own website is the primary source throughout:

  boe_speeches -> every /speech/ page on bankofengland.co.uk (1997-present), full
                  text pulled from each speech's PDF.
  boe_mpc      -> the "BoE MPC" composite: MPC minutes, the Monetary Policy Report
                  (and before 2019 the Inflation Report) split into its sections,
                  boxes and annexes, plus the press-conference transcript and the
                  Governor's opening remarks for each round.
  bis_boe      -> the BIS central bankers' speeches archive, used to cross-check
                  the scrape year by year and to backfill the handful of speeches
                  (mostly 1996-98) that the Bank's site no longer publishes.

Everything is already English, so there is no translation step. Records are
de-duplicated by Speech.id and written as JSONL to data/processed/corpus.jsonl.
"""
from __future__ import annotations

import collections

from ..config import PROCESSED
from ..schema import Speech, save_corpus
from . import boe_mpc, boe_speeches

CORPUS_PATH = PROCESSED / "corpus.jsonl"


def load_all(use_cache: bool = True, skip: tuple = (),
             start_year: int | None = None, end_year: int | None = None,
             concurrency: int = 12) -> list[Speech]:
    speeches: list[Speech] = []
    site: list[Speech] = []

    if "speeches" not in skip:
        print("[1/3] Bank of England speeches...")
        try:
            site = boe_speeches.load(use_cache=use_cache, start_year=start_year,
                                     end_year=end_year, concurrency=concurrency)
            speeches += site
        except Exception as e:  # noqa: BLE001 - one bad source must not kill the run
            print(f"    [warn] BoE speeches failed: {type(e).__name__}: {e}")

    if "mpc" not in skip:
        print("[2/3] BoE MPC composite (minutes / reports / press conferences)...")
        try:
            speeches += boe_mpc.load(use_cache=use_cache, start_year=start_year or 1997,
                                     end_year=end_year, concurrency=max(4, concurrency // 2))
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] MPC composite failed: {type(e).__name__}: {e}")

    if "bis" not in skip:
        print("[3/3] BIS cross-check...")
        try:
            from . import bis_boe
            bis = bis_boe.load(use_cache=use_cache, start_year=start_year,
                               end_year=end_year)
            speeches += bis_boe.crosscheck(site or speeches, bis)
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] BIS cross-check failed: {type(e).__name__}: {e}")

    seen: dict[str, Speech] = {}
    for s in speeches:
        seen.setdefault(s.id, s)
    deduped = list(seen.values())
    print(f"Combined {len(speeches)} -> {len(deduped)} after de-dup")
    return deduped


def merge(existing: list[Speech], incoming: list[Speech]) -> list[Speech]:
    """Add new records to an existing corpus, keeping what is already scored.

    Existing records win on id collision so classifier verdicts, anonymised text
    and tournament ratings survive a re-ingest.
    """
    by_id = {s.id: s for s in existing}
    added = 0
    for s in incoming:
        if s.id not in by_id:
            by_id[s.id] = s
            added += 1
    print(f"Merged {added} new records into {len(existing)} existing "
          f"-> {len(by_id)} total")
    return list(by_id.values())


def build(use_cache: bool = True, out_path=CORPUS_PATH, **kw) -> list[Speech]:
    corpus = load_all(use_cache=use_cache, **kw)
    save_corpus(corpus, out_path)
    print(f"Saved {len(corpus)} records to {out_path}")
    return corpus


def _summary(corpus: list[Speech]) -> None:
    by_type = collections.Counter(s.source_type for s in corpus)
    print("\nBy source type:", dict(by_type))
    print("Date range:", min(s.date for s in corpus), "..", max(s.date for s in corpus))
    print("Distinct speakers:", len({s.speaker for s in corpus}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip", default="",
                    help="comma-separated source keys to skip (speeches, mpc, bis)")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    corpus = build(use_cache=not args.no_cache, skip=tuple(x for x in args.skip.split(",") if x),
                   concurrency=args.concurrency)
    _summary(corpus)
