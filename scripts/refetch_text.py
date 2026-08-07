"""Re-scrape body text for corpus records that no longer carry it.

The corpus stores full text only for records that are, or could still become,
scored documents (see schema.keeps_text) so the file stays inside GitHub's
per-file limit. If the classifier changes and previously rejected speeches need a
second look, their text has to come back from source_url first — that is this.

Usage:
  python scripts/refetch_text.py --scope mpc      # MPC-roster records only
  python scripts/refetch_text.py --scope all
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mpclock.config import PROCESSED
from mpclock.corpus.boe_speeches import extract_speech
from mpclock.roster_mpc import is_mpc
from mpclock.schema import ST_INTERVIEW, ST_SPEECH, load_corpus, save_corpus

CORPUS = PROCESSED / "corpus.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="mpc", choices=("mpc", "all"))
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--path", default=str(CORPUS))
    args = ap.parse_args()

    corpus = load_corpus(args.path)
    todo = [s for s in corpus
            if not s.text and s.source_url
            and s.source_type in (ST_SPEECH, ST_INTERVIEW)
            and (args.scope == "all" or is_mpc(s.speaker))]
    print(f"{len(corpus)} records | {len(todo)} to re-fetch ({args.scope})")

    done = ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(extract_speech, s.source_url): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            try:
                fetched = fut.result()
            except Exception:  # noqa: BLE001
                fetched = None
            if fetched and fetched.text:
                s.text = fetched.text
                ok += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} fetched ({ok} recovered)")
    print(f"Recovered text for {ok}/{len(todo)} records")

    # keep what we just fetched: without this the save would strip it straight back
    save_corpus(corpus, args.path, keep_all_text=True)
    print(f"Saved {args.path}")


if __name__ == "__main__":
    main()
