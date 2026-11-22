"""Code sitting members' documents for the vote they signal at the next meeting.

    python scripts/code_guidance.py                 # everything since mid-2014 not yet coded
    python scripts/code_guidance.py --limit 20      # a trial batch
Needs OPENROUTER_API_KEY. Results: data/processed/guidance_codes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpclock.forecast.guidance import code_documents, member_documents
from mpclock.schema import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2014-06-01")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    rate = [(d, float(r)) for d, r in
            json.loads((ROOT / "site/macro.json").read_text())["series"]["bank_rate"]["data"]]
    docs = member_documents(corpus, args.since)
    print(f"{len(docs)} member documents since {args.since}")
    code_documents(docs, rate, max_workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
