"""Re-code members' documents with another decision model, for the model comparison
(scripts/test_guidance_models.py). Every arm is blind: no date, names replaced by
[OFFICIAL], years by [YEAR], so a model trained on later data cannot recall outcomes.

    python scripts/code_guidance_ab.py --arm jev113_blind
Arms: jev113_blind, jev114_blind, decider12_blind. Needs OPENROUTER_API_KEY.
Output: data/processed/guidance_ab/<arm>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpclock.forecast.guidance import code_documents, member_documents
from mpclock.process.anonymize import Anonymizer
from mpclock.process.roster import build_roster
from mpclock.schema import load_corpus

ARMS = {  # pinned versions, not "latest" aliases, so the comparison can be reproduced
    "jev113_blind": "typesafe/jev-1.13",
    "jev114_blind": "typesafe/jev-1.14",
    "decider12_blind": "perplexity/pplx-decider-v1.2",
}
OUT = ROOT / "data/processed/guidance_ab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--since", default="2014-06-01")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    rate = [(d, float(r)) for d, r in
            json.loads((ROOT / "site/macro.json").read_text())["series"]["bank_rate"]["data"]]
    docs = member_documents(corpus, args.since)
    anon = Anonymizer(build_roster([s.speaker for s in corpus]))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{args.arm}: {ARMS[args.arm]}, {len(docs)} member documents since {args.since}")
    codes = code_documents(docs, rate, max_workers=args.workers, limit=args.limit, model=ARMS[args.arm],
                           blind=True, path=OUT / f"{args.arm}.json", anonymizer=anon)
    print(f"{args.arm}: {sum(1 for d in docs if d.id in codes)} of {len(docs)} coded")


if __name__ == "__main__":
    main()
