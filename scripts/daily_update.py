"""Daily incremental update — ingest new BoE speeches and score only the new ones.

Re-fetches the Bank of England sitemap (no cache), finds speech URLs not already
in the corpus, scrapes + classifies them, then scores ONLY the new ones:
  - pairwise: resume the TrueSkill tournament; the new high-uncertainty speeches
    draw the comparisons while existing ratings are replayed from the log,
  - direct: score only speeches that don't have a direct score yet.
Finally rebuilds site/data.json and site/macro.json.

State (data/processed/corpus.jsonl with classifications + scores, and
tournament_log.jsonl) is the persistent memory between runs, so in CI it should
be committed back to the repo after each run. Cost is a few cents/day.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mpclock.config import PROCESSED
from mpclock.corpus import boe_mpc, boe_sitemap, boe_speeches
from mpclock.macro.uk_macro import MacroContext
from mpclock.output.build_data import era_adjust, write_data_json
from mpclock.process.anonymize import Anonymizer
from mpclock.process.classify import Classifier
from mpclock.process.roster import build_roster
from mpclock.roster_mpc import is_mpc
from mpclock.schema import load_corpus, save_corpus
from mpclock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"
SINCE = "1997-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearances", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="use mock scorers (no API spend)")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    existing = load_corpus(CORPUS) if CORPUS.exists() else []
    have_urls = {s.source_url for s in existing}

    # 1) re-fetch the sitemap fresh and scrape only speech URLs we don't yet have,
    #    then pick up any MPC minutes / report round published since the last run
    new = []
    try:
        urls = boe_sitemap.speech_urls(use_cache=False)
        new_urls = [u for u in urls if u not in have_urls]
        new = boe_speeches.load(new_urls, verbose=False) if new_urls else []
        print(f"sitemap: {len(urls)} speech URLs | {len(new_urls)} new | {len(new)} scraped")

        have_issues = {s.title.split(" — ")[0] for s in existing}
        new_mpc = boe_mpc.load_new(have_urls, have_issues, use_cache=False,
                                   start_year=datetime.date.today().year - 1)
        have_ids = {s.id for s in existing}
        new += [s for s in new_mpc if s.id not in have_ids]
        print(f"MPC composite: {len(new_mpc)} documents from new rounds")
    except Exception as e:  # noqa: BLE001 - a blocked scrape must not stop the deploy
        print(f"[warn] ingest failed: {type(e).__name__}: {e}; scoring what we have")

    def make_pool(c):
        return [s for s in c if s.is_policy and is_mpc(s.speaker) and SINCE <= s.date <= today]

    # 2-4) ingest + score the new speeches. Wrapped so an API failure (e.g. OpenRouter
    #      out of credits -> 402, or a network blip) does NOT fail the whole job/deploy:
    #      we log it and fall back to redeploying the existing scored data.
    corpus, scored_ok = existing, True
    try:
        if new:
            Classifier().classify_all(new)          # is_policy (composite types auto-pass)
            corpus = existing + new
            save_corpus(corpus, CORPUS)

        pool = make_pool(corpus)
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        for s in pool:
            if not s.text_anon:
                s.text_anon = anon(s.text)
        macro = MacroContext()
        new_urlset = {s.source_url for s in new}
        n_new_pool = sum(1 for s in pool if s.source_url in new_urlset)
        print(f"pool {len(pool)} MPC policy records | {n_new_pool} new to score")

        if args.dry_run:
            from run_full import MockDirectScorer
            from run_poc import MockJudge
            judge, scorer = MockJudge(), MockDirectScorer()
        else:
            from mpclock.judge.direct import DirectScorer
            from mpclock.judge.openrouter import Judge
            judge, scorer = Judge(), DirectScorer()

        if n_new_pool or any(s.mu is None for s in pool):
            run_tournament(pool, judge, appearances_per_speech=args.appearances,
                           macro=macro, resume=True)
        to_direct = [s for s in pool if s.direct_score is None]
        if to_direct:
            scorer.score_all(to_direct, macro, concurrency=6)
        save_corpus(corpus, CORPUS)   # persist classifications, ratings, direct scores
    except Exception as e:  # noqa: BLE001
        scored_ok = False
        print(f"[warn] update/scoring failed: {type(e).__name__}: {e}")
        print("[warn] redeploying existing scored data; new speeches retried next run")
        corpus = load_corpus(CORPUS)

    # 5) always rebuild outputs so the site redeploys (even on a degraded run)
    pool = make_pool(corpus)
    era_adjust(pool)
    meta = write_data_json(pool, args.out)
    print(f"wrote {args.out}: {meta['n_speeches']} speeches (scored_ok={scored_ok})")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_macro.py")], check=False)
    print("daily update complete")


if __name__ == "__main__":
    main()
