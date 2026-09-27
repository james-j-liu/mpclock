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

from mpclock.config import PROCESSED, cfg
from mpclock.corpus import (assemble, boe_interviews, boe_mpc, boe_sitemap, boe_speeches,
                            mpc_transcripts, tsc_evidence)
from mpclock.macro.uk_macro import MacroContext
from mpclock.output.build_data import era_adjust, write_data_json
from mpclock.process.anonymize import Anonymizer
from mpclock.process.classify import Classifier
from mpclock.process.roster import build_roster
from mpclock.roster_mpc import is_mpc
from mpclock.tenure import for_corpus
from mpclock.corpus.misses import Misses
from mpclock.schema import load_corpus, save_corpus
from mpclock.tournament.runner import run_tournament

CORPUS = PROCESSED / "corpus.jsonl"
SINCE = "1997-01-01"


def main():
    ap = argparse.ArgumentParser()
    # default: config.yaml tournament.target_appearances_per_speech, so new documents
    # get the same number of comparisons the rest of the pool was rated on
    ap.add_argument("--appearances", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="use mock scorers (no API spend)")
    ap.add_argument("--no-score", action="store_true",
                    help="ingest and classify new records, then stop (no scoring spend)")
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    existing = load_corpus(CORPUS) if CORPUS.exists() else []
    have_urls = {s.source_url for s in existing}

    # 1) re-fetch the sitemap fresh and scrape only speech URLs we don't yet have,
    #    then pick up any MPC minutes / report round published since the last run
    new = []
    try:
        # the sitemap is fetched fresh once here; every later loader reads the copy
        # this just cached (it is 2.5 MB and was being downloaded three times a run)
        urls = boe_sitemap.speech_urls(use_cache=False)
        # pages that came back empty before (placeholders for speeches and meetings
        # still to come) are retried daily for two weeks, then weekly — not every day
        misses = Misses(PROCESSED / "scrape_misses.json")
        new_urls = [u for u in urls if u not in have_urls and misses.due(u)]
        new = boe_speeches.load(new_urls, verbose=False) if new_urls else []
        misses.update(new_urls, {s.source_url for s in new})
        print(f"sitemap: {len(urls)} speech URLs | {len(new_urls)} fetched | {len(new)} scraped")

        have_issues = {s.title.split(" — ")[0] for s in existing}
        year0 = datetime.date.today().year - 1
        not_due = {u for u in misses.state if not misses.due(u)}
        tried_m = [u for u in boe_mpc.minutes_urls(True, year0) if u not in have_urls | not_due]
        tried_r = [u for u in boe_mpc.report_urls(True, year0)
                   if boe_mpc.issue_label(u) not in have_issues and u not in not_due]
        new_mpc = boe_mpc.load_new(have_urls | not_due, have_issues, use_cache=True,
                                   start_year=year0)
        got = {s.source_url for s in new_mpc} | {
            u for u in tried_r if any(s.title.startswith(boe_mpc.issue_label(u)) for s in new_mpc)}
        misses.update(tried_m + tried_r, got)
        misses.save()
        have_ids = {s.id for s in existing}
        new += [s for s in new_mpc if s.id not in have_ids]
        print(f"MPC composite: {len(new_mpc)} documents from new rounds")

        # Treasury Committee: only sessions not already held (a published transcript
        # never changes), found on the first listing pages, which are newest-first
        tsc = tsc_evidence.load(use_cache=False, start_year=datetime.date.today().year - 1,
                                verbose=False, skip_urls=have_urls, max_pages=2)
        fresh = [s for s in tsc if s.id not in have_ids]
        new += fresh
        print(f"Treasury Committee: {len(fresh)} new member-documents")

        # the Governor's broadcast interview (each decision day) and the verbatim
        # meeting transcripts (released each January, eight years on): unseen URLs only
        extra = boe_interviews.load(use_cache=True, skip_urls=have_urls, verbose=False)
        extra += mpc_transcripts.load(skip_urls=have_urls, verbose=False,
                                      first_year=datetime.date.today().year - 9)
        extra = [s for s in extra if s.id not in have_ids]
        new += extra
        print(f"interviews + meeting transcripts: {len(extra)} new")
    except Exception as e:  # noqa: BLE001 - a blocked scrape must not stop the deploy
        print(f"[warn] ingest failed: {type(e).__name__}: {e}; scoring what we have")

    def make_pool(c):
        # MPC members only, and only while they sat on it (tenure from the minutes)
        tenure = for_corpus(c)
        return [s for s in c if s.is_policy and is_mpc(s.speaker) and SINCE <= s.date <= today
                and tenure.active(s.speaker, s.date)]

    # 2-4) ingest + score the new speeches. Wrapped so an API failure (e.g. OpenRouter
    #      out of credits -> 402, or a network blip) does NOT fail the whole job/deploy:
    #      we log it and fall back to redeploying the existing scored data.
    corpus, scored_ok = existing, True
    try:
        # new records, plus any MPC record left unclassified (e.g. re-ingested by add_sources)
        pending = new + [s for s in existing if s.is_policy is None and is_mpc(s.speaker)]
        if pending:
            Classifier().classify_all(pending)      # is_policy (composite types auto-pass)
        if new:
            # a newly parsed site page can duplicate an old BIS backfill copy
            corpus = assemble.drop_duplicates(existing + new)
            save_corpus(corpus, CORPUS)

        # the Committee as the newest minutes record it (a new member included)
        from mpclock import roster_mpc, tenure
        state = tenure.sync_roster(corpus, PROCESSED / "roster_state.json")
        roster_mpc._apply_live_state()
        if state.get("added"):
            print(f"roster: new MPC member(s) from the minutes: {state['added']}")

        pool = make_pool(corpus)
        # one anonymiser for both judges; texts are anonymised lazily, only for the
        # documents actually sent to a model (not the whole pool every morning)
        anon = Anonymizer(build_roster([s.speaker for s in corpus]))
        macro = MacroContext()
        new_ids = {s.id for s in new}
        n_new_pool = sum(1 for s in pool if s.id in new_ids or s.mu is None)
        # re-ingested or previously short-changed records are topped up too
        floor = cfg()["tournament"].get("min_appearances", 10)
        n_thin = sum(1 for s in pool if s.mu is not None and s.id not in new_ids
                     and (s.n_comparisons or 0) < floor)
        print(f"pool {len(pool)} MPC policy records | {n_new_pool} new to score"
              + (f" | {n_thin} under {floor} comparisons to top up" if n_thin else ""))
        if args.no_score:
            print("--no-score: ingested and classified only")
            raise SystemExit(0)

        if args.dry_run:
            from run_full import MockDirectScorer
            from run_poc import MockJudge
            judge, scorer = MockJudge(), MockDirectScorer()
        else:
            from mpclock.judge.factory import make_direct_scorer, make_pairwise_judge
            judge, scorer = make_pairwise_judge(), make_direct_scorer()
            print(f"judges: pairwise={judge.model} direct={scorer.model}")

        if n_new_pool or n_thin:
            run_tournament(pool, judge, appearances_per_speech=args.appearances,
                           macro=macro, resume=True, anonymizer=anon)
        to_direct = [s for s in pool if s.direct_score is None]
        if to_direct:
            scorer.score_all(to_direct, macro, concurrency=6, anonymizer=anon)
        spent = getattr(judge, "cost", 0) + getattr(scorer, "cost", 0)
        if spent:
            print(f"model spend this run: ${spent:.4f}")
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
