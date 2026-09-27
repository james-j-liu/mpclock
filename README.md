# MPCLock — Bank of England MPC hawkishness

An LLM pairwise tournament that scores Bank of England monetary-policy
communication on a hawkish–dovish spectrum, and publishes the result as a static
site: **https://james-j-liu.github.io/mpclock/**

It is the Bank of England counterpart to [ECBLock](https://james-j-liu.github.io/ecblock/),
and both follow the method of [FedLock](https://jnathan9.github.io/fedlock/): an LLM
judge reads two anonymised documents side by side, with the macro conditions of
each, and picks the more hawkish *relative to those conditions*; TrueSkill turns
~30 such comparisons per document into a continuous score.

## What is scored

| | |
|---|---|
| **MPC members** | Every monetary-policy speech and interview by a Monetary Policy Committee member — Governor, Deputy Governors, Chief Economist, external members — 1997 to today, **only while they sat on the Committee** (tenure is read from the minutes' attendance lists). Plus, split per member: their **evidence to Parliament** (Treasury Committee Monetary Policy Report hearings, appointment hearings and inquiries; the Lords Economic Affairs Committee), their **words in the verbatim MPC meeting transcripts** (2015–, released eight years on), their **vote rationales** from the minutes (Nov 2025–), and the Governor's **broadcast interview** after each decision (2026–). Officials who never vote on Bank Rate (PRA, FPC, markets) are excluded entirely. |
| **"BoE MPC"** | The Committee's own output, as one composite speaker: **minutes** of every meeting; each **Monetary Policy Report** (the Inflation Report before 2019) with its annexes as one document; the **press-conference transcript** and the Governor's **opening remarks** for each round. |

## Sources

- **Primary — bankofengland.co.uk.** Speech URLs come from the site's sitemap API
  (`/_api/sitemap/getsitemap`); the listing page itself is JS-paginated and cannot
  be crawled. Each speech's full text is extracted from its PDF, not the short web
  summary. The same site supplies minutes, reports and press conferences.
- **Cross-check — BIS central bankers' speeches** (`speeches.zip`). An independent
  transcription of the same speeches, used to verify coverage year by year and to
  backfill the handful of speeches (mostly 1996–98) the Bank no longer publishes.
  Those records are marked `Bank of England (BIS)`.
- **Parliament — committees.parliament.uk.** Oral-evidence transcripts from the
  Treasury Committee (id 158) and the Lords Economic Affairs Committee (id 175),
  split per MPC witness (`corpus/tsc_evidence.py`).
- **MPC meeting transcripts** — `/monetary-policy/mpc-documentation/<year>`
  (`corpus/mpc_transcripts.py`), and the Governor's broadcast-interview transcripts
  under `/news/` (`corpus/boe_interviews.py`).
- **Macro context — ONS** (CPI, core CPI, unemployment, GDP) and the **Bank's own
  database** (Bank Rate, 10-year gilt yield).

## Pipeline

```
corpus/       boe_speeches · boe_mpc · tsc_evidence · mpc_transcripts ·
              boe_interviews · bis_boe             -> data/processed/corpus.jsonl
process/      classify (policy relevance) · anonymize (5 layers)
judge/        jev (pairwise + direct, default) · openrouter / direct (chat fallback)
              factory picks the model per job
tenure.py     who sat on the MPC when, and the live "current" roster
tournament/   TrueSkill engine + Swiss/uncertainty pairing
output/       era adjustment -> site/data.json
site/         static Plotly site (Timeline · Rankings · Speaker · Data · Methodology)
```

Scoring runs on TypeSafe's **Jev** decision model (`typesafe/jev-1.13`) via
OpenRouter's decisions API: it answers a typed choice or score question and returns
probabilities, bills input only ($0.042/M), and shows no slot-A bias. Each excerpt
is read up to `judge.jev_excerpt_chars` (56k — Jev's 32k-token window holds two);
longer documents are sampled 40/20/40 from opening, middle and close.
Classification uses a chat model (`judge.model`, Gemini 2.5 Flash-Lite).
`PAIRWISE_MODEL` / `DIRECT_MODEL` override the scoring models; a non-Jev id falls
back to the chat judges, which use `judge.max_excerpt_chars` and `uncapped_types`.

## Running it

```bash
pip install -r requirements.txt

python -m mpclock.corpus.assemble          # build the corpus (speeches + MPC + BIS)
python scripts/add_sources.py              # or: add MPC/BIS to an existing corpus
python scripts/classify_corpus.py --mpc-only
python scripts/classify_corpus.py --mpc-only --recheck   # after a classifier change
python scripts/build_macro.py
python scripts/run_full.py --appearances 30 --concurrency 20
python scripts/rescore_full.py --chunk 4000   # re-judge everything (e.g. a model switch),
python scripts/rescore_full.py --finalize     # resumable; swaps in only when complete
python -m http.server 8231 --directory site
```

`scripts/daily_update.py` is the incremental version of all of the above: it picks
up new speeches, minutes, report rounds, evidence sessions, interviews and meeting
transcripts, refreshes the current-member roster from the newest minutes, scores
only what is new, and rewrites `site/data.json`. It fetches only what it has not
seen: known transcripts are skipped, and pages that came back empty (placeholders
for meetings still to come) are remembered in `scrape_misses.json` and retried
daily for two weeks, then weekly. `--no-score` ingests and classifies without
spending on scoring. It runs from `.github/workflows/daily.yml` at 06:30 UTC and
deploys the site to GitHub Pages.

## State in the repo

`data/processed/corpus.jsonl` (text + classifier verdicts + ratings) and
`data/processed/tournament_log.jsonl` (every comparison ever paid for) are committed
deliberately: they are the pipeline's memory, they make the daily run incremental,
and the log means a re-run never re-pays for a comparison.

GitHub rejects any single file over 100 MB, so the corpus keeps text where it can
still be needed and drops it where it cannot (`schema.keeps_text`): everything by an
MPC member or the Committee keeps its text whatever the classifier decided, because a
change to the classifier has to be able to re-judge a speech it once rejected;
speeches by officials who never sit on the MPC keep metadata and `source_url` only,
and `scripts/refetch_text.py` can re-scrape them if the roster ever widens. The
anonymised copy of a text is never stored, being derived. Texts over 2,000
characters are stored zlib-compressed (base64 so the file stays JSONL): 88 MB of
text becomes ~41 MB. zlib is deterministic, so an unchanged record is byte-identical
on every save and git's daily deltas stay small.
