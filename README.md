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
| **MPC members** | Every speech and interview by a Monetary Policy Committee member — Governor, Deputy Governors, Chief Economist, external members — 1997 to today. Officials who never vote on Bank Rate (PRA, FPC, markets) are excluded from the scoring system entirely. |
| **"BoE MPC"** | The Committee's own output, as one composite speaker: **minutes** of every meeting; the **Monetary Policy Report** (the Inflation Report before 2019) split back into its Summary, Overview, chapters, In focus boxes and annexes; the **press-conference transcript** and the Governor's **opening remarks** for each round. |

## Sources

- **Primary — bankofengland.co.uk.** Speech URLs come from the site's sitemap API
  (`/_api/sitemap/getsitemap`); the listing page itself is JS-paginated and cannot
  be crawled. Each speech's full text is extracted from its PDF, not the short web
  summary. The same site supplies minutes, reports and press conferences.
- **Cross-check — BIS central bankers' speeches** (`speeches.zip`). An independent
  transcription of the same speeches, used to verify coverage year by year and to
  backfill the handful of speeches (mostly 1996–98) the Bank no longer publishes.
  Those records are marked `Bank of England (BIS)`.
- **Macro context — ONS** (CPI, core CPI, unemployment, GDP) and the **Bank's own
  database** (Bank Rate, 10-year gilt yield).

## Pipeline

```
corpus/       boe_speeches · boe_mpc · bis_boe   -> data/processed/corpus.jsonl
process/      classify (policy relevance) · anonymize (5 layers)
judge/        openrouter (pairwise) · direct (0-100 baseline)
tournament/   TrueSkill engine + Swiss/uncertainty pairing
output/       era adjustment -> site/data.json
site/         static Plotly site (Timeline · Rankings · Speaker · Data · Methodology)
```

Judge model: `google/gemini-2.5-flash-lite` via OpenRouter (set `OPENROUTER_API_KEY`
in `.env`; `JUDGE_MODEL` overrides the model).

`judge.max_excerpt_chars` in config.yaml is the cost dial: every comparison sends
two excerpts and the run makes ~15 per document, so it sets the token bill. Over
the cap a document is sampled 40/20/40 from its opening, middle and close rather
than truncated. `judge.uncapped_types` lists the types passed whole however long
they run — the minutes and the Monetary Policy Report with its annexes.

## Running it

```bash
pip install -r requirements.txt

python -m mpclock.corpus.assemble          # build the corpus (speeches + MPC + BIS)
python scripts/add_sources.py              # or: add MPC/BIS to an existing corpus
python scripts/classify_corpus.py --mpc-only
python scripts/classify_corpus.py --mpc-only --recheck   # after a classifier change
python scripts/build_macro.py
python scripts/run_full.py --appearances 30 --concurrency 20
python -m http.server 8231 --directory site
```

`scripts/daily_update.py` is the incremental version of all of the above: it picks
up new speeches, new minutes and new report rounds, scores only those, and rewrites
`site/data.json`. It runs from `.github/workflows/daily.yml` at 06:30 UTC and
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
anonymised copy of a text is never stored, being derived. That holds the file at
~76 MB.
