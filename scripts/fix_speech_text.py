"""Find and repair speeches whose text was taken from a PDF the page only cites.

Until the fix in boe_speeches._pdf_link, a speech page with no PDF of its own (its text
in the HTML) had its text replaced by the first PDF the page linked — usually an older
speech it cites. This re-reads each MPC member's speech page, compares the PDF the old
rule took with what the page's own speech is, and for every mismatch re-extracts the
text, resets the record's scores and drops its comparisons from the tournament log, so
the next daily run scores it afresh. Guidance codes for it are dropped too.

    python scripts/fix_speech_text.py --since 2015-01-01 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpclock.corpus import boe_speeches as B
from mpclock.roster_mpc import is_mpc
from mpclock.schema import load_corpus, save_corpus

CORPUS = ROOT / "data/processed/corpus.jsonl"
LOG = ROOT / "data/processed/tournament_log.jsonl"
CODES = ROOT / "data/processed/guidance_codes.json"


def check(s):
    html = B.fetch(s.source_url)
    if not html:
        return s, None, None
    links = [B._abs(m.group(1)) for m in B._PDF_RE.finditer(html)]
    old = links[0] if links else None
    new = B._pdf_link(html, s.source_url)
    return s, old, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    corpus = load_corpus(CORPUS)
    cands = [s for s in corpus if s.source_type in ("speech", "interview") and s.date >= args.since
             and "/speech/" in (s.source_url or "") and is_mpc(s.speaker) and (s.text or "")]
    print(f"checking {len(cands)} speech pages")
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(check, cands))
    def wrong(s, old):
        """Only when sure: the stored text came from slides/annex/Q&A, or does not name
        its own speaker in its opening pages."""
        if B._NOT_TEXT.search(old.rsplit("/", 1)[-1]):
            return "slides/annex/Q&A file"
        sur = s.speaker.replace(".", "").split()[-1].lower()
        if sur not in (s.text or "")[:6000].lower():
            return "text does not name the speaker"
        return ""
    bad = [(s, o, n, wrong(s, o)) for s, o, n in res if o and o != n]
    bad = [(s, o, n, w) for s, o, n, w in bad if w]
    for s, o, n, w in bad:
        print(f"  {s.date} {s.speaker}: took {o.rsplit('/', 1)[-1]} ({w}) -> "
              f"{'own PDF ' + n.rsplit('/', 1)[-1] if n else 'page text'}")
    print(f"{len(bad)} of {len(cands)} had text from another document")
    if args.dry_run or not bad:
        return
    fixed = set()
    for s, o, n, w in bad:
        new = B.extract_speech(s.source_url)
        nw = len((new.text if new else "").split())
        if not new or nw < 1200:
            print(f"  kept {s.date} {s.speaker}: the page's own text is too short ({nw} words) to be the speech")
            continue
        s.text, s.word_count = new.text, len(new.text.split())
        s.mu = s.mu_adj = s.sigma = s.direct_score = s.direct_adj = None
        s.n_comparisons = 0
        s.is_policy = None          # re-classified by the daily run
        s.text_anon = ""
        fixed.add(s.id)
    save_corpus(corpus, CORPUS)
    if LOG.exists():
        lines = LOG.read_text(encoding="utf-8").splitlines()
        keep = [ln for ln in lines if not any(f'"{i}"' in ln for i in fixed)]
        LOG.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        print(f"dropped {len(lines) - len(keep)} logged comparisons involving re-texted speeches")
    if CODES.exists():
        c = json.loads(CODES.read_text(encoding="utf-8"))
        for i in fixed:
            c.pop(i, None)
        CODES.write_text(json.dumps(c, sort_keys=True, indent=0), encoding="utf-8")
    print(f"re-texted {len(fixed)} speeches; the daily run will classify and score them")


if __name__ == "__main__":
    main()
