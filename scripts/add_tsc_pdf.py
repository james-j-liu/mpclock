"""Add a Treasury Committee evidence session supplied as a PDF transcript.

When the committee's site cannot be read directly, the transcript PDF (downloaded by
hand) is turned into the same per-witness documents tsc_evidence makes from the HTML,
with the same source URL, so the records get the same ids and the daily run treats
the session as already held.

    python scripts/add_tsc_pdf.py transcript.pdf --id 18358 --date 2026-11-25
Needs pdftotext (poppler-utils).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpclock.config import PROCESSED
from mpclock.corpus import tsc_evidence as T
from mpclock.schema import load_corpus, save_corpus

# "Q511 Chair: ...", "Dr Mann: ...", "Andrew Bailey: ..." at the start of a line
_LABEL = re.compile(r"^(Q\d+\s+)?((?:Dr |Dame |Ms |Mr |Mrs |Sir |Lord |Baroness )?"
                    r"[A-Z][A-Za-z.'’-]+(?: [A-Z][A-Za-z.'’-]+){0,3}):\s(.*)")


def parse_pdf_text(text: str) -> tuple[list, str, str]:
    """(turns, members-present line, witnesses line), as parse_turns gives for HTML."""
    text = text.replace("\x0c", "\n")
    members = re.search(r"Members present:\s*(.+?)\.\s*\n", text, re.S)
    members = re.sub(r"\s+", " ", members.group(1)) if members else ""
    wit = re.search(r"\nWitnesses\s*\n(.+?)\n\s*\n", text, re.S)
    witnesses = re.sub(r"\s+", " ", wit.group(1)) if wit else ""
    start = re.search(r"^Q\d+\s", text, re.M)
    body = text[start.start():] if start else text
    turns, label, para = [], None, []

    def flush():
        if label and para:
            # the HTML transcript carries the question number apart from the text
            text = re.sub(r"^Q\d+\s+", "", re.sub(r"\s+", " ", " ".join(para)).strip())
            turns.append(T.Turn(label, text))
    for line in body.split("\n"):
        m = _LABEL.match(line.strip())
        if m and m.group(2) != "Witnesses":
            flush()
            label, para = m.group(2), [line.strip()]
        elif not line.strip():
            # a page break mid-sentence is not a new paragraph
            if para and re.search(r'[.?!”"—:)]$', para[-1]):
                flush(); para = []
        elif label:
            para.append(line.strip())
    flush()
    return turns, members, witnesses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--id", required=True, help="the committee's oral evidence id (the URL number)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--category", default="Bank of England Monetary Policy Reports (Regular evidence sessions)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    text = subprocess.run(["pdftotext", args.pdf, "-"], capture_output=True, text=True, check=True).stdout
    turns, members, witnesses = parse_pdf_text(text)
    sess = T.Session(args.date, args.category, f"{T.BASE}/oralevidence/{args.id}/html/", "treasury",
                     bool(T.SERIES_RE.search(args.category)))
    recs = T.records_from_turns(sess, turns, members, witnesses)
    print(f"{len(turns)} paragraphs; members: {members[:80]}...; witnesses: {witnesses[:120]}...")
    for r in recs:
        print(f"  {r.id} {r.speaker:22} {r.word_count:6} words  {r.title}")
    corpus_path = PROCESSED / "corpus.jsonl"
    corpus = load_corpus(corpus_path)
    have = {s.id for s in corpus}
    new = [r for r in recs if r.id not in have]
    print(f"{len(new)} new of {len(recs)}")
    if new and not args.dry_run:
        save_corpus(corpus + new, corpus_path)
        print(f"added to {corpus_path}")


if __name__ == "__main__":
    main()
