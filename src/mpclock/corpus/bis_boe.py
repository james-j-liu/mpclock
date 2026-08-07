"""Bank of England speeches from the BIS archive — the cross-check on the scrape.

The Bank's own website is the primary source (see boe_speeches): it is complete,
it is the publisher of record, and its PDFs carry the full delivered text. The BIS
"central bankers' speeches" dataset is a second, independent transcription of the
same speeches, so it is used two ways:

  crosscheck()  compares the two corpora year by year, which is how gaps in a
                scrape show up (a run of missing months, a year where the PDF
                extractor silently returned nothing);
  load()        supplies the BIS records themselves, so anything the site scrape
                missed can still be added — flagged as institution "Bank of
                England (BIS)" and only when it does not duplicate a site record.

Source: https://www.bis.org/cbspeeches/download.htm  (speeches.zip, since 1996)
CSV columns: url,title,description,date,text,author. The institution is not a
column; it is named inside `description` ("… Governor of the Bank of England, at
…"), so the Bank is matched on that.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import date as _date

import requests

from ..config import RAW
from ..roster_mpc import canon
from ..schema import ST_SPEECH, Speech

csv.field_size_limit(10_000_000)

BIS_ZIP = "https://www.bis.org/speeches/speeches.zip"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CACHE = RAW / "bis"
INSTITUTION = "Bank of England (BIS)"

_BOE_RE = re.compile(r"Bank of England", re.I)
# other central banks whose speakers turn up in BoE-hosted events
_OTHER_BANK_RE = re.compile(
    r"(European Central Bank|Federal Reserve|Bank of Japan|Bank of Canada|"
    r"Reserve Bank of|Bundesbank|Banque de France|Bank for International)", re.I)
_WS = re.compile(r"\s+")
_TITLE_KEY_RE = re.compile(r"[^a-z0-9]+")


# BIS summary titles from the 1990s name the speaker ("Mr. George assesses how
# sustainable the current rate of growth …") and the author column of those rows
# is sometimes the wrong Bank official, so the title wins when the two disagree.
_TITLE_SPEAKER_RE = re.compile(r"^(?:Mr|Mrs|Ms|Dr|Sir|Lord)\.?\s+([A-Z][A-Za-z'’-]+)\b")


def _surname_index() -> dict[str, str]:
    from ..roster_mpc import CURRENT_MPC, FORMER_MPC
    idx: dict[str, str | None] = {}
    for full in CURRENT_MPC | FORMER_MPC:
        surname = canon(full).split()[-1].lower()
        # an ambiguous surname (two officials share it) resolves to nobody
        idx[surname] = None if surname in idx and idx[surname] != canon(full) else canon(full)
    return {k: v for k, v in idx.items() if v}


_SURNAMES = _surname_index()


def _speaker(author: str, title: str) -> str:
    name = canon(_WS.sub(" ", author.strip()))
    m = _TITLE_SPEAKER_RE.match(title.strip())
    if m:
        from_title = _SURNAMES.get(m.group(1).lower())
        if from_title and from_title.split()[-1].lower() != name.split()[-1].lower():
            return from_title
    return name


def _zip_bytes(use_cache: bool = True) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = CACHE / "speeches.zip"
    if use_cache and cache_f.exists():
        return cache_f.read_bytes()
    r = requests.get(BIS_ZIP, headers={"User-Agent": UA}, timeout=900)
    r.raise_for_status()
    cache_f.write_bytes(r.content)
    return r.content


def load(use_cache: bool = True, min_chars: int = 400,
         start_year: int | None = None, end_year: int | None = None) -> list[Speech]:
    """Every Bank of England speech in the BIS bulk extract."""
    z = zipfile.ZipFile(io.BytesIO(_zip_bytes(use_cache)))
    out: list[Speech] = []
    for member in z.namelist():
        if not member.endswith(".csv"):
            continue
        reader = csv.DictReader(io.StringIO(z.read(member).decode("utf-8", "replace")))
        for row in reader:
            desc = row.get("description") or ""
            if not _BOE_RE.search(desc) or _OTHER_BANK_RE.search(desc):
                continue
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            speaker = _speaker(row.get("author") or "", title)
            date = (row.get("date") or "")[:10]
            if len(text) < min_chars or not speaker or not re.match(r"\d{4}-\d{2}-\d{2}", date):
                continue
            year = int(date[:4])
            if (start_year and year < start_year) or (end_year and year > end_year):
                continue
            out.append(Speech(
                date=date,
                speaker=speaker,
                title=title[:220],
                text=text,
                source_type=ST_SPEECH,
                institution=INSTITUTION,
                source_url=(row.get("url") or "").strip(),
                orig_language="en",
            ))
    return out


def _title_key(title: str) -> str:
    return _TITLE_KEY_RE.sub("", title.lower())[:40]


def _days(a: str, b: str) -> int:
    try:
        return abs((_date.fromisoformat(a) - _date.fromisoformat(b)).days)
    except ValueError:
        return 999


def is_duplicate(bis: Speech, site: list[Speech], window_days: int = 21) -> bool:
    """True when a site record already covers this BIS speech.

    BIS dates a speech when it republishes it, up to a few weeks after delivery,
    so same-speaker records within a window count as the same speech unless their
    titles clearly differ.
    """
    key = _title_key(bis.title)
    for s in site:
        if canon(s.speaker) != canon(bis.speaker):
            continue
        d = _days(s.date, bis.date)
        if d == 0:
            return True
        if d <= window_days and (key[:20] in _title_key(s.title) or
                                 _title_key(s.title)[:20] in key):
            return True
    return False


def crosscheck(site: list[Speech], bis: list[Speech], verbose: bool = True) -> list[Speech]:
    """Report site-vs-BIS coverage and return the BIS speeches the site lacks."""
    by_speaker: dict[str, list[Speech]] = {}
    for s in site:
        by_speaker.setdefault(canon(s.speaker), []).append(s)
    missing = [b for b in bis if not is_duplicate(b, by_speaker.get(canon(b.speaker), []))]

    if verbose:
        import collections
        sy = collections.Counter(s.date[:4] for s in site)
        by = collections.Counter(s.date[:4] for s in bis)
        my = collections.Counter(s.date[:4] for s in missing)
        print(f"  cross-check: site {len(site)} vs BIS {len(bis)} "
              f"-> {len(missing)} BIS-only records")
        years = sorted(set(sy) | set(by))
        thin = [y for y in years if by[y] > sy[y]]
        if thin:
            print("  years where BIS has more than the site scrape "
                  "(year: site/BIS/BIS-only):")
            for y in thin:
                print(f"    {y}: {sy[y]}/{by[y]}/{my[y]}")
    return missing


if __name__ == "__main__":
    import collections
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sp = load()
    print(f"Loaded {len(sp)} Bank of England speeches from BIS")
    print("Date range:", min(s.date for s in sp), "..", max(s.date for s in sp))
    for name, n in collections.Counter(s.speaker for s in sp).most_common(15):
        print(f"  {n:4d}  {name}")
