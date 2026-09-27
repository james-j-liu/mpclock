"""The Governor's pooled broadcast interview after each MPC decision.

Since June 2026 the Bank publishes the transcript of the interview the Governor
gives the broadcasters (BBC, ITV, Sky, pooled) on decision day, under /news/ —
not /speech/, so the speech scraper never sees it. It is the Governor answering
unscripted media questions about the decision he has just announced, which is
exactly the "speaking to the media" register the rest of the corpus lacks.

Each transcript is one document for the Governor in office on that date,
questions included (as with Treasury Committee evidence: the question is what the
answer is responding to).
"""
from __future__ import annotations

import datetime as _dt
import json
import re

import trafilatura

from ..schema import ST_INTERVIEW, Speech
from . import boe_sitemap
from .boe_speeches import fetch

_URL_RE = re.compile(r"<loc>(https://www\.bankofengland\.co\.uk/news/\d{4}/[a-z]+/"
                     r"[^<]*governor[^<]*interview[^<]*transcript[^<]*)</loc>", re.I)
_DATE_RE = re.compile(r"(\d{1,2})-(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)-(\d{4})", re.I)
_MONTHS = ("january february march april may june july august september october "
           "november december").split()

# Governors and their last day in office (the site's Press-conf toggle uses the same)
_GOVERNORS = [("Edward George", "2003-06-30"), ("Mervyn King", "2013-06-30"),
              ("Mark Carney", "2020-03-15"), ("Andrew Bailey", "9999-12-31")]


def governor_on(day: str) -> str:
    return next(name for name, until in _GOVERNORS if day <= until)


def interview_urls(use_cache: bool = False) -> list[str]:
    return sorted(set(_URL_RE.findall(boe_sitemap._sitemap_xml(use_cache))))


def interview_record(url: str, min_words: int = 300) -> Speech | None:
    m = _DATE_RE.search(url)
    if not m:
        return None
    day = f"{int(m.group(3)):04d}-{_MONTHS.index(m.group(2).lower()) + 1:02d}-{int(m.group(1)):02d}"
    if day > _dt.date.today().isoformat():
        return None
    html = fetch(url)
    if not html:
        return None
    meta = json.loads(trafilatura.extract(html, output_format="json", with_metadata=True,
                                          favor_recall=True) or "{}")
    text = (meta.get("text") or "").strip()
    if len(text.split()) < min_words:
        return None
    return Speech(date=day, speaker=governor_on(day),
                  title=f"Governor's broadcast interview after the MPC decision, {day}",
                  text=text, source_type=ST_INTERVIEW, institution="Bank of England",
                  source_url=url, orig_language="en")


def load(use_cache: bool = False, skip_urls: set[str] | None = None,
         verbose: bool = True) -> list[Speech]:
    out = [r for u in interview_urls(use_cache)
           if not (skip_urls and u in skip_urls)
           for r in [interview_record(u)] if r]
    if verbose:
        print(f"Governor broadcast interviews: {len(out)}")
    return out
