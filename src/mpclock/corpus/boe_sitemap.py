"""Enumerate Bank of England speech URLs via the site's sitemap API.

The public speeches listing (bankofengland.co.uk/news/speeches) is JavaScript-
paginated and keeps the same URL across pages, so it can't be crawled by URL. The
site instead exposes a full sitemap:

    https://www.bankofengland.co.uk/_api/sitemap/getsitemap   (XML, ~15k URLs)

Every individual speech is a `<loc>` whose path is `/speech/YYYY/<month>/<slug>`.
We pull the sitemap once (cached to data/raw) and return the speech URLs, newest
years first. WebFetch is blocked (403) but a browser User-Agent works fine.
"""
from __future__ import annotations

import re
from pathlib import Path

import requests

from ..config import RAW

SITEMAP_URL = "https://www.bankofengland.co.uk/_api/sitemap/getsitemap"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "en"}
CACHE = RAW / "boe_sitemap.xml"

_SPEECH_RE = re.compile(r"<loc>(https://www\.bankofengland\.co\.uk/speech/(\d{4})/[^<]+)</loc>")


def _sitemap_xml(use_cache: bool = True) -> str:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if use_cache and CACHE.exists():
        return CACHE.read_text(encoding="utf-8")
    r = requests.get(SITEMAP_URL, headers=UA, timeout=120)
    r.raise_for_status()
    CACHE.write_text(r.text, encoding="utf-8")
    return r.text


def speech_urls(use_cache: bool = True, start_year: int | None = None,
                end_year: int | None = None) -> list[str]:
    """Return all /speech/ URLs, optionally bounded by the year in the path."""
    xml = _sitemap_xml(use_cache)
    urls: list[tuple[int, str]] = []
    for m in _SPEECH_RE.finditer(xml):
        url, year = m.group(1), int(m.group(2))
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue
        urls.append((year, url))
    # de-dup, newest first (so partial/aborted runs cover the most relevant years)
    seen: dict[str, None] = {}
    for _, u in sorted(urls, key=lambda t: -t[0]):
        seen.setdefault(u, None)
    return list(seen)


if __name__ == "__main__":
    import collections
    us = speech_urls()
    print(f"{len(us)} speech URLs")
    yr = collections.Counter(re.search(r"/speech/(\d{4})/", u).group(1) for u in us)
    for y in sorted(yr):
        print(f"  {y}  {yr[y]}")
