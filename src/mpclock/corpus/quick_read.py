"""A fast, dependency-free first look at a pasted link: can it be read, and when was it
published? Standard library only, so the intake job can run it before installing
anything and tell the page within seconds that a link is paywalled.

  date_from_url   /2026/11/10/, /2026-11-10, /2026/nov/10/, /20261110…
  read            fetch the page; count the words in its paragraphs; spot paywalls
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import urllib.request

_MON = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_URL_NUM = re.compile(r"(?<!\d)(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?=[/-]|$|\.|\?)")
_URL_NAME = re.compile(r"/(20\d{2})/([a-z]{3,9})/(\d{1,2})(?=/|$)", re.I)
_URL_COMPACT = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
_META_DATE = re.compile(
    r'(?:article:published_time|datePublished|pubdate|publish-date|date\.published|"dateCreated")'
    r'[^0-9]{0,40}(20\d{2}-\d{2}-\d{2})', re.I)
_PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_PAYWALL = re.compile(r"(?i)subscribe to (continue|read)|to continue reading|already a subscriber|"
                      r"this article is for subscribers|subscriber[- ]only|sign in to read|"
                      r"register to read|unlock this article|premium content|paywall")


def _valid(y, m, d) -> str | None:
    try:
        x = dt.date(int(y), int(m), int(d))
    except ValueError:
        return None
    return x.isoformat() if dt.date(2000, 1, 1) <= x <= dt.date.today() + dt.timedelta(days=1) else None


def date_from_url(url: str) -> str | None:
    u = url or ""
    for rx in (_URL_NUM,):
        m = rx.search(u)
        if m and _valid(*m.groups()):
            return _valid(*m.groups())
    m = _URL_NAME.search(u)
    if m and m.group(2)[:3].lower() in _MON:
        v = _valid(m.group(1), _MON[m.group(2)[:3].lower()], m.group(3))
        if v:
            return v
    m = _URL_COMPACT.search(u)
    if m:
        return _valid(*m.groups())
    return None


def read(url: str, timeout: int = 25) -> dict:
    """{'readable': bool, 'words': int, 'reason': str, 'date': str|None}"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept-Language": "en-GB,en;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(4_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", None)
        why = {401: "the site needs a login", 403: "the site blocks automated readers",
               404: "the page was not found", 451: "the site blocks this region"}.get(code, f"could not load it ({str(e)[:60]})")
        return {"readable": False, "words": 0, "reason": why, "date": date_from_url(url)}
    paras = [html.unescape(_TAG.sub(" ", p)) for p in _PARA.findall(raw)]
    text = " ".join(" ".join(p.split()) for p in paras)
    words = len(text.split())
    md = _META_DATE.search(raw)
    date = date_from_url(url) or (md.group(1) if md and _valid(*md.group(1).split("-")) else None)
    if words < 150:
        why = "the page is paywalled" if _PAYWALL.search(raw) else "too little article text on the page"
        return {"readable": False, "words": words, "reason": why, "date": date}
    if _PAYWALL.search(raw) and words < 400:
        return {"readable": False, "words": words, "reason": "only the free preview is readable (paywall)",
                "date": date}
    return {"readable": True, "words": words, "reason": "readable", "date": date}


def main():
    """Intake job's first step. Exit 3 when there is no pasted text and no link can be
    read, so the job stops at once and the page can say 'paste the text'."""
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    args = ap.parse_args()
    body = open(args.body, encoding="utf-8").read()
    m = re.search(r"^\s*transcript\s*:?\s*$(.*)", body, re.I | re.M | re.S)
    pasted = len((m.group(1) if m else "").split())
    urls = list(dict.fromkeys(u.rstrip(").,>]") for u in re.findall(r"https?://\S+", body.split("Transcript")[0])))
    results = {u: read(u) for u in urls}
    print(json.dumps({"pasted_words": pasted, "links": results}, indent=1))
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            for u, r in results.items():
                f.write(f"- {u}: {'readable' if r['readable'] else 'NOT readable — ' + r['reason']}"
                        f" ({r['words']} words){', dated ' + r['date'] if r['date'] else ''}\n")
    if pasted < 80 and not any(r["readable"] for r in results.values()):
        print("::error title=Link unreadable::" + "; ".join(f"{u}: {r['reason']}" for u, r in results.items())
              + " — paste the article text")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
