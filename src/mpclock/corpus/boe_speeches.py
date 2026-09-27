"""Scrape individual Bank of England speeches into Speech records.

Each speech has a landing page at /speech/YYYY/month/slug whose HTML is only a
short summary; the full text lives in a linked PDF at /-/media/boe/files/speech/…
So for each URL we:

  1. fetch the landing HTML (browser UA; the site 403s default agents),
  2. read the title (the speaker is in its tail: "… − speech by Andrew Bailey"),
     the delivery date, and the venue,
  3. download the PDF and extract its text (pypdf), stripping the repeated
     "Bank of England  Page N" running header/footer,
  4. fall back to the HTML body text if there is no usable PDF.

Speaker attribution is best-effort here; the downstream MPC-roster filter
(roster_mpc.is_mpc) is what actually decides who enters the scoring pool, so a
mis-parsed name simply fails to match and is dropped rather than polluting ratings.
"""
from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura

from ..schema import ST_INTERVIEW, ST_SPEECH, Speech
from . import boe_sitemap

UA = boe_sitemap.UA
BASE = "https://www.bankofengland.co.uk"

# speeches, and the occasional speech the Bank filed as a paper (Haldane's 2009
# "Banking on the State" lives under /files/paper/)
_PDF_RE = re.compile(r'href="((?:https://www\.bankofengland\.co\.uk)?/-/media/boe/files/(?:speech|paper)/[^"?#]+\.pdf)"', re.I)
# title/desc tail: "<topic> − speech by <Name>", "remarks given by <Name> at …",
# "slides by", "keynote address delivered by", "Speech by <Name> at …"
_ROLE = (r"(?:speech|remarks|slides|lecture|keynote|address|words|comments|"
         r"panel[\w ]*|introduction|welcome|opening|closing|presentation|talk|"
         r"speaking notes|contribution|intervention|q&a|discussion|monetary policy)")
_BY_RE = re.compile(
    rf"\b{_ROLE}\b[\w &,'’.-]*?\s+(?:given\s+|delivered\s+)?by\s+([^\n]+)$", re.I)
_TRAIL_BY_RE = re.compile(r"\bby\s+([A-Z][\w.\-'’]+(?:\s+[A-Za-zÀ-ÿ.\-'’]+){0,3})\s*$")
# slug: "…speech-by-charles-bean…", "remarks-given-by-mark-carney…"
_SLUG_BY_RE = re.compile(r"(?:by)-([a-z]+(?:-[a-z]+){1,2})")
_NAME_OK = re.compile(r"^[A-ZÀ-Ý][\w.\-'’]+(?:\s+[A-Za-zÀ-ÿ][\w.\-'’]*){0,3}$")
_INTERVIEW_RE = re.compile(r"interview|q&a|q-a|in conversation|talks to|spoke to", re.I)
_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_IDX = {m: i + 1 for i, m in enumerate(_MONTHS)}
_DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})\b", re.I)
_PAGE_HDR_RE = re.compile(r"^\s*Bank of England\s*(?:Page\s*\d+)?\s*$", re.I)
_PAGE_ONLY_RE = re.compile(r"^\s*Page\s*\d+\s*$", re.I)


def fetch(url: str, tries: int = 3, timeout: int = 30) -> str | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and r.text:
                if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1", "ascii"):
                    r.encoding = r.apparent_encoding or r.encoding
                return r.text
            if r.status_code in (403, 404, 410):
                return None
        except requests.RequestException:
            pass
    return None


def _abs(u: str) -> str:
    return u if u.startswith("http") else BASE + u


def _pdf_link(html: str) -> str | None:
    m = _PDF_RE.search(html)
    return _abs(m.group(1)) if m else None


def _clean_pdf_text(text: str) -> str:
    """Drop the running 'Bank of England / Page N' header and de-hyphenate."""
    lines = []
    for ln in text.splitlines():
        if _PAGE_HDR_RE.match(ln) or _PAGE_ONLY_RE.match(ln):
            continue
        lines.append(ln.rstrip())
    t = "\n".join(lines)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)      # join words split across line breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _pdf_text(url: str, timeout: int = 90) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pypdf is required for BoE PDF speeches; pip install pypdf") from e
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        raw = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:  # noqa: BLE001 - a bad/scanned PDF must not kill the run
        return ""
    return _clean_pdf_text(raw)


def _clean_name(cand: str) -> str:
    """Trim a captured 'by <X>' group down to a plausible person name."""
    # cut at the first venue clause, parenthetical, comma, or digit
    cand = re.split(r"\s+(?:at|to|before|during|given|for|of|on)\b|[(,\d]",
                    cand.strip(), maxsplit=1)[0]
    cand = cand.strip(" .-–—'\"")
    return cand if _NAME_OK.match(cand) else ""


_TRAIL_DASH_RE = re.compile(r"\s[-–—]\s*([A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+){1,2})\s*$")
_FROM_NAME_RE = re.compile(r"\bfrom\s+([A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+){1,2})\s*$")
_POSSESSIVE_RE = re.compile(r"^([A-Z][\w.'’-]+\s+[A-Z][\w'’-]+)['’]s\b")
_GOVERNORS_RE = re.compile(r"^Governor['’]s\b|\bGovernor['’]s (?:speech|remarks|lecture|address)", re.I)

# Since 2026 the Bank also titles pages "Dave Ramsden: Speech on quantitative
# tightening, …" — the name leads instead of trailing after "speech by".
_LEAD_NAME_RE = re.compile(r"^\s*([A-ZÀ-Ý][\w.'’-]+(?:\s+[A-ZÀ-Ý][\w.'’-]*){1,3})\s*:\s")


def _known_person(name: str) -> str:
    """The canonical roster name if `name` is someone the roster knows, else ""."""
    from ..roster_mpc import CURRENT_MPC, FORMER_MPC, canon
    person = canon(name)
    return person if person in CURRENT_MPC or person in FORMER_MPC else ""


def _speaker_from_slug(url: str) -> str:
    slug = url.rsplit("/", 1)[-1]
    m = _SLUG_BY_RE.search(slug)
    if m:
        return " ".join(w.capitalize() for w in m.group(1).split("-"))
    # "dave-ramsden-speech-on-quantitative-tightening": a slug that opens with the
    # name. Only trusted when the name is on the roster, since any two leading
    # words would otherwise pass as a person.
    words = slug.split("-")
    for n in (3, 2):
        person = _known_person(" ".join(w.capitalize() for w in words[:n]))
        if person:
            return person
    return ""


def _speaker(title: str, venue: str, url: str) -> str:
    for src in (title, venue):
        m = _BY_RE.search(src) or _TRAIL_BY_RE.search(src)
        if m:
            name = _clean_name(m.group(1))
            if name:
                return name
    m = _LEAD_NAME_RE.match(title)
    if m and _NAME_OK.match(m.group(1)):
        return m.group(1)
    # Older titles name the speaker in looser ways — "On counterparty risk - Andy
    # Haldane", "Introduction from Mark Carney", "Andy Haldane's opening remarks".
    # These patterns would match any capitalised words, so only a name on the MPC
    # roster is accepted from them.
    for rx in (_TRAIL_DASH_RE, _FROM_NAME_RE, _POSSESSIVE_RE):
        m = rx.search(title)
        if m and _known_person(m.group(1)):
            return _known_person(m.group(1))
    return _speaker_from_slug(url)


def _clean_title(title: str) -> str:
    """Remove the attribution — '… − speech by Name' or 'Name: …' — to leave the topic."""
    t = _BY_RE.sub("", title).strip(" -–—")
    t = _LEAD_NAME_RE.sub("", t, count=1).strip() if _LEAD_NAME_RE.match(t) else t
    return t or title


def _og(html: str, prop: str) -> str:
    m = re.search(rf'<meta[^>]+property="{re.escape(prop)}"[^>]+content="([^"]*)"', html, re.I)
    if not m:
        m = re.search(rf'<meta[^>]+name="{re.escape(prop)}"[^>]+content="([^"]*)"', html, re.I)
    return (m.group(1) if m else "").strip()


def _valid(date: str) -> str:
    """Return the date if it is a real calendar date, else "".

    Speech PDFs contain stray numbers next to month names ("… 90 October 2024" out
    of a footnote or a chart label), which would otherwise become the record's date.
    """
    try:
        import datetime
        datetime.date.fromisoformat(date)
        return date
    except (ValueError, TypeError):
        return ""


def _near_url_month(day: str, um) -> bool:
    """Whether `day` is within a month of the /speech/YYYY/<month>/ the URL files it
    under. The URL is the Bank's own filing and is never wrong about the month; a
    date read from the text can be a citation ("Bernanke, 27 January 2005")."""
    if not um or um.group(2) not in _MONTH_IDX:
        return True
    filed = int(um.group(1)) * 12 + _MONTH_IDX[um.group(2)]
    got = int(day[:4]) * 12 + int(day[5:7])
    return abs(got - filed) <= 1


def _pick_date(url: str, traf_date: str, body: str) -> str:
    um = re.search(r"/speech/(\d{4})/([a-z]+)/", url)
    # 1) the delivery date printed at the top of the speech PDF/page — the first
    #    one near the month the URL files the speech under
    for window in (body[:1500], body[:8000]):   # the header first, then further in
        for m in _DATE_TEXT_RE.finditer(window):
            cand = _valid(f"{m.group(3)}-{_MONTH_IDX[m.group(2).lower()]:02d}-{int(m.group(1)):02d}")
            if cand and _near_url_month(cand, um):
                return cand
    # 2) trafilatura's parsed metadata date, if it agrees with the URL year
    if re.match(r"\d{4}-\d{2}-\d{2}", traf_date or ""):
        if not um or traf_date[:4] == um.group(1):
            return traf_date
    # 3) the URL's year + month → first of month
    if um and um.group(2) in _MONTH_IDX:
        return f"{um.group(1)}-{_MONTH_IDX[um.group(2)]:02d}-01"
    if um:
        return f"{um.group(1)}-07-01"
    return traf_date[:10] if traf_date else ""


def extract_speech(url: str, min_chars: int = 800) -> Speech | None:
    html = fetch(url)
    if not html:
        return None
    raw = trafilatura.extract(html, output_format="json", with_metadata=True,
                              favor_recall=True)
    import json as _json
    meta = _json.loads(raw) if raw else {}
    title = (meta.get("title") or "").strip()
    if not title or title.lower().startswith("page not found"):
        return None
    html_text = (meta.get("text") or "").strip()

    pdf_url = _pdf_link(html)
    text = _pdf_text(pdf_url) if pdf_url else ""
    if len(text) < min_chars and len(html_text) > len(text):
        text = html_text                     # full-HTML speech (no/thin PDF)
    if len(text) < min_chars:
        return None

    venue = _og(html, "og:description")
    speaker = _speaker(title, venue, url)
    date = _pick_date(url, (meta.get("date") or ""), text)
    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return None
    if not speaker and _GOVERNORS_RE.search(title):
        # "Governor's speech at Mansion House": the office, not the name — the
        # Governor in office that day (these are the year's most important speeches)
        from .boe_interviews import governor_on
        speaker = governor_on(date)
    if not speaker:
        return None
    # The sitemap lists speeches before they are given ("Text to be published
    # Monday 28 September"); a date in the future is a placeholder, not a speech.
    import datetime as _dt
    if date > _dt.date.today().isoformat():
        return None
    stype = ST_INTERVIEW if _INTERVIEW_RE.search(title) else ST_SPEECH
    return Speech(
        date=date,
        speaker=speaker,
        title=_clean_title(title)[:220],
        text=text,
        source_type=stype,
        institution="Bank of England",
        source_url=url,
        orig_language="en",
    )


def load(urls: list[str] | None = None, *, use_cache: bool = True,
         start_year: int | None = None, end_year: int | None = None,
         concurrency: int = 12, verbose: bool = True) -> list[Speech]:
    urls = urls or boe_sitemap.speech_urls(use_cache=use_cache,
                                           start_year=start_year, end_year=end_year)
    out: list[Speech] = []
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(extract_speech, u): u for u in urls}
        for fut in as_completed(futs):
            done += 1
            try:
                s = fut.result()
            except Exception:  # noqa: BLE001
                s = None
            if s:
                out.append(s)
            if verbose and done % 100 == 0:
                print(f"  scraped {done}/{len(urls)} ({len(out)} kept)")
    if verbose:
        print(f"BoE speeches: {len(out)} extracted from {len(urls)} URLs")
    return out


if __name__ == "__main__":
    import collections
    sp = load(concurrency=12)
    print("Date range:", min(s.date for s in sp), "..", max(s.date for s in sp))
    print("Distinct speakers:", len({s.speaker for s in sp}))
    for name, c in collections.Counter(s.speaker for s in sp).most_common(20):
        print(f"  {c:4d}  {name}")
