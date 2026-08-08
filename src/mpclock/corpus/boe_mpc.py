"""The "BoE MPC" composite speaker: the Committee's own published output.

Individual MPC members speak for themselves; the Committee speaks through three
channels, and all of them are scored here under the single composite speaker
``BoE MPC`` (the analogue of ECBLock's "ECB council"):

  minutes   /minutes/YYYY/monetary-policy-committee-<month>-<year>      (1997-2015)
            /monetary-policy-summary-and-minutes/YYYY/<month>-<year>    (2015-)
            Since August 2015 the Monetary Policy Summary, the vote and the
            minutes are published together on decision day; before that the
            minutes appeared about two weeks after the meeting.

  reports   /inflation-report/YYYY/<month>-<year>                       (1997-2019)
            /monetary-policy-report/YYYY/<month>-<year>                 (2019-)
            The quarterly forecast round. The landing page links the full report
            PDF plus its satellites, and the report PDF is split back into its
            published sections — chapters, In focus boxes and annexes — using the
            document's own contents page, so each annex is judged on its own.

  briefing  the press-conference transcript and the Governor's opening remarks
            that accompany each report (linked from the same landing page).

Everything is a PDF; landing pages carry only a summary. Text extraction reuses
the speech scraper's pypdf path and running-header cleanup. Extracted text is
cached under data/raw/mpc_cache so re-runs and daily updates do not re-download
the ~4 MB report PDFs.
"""
from __future__ import annotations

import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ..config import RAW
from ..roster_mpc import canon, is_mpc
from ..schema import (ST_ACCOUNT, ST_MEMBER_VIEW, ST_QA, ST_REPORT, ST_STATEMENT,
                      BOE_MPC, Speech)
from . import boe_sitemap
from .boe_speeches import UA, _abs, _clean_pdf_text, fetch

BASE = "https://www.bankofengland.co.uk"
CACHE = RAW / "mpc_cache"

MONTHS = ("january february march april may june july august september "
          "october november december").split()
_MONTHS_TITLE = [m.capitalize() for m in MONTHS]
MONTH_IDX = {m: i + 1 for i, m in enumerate(MONTHS)}
_MON = "|".join(MONTHS)
# some report PDFs are filed under just an abbreviated month ("nov.pdf")
_MON_ABBR = "|".join(m[:3] for m in MONTHS)

# --- URL shapes -------------------------------------------------------------
_MINUTES_RE = re.compile(r"/minutes/(\d{4})/(?:[a-z]+/)?monetary-policy-committee-"
                         rf"({_MON})-(\d{{4}})\b", re.I)
_SUMMIN_RE = re.compile(rf"/monetary-policy-summary-and-minutes/(\d{{4}})/"
                        rf"(?:mpc-)?({_MON})-?(\d{{4}})?\b", re.I)
_REPORT_RE = re.compile(rf"/(inflation-report|monetary-policy-report)/(\d{{4}})/"
                        rf"({_MON})-(\d{{4}})/?$", re.I)
# a bare minutes PDF, used to fill years the sitemap omits (2003, most of 2005).
# Pre-2001 minutes sit under a "1900-2000" archive folder, later ones under the year.
_MINUTES_PDF = BASE + "/-/media/boe/files/minutes/{dir}/minutes-{month}-{year}.pdf"

_PDF_HREF_RE = re.compile(r'href="([^"]*?\.pdf)"', re.I)
# satellites we do not want as documents: slide decks and chart packs are images,
# and the minutes PDF linked from a report page is already loaded as minutes.
# (The minutes loader uses the narrower rule — since 2015 the minutes PDF is itself
# named "monetary-policy-summary-and-minutes-<month>-<year>.pdf".)
_SKIP_MEDIA_RE = re.compile(r"slides|chart|visual|data-annex", re.I)
_SKIP_PDF_RE = re.compile(rf"{_SKIP_MEDIA_RE.pattern}|monetary-policy-summary-and-minutes",
                          re.I)
_TRANSCRIPT_RE = re.compile(r"transcript", re.I)
_REMARKS_RE = re.compile(r"opening-remarks|opening_remarks", re.I)

# "held on 6 and 7 August 1997" / "meeting ending on 4 February 2026"
_MEETING_DATE_RE = re.compile(
    rf"(?:held on|ending on)\s+(?:\d{{1,2}}\s+(?:and|&)\s+)?(\d{{1,2}})\s+({_MON})\s+"
    r"((?:19|20)\d{2})", re.I)
_DATE_TEXT_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MON})\s+((?:19|20)\d{{2}})\b", re.I)


def _iso(day: str | int, month: str, year: str | int) -> str:
    return f"{int(year):04d}-{MONTH_IDX[month.lower()]:02d}-{int(day):02d}"


# --- fetching + caching -----------------------------------------------------
def _cache_path(url: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".txt")


def pdf_pages(url: str, use_cache: bool = True, timeout: int = 180) -> list[str]:
    """Per-page text of a PDF (page breaks are what the section splitter walks)."""
    path = _cache_path(url)
    if use_cache and path.exists():
        return path.read_text(encoding="utf-8").split("\f")
    try:
        from pypdf import PdfReader
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return []
        reader = PdfReader(io.BytesIO(r.content))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception:  # noqa: BLE001 - a bad/scanned PDF must not kill the run
        return []
    path.write_text("\f".join(pages), encoding="utf-8")
    return pages


def pdf_text(url: str, use_cache: bool = True) -> str:
    return unspace(_clean_pdf_text("\n".join(pdf_pages(url, use_cache=use_cache))))


# Reports typeset before ~2017 extract with a space inside almost every word
# ("In order to m aintain price stability"). Rejoining a stranded leading letter
# to the rest of its word makes them readable again; 'a' and 'I' are left alone
# because they are words in their own right.
_SPACED_RE = re.compile(r"(?<=\b[B-HJ-Zb-hj-z])\s(?=[A-Za-z]{2,})")
_SPACED_PROBE_RE = re.compile(r"\b[B-HJ-Zb-hj-z] [a-z]{2,}")


def is_letter_spaced(text: str) -> bool:
    probe = text[:20000]
    return len(_SPACED_PROBE_RE.findall(probe)) >= max(20, len(probe) // 600)


def unspace(text: str, spaced: bool | None = None) -> str:
    """Repair letter-spaced PDF text, if that is what this document has."""
    if spaced is None:
        spaced = is_letter_spaced(text)
    return _SPACED_RE.sub("", text) if spaced else text


# --- section splitting ------------------------------------------------------
# "… 49" — pre-2017 reports extract letter-spaced, so the page number can arrive
# as "4 9" and the title as "4 .1 C onsum er prices".
_CONTENTS_ENTRY_RE = re.compile(r"^(.{4,150}?)[\s.]+((?:\d\s?){1,3})$")
# top-level entries only: "Monetary Policy Summary", "1: Current economic
# conditions", "Box A: …", "Annex 2: …", "5 Prospects for inflation".
# "1.1: Inflation" and friends are folded into their parent chapter.
_SUBSECTION_RE = re.compile(r"^\d+\s*\.\s*\d")
_DROP_ENTRY_RE = re.compile(r"^(contents|glossary|index|monetary policy at the bank|"
                            r"other information|download|page\s*\d+)", re.I)
# same test on the squashed title, so letter-spaced front matter ("G lossary and
# other inform ation") is dropped too, along with pure furniture sections
_DROP_SQUASHED_RE = re.compile(r"^(contents|glossary|index|pressnotice|download|"
                               r"monetarypolicyatthebank|otherinformation|"
                               r"chartsandtables|listofcharts|references)")


def _page_entries(page: str) -> list[tuple[str, int]]:
    """Parse one page as a table of contents: [(title, printed page number)]."""
    entries: list[tuple[str, int]] = []
    carry = ""
    for raw in page.splitlines():
        line = raw.strip()
        if not line or _DROP_ENTRY_RE.match(line):
            carry = ""
            continue
        m = _CONTENTS_ENTRY_RE.match(line)
        if not m:
            # a contents entry wrapped onto a second line; hold it for the next
            carry = (carry + " " + line).strip() if carry else line
            continue
        title = ((carry + " " + m.group(1)).strip() if carry else m.group(1)).strip()
        carry = ""
        if len(title) < 4 or _DROP_ENTRY_RE.match(title) or \
                _DROP_SQUASHED_RE.match(_squash(title)[0]):
            continue
        entries.append((re.sub(r"\s+", " ", title), int(re.sub(r"\s", "", m.group(2)))))
    return entries


def _contents_entries(pages: list[str]) -> tuple[int, list[str]]:
    """(contents page index, top-level section titles) from a report's own contents.

    Reports up to 2016 print the contents without the word "Contents", so the page
    is identified structurally instead: the one that parses into the most
    "title … page-number" rows whose page numbers run forwards.
    """
    best: tuple[int, list[str]] = (-1, [])
    for i, page in enumerate(pages[:10]):
        entries = _page_entries(page)
        nums = [n for _, n in entries]
        # a real contents page runs forwards; allow a few strays from stray page
        # furniture rather than demanding a perfectly sorted column
        inversions = sum(1 for a, b in zip(nums, nums[1:]) if b < a)
        if len(entries) < 4 or inversions > 0.25 * len(entries):
            continue
        titles = [t for t, _ in entries if not _SUBSECTION_RE.match(t)]
        if len(titles) > len(best[1]):
            best = (i, titles)
    return best


def _squash(s: str) -> tuple[str, list[int]]:
    """Lowercase alphanumerics only, with a map back to offsets in `s`.

    Reports published before ~2017 extract with letter-spacing ("M oney and asset
    prices"), so headings are matched on the squashed text rather than literally.
    """
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s):
        if ch.isalnum():
            chars.append(ch.lower())
            idx.append(i)
    return "".join(chars), idx


def _heading_offsets(body: str, titles: list[str]) -> list[tuple[int, str]]:
    """Character offsets in `body` where each contents title appears as a heading."""
    squashed, imap = _squash(body)
    out: list[tuple[int, str]] = []
    cursor = 0
    for title in titles:
        key = _squash(title)[0][:60]
        if len(key) < 6:
            continue
        at = cursor
        while True:
            pos = squashed.find(key, at)
            if pos < 0:
                break
            start = imap[pos]
            # a heading starts its own line; a cross-reference in a sentence does not
            line_start = body.rfind("\n", 0, start) + 1
            if not body[line_start:start].strip():
                out.append((start, title))
                cursor = pos + len(key)
                break
            at = pos + 1
    return out


def split_sections(pages: list[str], min_chars: int = 700) -> list[tuple[str, str]]:
    """Split a report PDF into (section title, text) using its contents page.

    Falls back to a single whole-document section when the contents page cannot be
    parsed or its headings cannot be located in the body.
    """
    ci, titles = _contents_entries(pages)
    body = _clean_pdf_text("\n".join(pages[ci + 1:] if ci >= 0 else pages))
    spaced = is_letter_spaced(body)
    body = unspace(body, spaced)
    titles = [unspace(t, spaced) for t in titles]
    starts = _heading_offsets(body, titles) if titles else []
    if len(starts) < 2:
        whole = unspace(_clean_pdf_text("\n".join(pages)))
        return [("", whole)] if len(whole) >= min_chars else []

    out: list[tuple[str, str]] = []
    for n, (pos, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(body)
        text = body[pos:end].strip()
        if len(text) >= min_chars:
            out.append((title, text))
    return out


# --- MPC minutes ------------------------------------------------------------
def minutes_urls(use_cache: bool = True, start_year: int | None = None,
                 end_year: int | None = None) -> list[str]:
    xml = boe_sitemap._sitemap_xml(use_cache)
    urls: list[str] = []
    for m in re.finditer(r"<loc>(https://www\.bankofengland\.co\.uk[^<]+)</loc>", xml):
        u = m.group(1)
        hit = _MINUTES_RE.search(u) or _SUMMIN_RE.search(u)
        if not hit:
            continue
        year = int(hit.group(1))
        if (start_year and year < start_year) or (end_year and year > end_year):
            continue
        urls.append(u)
    return sorted(set(urls))


def _minutes_gaps(urls: list[str], start_year: int, end_year: int) -> list[str]:
    """Direct minutes-PDF URLs for months the sitemap does not list.

    The sitemap is missing 2003 and most of 2005, but the media path is regular,
    so the gaps are filled by construction and simply 404 where no meeting ran.
    """
    have = set()
    for u in urls:
        hit = _MINUTES_RE.search(u) or _SUMMIN_RE.search(u)
        if hit:
            year = hit.group(3) or hit.group(1)
            have.add((int(year), hit.group(2).lower()))
    out = []
    for year in range(max(start_year, 1997), min(end_year, 2015) + 1):
        folder = f"1900-2000/{year}" if year <= 2000 else str(year)
        for month in MONTHS:
            if (year, month) not in have:
                out.append(_MINUTES_PDF.format(dir=folder, month=month, year=year))
    return out


def _minutes_date(url: str, text: str, title: str) -> str:
    m = _MEETING_DATE_RE.search(title) or _MEETING_DATE_RE.search(text[:3000])
    if m:
        return _iso(m.group(1), m.group(2), m.group(3))
    m = _DATE_TEXT_RE.search(text[:1500])
    if m:
        return _iso(m.group(1), m.group(2), m.group(3))
    hit = _MINUTES_RE.search(url) or _SUMMIN_RE.search(url)
    if hit:
        year = hit.group(3) or hit.group(1)
        return f"{int(year):04d}-{MONTH_IDX[hit.group(2).lower()]:02d}-01"
    return ""


# --- "MPC members' views" (minutes from November 2025 on) -------------------
# Since the November 2025 meeting the minutes close with a section in which each
# member sets out, in their own paragraph, the reasoning behind their own vote.
# That is individual communication sitting inside a committee document, so it is
# cut out of the minutes and scored per member.
_VIEWS_HEAD_RE = re.compile(r"^\s*MPC members[’'’]?\s*views\s*$", re.I | re.M)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s", re.M)
_VOTE_GROUP_RE = re.compile(r"^\s*Votes? to [^\n]{3,80}$", re.I | re.M)
# "Andrew Bailey:", "Catherine L Mann:" — the middle token can be a bare initial
_MEMBER_PARA_RE = re.compile(r"^([A-Z][A-Za-z.'’-]*(?:\s+[A-Z][A-Za-z.'’-]*){1,3})\s*:\s+(?=[A-Z“\"])",
                             re.M)


def split_member_views(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Split "MPC members' views" out of the minutes.

    Returns the minutes without that section, and one (member, vote group,
    paragraph) triple per MPC member who set out a rationale.
    """
    head = _VIEWS_HEAD_RE.search(text)
    if not head:
        return text, []
    body_start = head.end()
    # the section runs to the next numbered paragraph after its own numbered
    # preamble ("20. Members set out the rationale …" … "21. On 5 November …")
    nums = [m.start() for m in _NUMBERED_RE.finditer(text, body_start)]
    end = nums[1] if len(nums) > 1 else (nums[0] if nums else len(text))
    section = text[body_start:end]

    views: list[tuple[str, str, str]] = []
    group = ""
    marks = sorted([(m.start(), m.end(), "group", m.group(0).strip())
                    for m in _VOTE_GROUP_RE.finditer(section)]
                   + [(m.start(), m.end(), "member", m.group(1).strip())
                      for m in _MEMBER_PARA_RE.finditer(section)])
    for i, (start, stop, kind, label) in enumerate(marks):
        if kind == "group":
            group = label
            continue
        person = canon(label)
        if not is_mpc(person) or person == BOE_MPC:
            continue
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(section)
        para = section[stop:nxt].strip()
        if len(para.split()) >= 60:      # a real rationale, not a stray line
            views.append((person, group, para))

    if not views:
        return text, []
    # drop the whole section, heading included, from the committee's own record
    cleaned = (text[:head.start()].rstrip() + "\n\n" + text[end:].lstrip()).strip()
    return cleaned, views


def _minutes_record(url: str, use_cache: bool = True) -> list[Speech]:
    """The minutes, plus one record per member view where the minutes carry them."""
    if url.lower().endswith(".pdf"):
        page_url, pdf_url, title = url, url, ""
    else:
        html = fetch(url)
        if not html:
            return []
        m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"\s*\|\s*Bank of England\s*$", "", (m.group(1) if m else "")).strip()
        if not title or title.lower().startswith("page not found"):
            return []
        pdfs = [_abs(h) for h in _PDF_HREF_RE.findall(html)
                if "/-/media/boe/files/" in h and not _SKIP_MEDIA_RE.search(h.rsplit("/", 1)[-1])]
        if not pdfs:
            return []
        page_url, pdf_url = url, pdfs[0]

    text = pdf_text(pdf_url, use_cache=use_cache)
    if len(text) < 2000:
        return []
    date = _minutes_date(url, text, title)
    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return []
    if not title:
        title = f"Minutes of the Monetary Policy Committee meeting, {date}"

    text, views = split_member_views(text)
    month = f"{_MONTHS_TITLE[int(date[5:7]) - 1]} {date[:4]}"
    out = [Speech(date=date, speaker=BOE_MPC, title=title[:220], text=text,
                  source_type=ST_ACCOUNT, institution="Bank of England",
                  source_url=page_url, orig_language="en")]
    for person, group, para in views:
        out.append(Speech(
            date=date, speaker=person,
            title=f"MPC minutes, individual vote rationale — {month}"[:220],
            text=(group + "\n\n" + para).strip() if group else para,
            source_type=ST_MEMBER_VIEW, institution="Bank of England",
            source_url=page_url, orig_language="en"))
    return out


# --- Monetary Policy Report / Inflation Report ------------------------------
def report_urls(use_cache: bool = True, start_year: int | None = None,
                end_year: int | None = None) -> list[str]:
    """Landing pages of each quarterly report issue (not their sub-pages)."""
    xml = boe_sitemap._sitemap_xml(use_cache)
    urls: list[str] = []
    for m in re.finditer(r"<loc>(https://www\.bankofengland\.co\.uk[^<]+)</loc>", xml):
        u = m.group(1)
        hit = _REPORT_RE.search(u)
        if not hit:
            continue
        year = int(hit.group(4))
        if (start_year and year < start_year) or (end_year and year > end_year):
            continue
        urls.append(u)
    return sorted(set(urls))


def _report_date(url: str, pages: list[str], html: str) -> str:
    # the report's Monetary Policy Summary dates the decision it accompanies
    # ("At its meeting ending on 4 February 2026, …"); it sits a few pages in.
    body = "\n".join(pages[:12])
    m = _MEETING_DATE_RE.search(body) or _DATE_TEXT_RE.search(body[:2000])
    if m:
        return _iso(m.group(1), m.group(2), m.group(3))
    m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)
    if m:
        return m.group(1)
    hit = _REPORT_RE.search(url)
    return f"{hit.group(4)}-{MONTH_IDX[hit.group(3).lower()]:02d}-01" if hit else ""


def issue_label(url: str) -> str:
    """"Monetary Policy Report February 2026" — the title prefix of every record
    cut from that issue, and so the key for "do we already have this round?"."""
    hit = _REPORT_RE.search(url)
    if not hit:
        return ""
    family = ("Monetary Policy Report" if hit.group(1) == "monetary-policy-report"
              else "Inflation Report")
    return f"{family} {hit.group(3).capitalize()} {hit.group(4)}"


def _report_records(url: str, use_cache: bool = True,
                    split: bool = True) -> list[Speech]:
    html = fetch(url)
    if not html:
        return []
    hit = _REPORT_RE.search(url)
    family = "Monetary Policy Report" if hit.group(1) == "monetary-policy-report" else "Inflation Report"
    issue = f"{hit.group(3).capitalize()} {hit.group(4)}"

    hrefs = [h for h in _PDF_HREF_RE.findall(html) if "/-/media/boe/files/" in h]
    main = transcript = remarks = ""
    extras: list[str] = []
    for h in dict.fromkeys(_abs(x) for x in hrefs):
        name = h.rsplit("/", 1)[-1]
        if _TRANSCRIPT_RE.search(name):
            transcript = transcript or h
        elif _REMARKS_RE.search(name):
            remarks = remarks or h
        elif _SKIP_PDF_RE.search(name):
            continue
        elif re.search(r"(monetary-policy|inflation)-report", name) or \
                re.fullmatch(rf"(?:{_MON}|{_MON_ABBR})(?:-?\d{{4}})?\.pdf", name, re.I):
            if not main:
                main = h
        else:
            extras.append(h)
    if not main and not transcript and not remarks:
        return []

    pages = pdf_pages(main, use_cache=use_cache) if main else []
    date = _report_date(url, pages, html)
    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return []

    def rec(title: str, text: str, stype: str, src: str) -> Speech:
        return Speech(date=date, speaker=BOE_MPC, title=title[:220], text=text,
                      source_type=stype, institution="Bank of England",
                      source_url=src, orig_language="en")

    def pretty_name(href: str) -> str:
        return href.rsplit("/", 1)[-1][:-4].replace("-", " ").replace("_", " ").strip()

    out: list[Speech] = []
    if pages:
        sections = split_sections(pages) if split else [("", unspace(_clean_pdf_text("\n".join(pages))))]
        if not split:
            # whole-report mode: the round's report IS one document, so the annex
            # papers published alongside it are appended rather than judged apart.
            body = sections[0][1] if sections else ""
            for h in extras:
                text = pdf_text(h, use_cache=use_cache)
                if len(text) >= 1500:
                    body += f"\n\n=== Annex: {pretty_name(h)} ===\n\n{text}"
            sections = [("", body)] if body else []
            extras = []
        for name, text in sections:
            label = f"{family} {issue}" + (f" — {name}" if name else "")
            # the report's own Monetary Policy Summary is the policy statement;
            # chapters, boxes and annexes are report sections.
            stype = ST_STATEMENT if re.match(r"monetary policy summary", name, re.I) else ST_REPORT
            out.append(rec(label, text, stype, main))
    for h in extras:
        text = pdf_text(h, use_cache=use_cache)
        if len(text) >= 1500:
            out.append(rec(f"{family} {issue} — {pretty_name(h)}", text, ST_REPORT, h))
    if transcript:
        text = pdf_text(transcript, use_cache=use_cache)
        if len(text) >= 1500:
            out.append(rec(f"{family} {issue} — press conference transcript", text, ST_QA, transcript))
    if remarks:
        text = pdf_text(remarks, use_cache=use_cache)
        if len(text) >= 800:
            out.append(rec(f"{family} {issue} — Governor's opening remarks", text,
                           ST_STATEMENT, remarks))
    return out


# --- driver -----------------------------------------------------------------
def _split_default() -> bool:
    from ..config import cfg
    return bool(cfg().get("corpus", {}).get("split_report_sections", False))


def load(use_cache: bool = True, start_year: int | None = 1997,
         end_year: int | None = None, concurrency: int = 8,
         split_sections_: bool | None = None, fill_gaps: bool = True,
         verbose: bool = True) -> list[Speech]:
    if split_sections_ is None:
        split_sections_ = _split_default()
    end = end_year or 2100
    murls = minutes_urls(use_cache, start_year, end_year)
    if fill_gaps:
        murls += _minutes_gaps(murls, start_year or 1997, end)
    rurls = report_urls(use_cache, start_year, end_year)
    if verbose:
        print(f"  MPC minutes: {len(murls)} URLs; reports: {len(rurls)} issues")

    out: list[Speech] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_minutes_record, u, use_cache): ("minutes", u) for u in murls}
        futs.update({ex.submit(_report_records, u, use_cache, split_sections_): ("report", u)
                     for u in rurls})
        done = 0
        for fut in as_completed(futs):
            done += 1
            kind, _ = futs[fut]
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                res = None
            if isinstance(res, Speech):
                out.append(res)
            elif res:
                out.extend(res)
            if verbose and done % 50 == 0:
                print(f"  fetched {done}/{len(futs)} MPC documents ({len(out)} records)")
    if verbose:
        import collections
        by = collections.Counter(s.source_type for s in out)
        print(f"BoE MPC composite: {len(out)} records {dict(by)}")
    return out


def load_new(seen_urls: set[str], seen_titles: set[str], use_cache: bool = False,
             start_year: int | None = None, end_year: int | None = None,
             concurrency: int = 6, verbose: bool = True) -> list[Speech]:
    """Only the Committee output the corpus does not already hold (daily update).

    Minutes are keyed by their page URL; a report round is keyed by its issue
    label, because one round yields many records whose source_url is a PDF.
    """
    split = _split_default()
    murls = [u for u in minutes_urls(use_cache, start_year, end_year) if u not in seen_urls]
    rurls = [u for u in report_urls(use_cache, start_year, end_year)
             if issue_label(u) not in seen_titles]
    if verbose:
        print(f"  MPC composite: {len(murls)} new minutes, {len(rurls)} new report rounds")
    if not murls and not rurls:
        return []

    out: list[Speech] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_minutes_record, u, use_cache) for u in murls]
        futs += [ex.submit(_report_records, u, use_cache, split) for u in rurls]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                res = None
            if isinstance(res, Speech):
                out.append(res)
            elif res:
                out.extend(res)
    return out


if __name__ == "__main__":
    import argparse
    import collections
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-split", action="store_true", help="one record per report PDF")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N reports")
    args = ap.parse_args()

    recs = load(use_cache=not args.no_cache, concurrency=args.concurrency,
                split_sections_=not args.no_split)
    print("Date range:", min(s.date for s in recs), "..", max(s.date for s in recs))
    print(collections.Counter(s.source_type for s in recs))
    for s in sorted(recs, key=lambda s: s.date)[-12:]:
        print(f"  {s.date}  {s.source_type:13} {s.word_count:6d}w  {s.title[:80]}")
