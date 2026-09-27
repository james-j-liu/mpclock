"""Verbatim transcripts of MPC policy meetings, one document per member.

Following the 2014 Warsh Review the Bank publishes, eight years after the event,
word-for-word transcripts of the meetings at which the MPC set policy (2015
onwards). They are the only record of what each member actually argued inside the
room, before the Committee's collective minutes smoothed it into consensus prose.

Each transcript covers a policy round — the pre-meeting data review and the
decision meeting — with turns marked by the speaker's name and a full stop:
"Andrew Haldane.  Thank you Governor. …". Every MPC member's turns are gathered
into one document per meeting; staff, the Treasury representative and the Chair's
procedural remarks by other names are left out because they are not MPC members.

Index: /monetary-policy/mpc-documentation/<year>  (one page per year, PDF links)
"""
from __future__ import annotations

import datetime as _dt
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..roster_mpc import canon, is_mpc
from ..schema import ST_MEETING, Speech
from .boe_mpc import MONTHS, _abs, _clean_pdf_text, fetch, pdf_pages

BASE = "https://www.bankofengland.co.uk"
INDEX = BASE + "/monetary-policy/mpc-documentation/{year}"
FIRST_YEAR = 2015          # the first year the Warsh commitment covers

_PDF_RE = re.compile(r'href="([^"]*mpc-transcript-[a-z]+-\d{4}\.pdf)"', re.I)
_MONTH_YEAR_RE = re.compile(r"mpc-transcript-([a-z]+)-(\d{4})\.pdf", re.I)
# "Governor Carney.    OK, morning" / "Andrew Haldane.  Thank you" — a turn opens a
# line with a name (optionally titled) followed by a full stop and a wide gap
_TURN_RE = re.compile(
    r"^[ \t]*((?:Governor|Deputy Governor|Dr|Professor|Sir|Dame)?\s*"
    r"[A-Z][A-Za-zÀ-ÿ'’-]+(?:\s+[A-Z][A-Za-zÀ-ÿ'’-]+){0,2})\.[ \t]{2,}", re.M)
_PRESENT_RE = re.compile(r"members\s+of\s+the\s+Committee\s+were\s+present:(.*?)"
                         r"(?:was present as|The following members of staff|$)", re.I | re.S)
_MEETING_DATE_RE = re.compile(r"Meeting on\s+\w+day\s+(\d{1,2})\s+(" + "|".join(MONTHS) +
                              r")\s+(\d{4})", re.I)


def transcript_urls(first_year: int = FIRST_YEAR, last_year: int | None = None) -> list[str]:
    """Every published meeting-transcript PDF; years not yet released simply 404."""
    last_year = last_year or _dt.date.today().year - 7
    urls: list[str] = []
    for year in range(first_year, last_year + 1):
        html = fetch(INDEX.format(year=year))
        if html:
            urls += [_abs(h) for h in _PDF_RE.findall(html)]
    return sorted(set(urls))


def _members(text: str) -> dict[str, str]:
    """surname -> canonical name, for the MPC members present at the round."""
    m = _PRESENT_RE.search(text[:4000])
    out: dict[str, str] = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        name = line.split(",")[0].strip()
        if len(name.split()) >= 2 and is_mpc(canon(name)):
            person = canon(name)
            out[person.split()[-1].lower()] = person
    return out


def _speaker(label: str, members: dict[str, str]) -> str | None:
    words = label.replace("Governor", "").split()
    return members.get(words[-1].lower()) if words else None


def transcript_records(url: str, use_cache: bool = True, min_words: int = 150) -> list[Speech]:
    pages = pdf_pages(url, use_cache=use_cache)
    # raw, not cleaned: the wide gap after "Andrew Haldane." is what marks a turn,
    # and cleaning collapses it; each turn is cleaned once it has been cut out
    text = "\n".join(pages)
    if len(text) < 5000:
        return []
    members = _members(text)
    if not members:
        return []
    # the decision meeting is the last one the transcript covers
    dates = [f"{int(y):04d}-{MONTHS.index(mo.lower()) + 1:02d}-{int(d):02d}"
             for d, mo, y in _MEETING_DATE_RE.findall(text)]
    if not dates:
        mm = _MONTH_YEAR_RE.search(url)
        dates = [f"{mm.group(2)}-{MONTHS.index(mm.group(1).lower()) + 1:02d}-01"] if mm else []
    if not dates:
        return []
    date = max(dates)

    turns = list(_TURN_RE.finditer(text))
    said: dict[str, list[str]] = {}
    for i, m in enumerate(turns):
        who = _speaker(m.group(1), members)
        if not who:
            continue                      # staff, the Treasury observer, a misread line
        end = turns[i + 1].start() if i + 1 < len(turns) else len(text)
        said.setdefault(who, []).append(_clean_pdf_text(text[m.end():end]))

    month = f"{MONTHS[int(date[5:7]) - 1].capitalize()} {date[:4]}"
    out = []
    for person, parts in said.items():
        body = "\n\n".join(parts)
        if len(body.split()) < min_words:
            continue
        out.append(Speech(date=date, speaker=person,
                          title=f"MPC meeting transcript — {month}",
                          text=body, source_type=ST_MEETING,
                          institution="Bank of England", source_url=url,
                          orig_language="en"))
    return out


def load(use_cache: bool = True, concurrency: int = 6, skip_urls: set[str] | None = None,
         verbose: bool = True, first_year: int = FIRST_YEAR) -> list[Speech]:
    """first_year: a released year never changes, so the daily run lists only the
    latest release window rather than every year back to 2015."""
    urls = [u for u in transcript_urls(first_year) if not skip_urls or u not in skip_urls]
    out: list[Speech] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(transcript_records, u, use_cache): u for u in urls}
        for fut in as_completed(futs):
            try:
                out.extend(fut.result())
            except Exception as e:  # noqa: BLE001 - one bad PDF must not stop the run
                if verbose:
                    print(f"    [warn] {futs[fut]}: {type(e).__name__}: {e}")
    if verbose:
        print(f"MPC meeting transcripts: {len(out)} member-documents from {len(urls)} meetings")
    return out


if __name__ == "__main__":
    import collections
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    recs = load()
    for name, n in collections.Counter(r.speaker for r in recs).most_common():
        print(f"  {n:3d}  {name}")
