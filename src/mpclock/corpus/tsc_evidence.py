"""Treasury Committee evidence sessions, split into one document per MPC member.

Every few months the Treasury Committee takes oral evidence from the Bank on the
Monetary Policy Report (called the Inflation Report before the 2019 rename). Four
or five MPC members appear together and are questioned for two hours, so the
transcript is not one speech but several interleaved ones.

This splits a session by witness: each MPC member's own answers, together with the
committee questions they were answering, become one document. Two members in the
same session therefore share some question text but no answer text — which is the
point, since the thing being scored is how each member responds under pressure,
unscripted, rather than the session as a whole.

Listing:    /committee/158/treasury-committee/publications/oral-evidence/
Transcript: /oralevidence/<id>/html/  — paragraphs whose speaker label is bold,
            questions numbered "Q391", witnesses named in a header line.
"""
from __future__ import annotations

import html as _html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

from ..config import RAW
from ..roster_mpc import canon, is_mpc
from ..schema import ST_TESTIMONY, Speech

BASE = "https://committees.parliament.uk"
LISTING = (BASE + "/committee/158/treasury-committee/publications/oral-evidence/"
           "?DateFrom=&DateTo=&SearchTerm=bank%20of%20england&SessionId=&page={page}")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
CACHE = RAW / "tsc"

# the monetary-policy series, under every name the committee has filed it as
SERIES_RE = re.compile(r"monetary policy report|inflation report", re.I)

_CARD_RE = re.compile(r'<div class="card card-button card-publication">(.*?)'
                      r'(?=<div class="card card-button card-publication">|'
                      r'<div class="section" id="pagination|$)', re.S)
_P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
_BOLD_RE = re.compile(r"<(?:b|strong)\b[^>]*>(.*?)</(?:b|strong)>"
                      r"|<span[^>]*font-weight:\s*bold[^>]*>(.*?)</span>", re.S)
_Q_RE = re.compile(r"^Q\d+\s*")
_TITLES_RE = re.compile(r"^(?:Rt\s+Hon\s+|Dr|Sir|Dame|Lord|Baroness|Professor|Prof|Mr|Mrs|Ms|Miss)\.?\s+",
                        re.I)
_MONTHS = ("january february march april may june july august september october "
           "november december").split()
_DATE_RE = re.compile(r"(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})", re.I)
_MEMBERS_RE = re.compile(r"Members present:\s*(.+?)(?:\.|$)", re.I)
_WITNESS_RE = re.compile(r"^Witnesses?:\s*(.+)$", re.I)


def _clean(fragment: str) -> str:
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", fragment))).strip()


def _fetch(url: str, cache_name: str | None = None, use_cache: bool = True) -> str:
    path = CACHE / cache_name if cache_name else None
    if path and use_cache and path.exists():
        return path.read_text(encoding="utf-8")
    r = requests.get(url, headers=UA, timeout=90)
    r.raise_for_status()
    # the transcripts are UTF-8 but often declare no charset, so requests falls
    # back to latin-1 and turns every curly quote into mojibake
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1", "ascii"):
        r.encoding = r.apparent_encoding or r.encoding
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return r.text


def _iso(text: str) -> str:
    m = _DATE_RE.search(text)
    if not m:
        return ""
    return f"{int(m.group(3)):04d}-{_MONTHS.index(m.group(2).lower())+1:02d}-{int(m.group(1)):02d}"


# --- session listing --------------------------------------------------------
@dataclass
class Session:
    date: str
    category: str
    url: str


def sessions(use_cache: bool = True, max_pages: int = 12) -> list[Session]:
    """Monetary-policy evidence sessions with an HTML transcript, newest first."""
    out: list[Session] = []
    for page in range(1, max_pages + 1):
        try:
            body = _fetch(LISTING.format(page=page), f"listing_{page}.html", use_cache)
        except requests.RequestException:
            break
        cards = _CARD_RE.findall(body)
        if not cards:
            break
        for card in cards:
            work = re.search(r'<span class="label">(?:Work|Inquiry)</span>([^<]*)', card)
            category = _clean(work.group(1)) if work else ""
            if not SERIES_RE.search(category):
                continue
            link = re.search(r'href="(/oralevidence/\d+/html/)"', card)
            date = re.search(r'<div class="primary-info">([^<]+)</div>', card)
            if not link or not date:
                continue   # PDF-only sessions (a handful of older ones) are skipped
            iso = _iso(_clean(date.group(1)))
            if iso:
                out.append(Session(iso, category, BASE + link.group(1)))
    seen: dict[str, Session] = {}
    for s in out:
        seen.setdefault(s.url, s)
    return sorted(seen.values(), key=lambda s: s.date, reverse=True)


# --- transcript parsing -----------------------------------------------------
@dataclass
class Turn:
    label: str      # speaker as printed ("Dr Mann", "Chair", "Andrew Bailey")
    text: str       # the paragraph, question number and label included


def parse_turns(body: str) -> tuple[list[Turn], str, str]:
    """(turns, members-present line, witnesses line) from a transcript's HTML."""
    turns: list[Turn] = []
    members = witnesses = ""
    current = ""
    for m in _P_RE.finditer(body):
        raw = m.group(1)
        text = _clean(raw)
        if not text:
            continue
        if not members:
            hit = _MEMBERS_RE.search(text)
            if hit:
                members = hit.group(1)
                continue
        hit = _WITNESS_RE.match(text)
        if hit:
            witnesses = (witnesses + "; " + hit.group(1)) if witnesses else hit.group(1)
            continue
        bold = _BOLD_RE.search(raw)
        label = _clean(bold.group(1) or bold.group(2) or "") if bold else ""
        label = _Q_RE.sub("", label).strip().rstrip(":").strip()
        if label and not label.startswith("Q"):
            current = label
        if not current:
            continue    # cover page: committee name, HC number, witness list, ordering
        turns.append(Turn(current, text))
    return turns, members, witnesses


def _surname(name: str) -> str:
    name = _TITLES_RE.sub("", name.strip()).strip()
    name = re.split(r"\s*[(,]", name)[0].strip()
    parts = [p for p in name.split() if len(p) > 1]
    return parts[-1].lower() if parts else ""


def _witness_names(witness_line: str, listing_witnesses: str = "") -> dict[str, str]:
    """surname -> canonical MPC name, for the MPC members giving evidence."""
    out: dict[str, str] = {}
    # "Witnesses: Andrew Bailey, Governor; Dr Swati Dhingra, External Member and
    # Dr Catherine L Mann." — names, roles and institutions all separated by the
    # same punctuation, so every fragment is tried and only roster names stick.
    for chunk in re.split(r"[;,]|(?<![A-Z])\band\b", witness_line + "; " + listing_witnesses):
        chunk = re.split(r"\s*\(", chunk)[0].strip(" .")
        chunk = re.sub(r"^(?:I{1,3}|IV)\s*:\s*", "", chunk).strip()
        name = _TITLES_RE.sub("", chunk).strip()
        if len(name.split()) < 2:
            continue
        person = canon(name)
        if is_mpc(person):
            out[_surname(person)] = person
    return out


def session_records(sess: Session, use_cache: bool = True,
                    min_words: int = 250) -> list[Speech]:
    body = _fetch(sess.url, f"transcript_{sess.url.rstrip('/').split('/')[-2]}.html", use_cache)
    turns, members, witness_line = parse_turns(body)
    if not turns:
        return []
    witnesses = _witness_names(witness_line)
    if not witnesses:
        return []
    mp_surnames = {_surname(n) for n in re.split(r"[;,]", members) if n.strip()}

    # A committee member and a witness can share a surname (John Mann MP vs
    # Catherine Mann), so a bare surname that belongs to someone on the committee
    # is treated as a question unless the label carries a title ("Dr Mann").
    def speaker_of(label: str) -> str | None:
        sur = _surname(label)
        if sur not in witnesses:
            return None
        if sur in mp_surnames and not _TITLES_RE.match(label.strip()):
            return None
        return witnesses[sur]

    docs: dict[str, list[str]] = {name: [] for name in witnesses.values()}
    words: dict[str, int] = {name: 0 for name in witnesses.values()}
    pending: list[str] = []      # question turns since the last new question block
    served: set[str] = set()     # witnesses already given the pending question
    last_was_witness = False

    for turn in turns:
        who = speaker_of(turn.label)
        if who is None:
            if last_was_witness:
                pending, served = [], set()
            pending.append(turn.text)
            last_was_witness = False
        else:
            if who not in served:
                docs[who].extend(pending)
                served.add(who)
            docs[who].append(turn.text)
            words[who] += len(turn.text.split())
            last_was_witness = True

    label = "Monetary Policy Report" if "monetary policy report" in sess.category.lower() \
        else "Inflation Report"
    out: list[Speech] = []
    for name, paragraphs in docs.items():
        if words[name] < min_words:      # a witness who barely spoke
            continue
        out.append(Speech(
            date=sess.date,
            speaker=name,
            title=f"Treasury Committee evidence — {label}, {sess.date}",
            text="\n\n".join(paragraphs),
            source_type=ST_TESTIMONY,
            institution="Bank of England",
            source_url=sess.url,
            orig_language="en",
        ))
    return out


def load(use_cache: bool = True, concurrency: int = 6, start_year: int | None = None,
         end_year: int | None = None, verbose: bool = True) -> list[Speech]:
    sess = sessions(use_cache)
    if start_year:
        sess = [s for s in sess if int(s.date[:4]) >= start_year]
    if end_year:
        sess = [s for s in sess if int(s.date[:4]) <= end_year]
    if verbose:
        print(f"  Treasury Committee: {len(sess)} monetary-policy sessions")
    out: list[Speech] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(session_records, s, use_cache): s for s in sess}
        for fut in as_completed(futs):
            try:
                out.extend(fut.result())
            except Exception as e:  # noqa: BLE001 - one bad transcript must not stop the run
                if verbose:
                    print(f"    [warn] {futs[fut].url}: {type(e).__name__}: {e}")
    if verbose:
        print(f"Treasury Committee: {len(out)} member-documents from {len(sess)} sessions")
    return out


if __name__ == "__main__":
    import collections
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    recs = load()
    if recs:
        print("Date range:", min(r.date for r in recs), "..", max(r.date for r in recs))
        for name, n in collections.Counter(r.speaker for r in recs).most_common():
            print(f"  {n:3d}  {name}")
        r = max(recs, key=lambda r: r.word_count)
        print(f"\nlongest: {r.date} {r.speaker} {r.word_count} words")
        print(r.text[:700])
