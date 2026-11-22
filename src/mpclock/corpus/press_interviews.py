"""Press interviews pasted in from the site's "Press interviews" tab.

MPC members talk to newspapers and broadcasters far more often than the Bank's own
site records. The tab files a GitHub issue holding a member, a date, the outlet, one
or more links and/or a pasted transcript; .github/workflows/press_interview.yml
hands the issue to scripts/press_interview.py, which calls ingest() here.

One interview is usually reported by several outlets ("…, she told the Guardian"),
so before anything is added the submission is checked against what is already held:

  same link          a link already registered (or already a corpus source) is a duplicate
  same interview     an interview by the same member within 4 days that shares a quoted
                     passage, or whose text overlaps (8-word shingles), is the same
                     interview: the new link is recorded as another report of it, and if
                     the new text is a fuller version it replaces the old one
  a speech report    an article whose quotes mostly come from a speech the member gave
                     within 2 days is a report of that speech, already in the corpus

What survives is classified for policy relevance (the same classifier as speeches)
and, if relevant, added to the corpus as source_type "interview" with the outlet in
`institution`, and coded for forward guidance. Every submission is logged in
data/processed/press_interviews.json with its outcome.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "data/processed/press_interviews.json"
SITE_COPY = ROOT / "site/press_interviews.json"

_FIELD = re.compile(r"^\s*(member|date|outlet|urls?|links?|title)\s*:\s*(.*)$", re.I)
_URL = re.compile(r"https?://\S+")
_QUOTE = re.compile(r"[“\"]([^”\"]{25,400})[”\"]")
_WORD = re.compile(r"[a-z0-9']+")


def parse_submission(body: str) -> dict:
    """Issue body -> {member, date, outlet, urls, transcript, title}. Fields are
    'Key: value' lines; everything after a line 'Transcript:' is the transcript."""
    out = {"member": "", "date": "", "outlet": "", "urls": [], "transcript": "", "title": ""}
    lines = (body or "").replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*transcript\s*:?\s*$", line, re.I) or re.match(r"^\s*#+\s*transcript", line, re.I):
            out["transcript"] = "\n".join(lines[i + 1:]).strip().strip("`").strip()
            break
        m = _FIELD.match(line)
        if m:
            k, v = m.group(1).lower(), m.group(2).strip()
            if k.startswith(("url", "link")):
                out["urls"] += _URL.findall(v)
            else:
                out[k] = v
        else:
            out["urls"] += _URL.findall(line)
    out["urls"] = list(dict.fromkeys(u.rstrip(").,>]") for u in out["urls"]))
    return out


def load_registry() -> list[dict]:
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save_registry(reg: list[dict]) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    SITE_COPY.write_text(json.dumps(reg, indent=1), encoding="utf-8")


def fetch_article(url: str) -> tuple[str, str | None]:
    """(article text, publication date YYYY-MM-DD if the page states one)."""
    import requests
    import trafilatura
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (compatible; MPCLock/1.0)"})
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
    date = None
    try:
        meta = trafilatura.extract_metadata(r.text)
        date = (meta.date or "")[:10] if meta else None
    except Exception:  # noqa: BLE001
        pass
    return text, (date if date and re.match(r"\d{4}-\d{2}-\d{2}$", date) else None)


# --- auto-detection: who and when ------------------------------------------------
_MONTHS = "january february march april may june july august september october november december".split()
_DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)
_DATE_MDY = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_DATE_URL = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|-|$)")


def detect_date(text: str, urls: list[str], today: str) -> str | None:
    """A date in the link's path, else the latest date written in the opening of the
    text that is not in the future."""
    from .quick_read import date_from_url
    for u in urls:
        d = date_from_url(u)
        if d:
            return d
    found = []
    head = (text or "")[:3000]
    for m in _DATE_DMY.finditer(head):
        found.append((int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1))))
    for m in _DATE_MDY.finditer(head):
        found.append((int(m.group(3)), _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2))))
    ok = []
    for y, mo, d in found:
        try:
            x = dt.date(y, mo, d).isoformat()
        except ValueError:
            continue
        if x <= today:
            ok.append(x)
    return max(ok) if ok else None


def detect_member(text: str, urls: list[str], date: str, corpus) -> tuple[str | None, str]:
    """The sitting MPC member the piece is about: full-name and surname mentions in the
    text and the link, plus 'the Governor' for whoever was Governor that day. Returns
    (member or None, explanation)."""
    from ..tenure import for_corpus
    from .boe_interviews import governor_on
    ten = for_corpus(corpus)
    sitting = [n for n in ten.table if ten.active(n, date)]
    low = (text or "").lower()
    slug = " ".join(urls).lower().replace("-", " ").replace("_", " ")
    score = {}
    for n in sitting:
        parts = n.split()
        sur = parts[-1].lower()
        full = len(re.findall(r"\b" + re.escape(n.lower()) + r"\b", low))
        surname = len(re.findall(r"\b" + re.escape(sur) + r"\b", low))
        in_slug = sur in slug.split()
        score[n] = 2 * full + surname + (3 if in_slug else 0)
    try:
        gov = governor_on(date)
        g = len(re.findall(r"\bgovernor\b(?!\s*general)", low)) - \
            len(re.findall(r"\bdeputy governor\b", low))
        if gov in score and g > 0:
            score[gov] += min(g, 5)
    except StopIteration:
        pass
    ranked = sorted(score.items(), key=lambda x: -x[1])
    if not ranked or ranked[0][1] < 2:
        return None, "no sitting MPC member is named clearly enough"
    if len(ranked) > 1 and ranked[0][1] < 1.5 * ranked[1][1]:
        return None, (f"ambiguous between {ranked[0][0]} ({ranked[0][1]}) and "
                      f"{ranked[1][0]} ({ranked[1][1]}) — choose the member")
    return ranked[0][0], f"detected from the text ({ranked[0][1]} mentions)"


def _shingles(text: str, n: int = 8) -> set:
    w = _WORD.findall((text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def _quotes(text: str) -> set:
    return {re.sub(r"\s+", " ", q.strip().lower()) for q in _QUOTE.findall(text or "")
            if len(q.split()) >= 6}


def overlap(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / max(1, min(len(sa), len(sb))) if sa and sb else 0.0


def _days(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days)


def find_duplicate(sub: dict, text: str, corpus, reg: list[dict]) -> tuple[str, dict | None, object]:
    """('url' | 'interview' | 'speech' | '', registry entry, corpus record)."""
    urls = set(sub["urls"])
    for e in reg:
        if urls & set(e.get("urls", [])) and e.get("status") in ("added", "duplicate", "speech_report"):
            return "url", e, None
    by_url = {s.source_url: s for s in corpus if s.source_url}
    for u in urls:
        if u in by_url:
            return "url", None, by_url[u]
    q_new = _quotes(text)
    mine = [s for s in corpus if s.speaker == sub["member"] and s.date and _days(s.date, sub["date"]) <= 4]
    for s in mine:
        if s.source_type != "interview":
            continue
        shared = q_new & _quotes(s.text) if q_new else set()
        q_in_text = sum(1 for q in q_new if q in (s.text or "").lower())
        if len(shared) >= 1 or q_in_text >= 2 or overlap(text, s.text) >= 0.10:
            entry = next((e for e in reg if e.get("corpus_id") == s.id), None)
            return "interview", entry, s
    for s in mine:
        if s.source_type != "speech" or _days(s.date, sub["date"]) > 2 or not q_new:
            continue
        hit = sum(1 for q in q_new if q in re.sub(r"\s+", " ", (s.text or "").lower()))
        if hit / len(q_new) >= 0.5:
            return "speech", None, s
    return "", None, None


def ingest(sub: dict, corpus, issue: int | None = None, classify=True, code=True,
           request_id: str | None = None) -> tuple[dict, list]:
    """Process one submission. Returns (registry entry, corpus) — the corpus with the
    interview added or updated where that applies."""
    from ..roster_mpc import canon, is_mpc
    from ..schema import ST_INTERVIEW, Speech
    from ..tenure import for_corpus

    reg = load_registry()
    entry = {"submitted": dt.date.today().isoformat(), "issue": issue, "request_id": request_id,
             "urls": sub["urls"],
             "outlet": sub.get("outlet", ""), "date": sub.get("date", ""), "member": sub.get("member", "")}

    def done(status, reason, **kw):
        entry.update(status=status, reason=reason, **kw)
        reg.append(entry)
        save_registry(reg)
        return entry, corpus

    auto_member = sub.get("member", "").strip().lower() in ("", "auto", "auto-detect")
    auto_date = sub.get("date", "").strip().lower() in ("", "auto", "auto-detect")
    if not auto_date:
        try:
            dt.date.fromisoformat(sub["date"])
        except ValueError:
            return done("rejected", "date must be YYYY-MM-DD (or blank to detect it)")

    text = sub.get("transcript", "").strip()
    fetched_from, page_date = None, None
    for u in sub["urls"]:
        try:
            t, d = fetch_article(u)
        except Exception as e:  # noqa: BLE001
            entry.setdefault("fetch_errors", []).append(f"{u}: {str(e)[:80]}")
            continue
        page_date = page_date or d
        if not text and len(t.split()) >= 150:
            text, fetched_from = t, u
        if text and page_date:
            break
    if len(text.split()) < 80:
        return done("needs_text", "could not read enough text from the link(s) — "
                                  "likely paywalled; paste the transcript or article text")

    notes = []
    if auto_date:
        today = dt.date.today().isoformat()
        from .quick_read import date_from_url
        url_date = next((d for d in map(date_from_url, sub["urls"]) if d), None)
        d = url_date or page_date or detect_date(text, sub["urls"], today)
        if not d:
            return done("needs_info", "could not find the interview's date — give it")
        sub["date"] = d
        entry["date"] = d
        notes.append(f"date {d} taken from the " + ("link" if url_date else "page" if page_date else "text"))
    if auto_member:
        m, why = detect_member(text, sub["urls"], sub["date"], corpus)
        if not m:
            return done("needs_info", f"could not tell which member this is: {why}")
        sub["member"] = m
        notes.append(f"member {m} {why}")
    member = canon(sub.get("member", ""))
    entry["member"] = member
    if notes:
        entry["detected"] = "; ".join(notes)
    if not member or not is_mpc(member):
        return done("rejected", f"'{sub.get('member')}' is not a known MPC member")
    sub["member"] = member
    if not for_corpus(corpus).active(member, sub["date"]):
        return done("rejected", f"{member} was not on the MPC on {sub['date']}")
    entry["words"] = len(text.split())
    entry["text_source"] = "pasted" if not fetched_from else fetched_from

    kind, e_dup, s_dup = find_duplicate(sub, text, corpus, reg)
    if kind == "url":
        return done("duplicate", "this link is already held" + (f" ({s_dup.title})" if s_dup else ""),
                    duplicate_of=(e_dup or {}).get("corpus_id") or (s_dup.id if s_dup else None))
    if kind == "speech":
        return done("speech_report", f"reports the speech '{s_dup.title}' ({s_dup.date}), already in the corpus",
                    duplicate_of=s_dup.id)
    if kind == "interview":
        # same interview reported again: record the link; keep the fuller text
        replaced = False
        if len(text.split()) > 1.5 * len((s_dup.text or "").split()):
            s_dup.text, s_dup.word_count = text, len(text.split())
            s_dup.mu = s_dup.sigma = None
            s_dup.n_comparisons = 0
            replaced = True
            _uncode(s_dup.id)
        if e_dup:
            e_dup["urls"] = list(dict.fromkeys(e_dup.get("urls", []) + sub["urls"]))
        return done("duplicate", f"same interview as '{s_dup.title}' ({s_dup.date})"
                    + ("; this version is fuller, so its text replaces the earlier one" if replaced else
                       "; recorded as another report of it"),
                    duplicate_of=s_dup.id, replaced_text=replaced)

    outlet = sub.get("outlet") or (re.sub(r"^www\.", "", sub["urls"][0].split("/")[2]) if sub["urls"] else "press")
    title = sub.get("title") or f"Interview with {outlet}"
    s = Speech(date=sub["date"], speaker=member, title=title, text=text, source_type=ST_INTERVIEW,
               institution=f"Press: {outlet}",
               source_url=sub["urls"][0] if sub["urls"] else f"press-interview:{issue or entry['submitted']}")
    if classify:
        from ..process.classify import Classifier
        s.is_policy = bool(Classifier().classify(s))
        if not s.is_policy:
            return done("not_policy", "not about monetary policy or the economic outlook — not added")
    else:
        s.is_policy = True
    corpus.append(s)
    if code:
        import json as _j
        from ..forecast.guidance import code_documents
        rate = [(d, float(r)) for d, r in
                _j.loads((ROOT / "site/macro.json").read_text())["series"]["bank_rate"]["data"]]
        codes = code_documents([s], rate, verbose=False)
        if s.id in codes:
            entry["signal"] = codes[s.id]
    return done("added", "added to the corpus; it is scored in the next daily run", corpus_id=s.id,
                title=title)


def _uncode(doc_id: str) -> None:
    from ..forecast.guidance import load_codes, save_codes
    c = load_codes()
    if c.pop(doc_id, None) is not None:
        save_codes(c)
