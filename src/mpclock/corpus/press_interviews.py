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


def fetch_article(url: str) -> str:
    import requests
    import trafilatura
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (compatible; MPCLock/1.0)"})
    r.raise_for_status()
    return trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""


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


def ingest(sub: dict, corpus, issue: int | None = None, classify=True, code=True) -> tuple[dict, list]:
    """Process one submission. Returns (registry entry, corpus) — the corpus with the
    interview added or updated where that applies."""
    from ..roster_mpc import canon, is_mpc
    from ..schema import ST_INTERVIEW, Speech
    from ..tenure import for_corpus

    reg = load_registry()
    entry = {"submitted": dt.date.today().isoformat(), "issue": issue, "urls": sub["urls"],
             "outlet": sub.get("outlet", ""), "date": sub.get("date", ""), "member": sub.get("member", "")}

    def done(status, reason, **kw):
        entry.update(status=status, reason=reason, **kw)
        reg.append(entry)
        save_registry(reg)
        return entry, corpus

    member = canon(sub.get("member", ""))
    entry["member"] = member
    if not member or not is_mpc(member):
        return done("rejected", f"'{sub.get('member')}' is not a known MPC member")
    try:
        dt.date.fromisoformat(sub.get("date", ""))
    except ValueError:
        return done("rejected", "date must be YYYY-MM-DD")
    sub["member"] = member
    if not for_corpus(corpus).active(member, sub["date"]):
        return done("rejected", f"{member} was not on the MPC on {sub['date']}")

    text = sub.get("transcript", "").strip()
    fetched_from = None
    if not text:
        for u in sub["urls"]:
            try:
                t = fetch_article(u)
            except Exception as e:  # noqa: BLE001
                entry.setdefault("fetch_errors", []).append(f"{u}: {str(e)[:80]}")
                continue
            if len(t.split()) >= 150:
                text, fetched_from = t, u
                break
    if len(text.split()) < 80:
        return done("needs_text", "could not read enough text from the link(s) — "
                                  "likely paywalled; paste the transcript or article text")
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
