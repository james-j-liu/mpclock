"""Code each member document for the vote it signals at the next meeting.

Every speech, interview, piece of evidence and vote rationale by a sitting MPC member
is put to Jev (the decision model the tournament uses) as one typed question —
"on this text alone, how is this member most likely to vote at the next meeting?" —
over five options, cut 50 / cut 25 / hold / hike 25 / hike 50. Jev returns a
probability for each, so a document that hedges spreads its mass and one that
gives explicit forward guidance concentrates it. A second question rates how much
the text is about the near-term policy outlook at all, used to weight documents.

Codes are cached by document id in data/processed/guidance_codes.json; each document
is paid for once (~$0.0005).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "data/processed/guidance_codes.json"
OPTIONS = {"cut_50": -2, "cut_25": -1, "hold": 0, "hike_25": 1, "hike_50": 2}
CRITERIA = {
    "cut_50": "Vote to cut Bank Rate by 50 basis points or more.",
    "cut_25": "Vote to cut Bank Rate by 25 basis points.",
    "hold": "Vote to keep Bank Rate unchanged.",
    "hike_25": "Vote to raise Bank Rate by 25 basis points.",
    "hike_50": "Vote to raise Bank Rate by 50 basis points or more.",
}
INSTRUCTIONS = (
    "The text is a speech, interview, parliamentary evidence or published vote rationale "
    "by a member of the Bank of England's Monetary Policy Committee, given on the date "
    "shown, when Bank Rate stood at the level shown. Based ONLY on what this member says, "
    "how are they most likely to vote at the NEXT MPC meeting after this text? Treat explicit "
    "forward guidance as the strongest evidence ('a further increase is likely to be needed "
    "in the near term', 'this is a one-for-now insurance hike', 'a gradual and careful "
    "approach to removing restraint', 'the case for a cut is building', 'policy needs to "
    "stay restrictive for longer'); then the member's stated balance of risks to inflation "
    "and how restrictive they judge policy to be. A member who describes a vote they have "
    "just cast and says nothing about the next one is most likely to repeat it only if they "
    "say the job is unfinished. If the text does not discuss the outlook for policy at all, "
    "choose hold."
)
FOCUS_LEVELS = ["Not about monetary policy", "Mentions policy in passing",
                "Some discussion of the policy outlook", "Substantial discussion of the policy outlook",
                "Mainly about the next policy decisions"]
FOCUS_INSTRUCTIONS = ("How much of this text is about the outlook for UK monetary policy — "
                      "the path of Bank Rate over the coming meetings?")


def load_codes(path: Path | None = None) -> dict:
    try:
        return json.loads((path or CODES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_codes(codes: dict, path: Path | None = None) -> None:
    (path or CODES).write_text(json.dumps(codes, sort_keys=True, indent=0), encoding="utf-8")


def _rate_on(rate, d):
    r = rate[0][1]
    for dd, v in rate:
        if dd <= d:
            r = v
        else:
            break
    return r


# --- v2: the policy passages coded on their own, and the direction of the next move ---
VERSION = 2
DIRECTION = {"lower": -1, "unchanged": 0, "higher": 1}
DIR_CRITERIA = {
    "lower": "The member's next vote for a change in Bank Rate is more likely to be a cut.",
    "unchanged": "The member expects to keep Bank Rate where it is for some time; no move is signalled either way.",
    "higher": "The member's next vote for a change in Bank Rate is more likely to be an increase.",
}
DIR_INSTRUCTIONS = (
    "Text by a member of the Bank of England's Monetary Policy Committee. Looking beyond the very "
    "next meeting to the next few, which way does this member lean on Bank Rate? A member who "
    "says a further increase is likely to be needed in the near term, while stressing that "
    "decisions are conditional and gradual, leans higher even if they might hold at the next "
    "meeting; a member who says policy is restrictive enough and they want to wait leans "
    "unchanged; one who says rates can come down leans lower.")
# v3 for vote rationales only: a box is written for the meeting just decided, so its "now"
# is that meeting, not the next one (a speech's "now" is usually the next decision)
BOX_VERSION = 3
BOX_NOTE = ("This is the member's published rationale for the vote they cast at the MPC meeting "
            "whose decision was announced on the date shown; the vote is stated at the top. Words "
            "like 'now', 'at this meeting', 'today' or 'rather than cutting/raising now' refer to "
            "THAT vote, already cast — not to the next meeting. Use what the member says about the "
            "conditions for a future move, and how close those conditions are, to judge their vote "
            "at the following meeting, about six weeks later. 'I would prefer to wait and see if X "
            "is confirmed in upcoming developments' signals a move at a coming meeting if X is "
            "confirmed, not a hold at the next one by default.")


def required_version(s) -> int:
    return BOX_VERSION if s.source_type == "member_view" else VERSION


PASSAGE_NOTE = ("These are the passages about monetary policy and the outlook for Bank Rate, taken "
                "from a longer text and joined by '…'. ")
_POLICY_SENT = re.compile(
    r"(?i)\bbank rate\b|\binterest rates?\b|monetary policy|policy (rate|stance|path|decision)|"
    r"\b(tighten|loosen|eas)(ing|ed)?\b|restrictive|further (increase|rise|cut|reduction|tightening)|"
    r"\brate (rise|cut|hike|increase)s?\b|\b(raising|cutting|raise|cut) (rates|bank rate)|"
    r"\bhike\b|\bthe (mpc|committee) (voted|decided)|\bmy (vote|judgement|view)\b|\bnext meeting")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")


def policy_passages(text: str, max_chars: int = 14000) -> str:
    """The sentences about policy, each with its neighbours, in order; '' if too little."""
    sents = [x.strip() for x in _SPLIT.split(re.sub(r"\s+", " ", text or "")) if x.strip()]
    keep = set()
    for i, x in enumerate(sents):
        if _POLICY_SENT.search(x):
            keep.update({i - 1, i, i + 1})
    out, prev = [], -2
    for i in sorted(k for k in keep if 0 <= k < len(sents)):
        out.append(("" if i == prev + 1 else " … ") + sents[i])
        prev = i
    joined = " ".join(out).strip()
    if len(joined) < 300:
        return ""
    if len(joined) > max_chars:          # keep the end: forward guidance usually closes a speech
        joined = "… " + joined[-max_chars:]
    return joined


def _choice(res, q, options):
    a = res["answers"][q]
    p = {k: float(a.get("probabilities", {}).get(k, 0.0)) for k in options}
    t = sum(p.values()) or 1.0
    p = {k: v / t for k, v in p.items()}
    return {k: round(v, 4) for k, v in p.items()}, round(sum(options[k] * v for k, v in p.items()), 4)


_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def blind_text(text: str, anonymizer) -> str:
    """Names replaced by [OFFICIAL] (the tournament's anonymizer) and years by [YEAR], so a
    model trained on later data cannot simply recall what the member went on to do."""
    return _YEAR.sub("[YEAR]", anonymizer(text))


def code_documents(docs, rate, max_workers: int = 8, cap: int = 90000, limit: int | None = None,
                   verbose: bool = True, model: str | None = None, blind: bool = False,
                   path: Path | None = None, anonymizer=None) -> dict:
    """Codes documents not yet coded at VERSION; returns the cache. Each document gets:
      p / ev            next-meeting vote, whole text
      dir / dir_ev      direction of the member's next move over the next few meetings
      p_pass / ev_pass, dir_pass / dir_ev_pass   the same on the policy passages alone
      focus             how much of the text is about the policy outlook"""
    import requests
    from ..judge.jev import MODEL as JEV, _Client
    from ..judge.openrouter import excerpt
    MODEL = model or JEV
    path = path or CODES

    codes = load_codes(path)
    todo = [s for s in docs if codes.get(s.id, {}).get("v") != required_version(s) and (s.text or "").strip()]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"guidance: {len(codes)} in cache, {len(todo)} to code (v{VERSION}; vote rationales v{BOX_VERSION})")
    if not todo:
        return codes
    client = _Client()

    def one(s):
        state = {"date": s.date, "bank_rate_percent": _rate_on(rate, s.date),
                 "document_type": s.source_type, "speaker_role": "MPC member"}
        text = s.text
        if blind:
            # no date, no names, no years: the model reads the text, it cannot look it up
            state.pop("date")
            text = blind_text(s.text, anonymizer)
        if s.source_type == "member_view":
            state["note"] = BOX_NOTE.replace("announced on the date shown", "being announced")
        instr = INSTRUCTIONS.replace("given on the date shown, when", "given when") if blind else INSTRUCTIONS
        vote_q = {"type": "choice", "instructions": instr, "criteria": CRITERIA}
        dir_q = {"type": "choice", "instructions": DIR_INSTRUCTIONS, "criteria": DIR_CRITERIA}

        def build(c):
            return {"model": MODEL, "state": {**state, "text": excerpt(text, c, s.source_type)},
                    "questions": {"next_vote": vote_q, "direction": dir_q,
                                  "focus": {"type": "score", "instructions": FOCUS_INSTRUCTIONS,
                                            "criteria": FOCUS_LEVELS}}}
        try:
            res = client.ask_shrinking(build, cap)
        except (requests.RequestException, RuntimeError) as e:
            return s, None, str(e)[:120]
        p, ev = _choice(res, "next_vote", OPTIONS)
        d, dev = _choice(res, "direction", DIRECTION)
        f = res["answers"]["focus"].get("score")
        rec = {"v": required_version(s), "model": MODEL, "served_model": res.get("model"), "blind": blind, "date": s.date, "speaker": s.speaker, "type": s.source_type,
               "p": p, "ev": ev, "dir": d, "dir_ev": dev,
               "focus": None if f is None else round(float(f) / (len(FOCUS_LEVELS) - 1), 3)}
        pas = policy_passages(text) if s.source_type != "member_view" else ""
        if pas:
            try:
                r2 = client.post({"model": MODEL,
                                  "state": {**state, "text": pas, "note": PASSAGE_NOTE},
                                  "questions": {"next_vote": vote_q, "direction": dir_q}})
                rec["p_pass"], rec["ev_pass"] = _choice(r2, "next_vote", OPTIONS)
                rec["dir_pass"], rec["dir_ev_pass"] = _choice(r2, "direction", DIRECTION)
                rec["pass_chars"] = len(pas)
            except (requests.RequestException, RuntimeError):
                pass
        return s, rec, None

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fu in as_completed([ex.submit(one, s) for s in todo]):
            s, rec, err = fu.result()
            if rec:
                prev = codes.get(s.id)
                if prev and prev.get("v") != rec["v"]:   # keep the earlier reading for comparison
                    rec["prev"] = {k: prev.get(k) for k in ("v", "p", "ev", "dir", "dir_ev")}
                codes[s.id] = rec
            elif verbose:
                print(f"  [skip] {s.date} {s.speaker}: {err}")
            done += 1
            if done % 100 == 0:
                save_codes(codes, path)
                if verbose:
                    print(f"  {done}/{len(todo)} coded, ${client.cost:.3f}")
    save_codes(codes, path)
    if verbose:
        print(f"guidance: coded {done}, spend ${client.cost:.4f}")
    return codes


def member_documents(corpus, since: str = "2014-06-01"):
    """Sitting members' own words: speeches, interviews, evidence, vote rationales."""
    from ..roster_mpc import is_mpc
    from ..tenure import for_corpus
    from .votes import OWN_TYPES
    ten = for_corpus(corpus)
    return [s for s in corpus if s.source_type in OWN_TYPES and s.date >= since
            and is_mpc(s.speaker) and ten.active(s.speaker, s.date)
            and (s.is_policy is not False or s.source_type == "member_view")]
