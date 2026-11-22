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


def load_codes() -> dict:
    try:
        return json.loads(CODES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_codes(codes: dict) -> None:
    CODES.write_text(json.dumps(codes, sort_keys=True, indent=0), encoding="utf-8")


def _rate_on(rate, d):
    r = rate[0][1]
    for dd, v in rate:
        if dd <= d:
            r = v
        else:
            break
    return r


def code_documents(docs, rate, max_workers: int = 8, cap: int = 90000, limit: int | None = None,
                   verbose: bool = True) -> dict:
    """docs: Speech records. Codes those not yet in the cache; returns the cache."""
    import requests
    from ..judge.jev import MODEL, _Client
    from ..judge.openrouter import excerpt

    codes = load_codes()
    todo = [s for s in docs if s.id not in codes and (s.text or "").strip()]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"guidance: {len(codes)} coded, {len(todo)} to code")
    if not todo:
        return codes
    client = _Client()

    def one(s):
        state = {"date": s.date, "bank_rate_percent": _rate_on(rate, s.date),
                 "document_type": s.source_type, "speaker_role": "MPC member"}

        def build(c):
            return {"model": MODEL,
                    "state": {**state, "text": excerpt(s.text, c, s.source_type)},
                    "questions": {
                        "next_vote": {"type": "choice", "instructions": INSTRUCTIONS,
                                      "criteria": CRITERIA},
                        "focus": {"type": "score", "instructions": FOCUS_INSTRUCTIONS,
                                  "criteria": FOCUS_LEVELS}}}
        try:
            res = client.ask_shrinking(build, cap)
        except (requests.RequestException, RuntimeError) as e:
            return s, None, str(e)[:120]
        a = res["answers"]["next_vote"]
        probs = {k: round(float(a.get("probabilities", {}).get(k, 0.0)), 4) for k in OPTIONS}
        tot = sum(probs.values()) or 1.0
        probs = {k: v / tot for k, v in probs.items()}
        ev = sum(OPTIONS[k] * v for k, v in probs.items())
        f = res["answers"]["focus"].get("score")
        focus = None if f is None else round(float(f) / (len(FOCUS_LEVELS) - 1), 3)
        return s, {"date": s.date, "speaker": s.speaker, "type": s.source_type,
                   "p": {k: round(v, 4) for k, v in probs.items()}, "ev": round(ev, 4),
                   "focus": focus}, None

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, s) for s in todo]
        for fu in as_completed(futs):
            s, rec, err = fu.result()
            if rec:
                codes[s.id] = rec
            elif verbose:
                print(f"  [skip] {s.date} {s.speaker}: {err}")
            done += 1
            if done % 100 == 0:
                save_codes(codes)
                if verbose:
                    print(f"  {done}/{len(todo)} coded, ${client.cost:.3f}")
    save_codes(codes)
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
