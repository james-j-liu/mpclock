"""Do MPC members' own words predict their own votes?

The aggregate test (vote_analysis.py) found that the Committee's overall tone adds
nothing to forecasting its next decision. This asks the sharper, member-level
question: at a given meeting, does the member who has sounded more hawkish than
their colleagues vote more hawkishly than them — and does it tell you something
their last vote did not?

Panel: one row per (meeting, member). The vote is the member's own preferred
change (checked_member_votes, validated against the counted vote), expressed as
the deviation from the Committee's decision in 25bp units, so a member wanting
+25bp at a hold scores +1. The predictor is the member's mean hawkishness over
their documents in the `window` days before the meeting, minus the same average
across the other members present — so the era, the data and the macro backdrop
cancel, and only "more hawkish than colleagues" remains.

By default only documents that were public at the time count (speeches,
interviews, Parliamentary evidence). --with-transcripts adds the verbatim meeting
transcripts, which were not public then but are the member's own earlier words.

    python scripts/member_vote_analysis.py [--window 90]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mpclock.corpus.boe_interviews import governor_on
from mpclock.macro.mpc_votes import checked_member_votes
from mpclock.roster_mpc import canon
from mpclock.schema import load_corpus
from mpclock.tenure import attendance

PUBLIC = {"speech", "interview", "testimony"}


def build_panel(window: int, with_transcripts: bool) -> pd.DataFrame:
    corpus = load_corpus(ROOT / "data/processed/corpus.jsonl")
    kinds = PUBLIC | ({"meeting"} if with_transcripts else set())
    docs = pd.DataFrame([{"p": canon(s.speaker), "d": pd.Timestamp(s.date), "m": s.mu}
                         for s in corpus if s.source_type in kinds and s.mu is not None])
    docs["m"] = (docs["m"] - 50) / 5
    rate = pd.DataFrame(json.loads((ROOT / "site/macro.json").read_text())["series"]
                        ["bank_rate"]["data"], columns=["d", "r"])
    rate["d"] = pd.to_datetime(rate["d"])

    rows = []
    for s in sorted((s for s in corpus if s.source_type == "mp_account"), key=lambda s: s.date):
        day = pd.Timestamp(s.date)
        before = rate[rate.d < day]
        votes = checked_member_votes(s.text, float(before.r.iloc[-1]) if len(before) else None,
                                     attendance(s.text), governor_on(s.date))
        if not votes:
            continue
        decision = pd.Series(list(votes.values())).mode().iloc[0]
        recent = docs[(docs.d < day) & (docs.d >= day - pd.Timedelta(days=window))]
        own = recent.groupby("p")["m"].mean()
        for person, bp in votes.items():
            rows.append({"meeting": s.date, "member": person,
                         "dev": (bp - decision) / 25.0,
                         "hawk": own.get(person, np.nan)})
    panel = pd.DataFrame(rows)
    # relative to colleagues present at the same meeting (who have documents)
    panel["rel"] = panel["hawk"] - panel.groupby("meeting")["hawk"].transform("mean")
    panel = panel.sort_values(["member", "meeting"])
    panel["prev_dev"] = panel.groupby("member")["dev"].shift(1)
    panel["next_dev"] = panel.groupby("member")["dev"].shift(-1)
    return panel


def ols(df: pd.DataFrame, y: str, xs: list[str]):
    d = df.dropna(subset=[y] + xs)
    fit = sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type="cluster",
                                                     cov_kwds={"groups": d["member"]})
    return fit, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--with-transcripts", action="store_true")
    args = ap.parse_args()

    p = build_panel(args.window, args.with_transcripts)
    have = p.dropna(subset=["rel"])
    print(f"{p.meeting.nunique()} meetings with named votes | {len(p)} member-votes | "
          f"{len(have)} with the member's own documents in the prior {args.window} days")
    print(f"dissents: {int((p.dev > 0).sum())} hawkish, {int((p.dev < 0).sum())} dovish, "
          f"{int((p.dev == 0).sum())} with the majority\n")

    print("mean hawkishness relative to colleagues at the same meeting, by the vote cast:")
    for label, mask in (("voted more hawkishly than the decision", have.dev > 0),
                        ("voted with the majority", have.dev == 0),
                        ("voted more dovishly than the decision", have.dev < 0)):
        sub = have[mask]
        print(f"   {label:40} {sub.rel.mean():+.2f}   (n={len(sub)})")

    print("\nregressions (standard errors clustered by member):")
    for title, y, xs in (
            ("this vote ~ relative tone", "dev", ["rel"]),
            ("this vote ~ relative tone + own previous vote", "dev", ["rel", "prev_dev"]),
            ("NEXT vote ~ relative tone + this vote", "next_dev", ["rel", "dev"])):
        fit, n = ols(have, y, xs)
        terms = "  ".join(f"{x} {fit.params[x]:+.3f} (p={fit.pvalues[x]:.3f})" for x in xs)
        print(f"   {title:48} n={n:4d}  {terms}")

    # the forecasting question that matters: someone in the majority last time —
    # does sounding hawkish or dovish relative to colleagues flag that they break away?
    was_in = have[have.prev_dev == 0]
    for side, sign in (("hawkish", 1), ("dovish", -1)):
        broke = (np.sign(was_in.dev) == sign)
        top = was_in.rel * sign >= was_in.rel.mul(sign).quantile(0.8)
        print(f"\nfrom the majority to a {side} dissent: {broke.mean():.1%} of all such member-votes;"
              f" {broke[top].mean():.1%} among the fifth most {side} in tone "
              f"({int(broke[top].sum())} of {int(top.sum())})")


if __name__ == "__main__":
    main()
