"""Pick the model for each job.

Pairwise judging and direct scoring run on a decision model (TypeSafe Jev), while
classification — which needs a chat model — keeps judge.model / JUDGE_MODEL.
Override per job with the PAIRWISE_MODEL / DIRECT_MODEL environment variables;
any non-Jev model id falls back to the chat judges.
"""
from __future__ import annotations

import os

from ..config import cfg, judge_model


def _model(key: str, env: str) -> str:
    return os.environ.get(env) or cfg()["judge"].get(key) or judge_model()


def pairwise_model() -> str:
    return _model("pairwise_model", "PAIRWISE_MODEL")


def direct_model() -> str:
    return _model("direct_model", "DIRECT_MODEL")


def make_pairwise_judge():
    m = pairwise_model()
    if m.startswith("typesafe/jev"):
        from .jev import JevJudge
        return JevJudge(model=m, max_excerpt_chars=cfg()["judge"].get("jev_excerpt_chars", 56000))
    from .openrouter import Judge
    return Judge(model=m)


def make_direct_scorer():
    m = direct_model()
    if m.startswith("typesafe/jev"):
        from .jev import JevScorer
        return JevScorer(model=m)
    from .direct import DirectScorer
    return DirectScorer(model=m)
