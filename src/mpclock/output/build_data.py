"""Era adjustment + emit the site's data.json (FedLock-compatible schema).

Era adjustment (FedLock): adjusted = raw_mu - quarterly_mean + 50, re-centring
50 to "neutral relative to contemporaries". Timeline uses raw mu; Rankings /
Speaker tabs use the adjusted score.

data.json schema (short keys, matching FedLock):
  d=date, m=mu raw, ma=mu era-adjusted, s=sigma, n=#comparisons,
  ds=direct raw score, dsa=direct era-adjusted score,
  a=speaker, tt=title, st=source type, wc=word count, inst=institution
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..schema import Speech

# LLMs scoring 0-100 cluster hard on multiples of 5/10 (verified across 4 models),
# so the direct method is quantised to ~23 levels. To present it as the continuous
# signal it approximates, we add a deterministic per-speech jitter spanning the
# quantisation width. It is zero-mean (corpus mean and rankings are unchanged) and
# seeded by the speech id, so it is stable and reversible (set to 0 to disable).
DIRECT_JITTER = 0.0  # display jitter for direct scores disabled (raw discrete levels)


def _jitter(speech_id: str) -> float:
    h = hashlib.sha1(speech_id.encode()).hexdigest()
    u1 = int(h[:8], 16) / 0xFFFFFFFF
    u2 = int(h[8:16], 16) / 0xFFFFFFFF
    return DIRECT_JITTER * (u1 - u2)  # symmetric triangular in [-w, w], mean 0


def _roster_dict() -> dict:
    from ..roster_mpc import to_dict
    return to_dict()


def _era_adjust_attr(speeches: list[Speech], src_attr: str, dst_attr: str,
                     recenter_to: float = 50.0) -> None:
    """Re-centre a per-speech score against its calendar-quarter mean to 50."""
    rows = [(s.id, s.date, getattr(s, src_attr)) for s in speeches
            if getattr(s, src_attr) is not None]
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["id", "date", "v"])
    df["q"] = pd.to_datetime(df["date"]).dt.to_period("Q")
    qmean = df.groupby("q")["v"].transform("mean")
    df["adj"] = df["v"] - qmean + recenter_to
    adj_by_id = dict(zip(df["id"], df["adj"]))
    for s in speeches:
        if getattr(s, src_attr) is not None:
            setattr(s, dst_attr, round(float(adj_by_id[s.id]), 2))


def era_adjust(speeches: list[Speech], recenter_to: float = 50.0) -> None:
    _era_adjust_attr(speeches, "mu", "mu_adj", recenter_to)
    _era_adjust_attr(speeches, "direct_score", "direct_adj", recenter_to)


def recenter_raw(speeches: list[Speech], attr: str, to: float = 50.0) -> None:
    """Shift a raw score so the whole-corpus mean equals `to` (50 -> normalised 0).

    The era-adjusted score is invariant to this global shift, so only the raw
    series moves. Used to put the corpus-wide average of each method at neutral.
    """
    vals = [getattr(s, attr) for s in speeches if getattr(s, attr) is not None]
    if not vals:
        return
    shift = to - sum(vals) / len(vals)
    for s in speeches:
        v = getattr(s, attr)
        if v is not None:
            setattr(s, attr, round(v + shift, 3))


def write_data_json(speeches: list[Speech], path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scored = [s for s in speeches if s.mu is not None or s.direct_score is not None]
    out = []
    for s in scored:
        rec = {
            "d": s.date,
            "m": round(float(s.mu), 2) if s.mu is not None else None,
            "ma": s.mu_adj if s.mu_adj is not None
                  else (round(float(s.mu), 2) if s.mu is not None else None),
            "s": round(float(s.sigma), 2) if s.sigma is not None else None,
            "n": s.n_comparisons,
            "ds": round(float(s.direct_score) + _jitter(s.id), 2) if s.direct_score is not None else None,
            "dsa": round((s.direct_adj if s.direct_adj is not None else float(s.direct_score)) + _jitter(s.id), 2)
                   if s.direct_score is not None else None,
            "a": s.speaker,
            "tt": s.title[:160],
            "st": s.source_type,
            "wc": s.word_count,
            "inst": s.institution,
        }
        out.append(rec)
    out.sort(key=lambda r: r["d"])
    meta = {
        "speeches": out,
        "meta": {
            "n_speeches": len(out),
            "n_speakers": len({r["a"] for r in out}),
            "n_pairwise": sum(1 for r in out if r["m"] is not None),
            "n_direct": sum(1 for r in out if r["ds"] is not None),
            "date_min": min((r["d"] for r in out), default=None),
            "date_max": max((r["d"] for r in out), default=None),
            "roster": _roster_dict(),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta["meta"]
