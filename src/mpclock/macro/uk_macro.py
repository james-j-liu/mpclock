"""UK macro context for the judge (analogue of FedLock's PCE/unemployment/GDP/VIX).

Series come from the ONS "timeseries" open-data endpoint (the old api.ons.gov.uk
was retired in Nov 2024; the live JSON now lives under www.ons.gov.uk). Each series
is addressed by a "{topic}|{cdid}|{dataset}" key. For each speech date we report the
most recent observation on or before that date, so the judge sees only information
available at the time.

Series:
  - core_cpi     : CPI excl. energy, food, alcohol & tobacco, annual % change (~Core PCE)
  - unemployment : LFS unemployment rate, aged 16+, %                         (~UNRATE)
  - gdp_growth   : real GDP, year-on-year % change                           (~GDP growth)
  - vix          : (optional) equity implied vol; left unset by default        (~VIX)

If a series can't be fetched it degrades to "n/a" rather than failing.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from ..config import RAW, cfg

ONS_BASE = "https://www.ons.gov.uk"
UA = {"User-Agent": "Mozilla/5.0 (compatible; MPCLock/1.0; +https://github.com/jamesjliu/mpclock)"}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _parse_key(series_key: str) -> tuple[str, str, str]:
    # "economy/inflationandpriceindices|dko8|mm23"
    topic, cdid, dataset = series_key.split("|")
    return topic, cdid, dataset


def _period_to_ts(row: dict) -> pd.Timestamp | None:
    """ONS rows carry year / month / quarter fields; build a timestamp."""
    y = row.get("year") or ""
    if not y:
        return None
    mon = (row.get("month") or "").strip().upper()[:3]
    q = (row.get("quarter") or "").strip().upper()
    if mon in _MONTHS:
        return pd.Timestamp(year=int(y), month=_MONTHS[mon], day=1)
    if q.startswith("Q"):
        return pd.Timestamp(year=int(y), month=(int(q[1]) - 1) * 3 + 1, day=1)
    return pd.Timestamp(year=int(y), month=1, day=1)


def _fetch_series(series_key: str) -> pd.Series | None:
    topic, cdid, dataset = _parse_key(series_key)
    cache = RAW / f"ons_{cdid}_{dataset}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        url = f"{ONS_BASE}/{topic}/timeseries/{cdid}/{dataset}/data"
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            print(f"[macro] failed to fetch {cdid}/{dataset}: {e}")
            return None
        rows = j.get("months") or j.get("quarters") or j.get("years") or []
        recs = []
        for row in rows:
            ts = _period_to_ts(row)
            try:
                v = float(row.get("value"))
            except (TypeError, ValueError):
                continue
            if ts is not None:
                recs.append((ts, v))
        if not recs:
            return None
        df = pd.DataFrame(recs, columns=["date", "value"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]))
    return s[~s.index.isna()].dropna().sort_index()


class MacroContext:
    def __init__(self):
        series_cfg = cfg()["macro"]["series"]
        self.series: dict[str, pd.Series] = {}
        for name, key in series_cfg.items():
            if not key:
                continue
            s = _fetch_series(key)
            if s is not None and len(s):
                self.series[name] = s

    def as_of(self, d: str) -> dict[str, float | None]:
        ts = pd.Timestamp(d)
        out: dict[str, float | None] = {}
        for name, s in self.series.items():
            prior = s[s.index <= ts]
            out[name] = float(prior.iloc[-1]) if len(prior) else None
        return out

    def string(self, d: str) -> str:
        v = self.as_of(d)
        labels = {
            "core_cpi": "Core CPI infl",
            "unemployment": "Unemployment",
            "gdp_growth": "GDP growth (YoY)",
            "vix": "Equity vol (VIX)",
        }
        bits = []
        for k, lab in labels.items():
            if v.get(k) is not None:
                suffix = "%" if k != "vix" else ""
                bits.append(f"{lab}: {v[k]:.1f}{suffix}")
        return "; ".join(bits) if bits else "n/a"


if __name__ == "__main__":
    mc = MacroContext()
    print("Loaded series:", list(mc.series))
    for d in ("2008-09-15", "2011-07-01", "2016-08-04", "2022-08-04", "2024-06-06"):
        print(d, "->", mc.string(d))
