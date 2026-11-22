"""Key UK data against the Bank's own expectations, for the site's "Data vs MPR" tab.

For CPI, services CPI, private-sector regular pay and unemployment: the ONS outturns,
each Report's Bank staff near-term projections, and every release since 2015 set
against the projection for that month in the latest Report published before it.
Everything comes from data/processed/forecast_inputs.json (ONS + MPR chart data).
"""
from __future__ import annotations

import re

from .inputs import release_date

SERIES = {
    "cpi": {"label": "CPI inflation", "unit": "% y/y", "source": "ONS D7G7", "scale": 0.2, "sign": 1},
    "services": {"label": "Services CPI inflation", "unit": "% y/y", "source": "ONS D7NN", "scale": 0.3, "sign": 1},
    "pay": {"label": "Private-sector regular pay growth", "unit": "% 3m y/y", "source": "ONS KAJ4", "scale": 0.3, "sign": 1},
    "unemp": {"label": "Unemployment rate", "unit": "%, 3 months to", "source": "ONS MGSX", "scale": 0.15, "sign": -1},
}


def _month(p: str) -> str:
    m = re.match(r"(\d{4})Q([1-4])", p)
    return f"{m.group(1)}-{3 * int(m.group(2)):02d}" if m else p


def build(inputs: dict, since: str = "2015-01") -> dict:
    out = inputs.get("outturns", {})
    mpr = inputs.get("mpr", {})
    dates = inputs.get("mpr_dates", {})
    reports = sorted((d, t) for t, d in dates.items() if t in mpr)
    res = {"series": {}, "reports": [{"tag": t, "date": d} for d, t in reports]}
    for k, meta in SERIES.items():
        obs = {p: v for p, v in sorted(out.get(k, {}).items()) if p >= since}
        projs = []
        for d, t in reports:
            pts = {_month(p): v for p, v in mpr[t].get(k, {}).items()}
            if pts:
                projs.append({"mpr": t, "date": d, "points": sorted(pts.items())})
        rel = []
        for p, a in obs.items():
            r = release_date(k, p)
            src = next(((d, t) for d, t in reversed(reports)
                        if d < r and p in {_month(q) for q in mpr[t].get(k, {})}), None)
            if not src:
                continue
            e = {_month(q): v for q, v in mpr[src[1]].get(k, {}).items()}[p]
            rel.append({"period": p, "released": r, "actual": a, "expected": round(e, 3), "mpr": src[1],
                        "diff": round(a - e, 3), "z": round(meta["sign"] * (a - e) / meta["scale"], 2)})
        res["series"][k] = {**meta, "outturns": sorted(obs.items()), "projections": projs, "releases": rel}
    return res
