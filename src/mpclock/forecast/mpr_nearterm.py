"""Bank staff near-term projections, read from each Monetary Policy Report's chart data.

Every MPR publishes the spreadsheets behind its charts. The near-term charts carry
staff projections for the coming months as extra rows after the outturns, labelled
e.g. "Oct 2026 (Bank staff projections)" or "2023 Q4\\n(Bank staff projection)", in
the same column as the data. Those rows are the Bank's own expectations for the data
that will be published before the next meetings — the benchmark against which a
"surprise" is measured.

Series read (where the Report charts them):
  cpi        headline CPI inflation, 12-month, monthly
  services   services CPI inflation, 12-month, monthly
  pay        private sector regular pay growth (AWE), 3m yoy, monthly/quarterly
  unemp      LFS unemployment rate, monthly/quarterly
  gdp        quarterly GDP growth, q/q

    python -m mpclock.forecast.mpr_nearterm DIR   # DIR holds one folder per MPR
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_MONTHLY = re.compile(r"^\s*([A-Za-z]{3})[a-z]*\.?\s+'?(\d{2,4})")
_QUARTERLY = re.compile(r"^\s*(\d{4})\s*Q([1-4])")
_PROJ = re.compile(r"(?i)projection|staff estimates?")   # 2015-18 Reports say "estimates"


def _period(label: str) -> str | None:
    """'Oct 2026 (Bank staff projections)' -> '2026-10'; '2023 Q4 (…)' -> '2023Q4'."""
    m = _QUARTERLY.match(label)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = _MONTHLY.match(label)
    if m and m.group(1).lower()[:3] in MONTHS:
        y = int(m.group(2))
        y = y + 2000 if y < 100 else y
        return f"{y}-{MONTHS[m.group(1).lower()[:3]]:02d}"
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# column header -> series. A header is matched after whitespace is collapsed.
_HEADERS = [
    ("cpi", re.compile(r"(?i)^(cpi( inflation)?( and .*projections?)?|inflation rate|annual cpi inflation|"
                       r"twelve-month cpi inflation)$")),
    ("services", re.compile(r"(?i)^services( cpi)?( price)?( inflation)?$")),
    ("pay", re.compile(r"(?i)^(awe )?private sector regular pay( growth)?( \(per cent\))?$")),
    ("unemp", re.compile(r"(?i)^(three-month )?(lfs )?unemployment rate$")),
]
# sheets that hold contributions, not rates: a 'Services' column there is in pp
_CONTRIB = re.compile(r"(?i)contribution|news to|relative to")


def _sheet_rows(ws):
    return [list(r) for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]


def _norm(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""


def _candidates(rows, title):
    """Every (series, {period: value}, rank) the sheet offers."""
    out = []
    hdr_rows = [i for i, r in enumerate(rows[:14])
                if any(_norm(c).lower() == "date" for c in r)]
    if not hdr_rows:
        return out
    hi = hdr_rows[0]
    hdr = [_norm(c) for c in rows[hi]]
    # a sheet carrying both contributions and component rates repeats the names:
    # the later block is the rates
    rates_block = bool(re.search(r"(?i)inflation rates?", title)) and not _CONTRIB.search(title.split("Inflation rate")[0] if "Inflation rate" in title else "")
    for j, h in enumerate(hdr):
        if j == 0 or not h:
            continue
        for series, rx in _HEADERS:
            if not rx.match(h):
                continue
            got = {}
            for r in rows[hi + 1:]:
                # the row label: the nearest cell to the left that names a projected
                # period (cells reading 'n.a.' sit between them in multi-series charts)
                lab = next((c for c in reversed(r[:j]) if isinstance(c, str) and _PROJ.search(c)), None)
                if not (lab and _PROJ.search(lab)):
                    continue
                p = _period(lab)
                v = _num(r[j]) if j < len(r) else None
                if v is None and j + 1 < len(hdr) and hdr[j + 1].lower() == "projection":
                    v = _num(r[j + 1]) if j + 1 < len(r) else None   # 2021-22 layout
                if p and v is not None:
                    got[p] = v
            if not got:
                continue
            contrib = bool(_CONTRIB.search(title))
            later_dup = sum(1 for k in hdr[:j] if k.lower() == h.lower()) > 0
            if series == "services" and contrib and not later_dup and not rates_block:
                continue          # a contribution in pp, not the services inflation rate
            if contrib and not later_dup and not (series == "cpi" and re.match(r"(?i)inflation rate|cpi inflation", h)):
                continue          # pp contributions / deviations from an earlier projection
            # keep the near term: the first six projected periods
            got = dict(sorted(got.items())[:6])
            rank = (bool(re.search(r"(?i)near-term|projection", title)), len(got))
            out.append((series, got, rank))
    return out


_LATER = re.compile(r"(?i)^(january|february|march|april|may|june|july|august|september|october|"
                    r"november|december) (\d{4}) projection$")


def earlier_projections(rows) -> dict[str, dict]:
    """A CPI chart in a later Report often re-draws earlier Reports' projections as
    columns headed 'July 2026 projection': {'mpr-july-2026': {period: value}}."""
    out: dict[str, dict] = {}
    hi = next((i for i, r in enumerate(rows[:14]) if any(_norm(c).lower() == "date" for c in r)), None)
    if hi is None:
        return out
    hdr = [_norm(c) for c in rows[hi]]
    for j, h in enumerate(hdr):
        m = _LATER.match(h)
        if not m:
            continue
        tag = f"mpr-{m.group(1).lower()}-{m.group(2)}"
        got = {}
        for r in rows[hi + 1:]:
            d = r[0]
            v = _num(r[j]) if j < len(r) else None
            if v is None:
                continue
            if hasattr(d, "year"):
                got[f"{d.year}-{d.month:02d}"] = v
            elif isinstance(d, str) and _period(d):
                got[_period(d)] = v
        if got:
            out[tag] = got
    return out


def extract(folder: Path) -> dict:
    import openpyxl
    best: dict[str, tuple] = {}
    for f in sorted(folder.iterdir()):
        if not re.search(r"(?i)\.xls[xm]?$", f.name) or re.search(r"(?i)databank|market profiles", f.name):
            continue
        try:
            wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        except Exception:  # noqa: BLE001
            continue
        for ws in wb.worksheets:
            rows = _sheet_rows(ws)
            title = " ".join(_norm(c) for r in rows[:3] for c in r if isinstance(c, str))
            if re.search(r"(?i)CPI inflation and projections", title):
                for tag, got in earlier_projections(rows).items():
                    _BACKFILL.setdefault(tag, {}).setdefault("cpi", {}).update(got)
            for series, got, rank in _candidates(rows, title):
                if series not in best or rank > best[series][1]:
                    best[series] = (got, rank)
    return {k: v[0] for k, v in best.items()}


_BACKFILL: dict[str, dict] = {}


def main(root: str, verbose: bool = True) -> dict:
    """{'mpr-november-2026': {'cpi': {'2026-10': 3.725, …}, …}, …}"""
    out = {}
    _BACKFILL.clear()
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            out[d.name] = extract(d)
    for tag, ser in _BACKFILL.items():      # fill gaps from later Reports' re-drawings
        for k, got in ser.items():
            if tag in out and not out[tag].get(k):
                out[tag][k] = dict(sorted(got.items())[:6])
    if verbose:
        for k, v in out.items():
            print(k, {s: len(x) for s, x in v.items()})
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=0)[:3000])
