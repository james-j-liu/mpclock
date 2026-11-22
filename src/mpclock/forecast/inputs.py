"""External inputs to the vote forecast: outturns, MPR expectations, markets.

Everything is cached in data/processed/forecast_inputs.json so the forecast and its
backtest are reproducible and the daily run only refreshes what moves.

  outturns   ONS monthly series, each point stamped with an (approximate) release
             date so the backtest only ever sees what had been published:
               cpi      D7G7  CPI 12-month rate                 (published ~3rd Wed of m+1)
               services D7NN  services CPI 12-month rate         (same release)
               pay      KAJ4  private sector regular AWE, 3m yoy (published ~mid m+2)
               unemp    MGSX  LFS unemployment rate, 3m          (same release)
  mpr        Bank staff near-term projections from each MPR's chart data
             (mpr_nearterm.py), keyed by the MPR's publication date
  markets    daily: Brent (FRED DCOILBRENTEU, Yahoo BZ=F for the latest days),
             TTF gas (Yahoo TTF=F), sterling ERI (BoE IADB XUDLBK67),
             Fed funds target upper bound (FRED DFEDTARU), ECB deposit rate (FRED ECBDFR)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/processed/forecast_inputs.json"
MPR_DIR_CACHE = ROOT / "data/raw/mpr_chartdata"     # downloaded spreadsheets (gitignored)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124 Safari/537.36 MPCLock/1.0"}

ONS = {
    "cpi": ("economy/inflationandpriceindices", "d7g7", "mm23"),
    "services": ("economy/inflationandpriceindices", "d7nn", "mm23"),
    "pay": ("employmentandlabourmarket/peopleinwork/earningsandworkinghours", "kaj4", "lms"),
    "unemp": ("employmentandlabourmarket/peoplenotinwork/unemployment", "mgsx", "lms"),
}
_MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


# hosts are particular about user agents: FRED refuses browser-like ones, Yahoo
# rate-limits anything that does not start "Mozilla/5.0"
_HOST_UA = {"fred.stlouisfed.org": "curl/8.5.0", "query1.finance.yahoo.com": "Mozilla/5.0"}


def _get(url, params=None, tries=4, timeout=90):
    last = None
    host = re.sub(r"^https?://([^/]+).*", r"\1", url)
    headers = {"User-Agent": _HOST_UA.get(host, UA["User-Agent"])}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:120]
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{url}: {last}")


def release_date(series: str, period: str) -> str:
    """Approximate publication date of an observation. CPI for month m: the
    Wednesday in the third week of m+1. Labour market (3 months to m): the Tuesday
    in the middle of m+2. Approximations err a few days late, never early."""
    y, m = int(period[:4]), int(period[5:7])
    lag = 1 if series in ("cpi", "services") else 2
    m2, y2 = m + lag, y
    while m2 > 12:
        m2, y2 = m2 - 12, y2 + 1
    d = dt.date(y2, m2, 15)
    want = 2 if series in ("cpi", "services") else 1          # Wed / Tue
    while d.weekday() != want:
        d += dt.timedelta(days=1)
    return d.isoformat()


def fetch_ons() -> dict:
    out = {}
    for key, (topic, cdid, ds) in ONS.items():
        j = _get(f"https://www.ons.gov.uk/{topic}/timeseries/{cdid}/{ds}/data").json()
        pts = {}
        for row in j.get("months", []):
            m = re.match(r"(\d{4}) ([A-Z]{3})", row["date"])
            try:
                v = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not m:
                continue
            y, mo = int(m.group(1)), _MON[m.group(2)]
            if key == "unemp":            # ONS labels a rolling quarter by its middle month
                mo += 1
                if mo > 12:
                    mo, y = 1, y + 1
            if y >= 2015:
                pts[f"{y}-{mo:02d}"] = v
        out[key] = pts
    return out


def _fred(sid: str) -> dict:
    df = pd.read_csv(io.StringIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                                      {"id": sid}).text))
    df.columns = ["d", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    df = df[df.d >= "2015-01-01"]
    return dict(zip(df.d, df.v.round(4)))


def _yahoo(ticker: str) -> dict:
    j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
             {"range": "10y", "interval": "1d"}).json()["chart"]["result"][0]
    out = {}
    for t, c in zip(j["timestamp"], j["indicators"]["quote"][0]["close"]):
        if c is not None:
            out[dt.datetime.utcfromtimestamp(t).date().isoformat()] = round(float(c), 4)
    return out


def _iadb(code: str) -> dict:
    params = {"csv.x": "yes", "Datefrom": "01/Jan/2015", "Dateto": "31/Dec/2030",
              "SeriesCodes": code, "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N"}
    df = pd.read_csv(io.StringIO(_get(
        "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp", params).text))
    df["d"] = pd.to_datetime(df["DATE"], format="%d %b %Y", errors="coerce")
    df = df.dropna(subset=["d"])
    return {d.date().isoformat(): float(v) for d, v in zip(df.d, pd.to_numeric(df[code], errors="coerce"))
            if pd.notna(v)}


def fetch_markets(prev: dict | None = None) -> dict:
    prev = prev or {}
    out = {}
    jobs = {
        "brent": lambda: {**_fred("DCOILBRENTEU"), **_safe(lambda: _yahoo("BZ=F"))},
        "gas_ttf": lambda: _yahoo("TTF=F"),
        "gbp_eri": lambda: _iadb("XUDLBK67"),
        "fed": lambda: _fred("DFEDTARU"),
        "ecb": lambda: _fred("ECBDFR"),
    }
    for k, fn in jobs.items():
        try:
            out[k] = fn()
        except Exception as e:  # noqa: BLE001 - keep yesterday's copy rather than nothing
            print(f"[inputs] {k} failed ({e}); keeping cached copy")
            out[k] = prev.get(k, {})
    return out


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return {}


# --- MPR near-term projections ---------------------------------------------------
_MPR_MONTHS = "january february march april may june july august september october november december".split()


def mpr_zip_url(page_html: str) -> str | None:
    m = re.search(r'href="([^"]*chart[^"]*\.zip)"', page_html, re.I)
    if not m:
        return None
    u = m.group(1)
    return u if u.startswith("http") else "https://www.bankofengland.co.uk" + u


def fetch_mpr_chartdata(reports: list[tuple[str, str]], have: set[str]) -> list[str]:
    """Download the chart-data spreadsheets of MPRs we have not read yet.
    `reports` is [(publication date, report page url)]. Returns the tags fetched."""
    import zipfile
    got = []
    for date, url in reports:
        y, mo = int(date[:4]), int(date[5:7])
        tag = _mpr_tag(url, date)
        if tag in have:
            continue
        try:
            page = _get(url).text
            z = mpr_zip_url(page)
            if not z:
                continue
            blob = _get(z, timeout=300).content
        except Exception as e:  # noqa: BLE001
            print(f"[inputs] MPR {tag}: {e}")
            continue
        d = MPR_DIR_CACHE / tag
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for n in zf.namelist():
                if re.search(r"(?i)\.xls[xm]?$", n) and not re.search(
                        r"(?i)fan chart|parameters|other forecast|projected level|visual summary", n):
                    (d / Path(n).name).write_bytes(zf.read(n))
        got.append(tag)
    return got


def _mpr_tag(url: str, date: str) -> str:
    m = re.search(r"/(\d{4})/([a-z]+)-\d{4}", url)
    if m:
        return f"mpr-{m.group(2)}-{m.group(1)}"
    return f"mpr-{_MPR_MONTHS[int(date[5:7]) - 1]}-{date[:4]}"


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(c: dict) -> None:
    CACHE.write_text(json.dumps(c, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def refresh(corpus=None, mpr_dir: Path | None = None) -> dict:
    """Refresh outturns, markets and MPR projections; returns the cache."""
    from .mpr_nearterm import main as read_mprs
    c = load_cache()
    try:
        c["outturns"] = fetch_ons()
    except Exception as e:  # noqa: BLE001
        print(f"[inputs] ONS failed ({e}); keeping cached outturns")
    c["markets"] = fetch_markets(c.get("markets"))
    mpr = c.setdefault("mpr", {})
    mpr_dates = c.setdefault("mpr_dates", {})
    if corpus is not None:
        # every Report since 2015 in the corpus, by publication month; the page that
        # links its chart data follows from the month (Inflation Report before Nov 2019)
        reports = []
        for s_ in corpus:
            if s_.source_type != "mp_report" or s_.date < "2015-01-01":
                continue
            y, mo = int(s_.date[:4]), int(s_.date[5:7])
            # a Report decided on 31 Jan is published (and filed) as February's
            cands = [(y, mo)] + ([(y + (mo == 12), mo % 12 + 1)] if int(s_.date[8:10]) >= 25 else [])
            have_dirs = {p.name for p in MPR_DIR_CACHE.iterdir()} if MPR_DIR_CACHE.exists() else set()
            yy, mm = next(((a, b) for a, b in cands if f"mpr-{_MPR_MONTHS[b - 1]}-{a}" in have_dirs),
                          cands[-1])
            y, month = yy, _MPR_MONTHS[mm - 1]
            base = "inflation-report" if s_.date < "2019-11-01" else "monetary-policy-report"
            tag = f"mpr-{month}-{y}"
            mpr_dates[tag] = min(mpr_dates.get(tag, s_.date), s_.date)
            reports.append((s_.date, f"https://www.bankofengland.co.uk/{base}/{y}/{month}-{y}"))
        have = {p.name for p in MPR_DIR_CACHE.iterdir()} if MPR_DIR_CACHE.exists() else set()
        fetch_mpr_chartdata(sorted(set(reports)), have | set(mpr))
    d = mpr_dir or MPR_DIR_CACHE
    if d.exists():
        for tag, ser in read_mprs(str(d), verbose=False).items():
            if ser:
                mpr[tag] = ser
    c["updated"] = dt.date.today().isoformat()
    save_cache(c)
    return c
