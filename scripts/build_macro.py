"""Build site/macro.json: macro/market series to overlay on the Timeline tab.

Series (UK analogues of the FedLock overlay):
  bank_rate     Bank of England official Bank Rate (%)   BoE IADB IUDBEDR (daily -> step)
  cpi_headline  CPI all-items, annual % change           ONS D7G7 / MM23 (monthly)
  cpi_core      CPI ex energy/food/alcohol/tobacco, YoY   ONS DKO8 / MM23 (monthly)
  rpi           RPI all-items, annual % change           ONS CZBH / MM23 (monthly)
  gilt_10y      10-year nominal gilt (par) yield (%)      BoE IADB IUDMNPY (daily -> monthly)

RPI matters historically as well as for comparison: the MPC's target was RPIX
(RPI excluding mortgage interest) at 2.5% until December 2003, when it moved to
CPI at 2%. RPI also still sets index-linked gilts and many regulated prices.

Bank Rate and gilt yields come from the Bank of England's Interactive Database
(IADB) CSV export; CPI comes from the ONS open-data timeseries endpoint.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from mpclock.macro.uk_macro import _fetch_series  # ONS fetcher (topic|cdid|dataset)

START = "1997-01-01"
TODAY = date.today().isoformat()
OUT = Path("site/macro.json")
UA = {"User-Agent": "Mozilla/5.0 (compatible; MPCLock/1.0; +https://github.com/jamesjliu/mpclock)"}

IADB = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"

# name: (source, key, label, unit, shape, extend_to_today)
#   source "iadb" -> BoE database series code; "ons" -> "topic|cdid|dataset"
SPEC = {
    "bank_rate":    ("iadb", "IUDBEDR",
                     "BoE Bank Rate", "%", "step", True),
    "cpi_headline": ("ons", "economy/inflationandpriceindices|d7g7|mm23",
                     "CPI headline (YoY)", "%", "line", False),
    "cpi_core":     ("ons", "economy/inflationandpriceindices|dko8|mm23",
                     "CPI core, ex energy/food/alcohol/tobacco (YoY)", "%", "line", False),
    "rpi":          ("ons", "economy/inflationandpriceindices|czbh|mm23",
                     "RPI all items (YoY)", "%", "line", False),
    "gilt_10y":     ("iadb", "IUDMNPY",
                     "10Y gilt (par) yield", "%", "line", False),
}


def iadb_series(code: str) -> pd.Series | None:
    params = {"csv.x": "yes", "Datefrom": "01/Jan/1997", "Dateto": "01/Jan/2027",
              "SeriesCodes": code, "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N"}
    try:
        r = requests.get(IADB, params=params, headers=UA, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:  # noqa: BLE001
        print(f"[iadb] failed {code}: {e}")
        return None
    if "DATE" not in df or code not in df:
        return None
    s = pd.Series(pd.to_numeric(df[code], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="%d %b %Y", errors="coerce"))
    return s[~s.index.isna()].dropna().sort_index()


def main():
    series_out = {}
    for name, (source, key, label, unit, shape, extend) in SPEC.items():
        s = iadb_series(key) if source == "iadb" else _fetch_series(key)
        if s is None or not len(s):
            print(f"[skip] {name}: no data")
            continue
        s = s.sort_index()
        anchor = s[s.index < pd.Timestamp(START)]
        anchor_val = float(anchor.iloc[-1]) if len(anchor) else None
        s = s[s.index >= pd.Timestamp(START)]
        if name == "gilt_10y":  # daily -> month-end for a clean, light line
            s = s.resample("ME").last().dropna()
        data = [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in s.items()]
        if shape == "step" and data:  # keep only rate-change points for a clean step line
            comp = [data[0]]
            for p in data[1:]:
                if p[1] != comp[-1][1]:
                    comp.append(p)
            if comp[-1] != data[-1]:
                comp.append(data[-1])
            data = comp
        if anchor_val is not None and (not data or data[0][0] > START):
            data.insert(0, [START, round(anchor_val, 3)])
        # carry a step series (Bank Rate) forward to today
        if extend and data and data[-1][0] < TODAY:
            data.append([TODAY, data[-1][1]])
        series_out[name] = {"label": label, "unit": unit, "shape": shape, "data": data}
        print(f"[ok]   {name}: {len(data)} points, {data[0][0]}..{data[-1][0]}, "
              f"last={data[-1][1]} (src={source})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"series": series_out}, ensure_ascii=False), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
