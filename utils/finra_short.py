"""
utils/finra_short.py
FINRA daily short-sale volume → weekly short-volume ratio per name.

FINRA publishes free, full-history consolidated daily short-sale volume files
(Reg SHO). For each symbol each day: ShortVolume, ShortExemptVolume,
TotalVolume. The signal of interest is the SHORT-VOLUME RATIO:
    short_ratio = ShortVolume / TotalVolume
the fraction of the day's volume executed short-side.

HONEST CAVEAT (carried into every use): a large share of reported short volume
is bona-fide market-maker hedging and internalised retail flow, NOT directional
bearish positioning — so the ratio is a noisy sentiment proxy. Treat as an
UNVALIDATED candidate; it must clear a standalone forward-edge event study
before any overlay test.

Consolidated file URL (immutable history, cache hard):
    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Pipe-delimited: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
(last line is a 'Grand Total' footer — dropped).

Network note: fetching ~250 files/yr is slow on first run; results are cached
(files never change). Default history is capped to keep the fetch tractable.
"""
from __future__ import annotations
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

CNMS_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"


def parse_cnms(text: str) -> pd.DataFrame:
    """Parse a CNMS short-volume file body → DataFrame indexed by Symbol with
    short_vol, short_exempt, total_vol. Drops header echo and 'Grand Total'."""
    if not text or "|" not in text:
        return pd.DataFrame(columns=["short_vol", "short_exempt", "total_vol"])
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    needed = {"Symbol", "ShortVolume", "TotalVolume"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=["short_vol", "short_exempt", "total_vol"])
    df = df[df["Symbol"].notna()]
    df = df[~df["Symbol"].str.contains("Grand Total", case=False, na=False)]
    out = pd.DataFrame({
        "short_vol": pd.to_numeric(df["ShortVolume"], errors="coerce"),
        "short_exempt": pd.to_numeric(df.get("ShortExemptVolume", 0), errors="coerce"),
        "total_vol": pd.to_numeric(df["TotalVolume"], errors="coerce"),
    })
    out.index = df["Symbol"].str.strip().str.upper()
    out = out[(out["total_vol"] > 0) & out["short_vol"].notna()]
    return out


def fetch_daily(d: date, timeout: int = 10) -> pd.DataFrame | None:
    """Download + parse one consolidated daily file. None on miss (weekend/
    holiday/404). Cached by the caller (Streamlit) since files are immutable."""
    import urllib.request
    import urllib.error
    url = CNMS_URL.format(ymd=d.strftime("%Y%m%d"))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    df = parse_cnms(text)
    return df if not df.empty else None


def daily_ratio_frame(symbols, daily: dict) -> pd.DataFrame:
    """Assemble {date: parsed_df} into a daily short_ratio frame
    (rows=dates, cols=symbols). Pure — testable without network."""
    syms = [s.upper() for s in symbols]
    rows = {}
    for d, df in sorted(daily.items()):
        if df is None or df.empty:
            continue
        sub = df.reindex(syms)
        ratio = sub["short_vol"] / sub["total_vol"]
        rows[pd.Timestamp(d)] = ratio
    if not rows:
        return pd.DataFrame(columns=syms, dtype=float)
    out = pd.DataFrame(rows).T
    out.columns = syms
    return out.sort_index()


def to_weekly(daily_ratio: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Resample a daily short_ratio frame to weekly (W-FRI). Mean over the
    week is the default (less day-noise than a single snapshot)."""
    if daily_ratio.empty:
        return daily_ratio
    g = daily_ratio.resample("W-FRI")
    return (g.mean() if how == "mean" else g.last()).dropna(how="all")


def weekly_short_ratio(symbols, start: date, end: date,
                       fetcher=fetch_daily) -> pd.DataFrame:
    """End-to-end: walk business days in [start, end], fetch each consolidated
    file, build the weekly short-ratio frame. `fetcher` is injectable for
    testing. Caller should wrap in st.cache_data and warn about fetch time."""
    daily = {}
    d = start
    one = timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:                      # Mon-Fri only
            df = fetcher(d)
            if df is not None:
                daily[d] = df
        d += one
    return to_weekly(daily_ratio_frame(symbols, daily))
