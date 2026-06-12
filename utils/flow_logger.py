"""
utils/flow_logger.py
Daily options-flow snapshot logger. PURPOSE: build the historical flow
dataset that does not currently exist, so a flow-based selection tilt can
be validated in 6-12 months. Logging now costs nothing; not logging costs
research that can never be recovered.

Each snapshot row (one per ticker per date, deduped):
    date, ticker, spot, expiry, call_prem, put_prem, prem_pcr, call_pct,
    n_vol_gt_oi  (contracts with Volume > OI — new positioning proxy)

Persistence mirrors utils/portfolio.py: Google Sheets worksheet
'flow_history' in the StratFlow Portfolio spreadsheet when secrets are set,
local JSON fallback otherwise (fallback resets on reboot — Sheets strongly
recommended for this module, the whole point is durability).
"""
import os
import json
import datetime as _dt
import pandas as pd
import streamlit as st

FLOW_COLS = ["date", "ticker", "spot", "expiry", "call_prem", "put_prem",
             "prem_pcr", "call_pct", "n_vol_gt_oi"]
_JSON = "/tmp/.stratflow_flow_history.json"


def _ws():
    """Worksheet 'flow_history' in the portfolio spreadsheet, or None."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scopes)
        gc = gspread.authorize(creds)
        key = st.secrets.get("portfolio_sheet_key")
        sh = gc.open_by_key(key) if key else gc.open("StratFlow Portfolio")
        try:
            return sh.worksheet("flow_history")
        except Exception:
            w = sh.add_worksheet(title="flow_history", rows=20000,
                                 cols=len(FLOW_COLS))
            w.update([FLOW_COLS])
            return w
    except Exception:
        return None


def storage_status() -> str:
    return "Google Sheets (durable)" if _ws() else "Local JSON (resets!)"


def snapshot_ticker(tk: str) -> dict | None:
    """One ticker's flow snapshot from CBOE (nearest expiry)."""
    from utils.data_fetcher import get_options_cboe
    from utils.order_flow import premium_flow
    try:
        spot, expmap = get_options_cboe(tk)
        if not expmap:
            return None
        expiry = sorted(expmap.keys())[0]
        calls, puts = expmap[expiry]
        pf = premium_flow(calls, puts)
        n_new = 0
        for df in (calls, puts):
            try:
                n_new += int((df["volume"].fillna(0)
                              > df["openInterest"].fillna(0)).sum())
            except Exception:
                pass
        return {"date": str(_dt.date.today()), "ticker": tk,
                "spot": round(float(spot), 2) if spot else None,
                "expiry": str(expiry),
                "call_prem": round(pf["call_premium"], 0),
                "put_prem": round(pf["put_premium"], 0),
                "prem_pcr": pf["prem_pcr"], "call_pct": pf["call_pct"],
                "n_vol_gt_oi": n_new}
    except Exception:
        return None


def log_flow_snapshot(tickers: list, progress_cb=None) -> dict:
    """Snapshot each ticker and append (deduped by date+ticker).
    Returns {logged, skipped, storage}."""
    snaps = []
    tickers = list(dict.fromkeys(t for t in tickers if t))
    for i, tk in enumerate(tickers):
        if progress_cb:
            progress_cb((i + 1) / len(tickers), tk)
        s = snapshot_ticker(tk)
        if s:
            snaps.append(s)
    logged = _append(snaps)
    return {"logged": logged, "skipped": len(tickers) - len(snaps),
            "storage": storage_status()}


def _append(snaps: list) -> int:
    if not snaps:
        return 0
    ws = _ws()
    if ws:
        try:
            seen = {(str(r["date"]), str(r["ticker"]))
                    for r in ws.get_all_records()}
            new = [[s.get(c, "") for c in FLOW_COLS] for s in snaps
                   if (s["date"], s["ticker"]) not in seen]
            if new:
                ws.append_rows(new)
            return len(new)
        except Exception:
            pass
    try:
        hist = json.load(open(_JSON)) if os.path.exists(_JSON) else []
    except Exception:
        hist = []
    seen = {(str(r["date"]), str(r["ticker"])) for r in hist}
    new = [s for s in snaps if (s["date"], s["ticker"]) not in seen]
    hist.extend(new)
    try:
        json.dump(hist, open(_JSON, "w"))
    except Exception:
        return 0
    return len(new)


def load_flow_history() -> pd.DataFrame:
    ws = _ws()
    rows = []
    if ws:
        try:
            rows = ws.get_all_records()
        except Exception:
            rows = []
    if not rows:
        try:
            rows = json.load(open(_JSON)) if os.path.exists(_JSON) else []
        except Exception:
            rows = []
    return pd.DataFrame(rows)
