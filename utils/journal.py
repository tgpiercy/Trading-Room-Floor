"""
utils/journal.py
The TRADE JOURNAL — durable capture of every decision the system makes,
with the full reason matrix per row: which gate decided, the quantitative
inputs, and a plain-language reason. The point: six months from now you can
reconstruct exactly why any position was entered, held, sized, or released
— and audit the system's live behavior against its validation.

One row per (date, ticker, decision); a REGIME summary row per run.

Decision vocabulary (the gate that decided):
  ENTER          selector  — composite rank ≤ N, passed redundancy
  HELD           selector  — already owned and still rank ≤ N
  HOLD-BAND      band      — owned, rank 11..30, decay unconfirmed
  RELEASE-DECAY  layer 2   — rank > 30 for 2+ consecutive weeks
  EXIT-TRAIL     layer 3   — close below 4×ATR chandelier
  SKIP-REDUNDANT filter    — >0.85 corr with an accepted stronger name
  REGIME         layer 1   — run-level exposure context row

Persistence mirrors flow_logger: 'trade_journal' worksheet in the StratFlow
Portfolio spreadsheet when secrets are configured; local JSON fallback
(resets on reboot — Sheets recommended).
"""
import os
import json
import pandas as pd
import streamlit as st

JOURNAL_COLS = ["date", "ticker", "decision", "gate", "rank", "mom_pct",
                "extadj_pct", "band", "weeks_breach", "blocked_by", "corr",
                "price", "stop", "weight", "exposure", "reason"]
_JSON = "/tmp/.stratflow_trade_journal.json"


def _ws():
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
            return sh.worksheet("trade_journal")
        except Exception:
            w = sh.add_worksheet(title="trade_journal", rows=20000,
                                 cols=len(JOURNAL_COLS))
            w.update([JOURNAL_COLS])
            return w
    except Exception:
        return None


def storage_status() -> str:
    return "Google Sheets (durable)" if _ws() else "Local JSON (resets!)"


def log_decisions(rows: list) -> int:
    """Append decision rows (deduped by date+ticker+decision).
    Each row: dict with JOURNAL_COLS keys (missing keys → blank)."""
    if not rows:
        return 0
    ws = _ws()
    keyf = lambda r: (str(r.get("date")), str(r.get("ticker")),
                      str(r.get("decision")))
    if ws:
        try:
            seen = {keyf(r) for r in ws.get_all_records()}
            new = [[r.get(c, "") for c in JOURNAL_COLS] for r in rows
                   if keyf(r) not in seen]
            if new:
                ws.append_rows(new)
            return len(new)
        except Exception:
            pass
    try:
        hist = json.load(open(_JSON)) if os.path.exists(_JSON) else []
    except Exception:
        hist = []
    seen = {keyf(r) for r in hist}
    new = [r for r in rows if keyf(r) not in seen]
    hist.extend(new)
    try:
        json.dump(hist, open(_JSON, "w"), default=str)
    except Exception:
        return 0
    return len(new)


def load_journal(ticker: str = None, last_n_days: int = None) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if ticker:
        df = df[df["ticker"].astype(str) == ticker]
    if last_n_days:
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df[d >= (pd.Timestamp.now() - pd.Timedelta(days=last_n_days))]
    return df
