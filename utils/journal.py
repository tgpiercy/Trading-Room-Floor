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

# Bump SYSTEM_VERSION whenever any frozen constant changes (selector, exit
# stack, risk, regime). Every journal row is stamped, so history segments
# cleanly by configuration era.
SYSTEM_VERSION = "SF-1.0 (composite_v1 + band30/2wk + trail4.0 + regime)"

JOURNAL_COLS = ["date", "ticker", "decision", "gate", "sleeve", "rank", "mom_pct",
                "extadj_pct", "band", "weeks_breach", "blocked_by", "corr",
                "price", "stop", "weight", "exposure", "reason",
                "system_version"]
EQUITY_COLS = ["date", "holdings_mtm", "n_positions", "exposure",
               "system_version", "note"]
_JSON = "/tmp/.stratflow_trade_journal.json"
_JSON_EQ = "/tmp/.stratflow_equity_curve.json"


def _open_sheet():
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
        return gc.open_by_key(key) if key else gc.open("StratFlow Portfolio")
    except Exception:
        return None


def _get_ws(name: str, cols: list):
    """Worksheet by name; creates it if missing and keeps the header row in
    sync with the current schema (older rows simply lack newer columns)."""
    sh = _open_sheet()
    if sh is None:
        return None
    try:
        try:
            w = sh.worksheet(name)
        except Exception:
            w = sh.add_worksheet(title=name, rows=20000, cols=len(cols))
            w.update([cols])
            return w
        try:
            header = w.row_values(1)
            if header != cols:
                w.update([cols])
        except Exception:
            pass
        return w
    except Exception:
        return None


def _ws():
    return _get_ws("trade_journal", JOURNAL_COLS)


def storage_status() -> str:
    return "Google Sheets (durable)" if _ws() else "Local JSON (resets!)"


def log_decisions(rows: list) -> int:
    """Append decision rows (deduped by date+ticker+decision).
    Each row: dict with JOURNAL_COLS keys (missing keys → blank)."""
    if not rows:
        return 0
    for r in rows:
        r.setdefault("system_version", SYSTEM_VERSION)
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


# ── Equity curve (live mark-to-market of the tracked book) ───────────────────
def log_equity(holdings_mtm: float, n_positions: int, exposure=None,
               note: str = "") -> bool:
    """One snapshot per date (dedup). The realized-performance series the
    live audit runs on."""
    import datetime as _dt
    row = {"date": str(_dt.date.today()),
           "holdings_mtm": round(float(holdings_mtm), 2),
           "n_positions": int(n_positions),
           "exposure": exposure, "system_version": SYSTEM_VERSION,
           "note": note}
    ws = _get_ws("equity_curve", EQUITY_COLS)
    if ws:
        try:
            seen = {str(r["date"]) for r in ws.get_all_records()}
            if row["date"] in seen:
                return False
            ws.append_rows([[row.get(c, "") for c in EQUITY_COLS]])
            return True
        except Exception:
            pass
    try:
        hist = json.load(open(_JSON_EQ)) if os.path.exists(_JSON_EQ) else []
    except Exception:
        hist = []
    if any(str(r["date"]) == row["date"] for r in hist):
        return False
    hist.append(row)
    try:
        json.dump(hist, open(_JSON_EQ, "w"), default=str)
        return True
    except Exception:
        return False


def load_equity() -> pd.DataFrame:
    ws = _get_ws("equity_curve", EQUITY_COLS)
    rows = []
    if ws:
        try:
            rows = ws.get_all_records()
        except Exception:
            rows = []
    if not rows:
        try:
            rows = json.load(open(_JSON_EQ)) if os.path.exists(_JSON_EQ) else []
        except Exception:
            rows = []
    return pd.DataFrame(rows)
