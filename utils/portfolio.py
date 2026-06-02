"""
utils/portfolio.py
Owned-position tracker + BALANCED decision engine + durable persistence.

Persistence: Google Sheets (durable, survives reboots) when
st.secrets['gcp_service_account'] (+ optional 'portfolio_sheet_key') are set;
otherwise a local JSON file (works within a container's life, wiped on reboot).
Identical interface either way, so the UI doesn't care which is active.

Decision engine (BALANCED — protect gains, trim on deterioration) reuses the
unified flow_outlook signal, the RS-trend state, and the GW2/ATR stop logic.
This is signal-based decision SUPPORT, not financial advice.
"""
import os
import json
import datetime as _dt
import pandas as pd
import numpy as np
import streamlit as st

from utils.flow_analysis import flow_outlook
from utils.rs_indicators import build_rs_df, classify_state
from utils.strategy import calc_price_indicators, calc_stops

HOLD_COLS = ["ticker", "shares", "entry_price", "entry_date", "stop", "notes"]
HIST_COLS = ["date", "ticker", "score", "decision", "price", "stop"]

DECISION_COLOR = {
    "ADD": "#00cc66", "HOLD": "#4fc3f7", "RAISE STOP": "#ffd700",
    "TRIM": "#ff9800", "REDUCE": "#ff7043", "EXIT": "#ff4444",
}


# ══════════════════════════════════════════════════════════════════════════════
# Persistence — Google Sheets primary, JSON fallback (same interface)
# ══════════════════════════════════════════════════════════════════════════════
def _json_path():
    for d in ("/mount/src", os.path.expanduser("~"), "/tmp"):
        try:
            if os.path.isdir(d) and os.access(d, os.W_OK):
                return os.path.join(d, ".stratflow_portfolio.json")
        except Exception:
            continue
    return "/tmp/.stratflow_portfolio.json"


def _sheets():
    """Return (holdings_ws, history_ws) or None if Sheets isn't configured."""
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

        def _ws(name, cols):
            try:
                return sh.worksheet(name)
            except Exception:
                w = sh.add_worksheet(title=name, rows=500, cols=max(len(cols), 6))
                w.update([cols])
                return w
        return _ws("holdings", HOLD_COLS), _ws("score_history", HIST_COLS)
    except Exception:
        return None


def storage_status() -> str:
    return "Google Sheets (durable)" if _sheets() else "Local JSON (resets on reboot)"


def _json_read():
    try:
        with open(_json_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _json_write(data):
    try:
        with open(_json_path(), "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def load_holdings() -> list:
    ws = _sheets()
    if ws:
        try:
            return [r for r in ws[0].get_all_records() if str(r.get("ticker", "")).strip()]
        except Exception:
            pass
    return _json_read().get("holdings", [])


def save_holdings(holdings: list) -> bool:
    ws = _sheets()
    if ws:
        try:
            w = ws[0]
            w.clear()
            w.update([HOLD_COLS] + [[h.get(c, "") for c in HOLD_COLS] for h in holdings])
            return True
        except Exception:
            pass
    data = _json_read()
    data["holdings"] = holdings
    return _json_write(data)


def append_scores(snaps: list) -> bool:
    """snaps: list of dicts with HIST_COLS keys. Dedupe by (date, ticker)."""
    if not snaps:
        return True
    ws = _sheets()
    if ws:
        try:
            w = ws[1]
            seen = {(str(r["date"]), str(r["ticker"])) for r in w.get_all_records()}
            new = [[s.get(c, "") for c in HIST_COLS] for s in snaps
                   if (str(s["date"]), str(s["ticker"])) not in seen]
            if new:
                w.append_rows(new)
            return True
        except Exception:
            pass
    data = _json_read()
    hist = data.get("history", [])
    seen = {(str(r["date"]), str(r["ticker"])) for r in hist}
    for s in snaps:
        if (str(s["date"]), str(s["ticker"])) not in seen:
            hist.append(s)
    data["history"] = hist
    return _json_write(data)


def load_score_history(ticker: str = None) -> pd.DataFrame:
    ws = _sheets()
    rows = []
    if ws:
        try:
            rows = ws[1].get_all_records()
        except Exception:
            rows = []
    if not rows:
        rows = _json_read().get("history", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if ticker:
        df = df[df["ticker"] == ticker]
    return df


def export_json() -> str:
    hist = load_score_history()
    return json.dumps({"holdings": load_holdings(),
                       "history": hist.to_dict("records") if not hist.empty else []},
                      indent=2)


def import_json(text: str) -> bool:
    data = json.loads(text)
    ok = save_holdings(data.get("holdings", []))
    if data.get("history"):
        append_scores(data["history"])
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Balanced decision engine
# ══════════════════════════════════════════════════════════════════════════════
def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                     "Close": "last", "Volume": "sum"}).dropna()


def decide_position(pos: dict, daily: pd.DataFrame, bench_daily: pd.DataFrame,
                    calls=None, puts=None, spot=None, expiry=None) -> dict:
    """
    Balanced decision for one owned position. Returns decision, reasons, score,
    RS state, P&L, R-multiple, and a suggested (trailing) stop.
    """
    tk = pos.get("ticker", "?")
    try:
        entry = float(pos.get("entry_price") or 0)
    except Exception:
        entry = 0.0
    try:
        stop = float(pos.get("stop") or 0)
    except Exception:
        stop = 0.0
    if spot is None or spot <= 0:
        spot = float(daily["Close"].iloc[-1]) if not daily.empty else 0.0

    # Unified flow signal (daily + options)
    o = flow_outlook(tk, daily, calls, puts, spot=spot, expiry=expiry)
    score = o.get("score", 0.0) if o.get("ok") else 0.0
    direction = o.get("direction", "Neutral")
    regime = o.get("regime", "—")
    levels = o.get("levels", {})

    # RS-trend state (weekly vs benchmark)
    rs_state, ext = "—", None
    try:
        wk, bwk = _weekly(daily), _weekly(bench_daily)
        rs = build_rs_df(wk["Close"], bwk["Close"])
        rs_state, _, _, _, ext = classify_state(rs)
    except Exception:
        pass

    # Suggested trailing stop (weekly GW2 program stop)
    sug_stop = None
    try:
        wk = calc_price_indicators(_weekly(daily))
        sug_stop = calc_stops(wk).get("program_stop")
    except Exception:
        pass

    pnl = (spot - entry) / entry if entry > 0 else None
    rmult = ((spot - entry) / (entry - stop)) if (entry > 0 and stop > 0 and entry > stop) else None

    # ── Balanced logic: protect gains, trim on deterioration ──────────────────
    reasons, decision = [], "HOLD"
    if stop > 0 and spot <= stop:
        decision = "EXIT"; reasons.append(f"price ${spot:.2f} at/below stop ${stop:.2f}")
    elif rs_state == "Broken Trend":
        decision = "EXIT"; reasons.append("RS trend broken")
    elif score <= -3:
        decision = "REDUCE"; reasons.append(f"flow turned bearish ({score:+.1f}/10)")
    elif rs_state == "Exhaustion":
        decision = "TRIM"; reasons.append(f"RS exhaustion (ext {ext:.0f}%)" if ext is not None else "RS exhaustion")
    elif pnl is not None and pnl > 0 and score < 0:
        decision = "TRIM"; reasons.append(f"gains +{pnl*100:.0f}% but flow deteriorating ({score:+.1f})")
    elif rs_state in ("Early Leadership", "Healthy Trend", "Recovery") and score >= 5:
        decision = "ADD"; reasons.append(f"strong confluence ({score:+.1f}/10), {rs_state.lower()}")
    else:
        decision = "HOLD"; reasons.append(f"thesis intact — {rs_state}, flow {score:+.1f}/10")

    # Stop management (balanced: ratchet up to protect gains; never loosen)
    raise_stop, new_stop = False, None
    if sug_stop and pnl is not None and pnl > 0 and sug_stop > stop:
        raise_stop, new_stop = True, round(float(sug_stop), 2)
        reasons.append(f"raise stop → ${new_stop:.2f}")

    return {
        "ticker": tk, "decision": decision, "score": round(float(score), 1),
        "direction": direction, "regime": regime, "rs_state": rs_state,
        "pnl_pct": round(pnl * 100, 1) if pnl is not None else None,
        "r_multiple": round(rmult, 2) if rmult is not None else None,
        "spot": round(float(spot), 2), "stop": stop,
        "suggested_stop": new_stop, "raise_stop": raise_stop,
        "reasons": "; ".join(reasons), "levels": levels,
    }
