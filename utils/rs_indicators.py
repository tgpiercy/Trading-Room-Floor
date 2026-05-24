"""
utils/rs_indicators.py
RS Trend v1.8 implementation.
Relative Strength ratio analysis vs a benchmark — state classification,
scoring, extension bands, and screener batch processing.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
STATE_CONFIG = {
    "Early Leadership": {
        "score": 2,  "emoji": "🚀", "color": "#00cc66",
        "description": "RS recently crossed W18; W8 > W18; low extension; emerging leadership",
        "action":      "Pilot entries; watch pullbacks; build carefully",
    },
    "Healthy Trend": {
        "score": 1,  "emoji": "✅", "color": "#00ff88",
        "description": "RS > W8/W18/W40; W8 > W18; W18 > W40; healthy structure",
        "action":      "Hold positions; add on pullbacks; normal management",
    },
    "Recovery": {
        "score": 0,  "emoji": "🔄", "color": "#ffd700",
        "description": "RS reclaimed W18; W8 turning higher; momentum improving",
        "action":      "Build watchlist; small pilots; wait for confirmation",
    },
    "Extended": {
        "score": -1, "emoji": "⚠️", "color": "#ff8c00",
        "description": "RS > +10% band; leadership extended; avoid chasing",
        "action":      "Avoid chasing; tighten management; consider partial profits",
    },
    "Exhaustion": {
        "score": -2, "emoji": "🔥", "color": "#ff4444",
        "description": "RS near extreme bands; momentum stretched; rollover risk",
        "action":      "Take profits; stop adding; use tighter stops",
    },
    "Broken Trend": {
        "score": -3, "emoji": "❌", "color": "#cc2222",
        "description": "RS below key averages; W18 structure weak; leadership deteriorating",
        "action":      "Exit weak positions; reduce exposure; focus on stronger leadership",
    },
}

EXT_BANDS = [
    (5,  "#4fc3f7", "5%"),
    (10, "#ef5350", "10%"),
    (15, "#ff8c00", "15%"),
    (20, "#ffd700", "20%"),
]


# ── Core Math ─────────────────────────────────────────────────────────────────

def build_rs_df(ticker_close: pd.Series, bench_close: pd.Series) -> pd.DataFrame:
    """
    Align ticker and benchmark closes, calculate RS ratio and all
    derived indicators. Returns a DataFrame indexed by date.

    Scoring / state classification uses the raw RS ratio + SMA8/18/40.

    Extension bands per spec: rsSma18 * (1 + pct/100) — dynamic, float with SMA18.

    Mansfield RS added as DISPLAY-ONLY comparison column — not used in scoring.
        Mansfield = (RS / SMA52_of_RS) - 1
    Zero-centred, cross-ticker comparable.
    """
    df = pd.DataFrame({"ticker": ticker_close, "bench": bench_close}).dropna()

    # ── Core RS & smoothed averages (per spec) ────────────────────────────────
    df["RS"]    = df["ticker"] / df["bench"]
    df["SMA8"]  = df["RS"].rolling(8).mean()
    df["SMA18"] = df["RS"].rolling(18).mean()
    df["SMA40"] = df["RS"].rolling(40).mean()

    # ── Extension bands per spec: rsSma18 * (1 + pct/100) ────────────────────
    df["ext5"]  = df["SMA18"] * (1 + 5  / 100)
    df["ext10"] = df["SMA18"] * (1 + 10 / 100)
    df["ext15"] = df["SMA18"] * (1 + 15 / 100)
    df["ext20"] = df["SMA18"] * (1 + 20 / 100)

    # Extension % above SMA18 baseline
    df["ExtPct"] = ((df["RS"] - df["SMA18"]) / df["SMA18"]) * 100

    # ── Mansfield RS — display/comparison only, not used in state scoring ─────
    # Measures how far RS ratio sits above/below its own 52-bar rolling mean.
    # Positive = outperforming long-term RS trend; negative = underperforming.
    # e.g. +0.12 means RS is 12% above its 52-bar average.
    sma52                 = df["RS"].rolling(52).mean()
    df["Mansfield"]       = (df["RS"] / sma52) - 1
    df["Mansfield_SMA18"] = df["Mansfield"].rolling(18).mean()   # smoothed trend line

    return df


def classify_state(df: pd.DataFrame) -> tuple:
    """
    Classify the most recent bar into an RS state.
    Returns (state_name, score, description, action, ext_pct).

    Logic matches Pine Script RS Trend v1.8 exactly.
    Priority cascade: climax > extended > healthyTrend >
                      earlyLeadership > setup > failure > neutral(0)
    """
    if len(df) < 42 or df["SMA40"].isna().all():
        return ("Insufficient Data", 0, "Need 40+ bars of data.", "", 0.0)

    clean = df.dropna(subset=["SMA8", "SMA18", "SMA40"])
    if len(clean) < 7:
        return ("Insufficient Data", 0, "Need more data.", "", 0.0)

    last    = clean.iloc[-1]
    prev    = clean.iloc[-2]
    bar5ago = clean.iloc[-6]   # [5] lookback in Pine Script

    rs    = last["RS"]
    sma8  = last["SMA8"]
    sma18 = last["SMA18"]
    sma40 = last["SMA40"]
    ext   = last["ExtPct"]

    # ── Structure flags — mirrors Pine Script variable names ──────────────────
    w8_above_18  = sma8  > sma18
    w18_above_40 = sma18 > sma40
    rs_above_8   = rs    > sma8
    rs_above_18  = rs    > sma18
    rs_above_40  = rs    > sma40
    w18_rising   = sma18 > prev["SMA18"]
    w8_rising    = sma8  > prev["SMA8"]

    # 5-bar lookback: was SMA18 above SMA40 five bars ago?
    # Used to confirm earlyLeadership is a FRESH cross, not established trend
    w18_above_40_5ago = bar5ago["SMA18"] > bar5ago["SMA40"]

    # ── State conditions — exact match to Pine Script ─────────────────────────

    # earlyLeadership: fresh cross — SMA18 must NOT have been above SMA40 5 bars ago
    early_leadership = (
        rs_above_18 and
        w8_above_18 and
        w18_rising  and
        ext < 5.0   and
        not w18_above_40_5ago
    )

    # healthyTrend: full bull stack + extension under 10%
    healthy_trend = (
        rs_above_8   and
        rs_above_18  and
        rs_above_40  and
        w8_above_18  and
        w18_above_40 and
        w18_rising   and
        ext < 10.0
    )

    # extended: between +10% and +20% bands, RS still above SMA18
    extended = (
        ext >= 10.0  and
        ext <  20.0  and
        rs_above_18
    )

    # climax (Exhaustion): >= +20% OR >= +15% while rolling over below SMA8
    climax = (
        ext >= 20.0 or
        (ext >= 15.0 and rs < sma8)
    )

    # failure (Broken Trend): SMA18 structurally below SMA40,
    # OR RS below SMA18 with SMA18 falling
    failure = (
        sma18 < sma40 or
        (rs < sma18 and not w18_rising)
    )

    # setup (Recovery): RS reclaimed SMA18, SMA8 turning up,
    # but SMA18 still below SMA40 (structure not yet repaired)
    setup = (
        rs_above_18 and
        w8_rising   and
        sma18 < sma40
    )

    # ── Score cascade — priority order matches Pine Script ────────────────────
    if   climax:           score = -2
    elif extended:         score = -1
    elif healthy_trend:    score =  1
    elif early_leadership: score =  2
    elif setup:            score =  0
    elif failure:          score = -3
    else:                  score =  0   # neutral / unclassified

    # Map score back to state name and config
    score_to_state = {
         2: "Early Leadership",
         1: "Healthy Trend",
         0: "Recovery",
        -1: "Extended",
        -2: "Exhaustion",
        -3: "Broken Trend",
    }
    state = score_to_state.get(score, "Recovery")
    cfg   = STATE_CONFIG.get(state, {})
    return (state, score, cfg.get("description", ""), cfg.get("action", ""), ext)


# ── Batch Screener ────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def run_screener(
    watchlist:  tuple,          # tuple so it's hashable for cache
    benchmark:  str  = "SPY",
    period:     str  = "1y",
    interval:   str  = "1wk",
) -> pd.DataFrame:
    """
    Download all tickers + benchmark in one batch call, calculate RS
    indicators for each, classify state, return results DataFrame.
    """
    tickers = list(watchlist) + [benchmark]
    tickers = list(dict.fromkeys(tickers))          # dedupe, preserve order

    try:
        raw = yf.download(
            tickers, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()

    if benchmark not in closes.columns:
        st.error(f"Benchmark {benchmark} data unavailable.")
        return pd.DataFrame()

    bench_close = closes[benchmark].dropna()
    rows = []

    for t in watchlist:
        if t == benchmark or t not in closes.columns:
            continue
        try:
            tk_close = closes[t].dropna()
            df_rs    = build_rs_df(tk_close, bench_close)
            state, score, desc, action, ext = classify_state(df_rs)

            last_price = float(tk_close.iloc[-1])
            last_rs    = df_rs["RS"].iloc[-1]   if not df_rs.empty else None
            last_sma18 = df_rs["SMA18"].iloc[-1] if not df_rs.empty else None

            # 4-week RS momentum (RS change over 4 bars)
            rs_mom = None
            if len(df_rs) >= 5:
                rs_4w = df_rs["RS"].iloc[-5]
                rs_mom = ((last_rs - rs_4w) / rs_4w * 100) if rs_4w else None

            rows.append({
                "Ticker":    t,
                "Price":     round(last_price, 2),
                "RS Score":  score,
                "RS State":  state,
                "Ext %":     round(ext, 1),
                "RS Mom 4W": round(rs_mom, 1) if rs_mom is not None else None,
                "Action":    action,
                "_emoji":    STATE_CONFIG.get(state, {}).get("emoji", ""),
                "_color":    STATE_CONFIG.get(state, {}).get("color", "#888"),
                "_df_rs":    df_rs,        # kept for drill-down (dropped before display)
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(
        ["RS Score", "Ext %"], ascending=[False, True]
    )
    return result
