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
    """
    if len(df) < 42 or df["SMA40"].isna().all():
        return ("Insufficient Data", 0, "Need 40+ bars of data.", "", 0.0)

    last = df.dropna(subset=["SMA8", "SMA18", "SMA40"]).iloc[-1]
    prev = df.dropna(subset=["SMA8", "SMA18"]).iloc[-2]

    rs    = last["RS"]
    sma8  = last["SMA8"]
    sma18 = last["SMA18"]
    sma40 = last["SMA40"]
    ext   = last["ExtPct"]

    sma18_rising = sma18 > prev["SMA18"]
    sma8_rising  = sma8  > prev["SMA8"]

    # Check if RS crossed above SMA18 recently (within last 8 bars)
    recent = df.dropna(subset=["RS", "SMA18"]).tail(9).iloc[:-1]
    crossed_recently = (recent["RS"] < recent["SMA18"]).any()

    cfg = STATE_CONFIG  # shorthand

    # ── Decision tree (priority order) ────────────────────────────────────────

    # Broken Trend — RS below SMA18
    if rs < sma18:
        s = "Broken Trend"
        return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)

    # Exhaustion — extreme extension or rolling over from high ext
    if ext >= 15 or (ext >= 10 and not sma8_rising):
        s = "Exhaustion"
        return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)

    # Extended — above +10% band but SMA8 still rising
    if ext >= 10:
        s = "Extended"
        return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)

    # Healthy Trend — full alignment: RS > SMA8 > SMA18 > SMA40
    if rs > sma8 and sma8 > sma18 and sma18 > sma40 and sma18_rising and not crossed_recently:
        s = "Healthy Trend"
        return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)

    # Early Leadership — fresh cross, SMA8 > SMA18 rising, low extension
    if rs > sma18 and sma8 > sma18 and sma18_rising and ext < 10:
        s = "Early Leadership"
        return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)

    # Recovery — RS above SMA18 but structure not yet confirmed
    s = "Recovery"
    return (s, cfg[s]["score"], cfg[s]["description"], cfg[s]["action"], ext)


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
