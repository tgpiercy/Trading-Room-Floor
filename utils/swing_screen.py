"""
utils/swing_screen.py
Multi-factor SWING / position screen for 1-week to 1-month+ holds. Clean-sheet
design grounded in published anomalies — intermediate momentum, trend
persistence, short-term (~1wk) reversal, and low-volatility/trend-quality.

Cross-sectional percentile ranking, gated by (a) market regime, (b) a per-name
trend + liquidity filter, with pullback entry timing and ATR risk sizing.

The factor weights are informed PRIORS to be validated out-of-sample — not
curve-fit performance claims. Every output is traceable to its inputs.
"""
import numpy as np
import pandas as pd
import streamlit as st

# Momentum lookbacks (trading days) → holding-period emphasis
LOOKBACK = {"short": 63, "medium": 126, "long": 200}     # ~3mo / 6mo / ~9-10mo
DEFAULT_WEIGHTS = {"momentum": 0.60, "entry": 0.30, "lowvol": 0.10}
MIN_DOLLAR_VOL = 3_000_000     # liquidity floor: median 20-day dollar volume


# ── factor primitives ─────────────────────────────────────────────────────────
def _sma(s, n):
    return s.rolling(n).mean()


def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def clenow_momentum(close, lookback=126):
    """Annualized exponential-regression slope × R² (Clenow). Rewards trends that
    are both strong and smooth. Returns (score, annualized_return, r2)."""
    s = close.dropna().tail(lookback)
    if len(s) < max(20, lookback // 2):
        return (np.nan, np.nan, np.nan)
    y = np.log(s.values)
    x = np.arange(len(y), dtype=float)
    b, a = np.polyfit(x, y, 1)               # slope (per-day log return), intercept
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ann = float(np.exp(b * 252) - 1)         # annualized from daily log slope
    return (ann * r2, ann, r2)


def compute_factors(df, lookback=126):
    """Per-ticker factor row, or None if data is insufficient."""
    if df is None or df.empty or len(df) < 70:
        return None
    c = df["Close"].dropna()
    if len(c) < 70:
        return None
    sma20, sma50 = _sma(c, 20), _sma(c, 50)
    last = float(c.iloc[-1])

    # Trend gate (regime of the individual name)
    if len(c) >= 200:
        sma200 = _sma(c, 200)
        trend_gate = (last > sma200.iloc[-1]) and (sma50.iloc[-1] > sma200.iloc[-1])
    else:  # short-history fallback: above rising 50d
        trend_gate = (last > sma50.iloc[-1]) and (sma50.iloc[-1] > sma50.iloc[-10])

    dollar_vol = float((df["Close"] * df["Volume"]).tail(20).median()) if "Volume" in df.columns else 0.0
    mom_score, ann, r2 = clenow_momentum(c, lookback)
    if np.isnan(mom_score):
        return None
    rsi = float(_rsi(c).iloc[-1])
    ret = c.pct_change()
    vstd = ret.tail(63).std()
    vol_ann = float(vstd * np.sqrt(252)) if vstd == vstd else np.nan
    atr = float(_atr(df).iloc[-1]) if {"High", "Low"}.issubset(df.columns) else last * 0.02
    sma20v = float(sma20.iloc[-1]) if sma20.iloc[-1] == sma20.iloc[-1] else last
    # Entry timing: pulled-back (lower RSI) within the uptrend scores higher
    entry_quality = float(np.clip((70 - rsi) / 40, 0, 1))

    return {"price": last, "trend_gate": bool(trend_gate), "dollar_vol": dollar_vol,
            "mom_score": float(mom_score), "ann_ret": float(ann), "r2": float(r2),
            "rsi": rsi, "vol_ann": vol_ann, "atr": atr, "sma20": sma20v,
            "sma50": float(sma50.iloc[-1]), "entry_quality": entry_quality}


def market_regime(spy_daily):
    """Top-level risk gate from SPY's own trend."""
    if spy_daily is None or spy_daily.empty or len(spy_daily) < 200:
        return {"risk_on": True, "spy_vs_200": None,
                "label": "Unknown (insufficient SPY history) — treating as neutral"}
    c = spy_daily["Close"].dropna()
    sma200 = c.rolling(200).mean().iloc[-1]
    sma50 = c.rolling(50).mean().iloc[-1]
    last = float(c.iloc[-1])
    risk_on = (last > sma200) and (sma50 > sma200)
    return {"risk_on": bool(risk_on), "spy_vs_200": round((last / sma200 - 1) * 100, 1),
            "label": "Risk-ON — SPY above a rising 200-day" if risk_on
                     else "Risk-OFF — SPY below 200-day (50d < 200d)"}


def run_swing_screen(daily_by_sym, lookback=126, weights=None,
                     min_dollar_vol=MIN_DOLLAR_VOL, top_n=25):
    """Rank a universe. Returns a DataFrame of the top_n gated candidates."""
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for sym, df in daily_by_sym.items():
        f = compute_factors(df, lookback)
        if not f or not f["trend_gate"] or f["dollar_vol"] < min_dollar_vol:
            continue
        if f["vol_ann"] != f["vol_ann"]:   # NaN vol
            continue
        rows.append({"ticker": sym, **f})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)

    d["mom_pct"] = d["mom_score"].rank(pct=True) * 100
    d["entry_pct"] = d["entry_quality"].rank(pct=True) * 100
    d["lowvol_pct"] = (1 - d["vol_ann"].rank(pct=True)) * 100
    d["score"] = (weights["momentum"] * d["mom_pct"]
                  + weights["entry"] * d["entry_pct"]
                  + weights["lowvol"] * d["lowvol_pct"]).round(1)
    d = d.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    inv = (1 / d["vol_ann"].replace(0, np.nan))
    d["weight_pct"] = (inv / inv.sum() * 100).round(1)
    d["entry_low"] = d[["sma20", "price"]].min(axis=1).round(2)
    d["entry_high"] = d["price"].round(2)
    d["stop"] = (d["price"] - 2.5 * d["atr"]).round(2)
    d["ann_ret_pct"] = (d["ann_ret"] * 100).round(1)
    d["vol_pct"] = (d["vol_ann"] * 100).round(1)
    d["r2"] = d["r2"].round(2)
    d["rsi"] = d["rsi"].round(0)
    return d


def hold_exit_note(row):
    return (f"Hold while > 50d (${row['sma50']:.2f}) and momentum stays top-tier; "
            f"exit on a 50-day break, stop ${row['stop']:.2f}, or momentum decay.")


# ── S&P 500 universe (dynamic with static fallback) ────────────────────────────
_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "JPM",
    "V", "AVGO", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ORCL",
    "MRK", "ABBV", "CVX", "AMD", "KO", "PEP", "ADBE", "WMT", "CRM", "BAC",
    "MCD", "NFLX", "ACN", "LIN", "TMO", "ABT", "CSCO", "INTC", "WFC", "DIS",
    "QCOM", "TXN", "DHR", "VZ", "INTU", "AMGN", "CAT", "NEE", "PM", "UNP",
    "LOW", "IBM", "GE", "SPGI", "HON", "RTX", "BA", "GS", "ISRG", "PFE",
    "NOW", "BKNG", "AXP", "ELV", "PLD", "T", "SYK", "MS", "BLK", "MDT",
    "GILD", "LMT", "ADI", "TJX", "VRTX", "SCHW", "MMC", "C", "CB", "AMT",
    "MO", "REGN", "DE", "SO", "ETN", "PGR", "BSX", "DUK", "MU", "PANW",
    "ZTS", "CME", "EQIX", "AON", "ITW", "SLB", "APD", "KLAC", "SHW", "CL",
]


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    """Live S&P 500 constituents from Wikipedia; static fallback if unavailable."""
    try:
        tbl = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        syms = [str(s).replace(".", "-").strip() for s in tbl["Symbol"].tolist()]
        syms = [s for s in syms if s and s.isascii()]
        if len(syms) > 100:
            return syms
    except Exception:
        pass
    return _SP500_FALLBACK
