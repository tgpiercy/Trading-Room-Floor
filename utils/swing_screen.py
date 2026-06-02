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
DEFAULT_WEIGHTS = {"momentum": 0.50, "entry": 0.22, "sector": 0.18, "lowvol": 0.10}
MIN_DOLLAR_VOL = 3_000_000     # liquidity floor: median 20-day dollar volume

# Sector bias — 11 SPDR sector ETFs + GICS→ETF mapping (for S&P 500 constituents)
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
GICS_TO_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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
    sma50v = float(sma50.iloc[-1])
    # Entry timing: pulled-back (lower RSI) within the uptrend scores higher
    entry_quality = float(np.clip((70 - rsi) / 40, 0, 1))

    # ── Trend STATE: active growth vs sideways chop vs deterioration ──────────
    ma_stack = bool(sma20v > sma50v and (len(c) < 200 or last > _sma(c, 200).iloc[-1]))
    # 50-day MA's own slope over ~1 month → rising / flat / falling
    slope50 = float(sma50.iloc[-1] / sma50.iloc[-21] - 1) if len(sma50.dropna()) >= 21 else 0.0
    # acceleration: short-window trend minus the full-window trend
    short_lb = max(42, lookback // 3)
    ann_short, _, _ = clenow_momentum(c, short_lb)
    accel = float((ann_short if ann_short == ann_short else 0.0) - ann)
    above50 = last > sma50v
    trend_state = _classify_trend(ann, r2, ma_stack, slope50, accel, above50)

    return {"price": last, "trend_gate": bool(trend_gate), "dollar_vol": dollar_vol,
            "mom_score": float(mom_score), "ann_ret": float(ann), "r2": float(r2),
            "rsi": rsi, "vol_ann": vol_ann, "atr": atr, "sma20": sma20v,
            "sma50": sma50v, "entry_quality": entry_quality,
            "slope50": slope50, "accel": accel, "ma_stack": ma_stack,
            "trend_state": trend_state}


def _classify_trend(ann_ret, r2, ma_stack, slope50, accel, above50):
    """Distinguish active markup from sideways chop and from rolling-over trends.
    Active growth = strong, smooth, rising MA. Deteriorating = momentum rolling
    over (falling 50d, decelerating, or lost the 50d). Chop = above 200d but no
    persistent direction (low R²)."""
    if (slope50 < -0.002) or (accel < -0.15) or (not above50):
        return "Deteriorating"
    if r2 < 0.30:                      # no persistent direction → chop, whatever the wiggle
        return "Chop"
    if ann_ret > 0.10 and r2 >= 0.45 and slope50 > 0.001:
        return "Active Growth"
    if accel > 0.10 and ann_ret > 0.0:
        return "Early"
    return "Chop"


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
                     min_dollar_vol=MIN_DOLLAR_VOL, top_n=25,
                     sector_of=None, sector_pct_by_etf=None, active_growth_only=True):
    """Rank a universe. Excludes deteriorating trends always, and sideways chop
    when active_growth_only. Applies a sector-momentum tilt. Returns top_n."""
    weights = weights or DEFAULT_WEIGHTS
    sector_of = sector_of or {}
    sector_pct_by_etf = sector_pct_by_etf or {}
    rows = []
    for sym, df in daily_by_sym.items():
        f = compute_factors(df, lookback)
        if not f or not f["trend_gate"] or f["dollar_vol"] < min_dollar_vol:
            continue
        if f["vol_ann"] != f["vol_ann"]:
            continue
        if f["trend_state"] == "Deteriorating":
            continue
        if active_growth_only and f["trend_state"] == "Chop":
            continue
        etf = sector_of.get(sym)
        f["sector"] = etf or "—"
        f["sector_pct"] = float(sector_pct_by_etf.get(etf, 50.0))   # neutral if unmapped
        rows.append({"ticker": sym, **f})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)

    d["mom_pct"] = d["mom_score"].rank(pct=True) * 100
    d["entry_pct"] = d["entry_quality"].rank(pct=True) * 100
    d["lowvol_pct"] = (1 - d["vol_ann"].rank(pct=True)) * 100
    d["sector_score"] = d["sector_pct"]
    w = {**DEFAULT_WEIGHTS, **weights}
    d["score"] = (w["momentum"] * d["mom_pct"] + w["entry"] * d["entry_pct"]
                  + w.get("sector", 0) * d["sector_score"]
                  + w["lowvol"] * d["lowvol_pct"]).round(1)
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


@st.cache_data(ttl=3600, show_spinner=False)
def sector_strength(lookback=126):
    """Percentile rank (0-100) of each sector ETF's Clenow momentum → sector bias."""
    from utils.data_fetcher import fetch_daily_batch
    data = fetch_daily_batch(tuple(SECTOR_ETFS), period="2y")
    scores = {}
    for etf in SECTOR_ETFS:
        df = data.get(etf)
        if df is not None and not df.empty:
            s, _, _ = clenow_momentum(df["Close"], lookback)
            if s == s:
                scores[etf] = s
    if not scores:
        return {}
    ser = pd.Series(scores)
    pct = ser.rank(pct=True) * 100
    return {etf: round(float(pct[etf]), 0) for etf in scores}


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_map():
    """{ticker: sector_ETF} for S&P 500 constituents from Wikipedia's GICS column."""
    try:
        tbl = pd.read_html(_WIKI_SP500)[0]
        m = {}
        for _, row in tbl.iterrows():
            t = str(row.get("Symbol", "")).replace(".", "-").strip()
            etf = GICS_TO_ETF.get(str(row.get("GICS Sector", "")).strip())
            if t and t.isascii() and etf:
                m[t] = etf
        if len(m) > 100:
            return m
    except Exception:
        pass
    return {}


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
