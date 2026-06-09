"""
utils/market_health.py
Market Health v2.5 — exact Pine Script logic.
SPY/IEF RS + RSP rescue + VIX + S5FI breadth → MH% + Target Risk.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

from utils.data_fetcher import get_stock_data

# Homegrown breadth basket (replaces the delisted ^SPXA50R)
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC"]


@st.cache_data(ttl=3600, show_spinner=False)
def sector_breadth() -> float:
    """% of sector ETFs trading above their own 50-day SMA. Robust, free breadth
    proxy that replaces the delisted ^SPXA50R. Uses Stooq-resilient OHLCV."""
    above = total = 0
    for sym in SECTOR_ETFS:
        try:
            d = get_stock_data(sym, period="6mo", interval="1d")
            if d is None or d.empty:
                continue
            c = d["Close"].dropna()
            if len(c) >= 50:
                total += 1
                if c.iloc[-1] > c.rolling(50).mean().iloc[-1]:
                    above += 1
        except Exception:
            continue
    return round(above / total * 100, 1) if total else 50.0


@st.cache_data(ttl=900, show_spinner=False)
def current_regime() -> dict:
    """THE canonical regime — latest exposure from the validated SPY/IEF/VIX gate
    (compute_regime_exposure). Shared by the Cockpit, Market Health and Rebalance
    so every surface shows the same verdict. Returns {exposure, label, color}."""
    try:
        from utils.strategy_backtest import compute_regime_exposure

        def wk(sym):
            d = get_stock_data(sym, period="2y", interval="1d")
            return (d["Close"].resample("W-FRI").last().dropna()
                    if (d is not None and not d.empty) else pd.Series(dtype=float))
        spy, ief, vix = wk("SPY"), wk("IEF"), wk("^VIX")
        if spy.empty or ief.empty or vix.empty:
            return {"exposure": None, "label": "⚪ Regime unknown — data unavailable", "color": "#888"}
        exp = float(compute_regime_exposure(spy, ief, vix).iloc[-1])
        if exp >= 0.99:
            return {"exposure": exp, "label": "🟢 Risk-ON — full exposure", "color": "#00cc66"}
        if exp <= 0.01:
            return {"exposure": exp, "label": "🔴 Risk-OFF — cash", "color": "#ff4444"}
        return {"exposure": exp, "label": f"🟡 Caution — ~{exp*100:.0f}% exposure", "color": "#ff9800"}
    except Exception:
        return {"exposure": None, "label": "⚪ Regime unknown", "color": "#888"}

# ── MH matrix (from Pine Script v2.5 — updated from PDF v2.4) ────────────────
_MH_MATRIX = {
    0: {0: 0,  1: 0,  2: 0,  3: 0},
    1: {0: 0,  1: 30, 2: 55, 3: 70},
    2: {0: 0,  1: 56, 2: 75, 3: 100},
}


def _rsi_n(series: pd.Series, n: int) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_health_data() -> dict:
    """
    Download all Market Health data sources in one call.
    Returns raw series dict — all daily, latest close.
    """
    symbols = ["SPY", "RSP", "IEF", "^VIX"]  # ^SPXA50R retired → homegrown sector_breadth()
    try:
        raw = yf.download(symbols, period="2y", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": symbols[0]})

        # Resample to weekly (includes current partial week)
        weekly = closes.resample("W-FRI").last().dropna(how="all")

        result = {}
        for sym in symbols:
            if sym in weekly.columns:
                result[sym] = weekly[sym].dropna()
        return result
    except Exception as e:
        return {}


def calc_market_health(target_risk_full: float = 4.5) -> dict:
    """
    Full Market Health v2.5 calculation.
    Returns MH%, target risk, component scores, and raw values.
    """
    data = fetch_market_health_data()
    if not data:
        return _empty_mh(target_risk_full)

    spy = data.get("SPY", pd.Series(dtype=float))
    rsp = data.get("RSP", pd.Series(dtype=float))
    ief = data.get("IEF", pd.Series(dtype=float))
    vix = data.get("^VIX", pd.Series(dtype=float))
    s5fi = pd.Series(dtype=float)  # breadth now via sector_breadth()

    if spy.empty or ief.empty:
        return _empty_mh(target_risk_full)

    # ── SPY/IEF RS ────────────────────────────────────────────────────────────
    rs = (spy / ief).dropna()
    rs_sma8  = rs.rolling(8).mean()
    rs_sma18 = rs.rolling(18).mean()

    # ── RSP vs its own SMA18 (rescue) ────────────────────────────────────────
    rsp_sma18 = rsp.rolling(18).mean() if not rsp.empty else pd.Series(dtype=float)

    if rs_sma18.dropna().empty:
        return _empty_mh(target_risk_full)

    last_rs    = float(rs.dropna().iloc[-1])
    last_sma18 = float(rs_sma18.dropna().iloc[-1])
    prev_sma18 = float(rs_sma18.dropna().iloc[-2]) if len(rs_sma18.dropna()) >= 2 else last_sma18
    last_sma8  = float(rs_sma8.dropna().iloc[-1])  if not rs_sma8.dropna().empty else last_sma18
    prev_sma8  = float(rs_sma8.dropna().iloc[-2])  if len(rs_sma8.dropna()) >= 2 else last_sma8

    last_rsp    = float(rsp.dropna().iloc[-1])       if not rsp.empty else 0
    last_rsp18  = float(rsp_sma18.dropna().iloc[-1]) if not rsp_sma18.dropna().empty else 0
    prev_rsp18  = float(rsp_sma18.dropna().iloc[-2]) if len(rsp_sma18.dropna()) >= 2 else last_rsp18

    # ── RS Score (b7 + b8) ────────────────────────────────────────────────────
    b7 = (last_rs > last_sma18) or (last_rsp > last_rsp18 if last_rsp18 > 0 else False)
    b8 = (last_sma18 > prev_sma18) or (last_rsp18 > prev_rsp18 if last_rsp18 > 0 else False)
    rs_score = int(b7) + int(b8)

    # RS band for display
    band_hi = last_sma18 * 1.01
    band_lo = last_sma18 * 0.99
    rs_regime = ("Strong" if last_rs > band_hi else
                 "Neutral" if last_rs >= band_lo else "Weak")

    # ── VIX Score ─────────────────────────────────────────────────────────────
    last_vix = float(vix.dropna().iloc[-1]) if not vix.empty else 25
    vix_score = (3 if last_vix < 17 else
                 2 if last_vix <= 20 else
                 1 if last_vix < 25 else 0)
    vix_regime = ("Low (<17)"   if last_vix < 17 else
                  "Normal (≤20)" if last_vix <= 20 else
                  "Caution (<25)" if last_vix < 25 else "Risk-Off (≥25)")

    # ── MH Base matrix ────────────────────────────────────────────────────────
    mh_base = _MH_MATRIX[rs_score][vix_score]

    # ── Adjustments (v2.5) ────────────────────────────────────────────────────
    rsp_above_18 = (last_rsp > last_rsp18) if last_rsp18 > 0 else False
    rs8_rising   = last_sma8 > prev_sma8
    rsp_adj  = 10 if rsp_above_18 else -10
    rs8_adj  = 5  if rs8_rising   else -5
    mh_raw   = max(0, min(100, mh_base + rsp_adj + rs8_adj))

    # ── Breadth: homegrown (% of sector ETFs above 50-day MA) ─────────────────
    s5fi_val = sector_breadth()

    k = (1.25 if s5fi_val > 60 else
         1.75 if s5fi_val <= 40 else 1.50)

    # ── Final MH with convex scaling ─────────────────────────────────────────
    mh_raw_frac  = mh_raw / 100.0
    mh_final     = round((mh_raw_frac ** k) * 100)
    target_risk  = round(target_risk_full * (mh_raw_frac ** k), 2)

    return dict(
        mh_pct        = mh_final,
        mh_raw        = mh_raw,
        mh_base       = mh_base,
        target_risk   = target_risk,
        rs_score      = rs_score,
        rs_regime     = rs_regime,
        vix           = round(last_vix, 1),
        vix_score     = vix_score,
        vix_regime    = vix_regime,
        s5fi          = round(s5fi_val, 1),
        k             = k,
        b7            = b7, b8 = b8,
        rsp_above_18  = rsp_above_18,
        rs8_rising    = rs8_rising,
        rsp_adj       = rsp_adj,
        rs8_adj       = rs8_adj,
        last_rs       = round(last_rs, 4),
        last_sma18    = round(last_sma18, 4),
        last_vix      = round(last_vix, 1),
        rs_series     = rs,
        rs_sma8       = rs_sma8,
        rs_sma18      = rs_sma18,
        vix_series    = vix,
        s5fi_series   = s5fi,
    )


def _empty_mh(full_risk: float) -> dict:
    return dict(mh_pct=0, mh_raw=0, mh_base=0, target_risk=0,
                rs_score=0, rs_regime="Unknown",
                vix=25, vix_score=0, vix_regime="Unknown",
                s5fi=50, k=1.5, b7=False, b8=False,
                rsp_above_18=False, rs8_rising=False,
                rsp_adj=-10, rs8_adj=-5,
                last_rs=0, last_sma18=0, last_vix=25,
                rs_series=pd.Series(dtype=float),
                rs_sma8=pd.Series(dtype=float),
                rs_sma18=pd.Series(dtype=float),
                vix_series=pd.Series(dtype=float),
                s5fi_series=pd.Series(dtype=float))
