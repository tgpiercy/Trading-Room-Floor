"""
utils/strategy_backtest.py
Phase 2 — combined strategy portfolio backtest.

Strategy (from validated Phase 1 signal):
  • RANK every name by RS extension (ExtPct = how far RS sits above its SMA18 —
    the continuous form of the 'Exhaustion'/leadership signal that tested +16%/12w).
  • HOLD the top N each rebalance.
  • GATE exposure by a regime filter (SPY trend + SPY/IEF RS + VIX), scaled:
    risk-on = full, caution = half, risk-off = cash. (Momentum crashed in 2022 —
    the gate is what protects against that.)
  • SIZE two ways for comparison: equal-weight and inverse-volatility.

No lookahead: ranks/exposure/vol use data ≤ T; returns realised T → T+cadence.
"""
import numpy as np
import pandas as pd

from utils.rs_indicators import build_rs_df
from utils.watchlist import yf_sym


def rank_metric_series(close_t: pd.Series, close_b: pd.Series) -> pd.Series:
    """Continuous RS-leadership score = ExtPct (RS vs its SMA18), causal."""
    rs_df = build_rs_df(close_t, close_b)
    return rs_df["ExtPct"]


def compute_regime_exposure(spy: pd.Series, ief: pd.Series, vix: pd.Series) -> pd.Series:
    """
    Weekly exposure multiplier from a regime score (0..100):
      SPY above 40w SMA (+40) · SPY/IEF RS rising (+30) · VIX calm (+30).
      score ≥ 66 → 1.0 (risk-on) · 33–66 → 0.5 (caution) · < 33 → 0.0 (risk-off).
    """
    idx = spy.index
    spy_sma = spy.rolling(40).mean()
    trend = (spy > spy_sma).astype(float) * 40

    ratio = (spy / ief.reindex(idx).ffill())
    ratio_rising = (ratio > ratio.rolling(13).mean()).astype(float) * 30

    v = vix.reindex(idx).ffill()
    vix_calm = np.where(v < 20, 30.0, np.where(v < 28, 15.0, 0.0))

    score = trend + ratio_rising + pd.Series(vix_calm, index=idx)
    exp = pd.Series(0.5, index=idx)
    exp[score >= 66] = 1.0
    exp[score < 33] = 0.0
    return exp


def _trailing_vol(close: pd.Series, lookback: int = 13) -> pd.Series:
    """Trailing weekly-return volatility (causal)."""
    return close.pct_change().rolling(lookback).std()


def run_portfolio_backtest(ohlcv: dict, pairs: list, top_n: int = 10,
                           cadence: int = 4, vol_lookback: int = 13,
                           min_hist: int = 45) -> dict:
    """
    Periodic-rebalance momentum portfolio with regime gating.
    Returns equity curves (equal-weight, vol-targeted, SPY benchmark),
    per-period returns, exposure path, and current holdings.
    """
    # Regime inputs
    spy = ohlcv.get("SPY", pd.DataFrame()).get("Close") if "SPY" in ohlcv else None
    ief = ohlcv.get("IEF", pd.DataFrame()).get("Close") if "IEF" in ohlcv else None
    vix = ohlcv.get("^VIX", pd.DataFrame()).get("Close") if "^VIX" in ohlcv else None
    if spy is None or ief is None or vix is None or spy.empty:
        return {"error": "Need SPY, IEF and ^VIX for the regime gate."}

    cal = spy.index                      # master weekly calendar
    exposure = compute_regime_exposure(spy, ief, vix)

    # Precompute per-pair: close (reindexed to calendar), ExtPct, trailing vol
    data = {}
    for dt, db, grp in pairs:
        yt, yb = yf_sym(dt), yf_sym(db)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        ct = ohlcv[yt]["Close"].reindex(cal).ffill()
        cb = ohlcv[yb]["Close"].reindex(cal).ffill()
        if ct.dropna().shape[0] < min_hist + cadence:
            continue
        ext = rank_metric_series(ct, cb)
        vol = _trailing_vol(ct, vol_lookback)
        data[dt] = {"close": ct, "ext": ext, "vol": vol}

    if not data:
        return {"error": "No tickers with enough history."}

    n = len(cal)
    start = min_hist
    reb_points = list(range(start, n - cadence, cadence))

    eq_ew, eq_vol, eq_spy = [1.0], [1.0], [1.0]
    dates, exp_path, period_ret_ew, period_ret_vol = [cal[start]], [], [], []

    for T in reb_points:
        e = float(exposure.iloc[T]) if T < len(exposure) else 0.5
        exp_path.append((cal[T], e))

        # Rank by ExtPct at T (no lookahead)
        scored = []
        for tk, d in data.items():
            val = d["ext"].iloc[T] if T < len(d["ext"]) else np.nan
            if pd.notna(val) and pd.notna(d["close"].iloc[T]):
                scored.append((tk, val))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [tk for tk, _ in scored[:top_n]]

        # Realised returns T → T+cadence
        rets, vols = [], []
        for tk in top:
            c = data[tk]["close"]
            r = c.iloc[T + cadence] / c.iloc[T] - 1
            rets.append(r)
            v = data[tk]["vol"].iloc[T]
            vols.append(v if pd.notna(v) and v > 0 else np.nan)
        if not rets:
            ret_ew = ret_vol = 0.0
        else:
            rets = np.array(rets)
            ret_ew = e * np.nanmean(rets)
            # inverse-vol weights (fallback to equal where vol missing)
            inv = np.array([1.0 / v if (v and not np.isnan(v)) else np.nan for v in vols])
            if np.all(np.isnan(inv)):
                w = np.ones_like(rets) / len(rets)
            else:
                inv = np.where(np.isnan(inv), np.nanmean(inv), inv)
                w = inv / inv.sum()
            ret_vol = e * float(np.nansum(w * rets))

        spy_ret = spy.iloc[T + cadence] / spy.iloc[T] - 1

        eq_ew.append(eq_ew[-1] * (1 + ret_ew))
        eq_vol.append(eq_vol[-1] * (1 + ret_vol))
        eq_spy.append(eq_spy[-1] * (1 + spy_ret))
        dates.append(cal[T + cadence])
        period_ret_ew.append(ret_ew)
        period_ret_vol.append(ret_vol)

    # Current holdings (latest rankable week)
    lastT = reb_points[-1] if reb_points else start
    cur = []
    for tk, d in data.items():
        val = d["ext"].iloc[lastT] if lastT < len(d["ext"]) else np.nan
        if pd.notna(val):
            cur.append((tk, round(float(val), 2)))
    cur.sort(key=lambda x: x[1], reverse=True)

    ppy = 52 / cadence
    return {
        "dates": dates,
        "equity": {"Equal-Weight": eq_ew, "Vol-Targeted": eq_vol, "SPY (benchmark)": eq_spy},
        "metrics": {
            "Equal-Weight": _perf(eq_ew, period_ret_ew, ppy),
            "Vol-Targeted": _perf(eq_vol, period_ret_vol, ppy),
            "SPY (benchmark)": _perf(eq_spy,
                [eq_spy[i+1]/eq_spy[i]-1 for i in range(len(eq_spy)-1)], ppy),
        },
        "exposure_path": exp_path,
        "current_holdings": cur[:top_n],
        "n_rebalances": len(reb_points),
        "cadence": cadence, "top_n": top_n,
    }


def _perf(equity: list, period_rets: list, ppy: float) -> dict:
    """Performance metrics from an equity curve + per-period returns."""
    eq = np.array(equity)
    if len(eq) < 2:
        return {}
    total = eq[-1] / eq[0] - 1
    years = len(period_rets) / ppy if ppy else 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0
    r = np.array(period_rets)
    vol = r.std() * np.sqrt(ppy) if len(r) > 1 else 0
    sharpe = (r.mean() * ppy) / vol if vol > 0 else 0
    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = dd.min()
    win = (r > 0).mean() * 100 if len(r) else 0
    return {
        "Total Return %": round(total * 100, 1),
        "CAGR %": round(cagr * 100, 1),
        "Volatility %": round(vol * 100, 1),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown %": round(maxdd * 100, 1),
        "Win Rate %": round(win, 0),
    }
