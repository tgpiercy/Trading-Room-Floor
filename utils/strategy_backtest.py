"""
utils/strategy_backtest.py
Phase 2 (portfolio backtest) + Phase 3 (walk-forward validation).

Strategy: rank by RS extension (ExtPct) → hold top N → regime-gate exposure
(scaled: risk-on full / caution half / risk-off cash) → size EW or inverse-vol.

Walk-forward (Phase 3): on each 2y train window pick the top-N/cadence with the
best in-sample Sharpe, then trade the next 6mo UNSEEN with those params. Stitch
all test windows into one true out-of-sample curve. Out-of-sample Sharpe vs
in-sample Sharpe (walk-forward efficiency) is the honest verdict.

No lookahead anywhere: all ranking/vol/regime series are causal (row T uses
data ≤ T); returns realise T → T+cadence; walk-forward params are chosen using
train-window data only, then applied forward.
"""
import numpy as np
import pandas as pd

from utils.rs_indicators import build_rs_df
from utils.watchlist import yf_sym


# ── Building blocks ───────────────────────────────────────────────────────────
def compute_regime_exposure(spy, ief, vix):
    idx = spy.index
    trend = (spy > spy.rolling(40).mean()).astype(float) * 40
    ratio = spy / ief.reindex(idx).ffill()
    ratio_rising = (ratio > ratio.rolling(13).mean()).astype(float) * 30
    v = vix.reindex(idx).ffill()
    vix_calm = pd.Series(np.where(v < 20, 30.0, np.where(v < 28, 15.0, 0.0)), index=idx)
    score = trend + ratio_rising + vix_calm
    exp = pd.Series(0.5, index=idx)
    exp[score >= 66] = 1.0
    exp[score < 33] = 0.0
    return exp


def precompute_series(ohlcv: dict, pairs: list, vol_lookback: int = 13,
                      min_hist: int = 45):
    """Precompute causal close/ExtPct/vol per ticker + regime exposure (once)."""
    if not all(k in ohlcv for k in ("SPY", "IEF", "^VIX")):
        return {"error": "Need SPY, IEF and ^VIX for the regime gate."}
    spy, ief, vix = ohlcv["SPY"]["Close"], ohlcv["IEF"]["Close"], ohlcv["^VIX"]["Close"]
    if spy.empty:
        return {"error": "SPY history empty."}
    cal = spy.index
    exposure = compute_regime_exposure(spy, ief, vix)

    data = {}
    for dt, db, grp in pairs:
        yt, yb = yf_sym(dt), yf_sym(db)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        ct = ohlcv[yt]["Close"].reindex(cal).ffill()
        cb = ohlcv[yb]["Close"].reindex(cal).ffill()
        if ct.dropna().shape[0] < min_hist + 5:
            continue
        ext = build_rs_df(ct, cb)["ExtPct"]
        vol = ct.pct_change().rolling(vol_lookback).std()
        data[dt] = {"close": ct, "ext": ext, "vol": vol}
    if not data:
        return {"error": "No tickers with enough history."}
    return {"data": data, "exposure": exposure, "spy": spy, "cal": cal,
            "min_hist": min_hist}


def simulate_window(pc: dict, start: int, end: int, top_n: int, cadence: int) -> dict:
    """Rebalance loop over [start, end). Returns EW + vol-targeted + SPY paths."""
    data, exposure, spy, cal = pc["data"], pc["exposure"], pc["spy"], pc["cal"]
    reb = list(range(start, end - cadence, cadence))
    eq_ew, eq_vol, eq_spy = [1.0], [1.0], [1.0]
    ret_ew, ret_vol, ret_spy, dates, exp_path = [], [], [], [cal[start]], []
    holdings_last = []

    for T in reb:
        e = float(exposure.iloc[T]) if T < len(exposure) else 0.5
        exp_path.append((cal[T], e))
        scored = [(tk, d["ext"].iloc[T]) for tk, d in data.items()
                  if T < len(d["ext"]) and pd.notna(d["ext"].iloc[T])
                  and pd.notna(d["close"].iloc[T])]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [tk for tk, _ in scored[:top_n]]
        holdings_last = [(tk, round(float(v), 2)) for tk, v in scored[:top_n]]

        rets, vols = [], []
        for tk in top:
            c = data[tk]["close"]
            rets.append(c.iloc[T + cadence] / c.iloc[T] - 1)
            v = data[tk]["vol"].iloc[T]
            vols.append(v if pd.notna(v) and v > 0 else np.nan)
        if rets:
            rets = np.array(rets)
            r_ew = e * np.nanmean(rets)
            inv = np.array([1.0 / v if (v and not np.isnan(v)) else np.nan for v in vols])
            if np.all(np.isnan(inv)):
                w = np.ones_like(rets) / len(rets)
            else:
                inv = np.where(np.isnan(inv), np.nanmean(inv), inv)
                w = inv / inv.sum()
            r_vol = e * float(np.nansum(w * rets))
        else:
            r_ew = r_vol = 0.0
        r_spy = spy.iloc[T + cadence] / spy.iloc[T] - 1

        eq_ew.append(eq_ew[-1] * (1 + r_ew))
        eq_vol.append(eq_vol[-1] * (1 + r_vol))
        eq_spy.append(eq_spy[-1] * (1 + r_spy))
        ret_ew.append(r_ew); ret_vol.append(r_vol); ret_spy.append(r_spy)
        dates.append(cal[T + cadence])

    return {"dates": dates, "eq_ew": eq_ew, "eq_vol": eq_vol, "eq_spy": eq_spy,
            "ret_ew": ret_ew, "ret_vol": ret_vol, "ret_spy": ret_spy,
            "exposure_path": exp_path, "holdings_last": holdings_last}


# ── Metrics ───────────────────────────────────────────────────────────────────
def _sharpe(rets, ppy):
    r = np.array(rets)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() * ppy) / (r.std() * np.sqrt(ppy)))


def metrics_from_equity(dates, equity, period_rets=None, ppy=None) -> dict:
    eq = np.array(equity)
    if len(eq) < 2:
        return {}
    total = eq[-1] / eq[0] - 1
    if dates is not None and len(dates) >= 2:
        years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1e-6)
    else:
        years = len(eq) / (ppy or 52)
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(eq)
    maxdd = ((eq - peak) / peak).min()
    if period_rets is not None and ppy:
        r = np.array(period_rets)
        vol = r.std() * np.sqrt(ppy) if len(r) > 1 else 0
        sharpe = _sharpe(period_rets, ppy)
        win = (r > 0).mean() * 100 if len(r) else 0
    else:
        vol = sharpe = win = 0
    return {"Total Return %": round(total*100,1), "CAGR %": round(cagr*100,1),
            "Volatility %": round(vol*100,1), "Sharpe": round(sharpe,2),
            "Max Drawdown %": round(maxdd*100,1), "Win Rate %": round(win,0)}


def run_portfolio_backtest(ohlcv, pairs, top_n=10, cadence=4, vol_lookback=13,
                           min_hist=45) -> dict:
    """Phase 2 full-sample backtest (output contract unchanged)."""
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist)
    if "error" in pc:
        return pc
    n = len(pc["cal"])
    sim = simulate_window(pc, min_hist, n, top_n, cadence)
    ppy = 52 / cadence
    return {
        "dates": sim["dates"],
        "equity": {"Equal-Weight": sim["eq_ew"], "Vol-Targeted": sim["eq_vol"],
                   "SPY (benchmark)": sim["eq_spy"]},
        "metrics": {
            "Equal-Weight": metrics_from_equity(sim["dates"], sim["eq_ew"], sim["ret_ew"], ppy),
            "Vol-Targeted": metrics_from_equity(sim["dates"], sim["eq_vol"], sim["ret_vol"], ppy),
            "SPY (benchmark)": metrics_from_equity(sim["dates"], sim["eq_spy"], sim["ret_spy"], ppy),
        },
        "exposure_path": sim["exposure_path"],
        "current_holdings": sim["holdings_last"],
        "n_rebalances": len(sim["ret_ew"]), "cadence": cadence, "top_n": top_n,
    }


# ── Phase 3: walk-forward parameter selection ────────────────────────────────
def walk_forward(ohlcv, pairs, train_weeks=104, test_weeks=26,
                 grid_top_n=(5, 8, 10, 15, 20), grid_cadence=(2, 4, 8),
                 vol_lookback=13, min_hist=45, progress_cb=None) -> dict:
    """
    Rolling train→test. Each train window picks the best (top_n, cadence) by
    in-sample Sharpe; those params trade the next unseen test window. Test
    returns are stitched into one out-of-sample equity curve.
    """
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist)
    if "error" in pc:
        return pc
    cal = pc["cal"]; n = len(cal)
    first = min_hist
    if first + train_weeks + test_weeks > n:
        return {"error": f"Not enough history: need ≥ {min_hist+train_weeks+test_weeks} "
                         f"weeks, have {n}. Use a longer period or shorter windows."}

    rolls = []
    oos_dates_all, oos_eq, oos_ret, spy_ret = [], [1.0], [], []
    spy_eq = [1.0]
    i = first
    starts = list(range(first, n - train_weeks - test_weeks + 1, test_weeks))
    for k, i in enumerate(starts):
        if progress_cb:
            progress_cb(k / max(len(starts), 1))
        tr_s, tr_e = i, i + train_weeks
        te_s, te_e = i + train_weeks, i + train_weeks + test_weeks

        # Optimize on train (in-sample Sharpe)
        best, best_sh = None, -1e9
        for tn in grid_top_n:
            for cad in grid_cadence:
                s = simulate_window(pc, tr_s, tr_e, tn, cad)
                sh = _sharpe(s["ret_ew"], 52 / cad)
                if sh > best_sh:
                    best_sh, best = sh, (tn, cad)
        tn, cad = best

        # Test on UNSEEN window with chosen params
        ste = simulate_window(pc, te_s, te_e, tn, cad)
        test_sh = _sharpe(ste["ret_ew"], 52 / cad)

        for j, r in enumerate(ste["ret_ew"]):
            oos_ret.append(r); spy_ret.append(ste["ret_spy"][j])
            oos_eq.append(oos_eq[-1] * (1 + r))
            spy_eq.append(spy_eq[-1] * (1 + ste["ret_spy"][j]))
            oos_dates_all.append(ste["dates"][j + 1])
        rolls.append({"Window start": str(cal[te_s].date()),
                      "Top N": tn, "Cadence": cad,
                      "Train Sharpe": round(best_sh, 2),
                      "Test Sharpe": round(test_sh, 2)})

    if len(oos_eq) < 3:
        return {"error": "Walk-forward produced too few out-of-sample points."}

    dates0 = [oos_dates_all[0]] + oos_dates_all
    # Approx ppy from median cadence chosen
    med_cad = int(np.median([r["Cadence"] for r in rolls]))
    ppy = 52 / med_cad
    oos_metrics = metrics_from_equity(dates0, oos_eq, oos_ret, ppy)
    spy_metrics = metrics_from_equity(dates0, spy_eq, spy_ret, ppy)

    is_sharpe = float(np.mean([r["Train Sharpe"] for r in rolls]))
    oos_sharpe = oos_metrics.get("Sharpe", 0)
    wfe = round(oos_sharpe / is_sharpe, 2) if is_sharpe > 0 else 0.0

    return {
        "oos_dates": dates0, "oos_equity": oos_eq, "spy_equity": spy_eq,
        "oos_metrics": oos_metrics, "spy_metrics": spy_metrics,
        "rolls": rolls, "is_sharpe": round(is_sharpe, 2), "oos_sharpe": oos_sharpe,
        "wfe": wfe, "n_rolls": len(rolls),
        "train_weeks": train_weeks, "test_weeks": test_weeks,
    }
