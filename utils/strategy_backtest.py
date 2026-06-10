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
import math

import numpy as np
import pandas as pd

from utils.rs_indicators import build_rs_df
from utils.watchlist import yf_sym
from utils.swing_screen import clenow_momentum

try:
    from utils.risk import apply_risk_layer
    _RISK_AVAIL = True
except Exception:
    _RISK_AVAIL = False


def _trend_ok_causal(close_w, lookback_w: int = 26, min_w: int = 20):
    """Boolean Series, point-in-time: True where the name passes the weekly trend
    gate, isn't deteriorating, and shows persistence (R² ≥ 0.30) — i.e. NOT chop
    and NOT rolling over. Used to FILTER another ranking (e.g. RS extension)."""
    c = close_w.astype(float)
    ok = pd.Series(False, index=c.index)
    sma10 = c.rolling(10).mean()
    sma40 = c.rolling(40).mean()
    for i in range(len(c)):
        if i < min_w:
            continue
        score, ann, r2 = clenow_momentum(c.iloc[:i + 1], lookback_w)
        if score != score:
            continue
        last = c.iloc[i]
        s10, s40 = sma10.iloc[i], sma40.iloc[i]
        gate = (last > s40 and s10 > s40) if s40 == s40 else ((s10 == s10) and last > s10)
        prev10 = sma10.iloc[i - 4] if i >= 14 else np.nan
        slope10 = (s10 / prev10 - 1) if (prev10 == prev10 and prev10 > 0) else 0.0
        deteriorating = (slope10 < -0.01) or (last < s10)
        if gate and not deteriorating and r2 >= 0.30:
            ok.iloc[i] = True
    return ok


def _swing_rank_causal(close_w, lookback_w: int = 26, min_w: int = 20):
    """Point-in-time (causal) weekly swing rank for backtesting. At each weekly
    index it uses ONLY past data: Clenow slope×R² over the trailing lookback_w
    bars, returned NaN unless the name passes a weekly trend gate, isn't
    deteriorating, and shows persistence (R² ≥ 0.30) — so chop/rollovers are
    auto-excluded by simulate_window's NaN filter. Reuses the live screen's
    clenow_momentum, so the validated score is identical math to the screen."""
    c = close_w.astype(float)
    out = pd.Series(np.nan, index=c.index)
    sma10 = c.rolling(10).mean()
    sma40 = c.rolling(40).mean()
    for i in range(len(c)):
        if i < min_w:
            continue
        score, ann, r2 = clenow_momentum(c.iloc[:i + 1], lookback_w)
        if score != score:
            continue
        last = c.iloc[i]
        s10, s40 = sma10.iloc[i], sma40.iloc[i]
        if s40 == s40:
            gate = (last > s40) and (s10 > s40)
        else:
            gate = (s10 == s10) and (last > s10)
        prev10 = sma10.iloc[i - 4] if i >= 14 else np.nan
        slope10 = (s10 / prev10 - 1) if (prev10 == prev10 and prev10 > 0) else 0.0
        deteriorating = (slope10 < -0.01) or (last < s10)
        if gate and not deteriorating and r2 >= 0.30:
            out.iloc[i] = score
    return out


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
                      min_hist: int = 45, signal: str = "extpct"):
    """Precompute causal close/rank/vol per ticker + regime exposure (once).
    signal='extpct' → rank by RS extension (vs benchmark); 'swing' → rank by the
    causal weekly swing momentum (Clenow slope×R², trend-gated, chop-excluded)."""
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
        if yt not in ohlcv:
            continue
        ct = ohlcv[yt]["Close"].reindex(cal).ffill()
        if ct.dropna().shape[0] < min_hist + 5:
            continue
        if signal == "swing":
            ext = _swing_rank_causal(ct)
        else:                                   # extpct or extpct_filtered (need benchmark)
            if yb not in ohlcv:
                continue
            cb = ohlcv[yb]["Close"].reindex(cal).ffill()
            ext = build_rs_df(ct, cb)["ExtPct"]
            if signal == "extpct_filtered":
                ext = ext.where(_trend_ok_causal(ct))   # drop chop / rollover names
        vol = ct.pct_change().rolling(vol_lookback).std()
        data[dt] = {"close": ct, "ext": ext, "vol": vol}
    if not data:
        return {"error": "No tickers with enough history."}
    return {"data": data, "exposure": exposure, "spy": spy, "cal": cal,
            "min_hist": min_hist, "sector": {dt: grp for dt, db, grp in pairs}}


def _turnover(old: dict, new: dict) -> float:
    """One-rebalance stock turnover = Σ|w_new − w_old| over all names (incl.
    moves to/from cash via exposure changes). Cost is charged on this."""
    keys = set(old) | set(new)
    return float(sum(abs(new.get(k, 0.0) - old.get(k, 0.0)) for k in keys))


def simulate_window(pc: dict, start: int, end: int, top_n: int, cadence: int,
                    cost_bps: float = 0.0, risk_params: dict = None) -> dict:
    """Rebalance loop over [start, end). Returns EW + vol-targeted + SPY paths.
    cost_bps = per-side transaction cost in basis points, charged on turnover.
    risk_params (optional) → also computes a risk-layered path (caps + vol target
    + per-trade risk) applied on top of the inverse-vol book, using causal vol and
    a vol-based stop proxy (the live system uses the actual GW2 program stop)."""
    data, exposure, spy, cal = pc["data"], pc["exposure"], pc["spy"], pc["cal"]
    sect_map = pc.get("sector", {})
    do_risk = bool(risk_params) and _RISK_AVAIL
    cost_side = cost_bps / 10000.0
    reb = list(range(start, end - cadence, cadence))
    eq_ew, eq_vol, eq_spy, eq_risk = [1.0], [1.0], [1.0], [1.0]
    ret_ew, ret_vol, ret_spy, ret_risk, dates, exp_path = [], [], [], [], [cal[start]], []
    holdings_last = []
    prev_ew, prev_vol, prev_risk = {}, {}, {}      # previous effective weights
    turn_ew_sum, turn_vol_sum, turn_risk_sum = 0.0, 0.0, 0.0

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
            # effective weights (sum to exposure e; remainder is cash)
            w_ew = {tk: e / len(top) for tk in top}
            inv = np.array([1.0 / v if (v and not np.isnan(v)) else np.nan for v in vols])
            if np.all(np.isnan(inv)):
                wv = np.ones_like(rets) / len(rets)
            else:
                inv = np.where(np.isnan(inv), np.nanmean(inv), inv)
                wv = inv / inv.sum()
            w_vol = {tk: e * float(wv[i]) for i, tk in enumerate(top)}

            t_ew, t_vol = _turnover(prev_ew, w_ew), _turnover(prev_vol, w_vol)
            turn_ew_sum += t_ew; turn_vol_sum += t_vol
            r_ew = e * float(np.nanmean(rets)) - t_ew * cost_side
            r_vol = e * float(np.nansum(wv * rets)) - t_vol * cost_side
            prev_ew, prev_vol = w_ew, w_vol

            if do_risk:
                hlist = []
                for i_, tk in enumerate(top):
                    vw = vols[i_]
                    dd = (min(0.25, max(0.03, 3.0 * vw)) if (vw == vw and vw and vw > 0)
                          else 0.10)
                    pxT = float(data[tk]["close"].iloc[T])
                    hlist.append({"ticker": tk, "weight": w_vol[tk], "price": pxT,
                                  "stop": pxT * (1 - dd),
                                  "vol": (float(vw) if vw == vw else None),
                                  "sector": sect_map.get(tk, "—")})
                nh, _ = apply_risk_layer(hlist, **risk_params)
                w_risk = {h["ticker"]: h["weight"] for h in nh}
                t_risk = _turnover(prev_risk, w_risk); turn_risk_sum += t_risk
                r_risk = float(sum(w_risk.get(tk, 0.0) * rets[i_]
                                   for i_, tk in enumerate(top))) - t_risk * cost_side
                prev_risk = w_risk
            else:
                r_risk = 0.0
        else:
            # de-risk fully to cash — pay to sell whatever was held
            t_ew, t_vol = _turnover(prev_ew, {}), _turnover(prev_vol, {})
            turn_ew_sum += t_ew; turn_vol_sum += t_vol
            r_ew = -t_ew * cost_side
            r_vol = -t_vol * cost_side
            prev_ew, prev_vol = {}, {}
            if do_risk:
                t_risk = _turnover(prev_risk, {}); turn_risk_sum += t_risk
                r_risk = -t_risk * cost_side
                prev_risk = {}
            else:
                r_risk = 0.0

        r_spy = spy.iloc[T + cadence] / spy.iloc[T] - 1
        eq_ew.append(eq_ew[-1] * (1 + r_ew))
        eq_vol.append(eq_vol[-1] * (1 + r_vol))
        eq_spy.append(eq_spy[-1] * (1 + r_spy))
        ret_ew.append(r_ew); ret_vol.append(r_vol); ret_spy.append(r_spy)
        if do_risk:
            eq_risk.append(eq_risk[-1] * (1 + r_risk)); ret_risk.append(r_risk)
        dates.append(cal[T + cadence])

    nreb = max(len(reb), 1)
    return {"dates": dates, "eq_ew": eq_ew, "eq_vol": eq_vol, "eq_spy": eq_spy,
            "eq_risk": eq_risk,
            "ret_ew": ret_ew, "ret_vol": ret_vol, "ret_spy": ret_spy,
            "ret_risk": ret_risk,
            "exposure_path": exp_path, "holdings_last": holdings_last,
            "avg_turnover_ew": turn_ew_sum / nreb,
            "avg_turnover_vol": turn_vol_sum / nreb,
            "avg_turnover_risk": turn_risk_sum / nreb,
            "annual_turnover_ew": turn_ew_sum / nreb * (52 / cadence)}


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
                           min_hist=45, cost_bps=0.0, signal="extpct") -> dict:
    """Phase 2 full-sample backtest (output contract unchanged)."""
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist, signal=signal)
    if "error" in pc:
        return pc
    n = len(pc["cal"])
    sim = simulate_window(pc, min_hist, n, top_n, cadence, cost_bps=cost_bps)
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
        "avg_turnover": round(sim["avg_turnover_ew"] * 100, 1),
        "annual_turnover": round(sim["annual_turnover_ew"] * 100, 0),
    }


def cost_sensitivity(ohlcv, pairs, top_n=10, cadence=4, vol_lookback=13,
                     min_hist=45, cost_levels=(0, 5, 10, 20, 40, 80), signal="extpct") -> dict:
    """
    Re-run the full-sample backtest at several per-side cost levels (bps).
    Returns net CAGR/Sharpe at each level + the break-even cost where the
    edge over SPY disappears. The make-or-break test for a thin edge.
    """
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist, signal=signal)
    if "error" in pc:
        return pc
    n = len(pc["cal"]); ppy = 52 / cadence
    spy_cagr = None
    rows = []
    for bps in cost_levels:
        sim = simulate_window(pc, min_hist, n, top_n, cadence, cost_bps=bps)
        m = metrics_from_equity(sim["dates"], sim["eq_ew"], sim["ret_ew"], ppy)
        if spy_cagr is None:
            spy_cagr = metrics_from_equity(sim["dates"], sim["eq_spy"],
                                           sim["ret_spy"], ppy).get("CAGR %", 0)
        rows.append({"Cost (bps/side)": bps, "Net CAGR %": m.get("CAGR %"),
                     "Net Sharpe": m.get("Sharpe"),
                     "Edge vs SPY %": round(m.get("CAGR %", 0) - spy_cagr, 1)})
    df = pd.DataFrame(rows)
    # Break-even: highest cost level where edge over SPY is still > 0
    pos = df[df["Edge vs SPY %"] > 0]
    breakeven = int(pos["Cost (bps/side)"].max()) if not pos.empty else 0
    return {"table": df, "spy_cagr": spy_cagr, "breakeven_bps": breakeven,
            "annual_turnover": round(sim["annual_turnover_ew"] * 100, 0),
            "top_n": top_n, "cadence": cadence}


# ── Phase 3: walk-forward parameter selection ────────────────────────────────
def walk_forward(ohlcv, pairs, train_weeks=104, test_weeks=26,
                 grid_top_n=(5, 8, 10, 15, 20), grid_cadence=(2, 4, 8),
                 vol_lookback=13, min_hist=45, progress_cb=None, signal="extpct",
                 param_mode="optimized", fixed_params=(10, 4)) -> dict:
    """
    Rolling train→test. Out-of-sample test returns stitched into one equity curve.
    param_mode controls how each window's (top_n, cadence) is chosen:
      • "optimized" — argmax in-sample Sharpe over the grid each window (can overfit;
        this is what produces an inflated IS Sharpe and a low WFE).
      • "frozen"    — optimize ONCE on the first train window, then reuse those params
        for every later window (realistic: pick params at deployment, never refit).
      • "fixed"     — use fixed_params for every window; no selection at all.
    Comparing the three shows how much of the headline is durable edge vs grid-search
    selection: if frozen/fixed OOS ≈ optimized OOS, the optimization added only overfit.
    """
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist, signal=signal)
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
    frozen = None
    for k, i in enumerate(starts):
        if progress_cb:
            progress_cb(k / max(len(starts), 1))
        tr_s, tr_e = i, i + train_weeks
        te_s, te_e = i + train_weeks, i + train_weeks + test_weeks

        if param_mode == "fixed":
            tn, cad = fixed_params
            best_sh = _sharpe(simulate_window(pc, tr_s, tr_e, tn, cad)["ret_ew"], 52 / cad)
        elif param_mode == "frozen":
            if frozen is None:
                b, bs = None, -1e9
                for t_ in grid_top_n:
                    for c_ in grid_cadence:
                        sh = _sharpe(simulate_window(pc, tr_s, tr_e, t_, c_)["ret_ew"], 52 / c_)
                        if sh > bs:
                            bs, b = sh, (t_, c_)
                frozen = b
            tn, cad = frozen
            best_sh = _sharpe(simulate_window(pc, tr_s, tr_e, tn, cad)["ret_ew"], 52 / cad)
        else:  # optimized — argmax over grid (the overfit-prone default)
            best, best_sh = None, -1e9
            for t_ in grid_top_n:
                for c_ in grid_cadence:
                    sh = _sharpe(simulate_window(pc, tr_s, tr_e, t_, c_)["ret_ew"], 52 / c_)
                    if sh > best_sh:
                        best_sh, best = sh, (t_, c_)
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
        "oos_returns": oos_ret,
        "oos_metrics": oos_metrics, "spy_metrics": spy_metrics,
        "rolls": rolls, "is_sharpe": round(is_sharpe, 2), "oos_sharpe": oos_sharpe,
        "wfe": wfe, "n_rolls": len(rolls), "param_mode": param_mode,
        "train_weeks": train_weeks, "test_weeks": test_weeks,
    }


# ── Robustness statistics (no scipy: erf-based PSR + block bootstrap) ──────────
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skew_kurt(r):
    n = len(r); m = r.mean(); s = r.std()
    if s == 0 or n < 3:
        return 0.0, 3.0
    g3 = float(((r - m) ** 3).mean() / s ** 3)
    g4 = float(((r - m) ** 4).mean() / s ** 4)   # non-excess kurtosis
    return g3, g4


def psr(rets, ppy, thr_annual=0.0):
    """Probabilistic Sharpe Ratio: P(true Sharpe > threshold), skew/kurtosis-adjusted.
    Returns a probability 0..1. thr_annual=0 → P(Sharpe > 0)."""
    r = np.asarray(rets, float); r = r[~np.isnan(r)]
    N = len(r)
    if N < 3 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()                  # per-period
    thr = thr_annual / math.sqrt(ppy)        # annual → per-period
    g3, g4 = _skew_kurt(r)
    denom = math.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr, 1e-9))
    z = (sr - thr) * math.sqrt(N - 1) / denom
    return float(_norm_cdf(z))


def block_bootstrap_metrics(rets, ppy, n_boot=800, block=4, seed=0):
    """Stationary-ish block bootstrap → distribution of annualized Sharpe & CAGR.
    Blocks preserve short-run autocorrelation so the CI isn't falsely tight."""
    r = np.asarray(rets, float); r = r[~np.isnan(r)]
    N = len(r)
    if N < block + 2:
        return {}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(N / block))
    yrs = N / ppy
    shs, cgs = [], []
    for _ in range(n_boot):
        st = rng.integers(0, N - block + 1, size=nblk)
        samp = np.concatenate([r[s:s + block] for s in st])[:N]
        sd = samp.std()
        shs.append((samp.mean() * ppy) / (sd * math.sqrt(ppy)) if sd > 0 else 0.0)
        eq = float(np.prod(1.0 + samp))
        cgs.append((eq ** (1.0 / yrs) - 1.0) if eq > 0 else -1.0)
    shs = np.array(shs); cgs = np.array(cgs)
    return {
        "sharpe_med": float(np.median(shs)),
        "sharpe_lo": float(np.percentile(shs, 5)),
        "sharpe_hi": float(np.percentile(shs, 95)),
        "p_sharpe_pos": float((shs > 0).mean()),
        "cagr_lo_pct": float(np.percentile(cgs, 5) * 100),
        "cagr_hi_pct": float(np.percentile(cgs, 95) * 100),
    }


def robustness_compare(ohlcv, pairs, signal="extpct", train_weeks=104, test_weeks=26,
                       vol_lookback=13, min_hist=45, fixed_params=(10, 4),
                       n_boot=800, progress_cb=None) -> dict:
    """Run the walk-forward under all three param modes and attach bootstrap + PSR.
    The comparison answers: is the edge durable, or grid-search selection?"""
    modes = [("Optimized (argmax/window)", "optimized"),
             ("Frozen (pick once)", "frozen"),
             (f"Fixed ({fixed_params[0]}/{fixed_params[1]})", "fixed")]
    rows, curves = [], {}
    spy_curve = None
    for li, (label, mode) in enumerate(modes):
        wf = walk_forward(ohlcv, pairs, train_weeks, test_weeks,
                          vol_lookback=vol_lookback, min_hist=min_hist, signal=signal,
                          param_mode=mode, fixed_params=fixed_params)
        if "error" in wf:
            return wf
        cads = [r["Cadence"] for r in wf["rolls"]] or [4]
        ppy = 52 / max(int(np.median(cads)), 1)
        boot = block_bootstrap_metrics(wf["oos_returns"], ppy, n_boot=n_boot)
        p0 = psr(wf["oos_returns"], ppy, 0.0)
        m = wf["oos_metrics"]
        rows.append({
            "Mode": label,
            "IS Sharpe": wf["is_sharpe"],
            "OOS Sharpe": wf["oos_sharpe"],
            "WFE": wf["wfe"],
            "CAGR %": m.get("CAGR %"),
            "MaxDD %": m.get("Max Drawdown %"),
            "PSR(>0)": round(p0, 2) if p0 == p0 else None,
            "Boot Sharpe 5–95%": (f"{boot['sharpe_lo']:.2f} … {boot['sharpe_hi']:.2f}"
                                  if boot else "—"),
            "P(Sh>0)": round(boot.get("p_sharpe_pos"), 2) if boot else None,
        })
        curves[label] = {"dates": wf["oos_dates"], "equity": wf["oos_equity"]}
        spy_curve = {"dates": wf["oos_dates"], "equity": wf["spy_equity"]}
        if progress_cb:
            progress_cb((li + 1) / len(modes))
    return {"rows": rows, "curves": curves, "spy": spy_curve}


def validate_risk_layer(ohlcv, pairs, signal="extpct", fixed_params=(10, 4),
                        train_weeks=104, test_weeks=26, vol_lookback=13, min_hist=45,
                        risk_params=None, n_boot=600, progress_cb=None) -> dict:
    """Frozen-config walk-forward comparing the RAW inverse-vol book vs the
    RISK-LAYERED book (same picks, same windows — only the risk layer differs).
    Answers: does the risk layer cut drawdown without gutting Sharpe?"""
    risk_params = risk_params or {"target_vol": 0.18, "max_pos": 0.20,
                                  "max_sector": 0.40, "per_trade_risk": 0.01}
    if not _RISK_AVAIL:
        return {"error": "Risk layer unavailable — utils/risk.py isn't imported by the "
                         "running strategy_backtest.py. Push utils/risk.py FIRST, delete any "
                         "committed utils/__pycache__/, then Reboot the app."}
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist, signal=signal)
    if "error" in pc:
        return pc
    cal = pc["cal"]; n = len(cal); first = min_hist
    tn, cad = fixed_params
    if first + train_weeks + test_weeks > n:
        return {"error": f"Not enough history: need ≥ {first+train_weeks+test_weeks} weeks, have {n}."}
    starts = list(range(first, n - train_weeks - test_weeks + 1, test_weeks))
    raw_ret, risk_ret, spy_ret, dates = [], [], [], []
    raw_eq, risk_eq, spy_eq = [1.0], [1.0], [1.0]
    for k, i in enumerate(starts):
        te_s, te_e = i + train_weeks, i + train_weeks + test_weeks
        s = simulate_window(pc, te_s, te_e, tn, cad, risk_params=risk_params)
        m = min(len(s["ret_vol"]), len(s["ret_risk"]), len(s["ret_spy"]))
        for j in range(m):
            rr, rk, sp = s["ret_vol"][j], s["ret_risk"][j], s["ret_spy"][j]
            raw_ret.append(rr); risk_ret.append(rk); spy_ret.append(sp)
            raw_eq.append(raw_eq[-1] * (1 + rr))
            risk_eq.append(risk_eq[-1] * (1 + rk))
            spy_eq.append(spy_eq[-1] * (1 + sp))
            dates.append(s["dates"][j + 1])
        if progress_cb:
            progress_cb((k + 1) / max(len(starts), 1))
    if len(raw_eq) < 3:
        return {"error": "Too few out-of-sample points."}
    d0 = [dates[0]] + dates
    ppy = 52 / cad
    return {
        "dates": d0, "raw_equity": raw_eq, "risk_equity": risk_eq, "spy_equity": spy_eq,
        "raw_metrics": metrics_from_equity(d0, raw_eq, raw_ret, ppy),
        "risk_metrics": metrics_from_equity(d0, risk_eq, risk_ret, ppy),
        "spy_metrics": metrics_from_equity(d0, spy_eq, spy_ret, ppy),
        "raw_boot": block_bootstrap_metrics(raw_ret, ppy, n_boot=n_boot),
        "risk_boot": block_bootstrap_metrics(risk_ret, ppy, n_boot=n_boot),
        "risk_params": risk_params, "fixed_params": fixed_params,
    }


def cost_stress(ohlcv, pairs, signal="extpct", fixed_params=(10, 4),
                cost_levels=(0, 5, 10, 25), train_weeks=104, test_weeks=26,
                vol_lookback=13, min_hist=45, risk_params=None, progress_cb=None) -> dict:
    """Frozen-config walk-forward swept across per-side transaction costs (bps).
    Shows how OOS Sharpe/CAGR/MaxDD decay with cost for both the raw inverse-vol
    book and the risk-layered book — the test of whether the edge survives reality.
    Turnover (cost-independent) is reported so the cost mechanism is visible."""
    risk_params = risk_params or {"target_vol": 0.18, "max_pos": 0.20,
                                  "max_sector": 0.40, "per_trade_risk": 0.01}
    pc = precompute_series(ohlcv, pairs, vol_lookback, min_hist, signal=signal)
    if "error" in pc:
        return pc
    cal = pc["cal"]; n = len(cal); first = min_hist
    tn, cad = fixed_params
    if first + train_weeks + test_weeks > n:
        return {"error": f"Not enough history: need ≥ {first+train_weeks+test_weeks} weeks, have {n}."}
    starts = list(range(first, n - train_weeks - test_weeks + 1, test_weeks))
    do_risk = _RISK_AVAIL
    rows, turn_vol_all, turn_risk_all = [], [], []
    for ci, cost in enumerate(cost_levels):
        raw_ret, risk_ret, spy_ret, dates = [], [], [], []
        raw_eq, risk_eq, spy_eq = [1.0], [1.0], [1.0]
        for i in starts:
            te_s, te_e = i + train_weeks, i + train_weeks + test_weeks
            s = simulate_window(pc, te_s, te_e, tn, cad, cost_bps=cost,
                                risk_params=(risk_params if do_risk else None))
            mlen = (min(len(s["ret_vol"]), len(s["ret_risk"]), len(s["ret_spy"]))
                    if do_risk else min(len(s["ret_vol"]), len(s["ret_spy"])))
            for j in range(mlen):
                rr, sp = s["ret_vol"][j], s["ret_spy"][j]
                rk = s["ret_risk"][j] if do_risk else rr
                raw_ret.append(rr); risk_ret.append(rk); spy_ret.append(sp)
                raw_eq.append(raw_eq[-1] * (1 + rr))
                risk_eq.append(risk_eq[-1] * (1 + rk))
                spy_eq.append(spy_eq[-1] * (1 + sp))
                dates.append(s["dates"][j + 1])
            if ci == 0:
                turn_vol_all.append(s["avg_turnover_vol"] * (52 / cad))
                if do_risk:
                    turn_risk_all.append(s["avg_turnover_risk"] * (52 / cad))
        if len(raw_eq) < 3:
            continue
        d0 = [dates[0]] + dates; ppy = 52 / cad
        rm = metrics_from_equity(d0, raw_eq, raw_ret, ppy)
        km = metrics_from_equity(d0, risk_eq, risk_ret, ppy)
        rows.append({
            "Cost bps/side": cost,
            "Raw Sharpe": rm.get("Sharpe"), "Raw CAGR %": rm.get("CAGR %"),
            "Raw MaxDD %": rm.get("Max Drawdown %"),
            "Risk Sharpe": km.get("Sharpe"), "Risk CAGR %": km.get("CAGR %"),
            "Risk MaxDD %": km.get("Max Drawdown %"),
        })
        if progress_cb:
            progress_cb((ci + 1) / max(len(cost_levels), 1))
    return {"rows": rows, "fixed_params": fixed_params, "do_risk": do_risk,
            "raw_turnover_pct": round(float(np.mean(turn_vol_all)) * 100, 0) if turn_vol_all else None,
            "risk_turnover_pct": round(float(np.mean(turn_risk_all)) * 100, 0) if turn_risk_all else None}
