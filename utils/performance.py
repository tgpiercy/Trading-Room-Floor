"""
utils/performance.py
Realized-performance analytics for the live account — the loop-closer that
answers "is the validated ~1.67-Sharpe edge actually showing up in reality?"

Works on a TOTAL-EQUITY series (invested mark-to-market + cash), because the
regime gate's cash periods are part of the strategy and the backtest's equity
curve includes them — comparing invested-value-only against a total-equity
backtest would be apples to oranges.

Snapshots are logged at an irregular cadence (whenever a page is visited), so
metrics resample to a regular weekly grid before annualising. Reuses
utils.significance for the honest "is this track record long enough to trust?"
read, so realized numbers carry the same statistical humility as the labs.

Pure / dependency-light (pandas, numpy) — unit-testable without network.
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

ANN = math.sqrt(52)
EXPECTED_SHARPE = 1.67          # the validated backtest headline


def _to_weekly(equity: pd.Series) -> pd.Series:
    """Irregular snapshots → regular weekly (W-FRI), forward-filled."""
    s = pd.Series(equity).dropna()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 2:
        return s
    return s.resample("W-FRI").last().ffill().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    s = _to_weekly(equity)
    if s.empty:
        return s
    return s / s.cummax() - 1.0


def equity_metrics(equity: pd.Series) -> dict:
    """Realized metrics from a (possibly irregular) total-equity series."""
    s = _to_weekly(equity)
    n = len(s)
    if n < 2:
        return {"n_weeks": n, "insufficient": True}
    rets = s.pct_change().dropna()
    span_days = (s.index[-1] - s.index[0]).days
    span_yrs = max(span_days / 365.25, 1e-9)
    total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / span_yrs) - 1) if s.iloc[0] > 0 else float("nan")
    vol = float(rets.std() * ANN) if rets.std() > 0 else 0.0
    sharpe = float(rets.mean() / rets.std() * ANN) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * ANN) if len(downside) > 1 and downside.std() > 0 else float("nan")
    dd = drawdown_series(s)
    return {
        "n_weeks": int(n), "span_days": int(span_days),
        "total_return": round(total_ret, 4), "cagr": round(cagr, 4),
        "vol_ann": round(vol, 4), "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if sortino == sortino else None,
        "max_dd": round(float(dd.min()), 4), "current_dd": round(float(dd.iloc[-1]), 4),
        "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
        "start_equity": round(float(s.iloc[0]), 2), "end_equity": round(float(s.iloc[-1]), 2),
        "insufficient": n < 8,
    }


def benchmark_compare(equity: pd.Series, bench: pd.Series) -> dict:
    """Compare the account against a benchmark equity series (e.g. SPY)."""
    p = _to_weekly(equity)
    b = _to_weekly(bench)
    idx = p.index.intersection(b.index)
    if len(idx) < 3:
        return {"insufficient": True}
    pr = p.reindex(idx).pct_change().dropna()
    br = b.reindex(idx).pct_change().dropna()
    j = pr.index.intersection(br.index)
    pr, br = pr.reindex(j), br.reindex(j)
    if len(j) < 2 or br.std() == 0:
        return {"insufficient": True}
    excess = pr - br
    beta = float(pr.cov(br) / br.var()) if br.var() > 0 else float("nan")
    up = br > 0
    down = br < 0
    up_cap = float(pr[up].mean() / br[up].mean()) if up.any() and br[up].mean() != 0 else float("nan")
    down_cap = float(pr[down].mean() / br[down].mean()) if down.any() and br[down].mean() != 0 else float("nan")
    return {
        "weeks": int(len(j)),
        "excess_cagr": round(float((pr.mean() - br.mean()) * 52), 4),
        "tracking_error": round(float(excess.std() * ANN), 4),
        "beta": round(beta, 3), "corr": round(float(pr.corr(br)), 3),
        "up_capture": round(up_cap, 3) if up_cap == up_cap else None,
        "down_capture": round(down_cap, 3) if down_cap == down_cap else None,
        "bench_sharpe": round(float(br.mean() / br.std() * ANN), 3) if br.std() > 0 else 0.0,
    }


def vs_expectation(realized_sharpe: float, n_weeks: int,
                   expected: float = EXPECTED_SHARPE) -> dict:
    """How does realized Sharpe compare to the validated backtest, and is the
    track record even long enough to tell? Reuses the min-track-record-length
    machinery so the verdict carries real statistical humility."""
    try:
        from utils.significance import min_track_record_length
        need = min_track_record_length(max(realized_sharpe, 0.01), target=0.95)
    except Exception:
        need = float("inf")
    enough = n_weeks >= need and n_weeks >= 8
    gap = realized_sharpe - expected
    if not enough:
        verdict = ("🟠 Track record too short to judge — need ~"
                   f"{need:.0f} weeks for the realized Sharpe to be "
                   f"statistically meaningful (have {n_weeks}). Keep logging.")
    elif gap >= -0.3:
        verdict = (f"✅ Realized Sharpe {realized_sharpe:.2f} is in line with "
                   f"the {expected:.2f} backtest (gap {gap:+.2f}). The edge is "
                   "showing up live.")
    else:
        verdict = (f"🛑 Realized Sharpe {realized_sharpe:.2f} is well below the "
                   f"{expected:.2f} backtest (gap {gap:+.2f}) — investigate "
                   "slippage, timing, tax drag, or the survivorship gap.")
    return {"realized": round(realized_sharpe, 3), "expected": expected,
            "gap": round(gap, 3), "weeks_needed_95pct": round(float(need), 1),
            "enough_data": bool(enough), "verdict": verdict}
