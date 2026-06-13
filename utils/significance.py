"""
utils/significance.py
Statistical significance for backtests — the formal version of the discipline
StratFlow already practices by hand (test whole families, refuse the nominal
best, document negatives). These are deterministic formulas, not ML.

Core tools (Bailey & López de Prado):
  • Probabilistic Sharpe Ratio (PSR) — P(true Sharpe > a benchmark), correcting
    for sample length, skew, and (non-normal) kurtosis.
  • Deflated Sharpe Ratio (DSR) — PSR against the EXPECTED-MAXIMUM Sharpe that
    N independent trials would produce by luck alone. This is the multiple-
    testing correction: the more configs you tried, the higher the bar.
  • Minimum Track Record Length — observations needed to call a Sharpe real.
  • Harvey-Liu factor-zoo hurdle — implied t-stat should clear 3.0, not 2.0.

Frequency convention: helpers take ANNUALISED Sharpe plus periods-per-year
(ppy, weekly = 52) and convert internally to the per-period units the formulas
require. kurtosis is non-excess (normal = 3); stats_from_returns handles the
pandas excess-kurtosis (+3) conversion for you.

No scipy dependency — uses statistics.NormalDist (stdlib) for the normal CDF
and its inverse.
"""
from __future__ import annotations
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
GAMMA = 0.5772156649015329   # Euler–Mascheroni


def _phi(x: float) -> float:
    return _N.cdf(x)


def _phinv(p: float) -> float:
    return _N.inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def probabilistic_sharpe_ratio(sr_periodic: float, n: int, skew: float = 0.0,
                               kurt: float = 3.0,
                               sr_star_periodic: float = 0.0) -> float:
    """P(true Sharpe > sr_star). All Sharpe inputs PER-PERIOD (same frequency
    as the n observations). kurt is non-excess (normal = 3)."""
    if n is None or n < 2:
        return float("nan")
    denom = 1.0 - skew * sr_periodic + ((kurt - 1.0) / 4.0) * sr_periodic ** 2
    if denom <= 0:
        denom = 1e-12
    z = (sr_periodic - sr_star_periodic) * math.sqrt(n - 1) / math.sqrt(denom)
    return _phi(z)


def expected_max_sharpe(n_trials: int, var_sr_periodic: float) -> float:
    """Expected maximum of n_trials i.i.d. Sharpe estimates whose cross-trial
    variance is var_sr_periodic (per-period). The luck threshold for DSR."""
    if n_trials is None or n_trials < 2 or var_sr_periodic <= 0:
        return 0.0
    z1 = _phinv(1.0 - 1.0 / n_trials)
    z2 = _phinv(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_periodic) * ((1.0 - GAMMA) * z1 + GAMMA * z2)


def deflated_sharpe_ratio(best_sr_ann: float, all_sr_ann, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0,
                          ppy: int = 52) -> dict:
    """Deflated Sharpe Ratio for the best config among the trials.

    best_sr_ann : annualised Sharpe of the chosen/best config
    all_sr_ann  : iterable of annualised Sharpes of EVERY config tried
    n_obs       : number of return observations (e.g. weeks)
    skew, kurt  : higher moments of the best config's returns (kurt non-excess)

    Returns {dsr, sr0_ann, n_trials, t_stat, harvey_pass, psr0}.
    sr0_ann is the luck threshold (the Sharpe N trials would beat by chance);
    a DSR near 1.0 means the result survives the multiple-testing correction.
    """
    a = np.asarray([s for s in all_sr_ann if s is not None and np.isfinite(s)],
                   dtype=float)
    n_trials = int(len(a))
    bs = best_sr_ann / math.sqrt(ppy)
    sp = a / math.sqrt(ppy)
    var_sr = float(np.var(sp, ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    dsr = probabilistic_sharpe_ratio(bs, n_obs, skew, kurt, sr_star_periodic=sr0)
    psr0 = probabilistic_sharpe_ratio(bs, n_obs, skew, kurt, 0.0)
    t_stat = bs * math.sqrt(n_obs) if (n_obs and n_obs > 0) else float("nan")
    return {"dsr": round(float(dsr), 4),
            "sr0_ann": round(float(sr0 * math.sqrt(ppy)), 4),
            "n_trials": n_trials,
            "t_stat": round(float(t_stat), 3),
            "harvey_pass": bool(t_stat > 3.0),
            "psr0": round(float(psr0), 4)}


def min_track_record_length(sr_ann: float, skew: float = 0.0, kurt: float = 3.0,
                            sr_star_ann: float = 0.0, target: float = 0.95,
                            ppy: int = 52) -> float:
    """Observations needed for PSR(sr_star) >= target. Inf if Sharpe <= star."""
    sr = sr_ann / math.sqrt(ppy)
    srs = sr_star_ann / math.sqrt(ppy)
    if sr <= srs:
        return float("inf")
    z = _phinv(target)
    return 1.0 + (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2) * (z / (sr - srs)) ** 2


def stats_from_returns(returns, ppy: int = 52):
    """(sr_ann, skew, kurt_nonexcess, n) from a periodic return series.
    Converts pandas excess kurtosis (+3) to the non-excess form the PSR uses."""
    r = pd.Series(returns).dropna()
    n = int(len(r))
    if n < 2 or r.std(ddof=1) == 0:
        return (float("nan"), 0.0, 3.0, n)
    sr_ann = r.mean() / r.std(ddof=1) * math.sqrt(ppy)
    return (float(sr_ann), float(r.skew()), float(r.kurt() + 3.0), n)


def verdict(dsr_result: dict) -> str:
    """One-line readout for a lab's JSON/UI."""
    d = dsr_result
    flag = ("✅" if d["dsr"] >= 0.95 and d["harvey_pass"] else
            "🟠" if d["dsr"] >= 0.90 else "🛑")
    return (f"{flag} DSR {d['dsr']:.3f} after {d['n_trials']} trials "
            f"(luck threshold SR0 {d['sr0_ann']:.2f}, t={d['t_stat']:.2f}, "
            f"Harvey t>3 {'pass' if d['harvey_pass'] else 'fail'})")
