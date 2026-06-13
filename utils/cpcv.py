"""
utils/cpcv.py
Combinatorially-Symmetric Cross-Validation → Probability of Backtest
Overfitting (Bailey, Borwein, López de Prado, Zhu 2014). The EMPIRICAL
complement to the analytic Deflated Sharpe Ratio in utils/significance.py.

The question PBO answers: when you pick the best config in-sample, how often
does it land BELOW the median config out-of-sample? That fraction IS the
probability your selection process is overfit.

  PBO → 0    : in-sample winners stay winners out-of-sample (robust selection)
  PBO → 0.5  : in-sample winners are random out-of-sample (overfit / no edge)

Method (CSCV):
  • Split the T return rows into S equal contiguous blocks (S even).
  • For every way to choose S/2 blocks as the in-sample (IS) set, the
    complement is the out-of-sample (OOS) set — C(S, S/2) symmetric splits.
  • Per split: pick the config with best IS Sharpe; find its OOS rank among
    all configs; logit of that relative rank. PBO = P(logit <= 0).
  • Block stats (sum, sumsq, count) are precomputed so each of the up-to
    ~12,870 splits is O(1) per config — fast, no scipy.

Also returns the OOS-Sharpe DISTRIBUTION of a designated production config
across all splits — a confidence interval on the frozen strategy, the
"distribution rather than a point estimate" benefit of combinatorial CV.

Caveat for StratFlow: this validates the CONFIG-SELECTION process over a
family of trials; it needs the per-config weekly return matrix (T x N). With
a weekly cadence the number of independent blocks is limited, so the lab
reports n_combinations and block size to judge whether the distribution is
trustworthy or too thin.
"""
from __future__ import annotations
import math
import itertools

import numpy as np
import pandas as pd


def _block_stats(R: np.ndarray, S: int):
    """Split T rows into S contiguous blocks; per-block sum, sumsq, count."""
    T = R.shape[0]
    m = (T // S) * S
    R = R[:m]
    blocks = np.array_split(R, S, axis=0)
    sums = np.array([b.sum(axis=0) for b in blocks])        # S x N
    sumsq = np.array([(b ** 2).sum(axis=0) for b in blocks])  # S x N
    counts = np.array([b.shape[0] for b in blocks], dtype=float)  # S
    return sums, sumsq, counts


def _pooled_sharpe(sums, sumsq, counts, block_idx) -> np.ndarray:
    """Per-period Sharpe per config over the union of the given blocks."""
    s = sums[block_idx].sum(axis=0)
    sq = sumsq[block_idx].sum(axis=0)
    n = counts[block_idx].sum()
    mean = s / n
    var = sq / n - mean ** 2
    var = np.where(var <= 0, np.nan, var)
    return mean / np.sqrt(var)


def cscv_pbo(returns_matrix, S: int = 16, prod_idx: int | None = None,
             ppy: int = 52) -> dict:
    """returns_matrix: T x N (rows=periods, cols=configs/trials).
    prod_idx: optional column index of the production config to track."""
    R = np.asarray(returns_matrix, dtype=float)
    R = R[~np.isnan(R).any(axis=1)]
    T, N = R.shape
    if N < 2:
        raise ValueError("PBO needs >= 2 configs")
    if S % 2 != 0:
        S -= 1
    if S < 4 or T < S * 2:
        raise ValueError(f"need T >= {S*2} rows for S={S}")
    sums, sumsq, counts = _block_stats(R, S)
    blocks = list(range(S))
    logits, oos_isbest, prod_oos = [], [], []
    is_best_count = np.zeros(N, dtype=int)
    for is_set in itertools.combinations(blocks, S // 2):
        oos_set = [b for b in blocks if b not in is_set]
        is_sr = _pooled_sharpe(sums, sumsq, counts, list(is_set))
        oos_sr = _pooled_sharpe(sums, sumsq, counts, oos_set)
        if np.all(np.isnan(is_sr)):
            continue
        n_star = int(np.nanargmax(is_sr))
        is_best_count[n_star] += 1
        order = np.argsort(np.where(np.isnan(oos_sr), -np.inf, oos_sr))
        rank = int(np.where(order == n_star)[0][0]) + 1     # 1..N (N=best)
        omega = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
        v = oos_sr[n_star]
        oos_isbest.append(float(v) if not np.isnan(v) else np.nan)
        if prod_idx is not None:
            pv = oos_sr[prod_idx]
            prod_oos.append(float(pv) if not np.isnan(pv) else np.nan)
    logits = np.array(logits)
    pbo = float((logits <= 0).mean()) if len(logits) else float("nan")
    out = {"pbo": round(pbo, 4),
           "n_combinations": int(len(logits)),
           "S": int(S), "n_configs": int(N),
           "logit_median": round(float(np.median(logits)), 3) if len(logits) else None,
           "oos_sharpe_isbest_median_ann":
               round(float(np.nanmedian(oos_isbest)) * math.sqrt(ppy), 3)
               if len(oos_isbest) else None,
           "is_best_distribution": is_best_count.tolist()}
    if prod_idx is not None and prod_oos:
        arr = np.array(prod_oos) * math.sqrt(ppy)
        arr = arr[~np.isnan(arr)]
        out["prod_oos_sharpe"] = {
            "median": round(float(np.median(arr)), 3),
            "p05": round(float(np.percentile(arr, 5)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "prob_negative": round(float((arr < 0).mean()), 4)}
    return out


def verdict(pbo_result: dict) -> str:
    p = pbo_result["pbo"]
    flag = "✅" if p <= 0.10 else "🟠" if p <= 0.25 else "🛑"
    extra = ""
    if "prod_oos_sharpe" in pbo_result:
        d = pbo_result["prod_oos_sharpe"]
        extra = (f" · production OOS Sharpe median {d['median']} "
                 f"(5–95%: {d['p05']}–{d['p95']}, P(<0)={d['prob_negative']})")
    return (f"{flag} PBO {p:.3f} over {pbo_result['n_combinations']} splits "
            f"(S={pbo_result['S']}, {pbo_result['n_configs']} configs){extra}")
