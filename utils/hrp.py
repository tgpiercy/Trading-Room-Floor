"""
utils/hrp.py
Hierarchical Risk Parity (López de Prado, 2016) — correlation-aware position
sizing. Self-contained: single-linkage agglomerative clustering + quasi-
diagonal ordering + recursive bisection. No scipy dependency (consistent with
utils/significance.py).

Why it might help: naive inverse-vol sizing ignores correlations, so a cluster
of correlated names quietly accumulates concentrated risk. HRP groups names by
correlation and allocates risk across clusters, down-weighting crowded groups.
It also never inverts the covariance matrix, so it stays robust when that
matrix is ill-conditioned (the regime where mean-variance optimisers blow up).

Why it might NOT help in StratFlow: the redundancy filter already removes the
worst correlation clusters at entry, so HRP may find little left to fix. The
HRP lab tests exactly that, against inverse-vol and equal-weight baselines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """López de Prado distance: d_ij = sqrt(0.5 * (1 - rho_ij))."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def _single_linkage_order(dist: np.ndarray) -> list:
    """Agglomerative single-linkage clustering; returns the quasi-diagonal
    leaf order (correlated names placed adjacent). O(n^3), fine for n<=~25."""
    n = dist.shape[0]
    if n <= 1:
        return list(range(n))
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    cdist = {}
    for i in range(n):
        for j in range(i + 1, n):
            cdist[(i, j)] = float(dist[i, j])
    nxt = n
    while len(active) > 1:
        best, bd = None, np.inf
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                ca, cb = active[ai], active[bi]
                d = cdist.get((min(ca, cb), max(ca, cb)), np.inf)
                if d < bd:
                    bd, best = d, (ca, cb)
        ca, cb = best
        members[nxt] = members[ca] + members[cb]
        for cx in active:
            if cx in (ca, cb):
                continue
            da = cdist.get((min(ca, cx), max(ca, cx)), np.inf)
            db = cdist.get((min(cb, cx), max(cb, cx)), np.inf)
            cdist[(min(nxt, cx), max(nxt, cx))] = min(da, db)
        active.remove(ca)
        active.remove(cb)
        active.append(nxt)
        nxt += 1
    return members[active[0]]


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Inverse-variance-weighted variance of a cluster (no matrix inversion)."""
    sub = cov.loc[items, items].values
    diag = np.diag(sub).astype(float)
    diag = np.where(diag <= 0, 1e-12, diag)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """HRP weights (sum to 1, long-only). cov/corr indexed by name."""
    names = list(cov.index)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=names)
    order_idx = _single_linkage_order(_corr_distance(corr.values))
    order = [names[i] for i in order_idx]
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c1, c2 = c[:half], c[half:]
            v1, v2 = _cluster_var(cov, c1), _cluster_var(cov, c2)
            alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            for it in c1:
                w[it] *= alpha
            for it in c2:
                w[it] *= (1.0 - alpha)
            nxt += [c1, c2]
        clusters = nxt
    w = w / w.sum()
    return w.reindex(names)


def hrp_from_returns(rets: pd.DataFrame) -> pd.Series:
    """HRP weights from a window of returns (columns = names). Drops zero-
    variance names; falls back to inverse-variance if clustering fails."""
    r = rets.dropna(axis=1, how="all")
    r = r.loc[:, r.std() > 0]
    if r.shape[1] == 0:
        return pd.Series(dtype=float)
    if r.shape[1] == 1:
        return pd.Series([1.0], index=list(r.columns))
    cov = r.cov()
    corr_vals = np.array(r.corr().fillna(0.0).values, dtype=float)  # writable copy
    np.fill_diagonal(corr_vals, 1.0)
    corr = pd.DataFrame(corr_vals, index=list(r.columns), columns=list(r.columns))
    try:
        w = hrp_weights(cov, corr)
        if w.isna().any() or w.sum() <= 0:
            raise ValueError("degenerate HRP weights")
        return w
    except Exception:
        iv = 1.0 / r.var()
        return iv / iv.sum()
