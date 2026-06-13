"""
utils/hrp.py
Hierarchical Risk Parity (Lopez de Prado), scipy-free.
Single-linkage clustering + quasi-diagonalization + recursive bisection.

Public API:
    hrp_weights(returns_df) -> dict[ticker -> weight]
    hrp_weights_from_cov(cov, tickers) -> dict[ticker -> weight]
"""

import math


# ---------------------------------------------------------------------------
# Covariance / correlation helpers (no numpy/pandas required, but pandas ok)
# ---------------------------------------------------------------------------

def _cov_to_corr(cov):
    """cov: list of lists (n x n). Returns correlation matrix (list of lists)."""
    n = len(cov)
    std = [math.sqrt(max(cov[i][i], 1e-16)) for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            denom = std[i] * std[j]
            if denom <= 0:
                corr[i][j] = 1.0 if i == j else 0.0
            else:
                c = cov[i][j] / denom
                c = max(-1.0, min(1.0, c))
                corr[i][j] = c
    return corr


def _corr_to_dist(corr):
    """Distance matrix from correlation, per Lopez de Prado: d = sqrt(0.5*(1-rho))."""
    n = len(corr)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = math.sqrt(max(0.0, 0.5 * (1.0 - corr[i][j])))
            dist[i][j] = d
    return dist


# ---------------------------------------------------------------------------
# Single-linkage clustering
# ---------------------------------------------------------------------------

def _single_linkage(dist):
    """
    Manual single-linkage agglomerative clustering.

    dist: n x n distance matrix (list of lists).
    Returns a linkage list of merges: [(cluster_a, cluster_b, distance, new_cluster_id), ...]
    in the order they were merged, plus a dict mapping cluster_id -> list of original leaf indices.

    Cluster ids 0..n-1 are original leaves; new clusters get ids n, n+1, ...
    """
    n = len(dist)
    if n == 1:
        return [], {0: [0]}

    # active clusters: id -> list of leaves
    members = {i: [i] for i in range(n)}
    # pairwise distance between active clusters (single linkage = min dist between members)
    active = list(range(n))

    # working distance cache between active cluster ids
    cdist = {}
    for i in range(n):
        for j in range(i + 1, n):
            cdist[(i, j)] = dist[i][j]

    next_id = n
    linkage = []

    while len(active) > 1:
        # find closest pair
        best = None
        best_d = None
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                a, b = active[a_idx], active[b_idx]
                key = (a, b) if a < b else (b, a)
                d = cdist[key]
                if best_d is None or d < best_d:
                    best_d = d
                    best = (a, b)

        a, b = best
        new_members = members[a] + members[b]
        new_id = next_id
        next_id += 1
        members[new_id] = new_members

        # compute single-linkage distance from new cluster to every remaining cluster
        remaining = [c for c in active if c not in (a, b)]
        for c in remaining:
            key_a = (a, c) if a < c else (c, a)
            key_b = (b, c) if b < c else (c, b)
            d_min = min(cdist[key_a], cdist[key_b])
            key_new = (new_id, c) if new_id < c else (c, new_id)
            cdist[key_new] = d_min

        # cleanup old keys involving a or b (optional, just leave them - harmless)
        active = remaining + [new_id]

        linkage.append((a, b, best_d, new_id))

    return linkage, members


def _quasi_diag_order(linkage, n):
    """
    Derive a leaf ordering (quasi-diagonalization) from the linkage list.
    Recursively expand the final merged cluster, always placing the two
    children's leaves contiguously.
    """
    if not linkage:
        return [0] if n == 1 else list(range(n))

    # build children map: cluster_id -> (left, right)
    children = {}
    for a, b, d, new_id in linkage:
        children[new_id] = (a, b)

    root = linkage[-1][3]

    def expand(cluster_id):
        if cluster_id < n:
            return [cluster_id]
        left, right = children[cluster_id]
        return expand(left) + expand(right)

    return expand(root)


# ---------------------------------------------------------------------------
# Recursive bisection for HRP weights
# ---------------------------------------------------------------------------

def _cluster_var(cov, indices):
    """Inverse-variance portfolio variance for a sub-cluster (indices into cov)."""
    if len(indices) == 1:
        i = indices[0]
        return max(cov[i][i], 1e-16)

    # inverse-variance weights within the cluster
    ivp = []
    for i in indices:
        v = max(cov[i][i], 1e-16)
        ivp.append(1.0 / v)
    s = sum(ivp)
    weights = [w / s for w in ivp]

    # portfolio variance = w' Cov w
    var = 0.0
    for a_pos, i in enumerate(indices):
        for b_pos, j in enumerate(indices):
            var += weights[a_pos] * weights[b_pos] * cov[i][j]
    return max(var, 1e-16)


def _recursive_bisection(cov, sorted_indices):
    """
    Returns dict: original_index -> weight, summing to 1.
    sorted_indices: leaf indices (0..n-1) in quasi-diagonal order.
    """
    n = len(sorted_indices)
    weights = {i: 1.0 for i in sorted_indices}

    if n == 1:
        return {sorted_indices[0]: 1.0}

    clusters = [sorted_indices]

    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left = c[:mid]
            right = c[mid:]

            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)

            denom = var_left + var_right
            if denom <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / denom  # alpha given to left

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters

    return weights


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def hrp_weights_from_cov(cov, tickers):
    """
    cov: n x n list of lists (covariance matrix), aligned to `tickers` order.
    tickers: list of n ticker strings.

    Returns dict {ticker: weight}, weights non-negative and summing to 1.
    """
    n = len(tickers)

    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}
    if n == 2:
        # closed-form inverse-variance split (single bisection step = HRP for n=2)
        v0 = max(cov[0][0], 1e-16)
        v1 = max(cov[1][1], 1e-16)
        denom = v0 + v1
        if denom <= 0:
            w0 = w1 = 0.5
        else:
            # alpha to asset 0 in inverse-variance cluster split
            w0 = 1.0 - v0 / denom
            w1 = 1.0 - w0
        return {tickers[0]: w0, tickers[1]: w1}

    corr = _cov_to_corr(cov)
    dist = _corr_to_dist(corr)
    linkage, _members = _single_linkage(dist)
    order = _quasi_diag_order(linkage, n)
    raw_weights = _recursive_bisection(cov, order)

    # normalize defensively (should already sum to 1)
    total = sum(raw_weights.values())
    if total <= 0:
        eq = 1.0 / n
        return {t: eq for t in tickers}

    return {tickers[i]: max(0.0, w / total) for i, w in raw_weights.items()}


def hrp_weights(returns_df):
    """
    returns_df: pandas DataFrame, columns = tickers, rows = periodic returns.
    Computes sample covariance and delegates to hrp_weights_from_cov.

    Returns dict {ticker: weight}.
    """
    tickers = list(returns_df.columns)
    n = len(tickers)

    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    # sample covariance, pandas builtin (no scipy)
    cov_df = returns_df.cov()
    cov = [[float(cov_df.iloc[i, j]) for j in range(n)] for i in range(n)]

    return hrp_weights_from_cov(cov, tickers)
