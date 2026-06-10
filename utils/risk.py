"""
utils/risk.py
The risk layer that turns the validated RS-Extension target book into a
risk-managed one. Four controls, applied in order:

  1. Per-trade risk budget — cap each name's weight so that being stopped out
     costs at most `per_trade_risk` of equity:  w_i · d_i ≤ per_trade_risk,
     where d_i = (price − stop)/price is the stop distance. Tight stop → larger
     allowed weight; wide stop → smaller. This is what ties sizing to the stop.
  2. Position cap — no single name above `max_pos`.
  3. Sector cap — no sector above `max_sector`.
  4. Volatility target — scale gross exposure so estimated portfolio vol ≈
     `target_vol` (single-correlation approximation), never levering past
     `max_gross`. De-risks in hot regimes, fills toward fully-invested in calm.

All transparent and rule-based — it reshapes weights, it does not predict.
"""
import math


def _est_port_vol(weights: dict, ann_vol: dict, corr: float) -> float:
    """Annualized portfolio vol under a single average-correlation assumption:
    var = (1-ρ)·Σ(wᵢσᵢ)² + ρ·(Σ wᵢσᵢ)²."""
    contribs = [weights[k] * ann_vol.get(k, 0.0) for k in weights]
    sum_sq = sum(c * c for c in contribs)
    sum_lin = sum(contribs)
    var = (1 - corr) * sum_sq + corr * sum_lin * sum_lin
    return math.sqrt(max(var, 0.0))


def _cap_redistribute(weights: dict, ceilings: dict, gross: float):
    """Cap weights at per-name ceilings; redistribute freed weight to names with
    headroom (proportional to current weight). Unplaceable excess → cash."""
    w = dict(weights)
    for _ in range(12):
        over = {k: w[k] - ceilings[k] for k in w if w[k] > ceilings[k] + 1e-12}
        if not over:
            break
        excess = sum(over.values())
        for k in over:
            w[k] = ceilings[k]
        head = [k for k in w if ceilings[k] - w[k] > 1e-12]
        hbase = sum(w[k] for k in head)
        if not head or hbase <= 1e-12 or excess <= 1e-12:
            break
        for k in head:
            w[k] = min(ceilings[k], w[k] + excess * (w[k] / hbase))
    invested = sum(w.values())
    return w, max(0.0, gross - invested)


def apply_risk_layer(holdings, *, target_vol: float = 0.15, max_pos: float = 0.20,
                     max_sector: float = 0.40, per_trade_risk: float = 0.01,
                     corr: float = 0.40, max_gross: float = 1.0,
                     periods_per_year: int = 52):
    """holdings: list of {ticker, weight, price, stop, vol (per-period std), sector}.
    Returns (new_holdings, report)."""
    if not holdings:
        return holdings, {"note": "no holdings"}

    w0 = {h["ticker"]: float(h.get("weight", 0.0)) for h in holdings}
    gross0 = sum(w0.values())
    ann_vol, dist, ceil = {}, {}, {}
    for h in holdings:
        tk = h["ticker"]
        v = h.get("vol")
        ann_vol[tk] = (float(v) * math.sqrt(periods_per_year)
                       if v is not None and v == v and v > 0 else 0.0)
        px, sp = h.get("price"), h.get("stop")
        d = ((px - sp) / px) if (px and sp and px > 0 and 0 < sp < px) else None
        dist[tk] = d
        risk_cap = (per_trade_risk / d) if (d and d > 0) else float("inf")
        ceil[tk] = min(max_pos, risk_cap)

    vol_raw = _est_port_vol(w0, ann_vol, corr)

    # 1+2. per-trade risk + position caps
    w1, cash_capped = _cap_redistribute(w0, ceil, gross0)

    # 3. sector caps (offending sectors scaled to cap; freed → cash)
    sec_of = {h["ticker"]: (h.get("sector") or "—") for h in holdings}
    sec_tot = {}
    for tk, wt in w1.items():
        sec_tot[sec_of[tk]] = sec_tot.get(sec_of[tk], 0.0) + wt
    capped_secs = {}
    w2 = dict(w1)
    for sec, tot in sec_tot.items():
        if sec != "—" and tot > max_sector + 1e-9 and tot > 0:
            scale = max_sector / tot
            for tk in [t for t in w2 if sec_of[t] == sec]:
                w2[tk] *= scale
            capped_secs[sec] = round(tot * 100, 1)

    # 4. volatility target (scale gross; respect ceilings on scale-up)
    gross2 = sum(w2.values())
    vol2 = _est_port_vol(w2, ann_vol, corr)
    k_vol = (target_vol / vol2) if vol2 > 0 else 1.0
    k_gross = (max_gross / gross2) if gross2 > 0 else 1.0
    k_ceil = min([ceil[t] / w2[t] for t in w2 if w2[t] > 1e-9] or [float("inf")])
    k = max(0.0, min(k_vol, k_gross, k_ceil))
    w_final = {t: w2[t] * k for t in w2}

    vol_final = _est_port_vol(w_final, ann_vol, corr)
    invested = sum(w_final.values())

    capped_names = sorted([h["ticker"] for h in holdings
                           if w0[h["ticker"]] > ceil[h["ticker"]] + 1e-6],
                          key=lambda t: -w0[t])
    new_holdings = []
    for h in holdings:
        tk = h["ticker"]
        nh = dict(h)
        nh["weight"] = round(w_final[tk], 4)
        d = dist[tk]
        nh["trade_risk_pct"] = round(w_final[tk] * d * 100, 2) if d else None
        new_holdings.append(nh)
    new_holdings.sort(key=lambda x: -x["weight"])

    report = {
        "target_vol_pct": round(target_vol * 100, 1),
        "vol_raw_pct": round(vol_raw * 100, 1),
        "vol_final_pct": round(vol_final * 100, 1),
        "gross_before_pct": round(gross0 * 100, 1),
        "gross_after_pct": round(invested * 100, 1),
        "vol_scale": round(k, 2),
        "cash_pct": round(max(0.0, 1 - invested) * 100, 1),
        "capped_names": capped_names,
        "capped_sectors": capped_secs,
        "max_trade_risk_pct": round(max([h.get("trade_risk_pct") or 0
                                         for h in new_holdings] + [0]), 2),
        "per_trade_budget_pct": round(per_trade_risk * 100, 2),
    }
    return new_holdings, report
