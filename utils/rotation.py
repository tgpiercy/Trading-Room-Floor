"""
utils/rotation.py
Early rotation detection — leading-edge signals that fire before
RS structure and price confirm. Designed to surface institutional
money movement (algorithmic accumulation) ahead of the trend.

Five signals:
  1. RS Acceleration   — 2nd derivative of RS (turn detection)
  2. RVOL z-score       — volume singled out vs sector peers
  3. Money-flow diverg. — A/D rising while price flat/falling
  4. Accumulation persist — consecutive weeks of net accumulation
  5. RRG quadrant        — RS-Ratio vs RS-Momentum (Improving = early)
"""
import pandas as pd
import numpy as np

# ── RRG: RS-Ratio & RS-Momentum (normalized ~100) ────────────────────────────
RRG_ENGINE_VERSION = "v2.1"

def calc_rrg(ticker_close: pd.Series, bench_close: pd.Series,
             window: int = 14, smooth: int = 4, mom_period: int = 5,
             presmooth: int = 3) -> tuple:
    """
    JdK-style RS-Ratio and RS-Momentum, normalized ~100 and smoothed.

    Clean (non-jagged) RRGs require FOUR things, all applied here:
      1. Pre-smooth the raw RS line (EMA) to kill micro-noise at the source.
      2. RS-Momentum measured over `mom_period` bars — NOT a 1-bar diff,
         which is pure noise once normalized.
      3. Both series smoothed with a rolling mean after normalization.
      4. Adequate normalization window so the z-score is stable.

    Returns (rs_ratio_series, rs_momentum_series). None if insufficient data.
    """
    rs = (100.0 * ticker_close / bench_close).dropna()
    if len(rs) < window + smooth + mom_period + presmooth + 2:
        return None, None

    # 1. Pre-smooth raw RS with EMA — removes single-week spikes before they
    #    propagate into the normalization and momentum calcs.
    if presmooth and presmooth > 1:
        rs = rs.ewm(span=presmooth, adjust=False).mean()

    # 2. RS-Ratio: z-score of RS vs rolling window, centered 100, then smoothed.
    rs_mean  = rs.rolling(window).mean()
    rs_std   = rs.rolling(window).std().replace(0, np.nan)
    rs_ratio = (100 + (rs - rs_mean) / rs_std).rolling(smooth).mean()

    # 3. RS-Momentum: normalized multi-bar change of RS-Ratio, centered 100.
    roc      = rs_ratio - rs_ratio.shift(mom_period)
    roc_mean = roc.rolling(window).mean()
    roc_std  = roc.rolling(window).std().replace(0, np.nan)
    rs_mom   = (100 + (roc - roc_mean) / roc_std).rolling(smooth).mean()

    return rs_ratio.dropna(), rs_mom.dropna()


def rrg_quadrant(ratio_val: float, mom_val: float) -> str:
    """Classify a point into one of the four RRG quadrants."""
    if ratio_val >= 100 and mom_val >= 100:   return "Leading"
    if ratio_val >= 100 and mom_val <  100:    return "Weakening"
    if ratio_val <  100 and mom_val <  100:    return "Lagging"
    return "Improving"


QUADRANT_COLOR = {
    "Leading":   "#00cc66",
    "Weakening": "#ffd700",
    "Lagging":   "#ff4444",
    "Improving": "#4fc3f7",
    "Unknown":   "#888888",
}


# ── RS Acceleration ───────────────────────────────────────────────────────────
def calc_rs_acceleration(rs_series: pd.Series, roc_period: int = 3) -> dict:
    """
    RS rate-of-change and acceleration (2nd derivative).
    Positive acceleration while RoC still negative = earliest turn signal.
    """
    rs = rs_series.dropna()
    if len(rs) < roc_period * 3:
        return {"roc": 0.0, "accel": 0.0, "roc_improving": False}
    roc   = rs.pct_change(roc_period) * 100
    accel = roc.diff()
    return {
        "roc":           round(float(roc.iloc[-1]), 2),
        "accel":         round(float(accel.iloc[-1]), 3),
        "roc_improving": bool(len(roc) >= 2 and roc.iloc[-1] > roc.iloc[-2]),
    }


# ── Money-flow divergence ─────────────────────────────────────────────────────
def calc_divergence(price: pd.Series, ad_line: pd.Series,
                    lookback: int = 8) -> str:
    """
    Compare price slope vs A/D slope over lookback.
    Bullish: price flat/down while A/D rising (accumulation under weakness).
    Bearish: price up while A/D falling (distribution into strength).
    """
    p = price.dropna()
    a = ad_line.dropna()
    if len(p) < lookback + 1 or len(a) < lookback + 1:
        return "None"
    p0, a0 = p.iloc[-lookback], a.iloc[-lookback]
    p_chg = (p.iloc[-1] - p0) / abs(p0) if p0 != 0 else 0
    a_chg = (a.iloc[-1] - a0) / abs(a0) if a0 != 0 else 0
    if p_chg < 0.01 and a_chg > 0.02:   return "Bullish"
    if p_chg > 0.01 and a_chg < -0.02:  return "Bearish"
    return "None"


# ── Accumulation persistence ──────────────────────────────────────────────────
def calc_accumulation_persistence(df: pd.DataFrame, ad_df: pd.DataFrame) -> int:
    """Consecutive weeks of net accumulation (A/D rising AND CMF positive)."""
    ad = ad_df["AD"].dropna()
    if len(ad) < 22:
        return 0
    hl  = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    cmf = (mfm * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    ad_rising = ad > ad.shift(1)
    cmf_pos   = cmf > 0
    accum     = (ad_rising & cmf_pos).dropna()
    count = 0
    for v in reversed(accum.values):
        if v: count += 1
        else: break
    return count


# ── RVOL (raw, for cross-sectional z-scoring) ────────────────────────────────
def calc_rvol(df: pd.DataFrame, period: int = 20) -> float:
    """Current relative volume vs N-week average."""
    if "Volume" not in df.columns or len(df) < period + 1:
        return None
    vol_ma = df["Volume"].rolling(period).mean()
    last_ma = vol_ma.dropna()
    if last_ma.empty or last_ma.iloc[-1] == 0:
        return None
    return round(float(df["Volume"].iloc[-1] / last_ma.iloc[-1]), 2)


def rvol_zscore(ticker_rvol: float, peer_rvols: list) -> float:
    """Z-score a ticker's RVOL against its sector peers."""
    peers = [r for r in peer_rvols if r is not None]
    if len(peers) < 2 or ticker_rvol is None:
        return 0.0
    arr = np.array(peers, dtype=float)
    mu, sd = arr.mean(), arr.std()
    if sd == 0:
        return 0.0
    return round(float((ticker_rvol - mu) / sd), 2)


# ── Composite Early Rotation Score ───────────────────────────────────────────
def calc_early_rotation_score(rs_accel: dict, rvol_z: float,
                              divergence: str, accum_persist: int,
                              quadrant: str) -> dict:
    """
    Early Rotation Score (0-10). Forward-looking; separate from GW2.
    High Early Rotation + low GW2 = money moving in before trend confirms.
    """
    score = 0
    detail = {}

    acc_ok = rs_accel["accel"] > 0
    detail["RS accelerating (2nd deriv > 0)"] = acc_ok
    if acc_ok: score += 2

    detail["RS RoC improving (3wk)"] = rs_accel["roc_improving"]
    if rs_accel["roc_improving"]: score += 1

    rvol_ok = rvol_z > 1.5
    detail[f"RVOL z-score > 1.5 vs peers ({rvol_z:.1f})"] = rvol_ok
    if rvol_ok: score += 2

    div_ok = divergence == "Bullish"
    detail["Bullish money-flow divergence"] = div_ok
    if div_ok: score += 2

    accum_ok = accum_persist >= 3
    detail[f"Accumulation persistence ≥ 3wk ({accum_persist}wk)"] = accum_ok
    if accum_ok: score += 2

    quad_ok = quadrant == "Improving"
    detail[f"Entering Improving quadrant ({quadrant})"] = quad_ok
    if quad_ok: score += 1

    # Classification
    if   score >= 7: tier = "🔥 Hot"
    elif score >= 5: tier = "⚡ Warming"
    elif score >= 3: tier = "👀 Stirring"
    else:            tier = "💤 Quiet"

    return {"score": score, "max": 10, "tier": tier, "detail": detail}


ROTATION_TIER_COLOR = {
    "🔥 Hot":      "#ff4444",
    "⚡ Warming":  "#ff8c00",
    "👀 Stirring": "#ffd700",
    "💤 Quiet":    "#888888",
}
