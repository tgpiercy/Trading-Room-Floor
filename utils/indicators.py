"""
utils/indicators.py
Pure pandas/numpy indicator calculations. All functions accept a
DataFrame with OHLCV columns and return the same DataFrame with new columns.
"""
import pandas as pd
import numpy as np


# ── Trend ─────────────────────────────────────────────────────────────────────

def sma(df: pd.DataFrame, periods: list[int] = [20, 50, 200]) -> pd.DataFrame:
    for p in periods:
        df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    return df


def ema(df: pd.DataFrame, periods: list[int] = [9, 21]) -> pd.DataFrame:
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid               = df["Close"].rolling(period).mean()
    s                 = df["Close"].rolling(period).std()
    df["BB_Mid"]      = mid
    df["BB_Upper"]    = mid + std * s
    df["BB_Lower"]    = mid - std * s
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / mid
    df["BB_PctB"]     = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).rolling(period).mean()
    loss      = (-delta.clip(upper=0)).rolling(period).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    return df


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    ef              = df["Close"].ewm(span=fast, adjust=False).mean()
    es              = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]      = ef - es
    df["MACD_Sig"]  = df["MACD"].ewm(span=sig, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]
    return df


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr    = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    pdm   = hi.diff().clip(lower=0)
    ndm   = (-lo.diff()).clip(lower=0)
    pdm   = pdm.where(pdm > ndm, 0)
    ndm   = ndm.where(ndm > pdm, 0)
    atr_  = tr.rolling(period).mean()
    pdi   = 100 * pdm.rolling(period).mean() / atr_
    ndi   = 100 * ndm.rolling(period).mean() / atr_
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi)
    df["ADX"] = dx.rolling(period).mean()
    df["+DI"] = pdi
    df["-DI"] = ndi
    return df


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    lo_min        = df["Low"].rolling(k).min()
    hi_max        = df["High"].rolling(k).max()
    df["Stoch_K"] = 100 * (df["Close"] - lo_min) / (hi_max - lo_min)
    df["Stoch_D"] = df["Stoch_K"].rolling(d).mean()
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tr       = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


# ── Flow / Volume ─────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.DataFrame:
    signed_vol = df["Volume"] * np.sign(df["Close"].diff()).fillna(0)
    df["OBV"]  = signed_vol.cumsum()
    return df


def mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tp        = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf       = tp * df["Volume"]
    pos       = rmf.where(tp > tp.shift(1), 0.0)
    neg       = rmf.where(tp < tp.shift(1), 0.0)
    mfr       = pos.rolling(period).sum() / neg.rolling(period).sum()
    df["MFI"] = 100 - 100 / (1 + mfr)
    return df


def cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    mfm        = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    df["CMF"]  = (mfm * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return df


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(period).mean()
    return df


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP — most meaningful on intraday; included for daily context too."""
    tp         = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def force_index(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    fi            = df["Close"].diff() * df["Volume"]
    df["FI"]      = fi.ewm(span=period, adjust=False).mean()
    return df


# ── Options ───────────────────────────────────────────────────────────────────

def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Strike where total options writer pain is minimised."""
    strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
    pain    = {}
    c_oi    = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
    p_oi    = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))
    for s in strikes:
        cp = sum(max(0, s - k) * v for k, v in c_oi.items() if s > k)
        pp = sum(max(0, k - s) * v for k, v in p_oi.items() if s < k)
        pain[s] = cp + pp
    return float(min(pain, key=pain.get)) if pain else 0.0


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density φ(x) — no scipy dependency."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x) -> np.ndarray:
    """Standard normal CDF N(x) via erf — vectorized, no scipy dependency."""
    import math
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def bs_gamma(S: float, K, T: float, sigma, r: float = 0.045) -> np.ndarray:
    """
    Black-Scholes gamma, computed from implied volatility.

    yfinance does NOT supply option Greeks, but it DOES supply
    impliedVolatility — and gamma is identical for calls and puts, so we
    can reconstruct it analytically:

        d1    = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        gamma = φ(d1) / (S · σ · √T)

    S=spot, K=strike(s), T=years to expiry, sigma=IV (decimal), r=rate.
    Returns 0 where inputs are invalid (T≤0, σ≤0, etc.).
    """
    K     = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)

    sig_sqrtT = np.where(valid, sigma * np.sqrt(T), np.nan)
    d1 = np.where(valid,
                  (np.log(np.where(valid, S / K, 1.0)) + (r + 0.5 * sigma**2) * T) / sig_sqrtT,
                  0.0)
    gamma = np.where(valid, _norm_pdf(d1) / (S * sig_sqrtT), 0.0)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def gamma_exposure(calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
                   expiry=None, r: float = 0.045) -> pd.DataFrame:
    """
    Net dealer Gamma Exposure (GEX) by strike, using Black-Scholes gamma
    reconstructed from implied volatility (Yahoo has no native gamma).

    Convention (SqueezeMetrics-style): calls contribute +gamma, puts −gamma.
    GEX per strike is expressed as $ notional per 1% spot move:
        GEX = sign · gamma · OI · 100 · S² · 0.01

    Positive net GEX → dealers long gamma (volatility-dampening / pinning).
    Negative net GEX → dealers short gamma (volatility-amplifying).
    """
    # Time to expiry in years
    if expiry is not None:
        try:
            days = (pd.Timestamp(expiry) - pd.Timestamp.now()).days
            T = max(days, 0) / 365.0
        except Exception:
            T = 30 / 365.0
    else:
        T = 30 / 365.0
    if T <= 0:
        T = 1 / 365.0   # expiry day — avoid division by zero

    def _side(df, sign):
        if df.empty or "strike" not in df.columns or "openInterest" not in df.columns:
            return pd.DataFrame(columns=["strike", "gex"])
        d = df.copy()
        d = d[d["openInterest"] > 0]
        if d.empty:
            return pd.DataFrame(columns=["strike", "gex"])
        # Prefer a broker-provided gamma (e.g. CBOE greeks); else reconstruct via BS.
        if "gamma" in d.columns and pd.to_numeric(d["gamma"], errors="coerce").abs().sum() > 0:
            g = pd.to_numeric(d["gamma"], errors="coerce").fillna(0.0).values
        else:
            if "impliedVolatility" not in d.columns:
                return pd.DataFrame(columns=["strike", "gex"])
            d = d[d["impliedVolatility"] > 0]
            if d.empty:
                return pd.DataFrame(columns=["strike", "gex"])
            g = bs_gamma(spot, d["strike"].values, T, d["impliedVolatility"].values, r)
        d = d.assign(gex=sign * g * d["openInterest"].values * 100 * spot * spot * 0.01)
        return d[["strike", "gex"]]

    c = _side(calls, +1).rename(columns={"gex": "call_gex"})
    p = _side(puts,  -1).rename(columns={"gex": "put_gex"})
    if c.empty and p.empty:
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex"])

    merged = pd.merge(c, p, on="strike", how="outer").fillna(0)
    merged["net_gex"] = merged["call_gex"] + merged["put_gex"]
    return merged.sort_values("strike").reset_index(drop=True)


def bs_delta(S, K, T, sigma, r, is_call: bool) -> np.ndarray:
    """Black-Scholes delta (fallback when a feed gives no native delta)."""
    K = np.asarray(K, float); sigma = np.asarray(sigma, float)
    valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)
    sq = np.where(valid, sigma * np.sqrt(T), np.nan)
    d1 = np.where(valid, (np.log(np.where(valid, S / K, 1.0)) + (r + 0.5 * sigma**2) * T) / sq, 0.0)
    nd1 = _norm_cdf(d1)
    delta = nd1 if is_call else (nd1 - 1.0)
    return np.where(valid, delta, 0.0)


def bs_charm(S, K, T, sigma, r=0.045) -> np.ndarray:
    """
    Charm = ∂Δ/∂t (delta decay) per CALENDAR DAY. Identical for calls/puts (q=0).
    Drives the pin-into-expiry effect: as time passes, OTM deltas bleed toward 0
    and ITM toward ±1, forcing dealers to re-hedge — strongest near expiry.
    """
    K = np.asarray(K, float); sigma = np.asarray(sigma, float)
    valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)
    sq = np.where(valid, sigma * np.sqrt(T), np.nan)
    d1 = np.where(valid, (np.log(np.where(valid, S / K, 1.0)) + (r + 0.5 * sigma**2) * T) / sq, 0.0)
    d2 = d1 - sq
    charm = -_norm_pdf(d1) * (2 * r * T - d2 * sq) / (2 * T * sq)
    charm = np.where(valid, charm / 365.0, 0.0)
    return np.nan_to_num(charm, nan=0.0, posinf=0.0, neginf=0.0)


def bs_vanna(S, K, T, sigma, r=0.045) -> np.ndarray:
    """
    Vanna = ∂Δ/∂σ = ∂vega/∂S, per 1.00 vol change. Identical for calls/puts.
    Drives vol-triggered hedging: when IV moves, dealer delta shifts and must be
    re-hedged — the engine behind 'vanna rallies' as vol mean-reverts.
    """
    K = np.asarray(K, float); sigma = np.asarray(sigma, float)
    valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)
    sq = np.where(valid, sigma * np.sqrt(T), np.nan)
    d1 = np.where(valid, (np.log(np.where(valid, S / K, 1.0)) + (r + 0.5 * sigma**2) * T) / sq, 0.0)
    d2 = d1 - sq
    vanna = np.where(valid, -_norm_pdf(d1) * d2 / sigma, 0.0)
    return np.nan_to_num(vanna, nan=0.0, posinf=0.0, neginf=0.0)


def greek_exposures(calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
                    expiry=None, r: float = 0.045) -> dict:
    """
    Aggregate dealer-greek exposures by strike + net totals:
      • DEX   — net dollar delta (positioning lean): Σ delta·OI·100·S, natural
                delta signs (calls +, puts −). Positive = call-delta dominant.
      • Charm — Σ sign·charm·OI·100·S  (calls +, puts −): per-day delta-hedge
                drift (pin pressure into expiry).
      • Vanna — Σ sign·vanna·OI·100·S: delta-hedge flow per 1.00 vol change.
    Uses native delta when present (CBOE); BS fallback otherwise. Charm/Vanna
    always BS from IV. Estimate under the standard dealer convention.
    """
    if expiry is not None:
        try:
            T = max((pd.Timestamp(expiry) - pd.Timestamp.now()).days, 0) / 365.0
        except Exception:
            T = 30 / 365.0
    else:
        T = 30 / 365.0
    if T <= 0:
        T = 1 / 365.0

    def _side(df, is_call):
        cols = ["strike", "openInterest"]
        if df.empty or any(c not in df.columns for c in cols):
            return pd.DataFrame(columns=["strike", "dex", "charm", "vanna"])
        d = df[df["openInterest"] > 0].copy()
        if d.empty:
            return pd.DataFrame(columns=["strike", "dex", "charm", "vanna"])
        iv = pd.to_numeric(d.get("impliedVolatility", 0), errors="coerce").fillna(0).values
        K = d["strike"].values; OI = d["openInterest"].values
        sign = 1.0 if is_call else -1.0
        # Delta: native if present, else BS
        if "delta" in d.columns and pd.to_numeric(d["delta"], errors="coerce").abs().sum() > 0:
            delta = pd.to_numeric(d["delta"], errors="coerce").fillna(0).values
        else:
            delta = bs_delta(spot, K, T, iv, r, is_call)
        charm = bs_charm(spot, K, T, iv, r)
        vanna = bs_vanna(spot, K, T, iv, r)
        notional = OI * 100 * spot
        out = pd.DataFrame({
            "strike": K,
            "dex": delta * notional,                  # natural sign (direction)
            "charm": sign * charm * notional,         # dealer convention
            "vanna": sign * vanna * notional,
        })
        return out

    c = _side(calls, True); p = _side(puts, False)
    if c.empty and p.empty:
        return {"net_dex": 0.0, "net_charm": 0.0, "net_vanna": 0.0,
                "by_strike": pd.DataFrame(columns=["strike", "dex", "charm", "vanna"])}
    allk = pd.concat([c, p], ignore_index=True)
    by_strike = allk.groupby("strike", as_index=False)[["dex", "charm", "vanna"]].sum()
    by_strike = by_strike.sort_values("strike").reset_index(drop=True)
    net_dex = float(by_strike["dex"].sum())
    gross_dex = float(by_strike["dex"].abs().sum())
    return {
        "net_dex": net_dex, "gross_dex": gross_dex,
        "dex_tilt": (net_dex / gross_dex) if gross_dex else 0.0,
        "net_charm": float(by_strike["charm"].sum()),
        "net_vanna": float(by_strike["vanna"].sum()),
        "by_strike": by_strike,
    }


def gamma_flip(gex_df: pd.DataFrame) -> float:
    """
    Estimate the zero-gamma (flip) strike: the level where net GEX crosses
    zero, separating call-dominated strikes (above, dealers long gamma →
    stabilizing) from put-dominated strikes (below → destabilizing).

    Finds the net-GEX sign change nearest the middle of the chain and
    linearly interpolates the crossing strike.
    """
    if gex_df.empty or "net_gex" not in gex_df.columns or len(gex_df) < 2:
        return None
    g = gex_df.sort_values("strike").reset_index(drop=True)
    strikes = g["strike"].values
    net = g["net_gex"].values

    crossings = []
    for i in range(1, len(net)):
        if (net[i - 1] < 0 <= net[i]) or (net[i - 1] > 0 >= net[i]):
            x0, x1 = strikes[i - 1], strikes[i]
            y0, y1 = net[i - 1], net[i]
            x = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
            crossings.append(float(x))
    if not crossings:
        return None
    # Return the crossing nearest the chain's mid-strike (most relevant flip)
    mid = strikes[len(strikes) // 2]
    return round(min(crossings, key=lambda x: abs(x - mid)), 2)


def pcr(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """Put/Call ratios by volume and open interest. Safe against empty/missing columns."""
    def _sum(df, col):
        if df.empty or col not in df.columns:
            return 0
        return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()

    total_call_vol = _sum(calls, "volume")
    total_put_vol  = _sum(puts,  "volume")
    total_call_oi  = _sum(calls, "openInterest")
    total_put_oi   = _sum(puts,  "openInterest")
    return {
        "pcr_volume": round(total_put_vol / total_call_vol, 2) if total_call_vol else 0,
        "pcr_oi":     round(total_put_oi  / total_call_oi,  2) if total_call_oi  else 0,
        "call_vol":   int(total_call_vol),
        "put_vol":    int(total_put_vol),
        "call_oi":    int(total_call_oi),
        "put_oi":     int(total_put_oi),
    }


def gamma_horizons(chains: list, spot: float, r: float = 0.045,
                   horizons=(7, 14, 30)) -> dict:
    """
    Aggregate dealer gamma across multiple expirations into time-horizon buckets.

    chains: list of (expiry_str, calls_df, puts_df) — typically all expiries
            within ~35 days.
    Returns per-expiry net GEX + flip, and bucketed net GEX/flip for each horizon
    (≤7d, ≤14d, ≤30d). Sign of bucket net GEX = dealer gamma over that horizon:
    positive → long gamma (vol-dampening / pinning); negative → short gamma
    (vol-amplifying / squeezy). Each bucket also reports a tilt ratio
    (net ÷ gross) so the regime read is comparable across tickers.
    """
    today = pd.Timestamp.now().normalize()
    per_expiry = []
    for expiry, calls, puts in chains:
        gdf = gamma_exposure(calls, puts, spot, expiry=expiry, r=r)
        if gdf.empty:
            continue
        gross = float(gdf["call_gex"].abs().sum() + gdf["put_gex"].abs().sum())
        days = max((pd.Timestamp(expiry) - today).days, 0)
        per_expiry.append({
            "expiry": str(expiry), "days": days,
            "net_gex": float(gdf["net_gex"].sum()), "gross_gex": gross,
            "flip": gamma_flip(gdf), "_gdf": gdf[["strike", "net_gex"]],
        })
    per_expiry.sort(key=lambda e: e["days"])

    buckets, bucket_flip, bucket_tilt = {}, {}, {}
    for h in horizons:
        members = [e for e in per_expiry if e["days"] <= h]
        net = float(sum(e["net_gex"] for e in members))
        gross = float(sum(e["gross_gex"] for e in members))
        buckets[h] = net
        bucket_tilt[h] = (net / gross) if gross > 0 else 0.0
        if members:
            combined = pd.concat([e["_gdf"] for e in members])
            agg = combined.groupby("strike", as_index=False)["net_gex"].sum()
            bucket_flip[h] = gamma_flip(agg)
        else:
            bucket_flip[h] = None

    for e in per_expiry:
        e.pop("_gdf", None)
    return {"per_expiry": per_expiry, "buckets": buckets, "bucket_flip": bucket_flip,
            "bucket_tilt": bucket_tilt, "spot": spot}


def gamma_regime_label(tilt: float, tilt_threshold: float = 0.05) -> tuple:
    """Map a gamma tilt ratio (net ÷ gross, −1..+1) to (emoji, label, read)."""
    if tilt > tilt_threshold:
        return ("🟢", "Long gamma", "pinned / mean-reverting — fade extremes")
    if tilt < -tilt_threshold:
        return ("🔴", "Short gamma", "squeezy / trending — respect breakouts")
    return ("🟡", "Near-flat", "transitional — gamma not decisive")


# ══════════════════════════════════════════════════════════════════════════════
# Dealer greek exposures: DEX (delta), Charm (delta decay), Vanna (delta-vs-vol)
# DEX uses native delta when present (CBOE); charm/vanna are Black-Scholes from IV
# (CBOE supplies only first-order greeks). Sign convention mirrors GEX: calls +,
# puts − for the hedging-flow greeks (charm/vanna). DEX is reported as net OI
# delta (calls +, puts − via delta's own sign) — dealers hold the inverse.
# ══════════════════════════════════════════════════════════════════════════════
from math import erf as _erf
_SQRT2 = np.sqrt(2.0)


def _ncdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(_erf)(x / _SQRT2))


def _npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _d1d2(S, K, T, sigma, r):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sig = np.where(sigma <= 0, np.nan, sigma)
    sT = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / sT
    return d1, d1 - sT


def bs_delta(S, K, T, sigma, r=0.045, is_call=True):
    d1, _ = _d1d2(S, K, T, sigma, r)
    nd1 = _ncdf(d1)
    return nd1 if is_call else nd1 - 1.0


def bs_charm(S, K, T, sigma, r=0.045):
    """∂Δ/∂t per YEAR (q=0; identical for calls & puts). Divide by 365 for per-day."""
    d1, d2 = _d1d2(S, K, T, sigma, r)
    sig = np.asarray(sigma, dtype=float)
    sT = sig * np.sqrt(T)
    return -_npdf(d1) * (2 * r * T - d2 * sT) / (2 * T * sT)


def bs_vanna(S, K, T, sigma, r=0.045):
    """∂Δ/∂σ per 1.00 vol (q=0; identical for calls & puts). ×0.01 for per-1%-vol."""
    d1, d2 = _d1d2(S, K, T, sigma, r)
    sig = np.asarray(sigma, dtype=float)
    return -_npdf(d1) * d2 / sig


def greek_exposures(chains: list, spot: float, r: float = 0.045) -> dict:
    """
    Net dealer greek exposures aggregated across the supplied expirations.
      net_dex   : net OI delta in $ (calls +, puts − via delta's sign).
                  Positive = call-delta-heavy positioning; dealers hold the inverse.
      net_charm : $ of delta hedging per DAY (sign mirrors GEX). Intensifies into
                  expiry — the mechanical 'pin toward high-OI strikes'.
      net_vanna : $ of delta shift per 1% IV move (sign mirrors GEX). Drives
                  vol-triggered hedging (vanna rallies / vol-down selling).
    """
    today = pd.Timestamp.now().normalize()
    per, net_dex, net_charm, net_vanna = [], 0.0, 0.0, 0.0
    for expiry, calls, puts in chains:
        days = max((pd.Timestamp(expiry) - today).days, 0)
        T = max(days, 1) / 365.0
        dex = charm = vanna = 0.0
        for df, sign, is_call in [(calls, +1, True), (puts, -1, False)]:
            if df.empty or "openInterest" not in df.columns or "strike" not in df.columns:
                continue
            d = df[df["openInterest"] > 0]
            if d.empty:
                continue
            oi = d["openInterest"].values.astype(float)
            K = d["strike"].values.astype(float)
            iv = (d["impliedVolatility"].values.astype(float)
                  if "impliedVolatility" in d.columns else None)
            # DEX — native delta if available, else BS
            if "delta" in d.columns and pd.to_numeric(d["delta"], errors="coerce").abs().sum() > 0:
                dl = pd.to_numeric(d["delta"], errors="coerce").fillna(0.0).values
            elif iv is not None:
                dl = bs_delta(spot, K, T, iv, r, is_call)
            else:
                dl = np.zeros_like(oi)
            dex += np.nansum(dl * oi) * 100 * spot
            # Charm / Vanna need IV
            if iv is not None:
                ch = bs_charm(spot, K, T, iv, r) / 365.0
                vn = bs_vanna(spot, K, T, iv, r) * 0.01
                charm += sign * np.nansum(ch * oi) * 100 * spot
                vanna += sign * np.nansum(vn * oi) * 100 * spot
        per.append({"expiry": str(expiry), "days": days,
                    "dex": dex, "charm": charm, "vanna": vanna})
        net_dex += dex; net_charm += charm; net_vanna += vanna
    per.sort(key=lambda e: e["days"])
    return {"net_dex": net_dex, "net_charm": net_charm, "net_vanna": net_vanna,
            "per_expiry": per, "spot": spot}
