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
        need = ["strike", "openInterest", "impliedVolatility"]
        if df.empty or any(c not in df.columns for c in need):
            return pd.DataFrame(columns=["strike", "gex"])
        d = df[need].copy()
        d = d[(d["impliedVolatility"] > 0) & (d["openInterest"] > 0)]
        if d.empty:
            return pd.DataFrame(columns=["strike", "gex"])
        g = bs_gamma(spot, d["strike"].values, T, d["impliedVolatility"].values, r)
        d["gex"] = sign * g * d["openInterest"].values * 100 * spot * spot * 0.01
        return d[["strike", "gex"]]

    c = _side(calls, +1).rename(columns={"gex": "call_gex"})
    p = _side(puts,  -1).rename(columns={"gex": "put_gex"})
    if c.empty and p.empty:
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex"])

    merged = pd.merge(c, p, on="strike", how="outer").fillna(0)
    merged["net_gex"] = merged["call_gex"] + merged["put_gex"]
    return merged.sort_values("strike").reset_index(drop=True)


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
