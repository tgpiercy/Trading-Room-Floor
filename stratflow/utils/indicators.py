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


def gamma_exposure(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Estimate net dealer GEX by strike. Requires gamma column."""
    c = calls[["strike", "gamma", "openInterest"]].copy()
    p = puts[["strike",  "gamma", "openInterest"]].copy()
    c["GEX"] =  c["gamma"].fillna(0) * c["openInterest"].fillna(0) * 100 * spot
    p["GEX"] = -p["gamma"].fillna(0) * p["openInterest"].fillna(0) * 100 * spot
    merged   = pd.merge(
        c[["strike", "GEX"]].rename(columns={"GEX": "call_gex"}),
        p[["strike", "GEX"]].rename(columns={"GEX": "put_gex"}),
        on="strike", how="outer"
    ).fillna(0)
    merged["net_gex"] = merged["call_gex"] + merged["put_gex"]
    return merged.sort_values("strike")


def pcr(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """Put/Call ratios by volume and open interest."""
    total_call_vol = calls["volume"].sum()
    total_put_vol  = puts["volume"].sum()
    total_call_oi  = calls["openInterest"].sum()
    total_put_oi   = puts["openInterest"].sum()
    return {
        "pcr_volume": round(total_put_vol  / total_call_vol,  2) if total_call_vol  else 0,
        "pcr_oi":     round(total_put_oi   / total_call_oi,   2) if total_call_oi   else 0,
        "call_vol":   int(total_call_vol),
        "put_vol":    int(total_put_vol),
        "call_oi":    int(total_call_oi),
        "put_oi":     int(total_put_oi),
    }
