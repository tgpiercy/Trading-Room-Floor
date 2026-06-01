"""
utils/order_flow.py
Order-flow inference from free data. True tape/order-book isn't available,
so we infer institutional footprints from:
  - Premium-weighted options flow (dollars, not contracts)
  - New positioning (Vol > OI)
  - Sweep detection (same-side strikes across the chain)
  - Intraday volume profile + auction imbalance (5-min bars)
"""
import pandas as pd
import numpy as np


# ── Premium-weighted options flow ─────────────────────────────────────────────
def premium_flow(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """
    Dollar-weighted options flow: volume × midprice × 100.
    Shows where MONEY went, not just contract count.
    """
    def _prem(df):
        df = df.copy()
        mid = df.get("lastPrice", pd.Series(0, index=df.index)).fillna(0)
        # Prefer midpoint of bid/ask when available
        if "bid" in df.columns and "ask" in df.columns:
            ba_mid = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
            mid = ba_mid.where(ba_mid > 0, mid)
        df["premium"] = df["volume"].fillna(0) * mid * 100
        return df

    c = _prem(calls)
    p = _prem(puts)
    call_prem = float(c["premium"].sum())
    put_prem  = float(p["premium"].sum())
    total     = call_prem + put_prem

    return {
        "call_premium": call_prem,
        "put_premium":  put_prem,
        "total_premium": total,
        "call_pct": round(call_prem / total * 100, 1) if total else 0,
        "put_pct":  round(put_prem / total * 100, 1) if total else 0,
        "prem_pcr": round(put_prem / call_prem, 2) if call_prem else 0,
        "calls_df": c,
        "puts_df":  p,
    }


def new_positioning(calls: pd.DataFrame, puts: pd.DataFrame,
                    min_premium: float = 25_000) -> pd.DataFrame:
    """
    Contracts where Volume > Open Interest = new positioning today
    (not closing existing). Ranked by premium (dollar size).
    """
    frames = []
    for df, label in [(calls, "CALL"), (puts, "PUT")]:
        d = df.copy()
        if "premium" not in d.columns:
            mid = d.get("lastPrice", pd.Series(0, index=d.index)).fillna(0)
            d["premium"] = d["volume"].fillna(0) * mid * 100
        d = d[d["volume"].fillna(0) > d["openInterest"].fillna(0)]   # new > existing
        d = d[d["premium"] >= min_premium]
        d["type"] = label
        frames.append(d)
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["type","strike","lastPrice","volume","openInterest",
                        "premium","impliedVolatility","inTheMoney"] if c in out.columns]
    return out[cols].sort_values("premium", ascending=False).reset_index(drop=True)


def detect_sweeps(calls: pd.DataFrame, puts: pd.DataFrame,
                  spot: float, min_volume: int = 200) -> dict:
    """
    Sweep proxy: clusters of high-volume same-side strikes near spot,
    suggesting an aggressive directional order worked across strikes.
    """
    def _cluster(df, side):
        d = df.copy()
        d = d[d["volume"].fillna(0) >= min_volume]
        if d.empty:
            return 0, 0.0
        # Count strikes with elevated volume; sum premium
        if "premium" not in d.columns:
            mid = d.get("lastPrice", pd.Series(0, index=d.index)).fillna(0)
            d["premium"] = d["volume"].fillna(0) * mid * 100
        return len(d), float(d["premium"].sum())

    n_call, call_prem = _cluster(calls, "call")
    n_put,  put_prem  = _cluster(puts, "put")

    bias = ("Bullish sweep cluster" if call_prem > put_prem * 1.5 else
            "Bearish sweep cluster" if put_prem > call_prem * 1.5 else
            "Mixed / balanced")
    return {
        "call_strikes_hot": n_call, "put_strikes_hot": n_put,
        "call_sweep_prem": call_prem, "put_sweep_prem": put_prem,
        "bias": bias,
    }


# ── Intraday volume profile ───────────────────────────────────────────────────
def volume_by_price(intraday: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    """
    Volume-at-price profile. High-volume nodes = accumulation/distribution zones
    where institutions worked size.
    """
    if intraday.empty or "Volume" not in intraday.columns:
        return pd.DataFrame()
    d = intraday.dropna(subset=["Close","Volume"])
    if d.empty:
        return pd.DataFrame()
    lo, hi = d["Low"].min(), d["High"].max()
    if hi <= lo:
        return pd.DataFrame()
    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    # Assign each bar's volume to its typical-price bin
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    vol = np.zeros(bins)
    for i, v in zip(idx, d["Volume"].values):
        vol[i] += v
    prof = pd.DataFrame({"price": mids, "volume": vol})
    # Point of Control = highest-volume price node
    poc = prof.loc[prof["volume"].idxmax(), "price"] if prof["volume"].sum() > 0 else None
    prof.attrs["poc"] = poc
    return prof


def auction_analysis(intraday: pd.DataFrame) -> dict:
    """
    Opening / closing auction proxy from intraday bars.
    Late-day volume concentration often = institutions working orders into close.
    """
    if intraday.empty:
        return {}
    d = intraday.dropna(subset=["Close","Volume"]).copy()
    if d.empty:
        return {}
    # Group by date
    d["date"] = d.index.date
    last_day = d[d["date"] == d["date"].max()]
    if last_day.empty:
        return {}

    total_vol = last_day["Volume"].sum()
    n = len(last_day)
    if n < 4 or total_vol == 0:
        return {}

    open_chunk  = last_day.iloc[:max(1, n // 6)]    # first ~1/6 of session
    close_chunk = last_day.iloc[-max(1, n // 6):]   # last ~1/6 of session
    open_vol_pct  = open_chunk["Volume"].sum()  / total_vol * 100
    close_vol_pct = close_chunk["Volume"].sum() / total_vol * 100

    # Late-day price drift (accumulation vs distribution into close)
    close_drift = ((close_chunk["Close"].iloc[-1] - close_chunk["Close"].iloc[0])
                   / close_chunk["Close"].iloc[0] * 100) if len(close_chunk) >= 2 else 0

    signal = "Neutral"
    if close_vol_pct > 25 and close_drift > 0.2:
        signal = "Late-day accumulation (institutional buy-into-close)"
    elif close_vol_pct > 25 and close_drift < -0.2:
        signal = "Late-day distribution (selling into close)"
    elif open_vol_pct > 30:
        signal = "Opening-auction heavy (gap positioning)"

    return {
        "open_vol_pct":  round(open_vol_pct, 1),
        "close_vol_pct": round(close_vol_pct, 1),
        "close_drift":   round(close_drift, 2),
        "total_vol":     int(total_vol),
        "signal":        signal,
        "session_date":  str(d["date"].max()),
    }


def vwap_deviation(intraday: pd.DataFrame) -> dict:
    """
    Persistent trading away from VWAP can hint at iceberg/working orders.
    Returns current price vs session VWAP and how long price held above/below.
    """
    if intraday.empty:
        return {}
    d = intraday.dropna(subset=["Close","Volume"]).copy()
    d["date"] = d.index.date
    last_day = d[d["date"] == d["date"].max()]
    if last_day.empty or last_day["Volume"].sum() == 0:
        return {}
    tp = (last_day["High"] + last_day["Low"] + last_day["Close"]) / 3
    vwap = (tp * last_day["Volume"]).cumsum() / last_day["Volume"].cumsum()
    last_close = last_day["Close"].iloc[-1]
    last_vwap  = vwap.iloc[-1]
    dev_pct = (last_close - last_vwap) / last_vwap * 100 if last_vwap else 0
    pct_above = (last_day["Close"].values > vwap.values).mean() * 100
    return {
        "vwap": round(float(last_vwap), 2),
        "price": round(float(last_close), 2),
        "dev_pct": round(float(dev_pct), 2),
        "pct_session_above_vwap": round(float(pct_above), 1),
    }
