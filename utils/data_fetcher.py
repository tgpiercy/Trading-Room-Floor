"""
utils/data_fetcher.py
All API calls via yfinance with Streamlit caching.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

MARKET_INDICES = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "DOW":          "^DJI",
    "Russell 2000": "^RUT",
    "VIX":          "^VIX",
    "10Y Yield":    "^TNX",
    "Gold":         "GC=F",
    "Oil (WTI)":    "CL=F",
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_market_overview() -> dict:
    """
    Return dict of index -> {price, change_pct, symbol, date}.
    Uses a 1-month window so weekends/holidays never produce empty results.
    Always shows the most recent available close.
    """
    result = {}
    for name, sym in MARKET_INDICES.items():
        try:
            hist = yf.download(sym, period="1mo", interval="1d",
                               progress=False, auto_adjust=True)
            hist = _flatten(hist).dropna(subset=["Close"])
            if hist.empty:
                continue
            curr = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr
            chg  = (curr - prev) / prev * 100 if prev else 0.0
            date = hist.index[-1].strftime("%b %d")
            result[name] = {
                "price":      curr,
                "change_pct": chg,
                "symbol":     sym,
                "date":       date,
            }
        except Exception:
            pass
    return result


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data. Returns empty DataFrame on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df = _flatten(df).dropna(how="all")
        return df
    except Exception as e:
        st.error(f"Data error for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    """Fetch fundamental info dict."""
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_options_chain(ticker: str, expiry: str | None = None):
    """
    Returns (calls_df, puts_df, expirations_list).
    Adds a 'type' column and normalises numeric columns.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = list(t.options)
        if not expirations:
            return pd.DataFrame(), pd.DataFrame(), []

        expiry = expiry if expiry in expirations else expirations[0]
        chain  = t.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        for df, label in [(calls, "call"), (puts, "put")]:
            df["type"] = label
            for col in ["volume", "openInterest"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            for col in ["impliedVolatility", "lastPrice", "bid", "ask"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df["vol_oi_ratio"] = df.apply(
                lambda r: round(r["volume"] / r["openInterest"], 2)
                if r["openInterest"] > 0 else 0.0, axis=1
            )

        return calls, puts, expirations
    except Exception as e:
        st.error(f"Options error for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), []


@st.cache_data(ttl=300, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Most recent available close price."""
    try:
        hist = yf.download(ticker, period="5d", interval="1d",
                           progress=False, auto_adjust=True)
        hist = _flatten(hist).dropna(subset=["Close"])
        return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlcv_batch(symbols: tuple, period: str = "2y") -> dict:
    """
    Download daily OHLCV for all symbols, resample to weekly (W-FRI).
    Returns dict of {yf_symbol: weekly_ohlcv_df}.
    Includes current partial week via daily→weekly resample.
    """
    syms = list(symbols)
    try:
        raw = yf.download(syms, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return {}

        result = {}
        for sym in syms:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    avail = raw.columns.get_level_values(1).unique()
                    if sym not in avail:
                        continue
                    df = pd.DataFrame({
                        "Open":   raw["Open"][sym],
                        "High":   raw["High"][sym],
                        "Low":    raw["Low"][sym],
                        "Close":  raw["Close"][sym],
                        "Volume": raw["Volume"][sym],
                    }).dropna(subset=["Close"])
                else:
                    df = raw[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])

                weekly = df.resample("W-FRI").agg({
                    "Open":   "first",
                    "High":   "max",
                    "Low":    "min",
                    "Close":  "last",
                    "Volume": "sum",
                }).dropna(subset=["Close"])

                if not weekly.empty:
                    result[sym] = weekly
            except Exception:
                continue
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_intraday_data(ticker: str, interval: str = "5m", days: int = 5) -> pd.DataFrame:
    """
    Fetch intraday bars (5m/15m). yfinance allows ~60d of intraday history.
    interval: '1m','2m','5m','15m','30m','60m'. days capped per interval limits.
    """
    period_map = {"1m": "5d", "2m": "5d", "5m": "1mo",
                  "15m": "1mo", "30m": "1mo", "60m": "3mo"}
    period = period_map.get(interval, "1mo")
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df = _flatten(df).dropna(how="all")
        # Trim to requested days
        if not df.empty and days:
            cutoff = df.index.max() - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]
        return df
    except Exception as e:
        st.error(f"Intraday error for {ticker}: {e}")
        return pd.DataFrame()
