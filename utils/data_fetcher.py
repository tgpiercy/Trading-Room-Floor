"""
utils/data_fetcher.py
All API calls via yfinance with Streamlit caching.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time

# ── Anti-rate-limit infrastructure ────────────────────────────────────────────
# Yahoo throttles by IP; Streamlit Cloud shares IPs, so the options endpoint in
# particular gets 429s. A browser-impersonating curl_cffi session avoids most of
# them. Newer yfinance auto-detects curl_cffi when installed; we also try to pass
# a session explicitly and fall back cleanly if the version doesn't accept it.
try:
    from curl_cffi import requests as _cffi
    _SESSION = _cffi.Session(impersonate="chrome")
except Exception:
    _SESSION = None


def _retry(fn, tries: int = 4, base_delay: float = 1.5):
    """Call fn() with exponential backoff on rate-limit / transient errors."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("too many requests", "rate lim", "429", "timed out")):
                time.sleep(base_delay * (2 ** i))   # 1.5, 3, 6, 12s
                continue
            raise
    raise last


def _ticker(symbol: str):
    """yf.Ticker with the shared session when supported."""
    if _SESSION is not None:
        try:
            return yf.Ticker(symbol, session=_SESSION)
        except TypeError:
            pass   # this yfinance version manages its own session
    return yf.Ticker(symbol)


def _download(*args, **kwargs):
    """yf.download with the shared session when supported, plus retry/backoff."""
    def _do():
        if _SESSION is not None:
            try:
                return yf.download(*args, session=_SESSION, **kwargs)
            except TypeError:
                pass   # session param not accepted in this version
        return yf.download(*args, **kwargs)
    return _retry(_do)

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
            hist = _download(sym, period="1mo", interval="1d",
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


@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data. Returns empty DataFrame on failure."""
    try:
        df = _download(ticker, period=period, interval=interval,
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
        return _ticker(ticker).info
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# CBOE delayed options feed (free, no key, includes greeks) — PRIMARY source.
# One request returns every expiration+strike. Avoids Yahoo's shared-IP throttle.
# yfinance remains the automatic fallback if CBOE is unavailable.
# ══════════════════════════════════════════════════════════════════════════════
_CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"


def _cboe_symbol(sym: str) -> str:
    """CBOE uses an underscore prefix for index products (^VIX → _VIX)."""
    s = sym.strip().upper()
    return "_" + s[1:] if s.startswith("^") else s


def _http_json(url: str):
    """GET JSON via the chrome-impersonating session if available, else requests."""
    if _SESSION is not None:
        r = _SESSION.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    import requests
    r = requests.get(url, timeout=15,
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _parse_occ(occ: str):
    """OCC symbol → (expiry 'YYYY-MM-DD', 'C'/'P', strike float). Positional parse."""
    occ = occ.strip()
    if len(occ) < 16:
        return None
    try:
        strike = int(occ[-8:]) / 1000.0
        cp = occ[-9]
        ymd = occ[-15:-9]
        if cp not in ("C", "P"):
            return None
        return f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}", cp, strike
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_options_cboe(ticker: str):
    """
    Fetch CBOE delayed options. Returns (spot, {expiry: (calls_df, puts_df)}).
    DataFrames match the yfinance schema plus native greeks (gamma/delta/theta/vega).
    Returns (None, {}) on any failure so callers can fall back to yfinance.
    """
    try:
        data = _http_json(_CBOE_BASE.format(_cboe_symbol(ticker))).get("data", {})
    except Exception:
        return None, {}
    options = data.get("options") or []
    if not options:
        return None, {}
    spot = data.get("current_price") or data.get("close") or data.get("last")
    buckets = {}   # expiry -> {"C": [...], "P": [...]}
    for o in options:
        occ = o.get("option") or o.get("symbol")
        if not occ:
            continue
        parsed = _parse_occ(occ)
        if not parsed:
            continue
        expiry, cp, strike = parsed
        row = {
            "strike": strike,
            "impliedVolatility": float(o.get("iv") or 0.0),
            "openInterest": int(o.get("open_interest") or 0),
            "volume": int(o.get("volume") or 0),
            "lastPrice": float(o.get("last_trade_price") or 0.0),
            "bid": float(o.get("bid") or 0.0),
            "ask": float(o.get("ask") or 0.0),
            "gamma": float(o.get("gamma") or 0.0),
            "delta": float(o.get("delta") or 0.0),
            "theta": float(o.get("theta") or 0.0),
            "vega": float(o.get("vega") or 0.0),
        }
        buckets.setdefault(expiry, {"C": [], "P": []})[cp].append(row)

    expmap = {}
    for expiry, sides in buckets.items():
        calls = pd.DataFrame(sides["C"]); puts = pd.DataFrame(sides["P"])
        for df, label in [(calls, "call"), (puts, "put")]:
            if df.empty:
                continue
            df["type"] = label
            df["vol_oi_ratio"] = df.apply(
                lambda r: round(r["volume"] / r["openInterest"], 2)
                if r["openInterest"] > 0 else 0.0, axis=1)
        expmap[expiry] = (calls.sort_values("strike").reset_index(drop=True) if not calls.empty else calls,
                          puts.sort_values("strike").reset_index(drop=True) if not puts.empty else puts)
    try:
        spot = float(spot) if spot else None
    except Exception:
        spot = None
    return spot, expmap


def get_options_chain(ticker: str, expiry: str | None = None):
    """
    Returns (calls_df, puts_df, expirations_list).
    Tries CBOE (free, fast, has greeks) first; falls back to yfinance.
    """
    try:
        spot, expmap = get_options_cboe(ticker)
        if expmap:
            exps = sorted(expmap.keys())
            use = expiry if expiry in expmap else exps[0]
            calls, puts = expmap[use]
            if not calls.empty or not puts.empty:
                return calls, puts, exps
    except Exception:
        pass
    return _get_options_chain_yf(ticker, expiry)


@st.cache_data(ttl=900, show_spinner=False)
def _get_options_chain_yf(ticker: str, expiry: str | None = None):
    """
    Returns (calls_df, puts_df, expirations_list).
    Uses the shared session + retry/backoff; options is the most throttled
    Yahoo endpoint. Cached 15 min so repeated views don't re-hit Yahoo.
    """
    try:
        t = _ticker(ticker)
        expirations = _retry(lambda: list(t.options))
        if not expirations:
            return pd.DataFrame(), pd.DataFrame(), []

        expiry = expiry if expiry in expirations else expirations[0]
        chain  = _retry(lambda: t.option_chain(expiry))
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
        msg = str(e).lower()
        if any(k in msg for k in ("too many requests", "rate lim", "429")):
            st.warning(f"⏳ Yahoo is rate-limiting options data for {ticker}. "
                       f"This is a shared-IP throttle on Streamlit Cloud, not your app. "
                       f"Wait ~30–60s and rerun — cached data is reused for 15 min once it loads.")
        else:
            st.error(f"Options error for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), []


@st.cache_data(ttl=600, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Most recent available close price."""
    try:
        hist = _download(ticker, period="5d", interval="1d",
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
        raw = _download(syms, period=period, interval="1d",
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
        df = _download(ticker, period=period, interval=interval,
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


@st.cache_data(ttl=900, show_spinner=False)
def get_options_chains_multi(ticker: str, max_days: int = 35):
    """
    Fetch all option expirations within max_days as (expiry, calls, puts).
    CBOE primary — ONE request returns every expiry (no per-expiry throttle);
    yfinance is the fallback.
    """
    try:
        spot, expmap = get_options_cboe(ticker)
        if expmap:
            today = pd.Timestamp.now().normalize()
            out = []
            for exp in sorted(expmap.keys()):
                try:
                    days = (pd.Timestamp(exp) - today).days
                except Exception:
                    continue
                if 0 <= days <= max_days:
                    calls, puts = expmap[exp]
                    out.append((exp, calls, puts))
            if out:
                return out
    except Exception:
        pass
    return _get_options_chains_multi_yf(ticker, max_days)


@st.cache_data(ttl=900, show_spinner=False)
def _get_options_chains_multi_yf(ticker: str, max_days: int = 35):
    """yfinance fallback for multi-expiry chains (per-expiry calls; throttled)."""
    try:
        t = _ticker(ticker)
        expirations = _retry(lambda: list(t.options))
        if not expirations:
            return []
        today = pd.Timestamp.now().normalize()
        out = []
        for exp in expirations:
            try:
                days = (pd.Timestamp(exp) - today).days
            except Exception:
                continue
            if days < 0 or days > max_days:
                continue
            chain = _retry(lambda exp=exp: t.option_chain(exp))
            calls, puts = chain.calls.copy(), chain.puts.copy()
            for df in (calls, puts):
                for col in ["volume", "openInterest"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                for col in ["impliedVolatility", "lastPrice", "bid", "ask"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            out.append((exp, calls, puts))
        return out
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("too many requests", "rate lim", "429")):
            st.warning(f"⏳ Yahoo rate-limited multi-expiry options for {ticker}. "
                       f"Wait ~30–60s and rerun — cached 15 min once loaded.")
        else:
            st.error(f"Multi-expiry options error for {ticker}: {e}")
        return []
