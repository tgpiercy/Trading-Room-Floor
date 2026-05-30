"""
utils/fred.py
FRED (St. Louis Fed) macro data client. Powers the macro/credit regime
panel and supplies the live risk-free rate for the GEX gamma calc.

API key must live in Streamlit secrets:  st.secrets["FRED_API_KEY"]
Never hardcode it in a committed file.

FRED is macro/economic data only — it does NOT provide option Greeks,
intraday, or order flow. Its value here is the regime layer, especially
credit spreads, which lead equity weakness.
"""
import pandas as pd
import requests
import streamlit as st

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series we care about
SERIES = {
    "hy_spread":   "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS (%)
    "ig_spread":   "BAMLC0A0CM",     # ICE BofA US Corporate (IG) OAS (%)
    "curve_10y2y": "T10Y2Y",         # 10yr − 2yr Treasury (%)
    "curve_10y3m": "T10Y3M",         # 10yr − 3mo Treasury (%)
    "stress":      "STLFSI4",        # St. Louis Fed Financial Stress Index
    "nfci":        "NFCI",           # Chicago Fed National Financial Conditions
    "rate_3mo":    "DGS3MO",         # 3-month Treasury yield (%) — risk-free
}


def fred_key():
    """Read FRED key from Streamlit secrets; None if absent."""
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return None


def fred_available() -> bool:
    return fred_key() is not None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, days: int = 760) -> pd.Series:
    """Fetch one FRED series as a date-indexed Series. Empty on any failure."""
    key = fred_key()
    if not key:
        return pd.Series(dtype=float)
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start,
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=12)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate(default: float = 0.045) -> float:
    """Latest 3-month Treasury as a decimal (e.g. 0.0438). Falls back to default."""
    s = fetch_fred_series(SERIES["rate_3mo"], days=30)
    if s.empty:
        return default
    try:
        return round(float(s.iloc[-1]) / 100.0, 4)
    except Exception:
        return default


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data() -> dict:
    """Fetch all macro series. Returns {key: Series} (empty Series if unavailable)."""
    return {k: fetch_fred_series(v) for k, v in SERIES.items()}


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def calc_macro_regime(macro: dict) -> dict:
    """
    Composite macro regime from credit spreads, yield curve, and financial
    stress. Produces a 0-100 Macro Health score (comparable to the price-based
    Market Health %) plus a regime label and component breakdown.

    This is a PARALLEL read — intentionally NOT folded into the price-based
    MH% — so the two can confirm or diverge.
    """
    comps = {}
    detail = {}

    # ── Credit: HY spread level + 1-month direction ───────────────────────────
    hy = macro.get("hy_spread", pd.Series(dtype=float))
    if not hy.empty:
        last   = float(hy.iloc[-1])
        mo_ago = float(hy.iloc[-22]) if len(hy) >= 22 else float(hy.iloc[0])
        widening = last > mo_ago
        # Tight spread → high score. ~2.5% → 100, ~8% → 0.
        lvl_score = _clamp(100 * (8.0 - last) / (8.0 - 2.5))
        dir_adj   = -12 if widening else +8
        comps["credit"] = _clamp(lvl_score + dir_adj)
        detail["HY Credit Spread"] = (f"{last:.2f}% · "
            f"{'widening ⚠️' if widening else 'tightening'} "
            f"(1mo: {mo_ago:.2f}%)")

    # ── Yield curve: 10y-2y ───────────────────────────────────────────────────
    cv = macro.get("curve_10y2y", pd.Series(dtype=float))
    if not cv.empty:
        c = float(cv.iloc[-1])
        # Inverted (<0) low, steep (>1.5) high. 0 → ~50.
        comps["curve"] = _clamp(50 + c * 30)
        detail["Yield Curve (10y−2y)"] = (f"{c:+.2f}% · "
            f"{'inverted ⚠️' if c < 0 else 'normal' if c < 1 else 'steep'}")

    # ── Financial stress: STLFSI4 (0 = normal, + = stress) ───────────────────
    ss = macro.get("stress", pd.Series(dtype=float))
    if not ss.empty:
        s = float(ss.iloc[-1])
        comps["stress"] = _clamp(50 - s * 28)
        detail["Financial Stress (STLFSI)"] = (f"{s:+.2f} · "
            f"{'elevated ⚠️' if s > 0.5 else 'calm' if s < -0.3 else 'normal'}")

    # ── NFCI (Chicago Fed): + = tight conditions ─────────────────────────────
    nf = macro.get("nfci", pd.Series(dtype=float))
    if not nf.empty:
        n = float(nf.iloc[-1])
        detail["Financial Conditions (NFCI)"] = (f"{n:+.2f} · "
            f"{'tight ⚠️' if n > 0 else 'loose'}")

    if not comps:
        return {"available": False, "score": 0, "regime": "No Data", "detail": {}}

    score = round(sum(comps.values()) / len(comps))
    regime = ("Risk-On"  if score >= 70 else
              "Neutral"  if score >= 50 else
              "Caution"  if score >= 30 else "Risk-Off")
    return {"available": True, "score": score, "regime": regime,
            "components": comps, "detail": detail}


MACRO_REGIME_COLOR = {
    "Risk-On":  "#00cc66",
    "Neutral":  "#ffd700",
    "Caution":  "#ff8c00",
    "Risk-Off": "#ff4444",
    "No Data":  "#888888",
}
