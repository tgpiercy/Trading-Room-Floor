"""
StratFlow Adapter — single integration surface for the exit-validation stack
============================================================================
This is the ONLY file you ever edit to connect StratFlow to the exit
workstream (sweep page, Stage-2 comparison, future utils/exits.py).
Everything downstream imports from here and never changes.

Three things live here:
  1. get_universe()       -> list[str]
  2. build_signals(closes)-> (entry_df, decay_df)  boolean DataFrames
  3. get_regime()         -> str  ("risk_on" | "caution" | "risk_off")

Each tries StratFlow's known modules first and falls back to a working
stand-in, so the validation pages run out of the box and get *more*
accurate as you wire each hook. Wiring instructions sit directly above
each function.
"""

from __future__ import annotations
import importlib
import pandas as pd

# ---------------------------------------------------------------------------
# 1) UNIVERSE
# Edit CANDIDATE_UNIVERSE_SOURCES to match StratFlow: each tuple is
# (module_path, attribute). First import that succeeds wins. If your
# universe lives in a function or Google Sheet loader, just replace the
# body of get_universe() with that call.
# ---------------------------------------------------------------------------
CANDIDATE_UNIVERSE_SOURCES = [
    ("utils.universe", "UNIVERSE"),
    ("utils.tickers", "UNIVERSE"),
    ("config", "UNIVERSE"),
    ("core.universe", "UNIVERSE"),
]

_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "JPM", "XOM", "UNH", "COST", "LLY", "CAT", "DE", "LMT", "NOC",
    "PWR", "VRT", "ETN", "GEV", "CEG", "RTX", "PLTR", "ANET",
    "FCX", "GLD", "TLT", "XLE", "XLF",
]


def get_universe() -> tuple[list[str], str]:
    """Returns (tickers, source_label)."""
    for mod_path, attr in CANDIDATE_UNIVERSE_SOURCES:
        try:
            mod = importlib.import_module(mod_path)
            tickers = list(getattr(mod, attr))
            if tickers:
                return tickers, f"{mod_path}.{attr} (StratFlow)"
        except Exception:
            continue
    return _FALLBACK_UNIVERSE, "fallback stand-in (30 names)"


# ---------------------------------------------------------------------------
# 2) SIGNALS
# Replace the body below with StratFlow's validated entry signal and
# signal-decay exit. Contract (do not change):
#   input : closes — weekly close DataFrame, index=W-FRI dates, cols=tickers
#   output: (entry, decay) — boolean DataFrames, same shape as closes,
#           computed using data through bar t ONLY (no lookahead),
#           NaN-warmup filled entry=False / decay=True.
# If StratFlow's screener works from richer inputs (flow, volume), import
# and call it here — the contract only fixes the output shape.
# ---------------------------------------------------------------------------
MOM_LOOKBACK = 12   # weeks  (stand-in only)
TREND_MA = 30       # weeks  (stand-in only)
MOM_QUANTILE = 0.80


def build_signals(closes: pd.DataFrame):
    """STAND-IN: momentum top-quintile + 30w trend filter.
    Decay: rank below median OR close under 30w MA.
    Replace with StratFlow's validated logic."""
    mom = closes.pct_change(MOM_LOOKBACK)
    rank = mom.rank(axis=1, pct=True)
    ma = closes.rolling(TREND_MA).mean()
    entry = (rank >= MOM_QUANTILE) & (closes > ma)
    decay = (rank < 0.50) | (closes < ma)
    return entry.fillna(False), decay.fillna(True)


SIGNALS_ARE_STANDIN = build_signals.__doc__.startswith("STAND-IN")


# ---------------------------------------------------------------------------
# 3) REGIME
# Wired to utils/market_health.current_regime() per StratFlow v59. Maps
# whatever labels StratFlow uses onto the three-state contract used by the
# graduated exposure layer. Edit REGIME_MAP if your labels differ.
# ---------------------------------------------------------------------------
REGIME_MAP = {
    "risk_on": "risk_on", "bull": "risk_on", "healthy": "risk_on",
    "caution": "caution", "neutral": "caution", "mixed": "caution",
    "risk_off": "risk_off", "bear": "risk_off", "unhealthy": "risk_off",
}

# Frozen graduated exposure caps (Layer 1 of the exit stack)
REGIME_EXPOSURE = {"risk_on": 1.00, "caution": 0.60, "risk_off": 0.25}


def get_regime() -> str:
    try:
        mh = importlib.import_module("utils.market_health")
        raw = str(mh.current_regime()).strip().lower()
        return REGIME_MAP.get(raw, "caution")
    except Exception:
        return "caution"  # safe default when accessor unavailable
