"""
StratFlow Adapter — single integration surface for the exit-validation stack
============================================================================
v2 — WIRED TO THE VALIDATED STRATFLOW MACHINERY (no stand-ins).

This file is the only place the exit workstream (sweep page, Stage-2
comparison, future utils/exits.py) touches StratFlow internals. It calls the
exact code paths the walk-forward validated:

  * Universe + pairs ......... utils.watchlist (PORTFOLIO / ALL_YF_SYMBOLS)
  * Signal ................... utils.strategy_backtest.precompute_series
                               (causal ExtPct vs each name's own benchmark)
  * Regime ................... utils.strategy_backtest.compute_regime_exposure
                               (same SPY/IEF/VIX gate as current_regime())

Frozen signal→trade mapping for the per-name exit study
-------------------------------------------------------
The production strategy is a top-N rotation. For trade-level exit analysis
that maps to:
  ENTRY : name's causal ExtPct rank enters the top ENTRY_TOP_N (= the
          validated fixed top-N of 10)
  DECAY : rank falls below EXIT_RANK (2× buffer — hysteresis so ordinary
          rank jitter doesn't churn) OR ExtPct < 0 (RS lost its SMA18 —
          leadership gone) OR ExtPct unavailable.
These two constants are FROZEN for Stage 1. Stage 2 sensitivity-checks them.

Regime is intentionally NOT applied to entries here — exposure gating is
Layer 1 of the exit stack and is studied separately. The exposure series is
exported for that purpose.
"""

from __future__ import annotations
import pandas as pd

from utils.watchlist import PORTFOLIO, ALL_YF_SYMBOLS, yf_sym
from utils.strategy_backtest import precompute_series

# ── Frozen constants ─────────────────────────────────────────────────────────
ENTRY_TOP_N = 10        # validated fixed top-N
EXIT_RANK = 20          # LEGACY BASELINE for the exit sweep's pre-widening arm
                        # only. The FROZEN production band is 30 (utils/exits.py,
                        # used by Rebalance and the selection/HRP/CPCV labs). Do
                        # not treat this 20 as the production exit rank.
SIGNAL = "extpct"       # the validated primary signal
SIGNALS_ARE_STANDIN = False  # real StratFlow signals — sweep results count

# Graduated exposure caps consumed by Layer 1 (regime) of the exit stack.
# compute_regime_exposure already emits {1.0, 0.5, 0.0}; the caution cap is
# read from it directly, these are labels for UI surfaces.
REGIME_EXPOSURE = {"risk_on": 1.00, "caution": 0.50, "risk_off": 0.00}


def get_download_symbols() -> tuple:
    """Every yf symbol the signal stack needs: all watchlist names + their
    benchmarks + the regime trio."""
    return tuple(dict.fromkeys(list(ALL_YF_SYMBOLS) + ["SPY", "IEF", "^VIX"]))


def universe_label() -> str:
    n = len({p[0] for p in PORTFOLIO})
    return f"utils.watchlist.PORTFOLIO ({n} names, validated ExtPct signal)"


def prepare(ohlcv: dict):
    """ohlcv: {yf_symbol: weekly W-FRI OHLCV df} from fetch_ohlcv_batch.

    Returns dict:
      data     : {display_ticker: df[open, high, low, close]} on the common
                 weekly calendar (leading NaN = not yet listed)
      entry    : bool DataFrame  — ExtPct rank ≤ ENTRY_TOP_N
      decay    : bool DataFrame  — default decay (rank > EXIT_RANK or
                 ExtPct < 0 / missing), immediate (1-week) confirmation
      rank,ext : the causal rank / ExtPct frames (for decay variants)
      exposure : causal regime exposure Series (1.0 / 0.5 / 0.0)
    Raises ValueError with the engine's message if inputs are unusable.
    """
    pc = precompute_series(ohlcv, list(PORTFOLIO), signal=SIGNAL)
    if "error" in pc:
        raise ValueError(pc["error"])

    cal = pc["cal"]
    ext = pd.DataFrame({tk: d["ext"] for tk, d in pc["data"].items()},
                       index=cal)
    rank = ext.rank(axis=1, ascending=False)        # 1 = strongest ExtPct
    entry = (rank <= ENTRY_TOP_N).fillna(False).astype(bool)
    decay = build_decay(rank, ext)

    data = {}
    for tk in pc["data"]:
        yt = yf_sym(tk)
        df = ohlcv[yt][["Open", "High", "Low", "Close"]].reindex(cal).ffill()
        data[tk] = df.rename(columns=str.lower)
    return {"data": data, "entry": entry, "decay": decay,
            "rank": rank, "ext": ext, "exposure": pc["exposure"]}


def build_decay(rank: pd.DataFrame, ext: pd.DataFrame,
                exit_rank: int = EXIT_RANK, confirm_weeks: int = 1,
                use_ext_neg: bool = True) -> pd.DataFrame:
    """Decay-exit variants for Stage 2. A name decays when the bad condition
    (rank beyond exit_rank, optionally OR ExtPct<0; missing data always bad)
    has held for `confirm_weeks` consecutive weeks. Causal: row T uses ≤ T."""
    bad = (rank > exit_rank) | rank.isna()
    if use_ext_neg:
        bad = bad | (ext < 0)
    if confirm_weeks <= 1:
        return bad.fillna(True).astype(bool)
    conf = bad.rolling(confirm_weeks).sum() >= confirm_weeks
    return conf.fillna(False).astype(bool)  # warmup: not yet confirmed


def aux_frames(ohlcv: dict, cal: pd.DatetimeIndex) -> dict:
    """Auxiliary causal frames for the Selection Lab (display-ticker columns
    on the supplied calendar):
      rs       : RS ratio vs each name's OWN designated benchmark
      ext_spy  : ExtPct computed vs SPY for ALL names (single-benchmark
                 control; CAD names mix currency vs SPY — control arm only)
      close    : weekly closes per display ticker
    """
    closes, rs, rs_spy = {}, {}, {}
    spy = ohlcv.get("SPY")
    spy_c = spy["Close"].reindex(cal).ffill() if spy is not None else None
    for dt_tk, db_tk, _g in PORTFOLIO:
        yt, yb = yf_sym(dt_tk), yf_sym(db_tk)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        c = ohlcv[yt]["Close"].reindex(cal).ffill()
        b = ohlcv[yb]["Close"].reindex(cal).ffill()
        closes[dt_tk] = c
        rs[dt_tk] = c / b
        if spy_c is not None:
            rs_spy[dt_tk] = c / spy_c
    rs = pd.DataFrame(rs, index=cal)
    close_df = pd.DataFrame(closes, index=cal)
    ext_spy = pd.DataFrame(index=cal)
    if rs_spy:
        rsp = pd.DataFrame(rs_spy, index=cal)
        sma18 = rsp.rolling(18).mean()
        ext_spy = (rsp - sma18) / sma18 * 100.0
    return {"rs": rs, "ext_spy": ext_spy, "close": close_df}


def get_regime() -> str:
    """Live three-state regime from the canonical accessor (dict-returning)."""
    try:
        from utils.market_health import current_regime
        exp = current_regime().get("exposure")
        if exp is None:
            return "caution"
        return ("risk_on" if exp >= 0.99 else
                "risk_off" if exp <= 0.01 else "caution")
    except Exception:
        return "caution"
