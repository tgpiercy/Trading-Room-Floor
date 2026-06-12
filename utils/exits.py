"""
utils/exits.py
The validated exit stack — frozen specification from the Stage-1/Stage-2
exit studies (exit_sweep_v1, exit_stage2_v1, 10y, costed at 10 bps/side).

Three layers, monotonic (each can only reduce or close exposure):

  Layer 1 — REGIME (portfolio): graduated exposure from the validated
            SPY/IEF/VIX gate (utils.strategy_backtest.compute_regime_exposure).
            Validated to compose cleanly with Layers 2-3 (config F).
  Layer 2 — DECAY (per name, primary): exit when ExtPct rank has been worse
            than EXIT_RANK for CONFIRM_WEEKS consecutive weeks. Fires ~97%
            of exits. The Stage-2 winner (config D); the ExtPct<0 condition
            was proven redundant (config E) and is deliberately absent.
  Layer 3 — TRAIL (per name, backstop): chandelier stop at TRAIL_K × Wilder
            ATR(ATR_PERIOD) below the highest weekly close since entry.
            Fires ~3% of exits — disaster insurance, not the workhorse.

Re-entry is inherent: the rotation entry (rank ≤ ENTRY_TOP_N) re-admits any
exited name automatically; no special-case logic exists or should be added.

FROZEN means frozen: changing any constant below invalidates the validation
and requires re-running the Exit Lab page (Stage 1 + Stage 2 + folds).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ── Frozen, validated configuration ──────────────────────────────────────────
ENTRY_TOP_N   = 10     # rotation entry: ExtPct rank ≤ 10
EXIT_RANK     = 30     # decay boundary (hysteresis band 11-30 = hold zone)
CONFIRM_WEEKS = 2      # consecutive weeks beyond EXIT_RANK before decay fires
TRAIL_K       = 4.0    # chandelier multiplier (Stage-1 plateau)
ATR_PERIOD    = 14     # weekly Wilder ATR

EXIT_LAYER_COLOR = {"decay": "#ff9800", "trail": "#ff4444", "hold": "#4fc3f7"}


# ── Primitives (identical math to the validation harness) ────────────────────
def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               n: int = ATR_PERIOD) -> np.ndarray:
    """Wilder-smoothed ATR on weekly bars. NaN during warmup."""
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    atr = np.full_like(tr, np.nan)
    if len(tr) <= n:
        return atr
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def trail_level(peak_close: float, atr: float,
                k: float = TRAIL_K) -> float | None:
    """Chandelier stop level for display / order placement."""
    if atr is None or atr != atr or peak_close is None:
        return None
    return float(peak_close - k * atr)


def decay_matrix(rank: pd.DataFrame,
                 exit_rank: int = EXIT_RANK,
                 confirm_weeks: int = CONFIRM_WEEKS) -> pd.DataFrame:
    """Vectorized Layer-2 decay signal (causal): True at (t, name) when the
    name's rank has been > exit_rank (or missing) for confirm_weeks
    consecutive weeks ending at t. Matches the validated build."""
    bad = (rank > exit_rank) | rank.isna()
    if confirm_weeks <= 1:
        return bad.fillna(True).astype(bool)
    conf = bad.rolling(confirm_weeks).sum() >= confirm_weeks
    return conf.fillna(False).astype(bool)


# ── Per-position decision (live surfaces: Portfolio / Rebalance) ─────────────
def evaluate_position(rank_recent: pd.Series,
                      weekly_ohlc: pd.DataFrame,
                      entry_date=None) -> dict:
    """Decision for ONE held name at the latest bar.

    rank_recent : the name's causal ExtPct-rank Series (at least the last
                  CONFIRM_WEEKS values; lower = stronger).
    weekly_ohlc : weekly df with High/Low/Close (any casing), enough history
                  for ATR warmup; index = W-FRI dates.
    entry_date  : position entry date (peak tracked from there). If None,
                  peak is tracked over the available history (conservative).

    Returns {action: HOLD|EXIT, layer: hold|decay|trail, trail_stop,
             rank_now, weeks_in_breach, note}.
    """
    cols = {c.lower(): c for c in weekly_ohlc.columns}
    h = weekly_ohlc[cols["high"]].to_numpy(float)
    l = weekly_ohlc[cols["low"]].to_numpy(float)
    c = weekly_ohlc[cols["close"]].to_numpy(float)
    atr = wilder_atr(h, l, c)
    last_close, last_atr = float(c[-1]), float(atr[-1])

    win = weekly_ohlc if entry_date is None else weekly_ohlc.loc[entry_date:]
    peak = float(win[cols["close"]].max()) if len(win) else last_close
    stop = trail_level(peak, last_atr)

    r = rank_recent.dropna() if rank_recent is not None else pd.Series(dtype=float)
    rank_now = float(r.iloc[-1]) if len(r) else None
    tail = rank_recent.iloc[-CONFIRM_WEEKS:] if rank_recent is not None else pd.Series(dtype=float)
    breach = bool(len(tail) >= CONFIRM_WEEKS
                  and ((tail > EXIT_RANK) | tail.isna()).all())
    weeks_breach = 0
    if rank_recent is not None:
        for v in reversed(rank_recent.tolist()):
            if (v != v) or v > EXIT_RANK:
                weeks_breach += 1
            else:
                break

    if breach:
        return {"action": "EXIT", "layer": "decay", "trail_stop": stop,
                "rank_now": rank_now, "weeks_in_breach": weeks_breach,
                "note": f"rank > {EXIT_RANK} for {CONFIRM_WEEKS}+ wks"}
    if stop is not None and last_close < stop:
        return {"action": "EXIT", "layer": "trail", "trail_stop": stop,
                "rank_now": rank_now, "weeks_in_breach": weeks_breach,
                "note": f"close {last_close:.2f} < chandelier {stop:.2f}"}
    note = (f"in hold band (rank {rank_now:.0f})" if rank_now is not None
            else "rank unavailable — monitoring trail only")
    if weeks_breach == 1:
        note += f" · 1 wk beyond {EXIT_RANK}, needs {CONFIRM_WEEKS}"
    return {"action": "HOLD", "layer": "hold", "trail_stop": stop,
            "rank_now": rank_now, "weeks_in_breach": weeks_breach,
            "note": note}


def hold_band_filter(current_tickers: list, rank: pd.DataFrame,
                     decay: pd.DataFrame | None = None) -> dict:
    """Split held names into keep/release at the latest bar — the hold-band
    logic for the Rebalance target book: entries need rank ≤ ENTRY_TOP_N,
    but holdings persist until Layer-2 decay confirms.
    Returns {ticker: 'keep'|'release'} for names present in rank."""
    if decay is None:
        decay = decay_matrix(rank)
    out = {}
    last = decay.iloc[-1]
    for tk in current_tickers:
        if tk in last.index:
            out[tk] = "release" if bool(last[tk]) else "keep"
    return out
