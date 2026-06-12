"""
utils/selection.py
The validated production SELECTOR — frozen from the Selection Lab
(selection_lab_v1) and its Gate-3 confirmation sweep (selection_confirm_v1).

Composite rank, two equal-weighted components (cross-sectional percentile):
  1. 26-week RS momentum        — slow relative trend vs each name's own
                                  designated benchmark (the core contributor)
  2. vol-adjusted ExtPct        — RS extension above its 18w SMA divided by
                                  26w RS-return volatility (signal-to-noise)
plus a REDUNDANCY FILTER at entry: walking candidates strongest-first, skip
any whose trailing 26w return correlation with an already-accepted stronger
name exceeds 0.85 (caps hidden concentration the sleeve labels can't see).

Validated vs raw-ExtPct baseline: +0.08-0.12 Sharpe, fewer trades, longer
holds, better F1-F3 folds; Gate-3 sweep flat across mom∈{13,26,39} ×
corr∈{0.80,0.85,0.90} (plateau, not peak). Constants below are FROZEN at
the a-priori mid-plateau choice — changing them invalidates the validation.

Negative results encoded here by absence:
  - ExtPct<0 exit condition (redundant — exit_stage2_v1 config E)
  - single-benchmark ranking (worse on every metric — selection_lab_v1 E)
  - defensive bond overlay (hurt all folds incl. 2022 — selection_lab_v1 G)
"""
from __future__ import annotations
import pandas as pd

from utils.watchlist import yf_sym

# ── Frozen, validated configuration ──────────────────────────────────────────
MOM_WEEKS = 26       # slow RS-momentum lookback
VOL_WEEKS = 26       # RS-return vol window for the adjustment
CORR_MAX = 0.85      # redundancy ceiling at entry
CORR_WIN = 26        # weeks of returns for the correlation
CAND_POOL = 25       # filter walks this many top candidates
SELECTOR_VERSION = "composite_v1 (mom26 ⊕ vol-adj ext · redundancy 0.85)"


def rs_frames(ohlcv: dict, pairs: list, cal: pd.DatetimeIndex):
    """(rs, close) DataFrames — display-ticker columns on `cal`.
    rs = close / designated-benchmark close per utils.watchlist pairs."""
    closes, rs = {}, {}
    for dt_tk, db_tk, _g in pairs:
        yt, yb = yf_sym(dt_tk), yf_sym(db_tk)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        c = ohlcv[yt]["Close"].reindex(cal).ffill()
        b = ohlcv[yb]["Close"].reindex(cal).ffill()
        closes[dt_tk] = c
        rs[dt_tk] = c / b
    return pd.DataFrame(rs, index=cal), pd.DataFrame(closes, index=cal)


def composite_rank(ext: pd.DataFrame, rs: pd.DataFrame) -> pd.DataFrame:
    """Causal composite rank frame (1 = strongest).
    ext: the validated ExtPct frame (from precompute_series); rs: rs_frames."""
    rs = rs.reindex(columns=ext.columns)
    mom = rs.pct_change(MOM_WEEKS)
    rs_vol = (rs.pct_change().rolling(VOL_WEEKS).std() * 100).replace(0, pd.NA)
    ext_adj = ext / rs_vol
    score = (mom.rank(axis=1, pct=True) + ext_adj.rank(axis=1, pct=True))
    return score.rank(axis=1, ascending=False)


def redundancy_filter_today(rank_row: pd.Series, close_df: pd.DataFrame,
                            top_n: int, corr_max: float = CORR_MAX,
                            window: int = CORR_WIN,
                            pool: int = CAND_POOL) -> list:
    """Today's entry list: candidates strongest-first, skipping any with
    trailing-`window` return correlation > corr_max to an accepted name.
    Returns up to top_n display tickers, ordered by rank."""
    row = rank_row.dropna().sort_values()
    cands = list(row[row <= pool].index)
    if not cands:
        return []
    seg = close_df.pct_change().iloc[-window:]
    accepted = []
    for tk in cands:
        if tk not in seg.columns:
            continue
        s1 = seg[tk]
        if s1.isna().all() or s1.std() == 0:
            continue
        clash = False
        for a in accepted:
            c = s1.corr(seg[a])
            if c == c and c > corr_max:
                clash = True
                break
        if not clash:
            accepted.append(tk)
            if len(accepted) >= top_n:
                break
    return accepted
