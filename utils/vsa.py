"""
utils/vsa.py
Volume Spread Analysis (Wyckoff/Tom-Williams style) — reads the relationship
between a bar's SPREAD (range), CLOSE LOCATION, and VOLUME relative to recent
norms to infer smart-money accumulation vs distribution.

This is a transparent heuristic classifier over recent bars, not a black box:
each flagged signal maps to a textbook VSA pattern. Bullish patterns (no-supply,
tests/springs, stopping & selling climax) net against bearish ones (no-demand,
upthrusts, buying climax) into a bias.
"""
import numpy as np
import pandas as pd

# signal → (points, label)  ; +bullish / -bearish
_BULL = {"No Supply", "Test", "Spring", "Stopping Volume", "Selling Climax"}
_BEAR = {"No Demand", "Upthrust", "Buying Climax", "Effort↓"}


def vsa_analysis(daily: pd.DataFrame, recent: int = 8, vol_win: int = 20) -> dict:
    """Return {bias, score, signal (latest notable), note, signals:[...]}.
    bias ∈ {Accumulation, Distribution, Neutral}."""
    out = {"bias": "Neutral", "score": 0, "signal": "—", "note": "insufficient data", "signals": []}
    if daily is None or daily.empty or len(daily) < vol_win + recent + 2:
        return out
    if not {"High", "Low", "Close", "Volume"}.issubset(daily.columns):
        return out

    h, l, c, v = daily["High"], daily["Low"], daily["Close"], daily["Volume"]
    spread = (h - l).replace(0, np.nan)
    rng_pos = ((c - l) / spread).clip(0, 1)              # 0 = close at low, 1 = at high
    avg_vol = v.rolling(vol_win).mean()
    avg_spr = spread.rolling(vol_win).mean()
    prevc = c.shift(1)

    n = len(daily)
    flagged = []
    score = 0
    for i in range(n - recent, n):
        if i < vol_win + 1:
            continue
        av, asp = avg_vol.iloc[i], avg_spr.iloc[i]
        if not (av == av and asp == asp and av > 0 and asp > 0):
            continue
        vol, sp, rp = v.iloc[i], spread.iloc[i], rng_pos.iloc[i]
        if not (sp == sp and rp == rp):
            continue
        up = c.iloc[i] > prevc.iloc[i]
        down = c.iloc[i] < prevc.iloc[i]
        hi_vol, ultra = vol > 1.5 * av, vol > 2.0 * av
        low_vol = vol < 0.7 * av
        wide, narrow = sp > 1.3 * asp, sp < 0.7 * asp
        new_hi = h.iloc[i] >= h.iloc[max(0, i - recent):i + 1].max()
        new_lo = l.iloc[i] <= l.iloc[max(0, i - recent):i + 1].min()

        sig = None
        if wide and ultra and down and rp > 0.6:
            sig = "Selling Climax"      # capitulation, close strong → bullish
        elif wide and ultra and down and rp >= 0.45:
            sig = "Stopping Volume"     # big vol absorbing supply → bullish
        elif wide and ultra and up and rp < 0.5:
            sig = "Buying Climax"       # huge vol up but weak close → bearish
        elif wide and hi_vol and rp < 0.35 and new_hi:
            sig = "Upthrust"            # new high rejected on volume → bearish
        elif new_lo and rp > 0.6 and low_vol:
            sig = "Spring"             # new low rejected on low vol → bullish
        elif down and rp > 0.6 and low_vol:
            sig = "Test"               # low-vol test of supply, close up → bullish
        elif up and narrow and low_vol:
            sig = "No Demand"          # rally on no volume → bearish
        elif down and narrow and low_vol:
            sig = "No Supply"          # decline on no volume → bullish
        elif hi_vol and narrow:
            sig = "Effort↓"            # effort, no result (churn) → caution/bearish

        if sig:
            w = 2 if i >= n - 3 else 1   # recency weight (last 3 bars count double)
            score += w if sig in _BULL else (-w if sig in _BEAR else 0)
            flagged.append((daily.index[i], sig))

    if score >= 2:
        bias = "Accumulation"
    elif score <= -2:
        bias = "Distribution"
    else:
        bias = "Neutral"
    latest = flagged[-1][1] if flagged else "—"
    note = (f"{bias.lower()} — latest: {latest}" if flagged
            else "no notable VSA signals recently")
    out.update(bias=bias, score=int(score), signal=latest, note=note,
               signals=[s for _, s in flagged])
    return out
