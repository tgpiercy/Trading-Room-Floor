"""
utils/flow_analysis.py
Interpretation layer — turns raw flow metrics into a multi-day pattern read:
a scored confluence table + a plain-language verdict.

Two history sources:
  • LIVE (always available): price-derived flow — OBV, CMF, RVOL, money-flow
    divergence, accumulation persistence — computed from the daily OHLCV
    download (2y available), so trends are visible immediately.
  • PERSISTED (builds going forward): options-derived flow — premium call/put,
    GEX sign, max pain — snapshotted daily, since Yahoo only returns "now".
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import date

from utils.indicators import obv, cmf, relative_volume


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE  (best-effort; ephemeral on Streamlit Cloud redeploys)
# ══════════════════════════════════════════════════════════════════════════════
def _hist_path() -> str:
    for d in (os.getcwd(), "/tmp"):
        try:
            if os.path.isdir(d) and os.access(d, os.W_OK):
                return os.path.join(d, ".stratflow_flow_history.json")
        except Exception:
            continue
    return "/tmp/.stratflow_flow_history.json"


def load_history() -> dict:
    try:
        with open(_hist_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def save_snapshot(ticker: str, snap: dict, keep_days: int = 90) -> None:
    """Save today's options-flow snapshot for a ticker (idempotent per day)."""
    try:
        h = load_history()
        h.setdefault(ticker, {})[date.today().isoformat()] = snap
        days = sorted(h[ticker].keys())[-keep_days:]
        h[ticker] = {d: h[ticker][d] for d in days}
        with open(_hist_path(), "w") as f:
            json.dump(h, f)
    except Exception:
        pass


def get_persisted(ticker: str) -> pd.DataFrame:
    """Return persisted options-flow history for a ticker as a DataFrame."""
    h = load_history().get(ticker, {})
    if not h:
        return pd.DataFrame()
    rows = [{"date": d, **v} for d, v in sorted(h.items())]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION SCORING  (each directional dim scores −2..+2)
# ══════════════════════════════════════════════════════════════════════════════
def _score_band(x, t2, t1):
    """+2/+1/0/−1/−2 by symmetric thresholds t2>t1."""
    if   x >=  t2: return 2
    elif x >=  t1: return 1
    elif x <= -t2: return -2
    elif x <= -t1: return -1
    return 0


def analyze_flow(ticker: str, daily: pd.DataFrame, window: int = 20,
                 opt_snapshot: dict | None = None) -> dict:
    """
    Score multi-day flow confluence and produce a verdict.

    daily        : daily OHLCV (>= window+30 rows ideal)
    opt_snapshot : optional {call_pct, put_pct, prem_pcr, gex_sign, max_pain}
                   from today's options chain (also gets persisted by the page)
    """
    out = {"dimensions": [], "net": 0, "confidence": 0, "direction": "Neutral",
           "pattern": "Insufficient Data", "verdict": "", "confirm": "", "negate": "",
           "series": {}}

    if daily.empty or len(daily) < window + 5:
        return out

    df = daily.copy()
    obv_s = obv(df)["OBV"]
    cmf_s = cmf(df)["CMF"]
    rvol_s = relative_volume(df)["RVOL"]
    close = df["Close"]
    vol = df["Volume"]

    dims = []

    # ── 1. OBV net flow ratio (fraction of window volume that was net buying) ─
    net_obv = obv_s.iloc[-1] - obv_s.iloc[-window]
    gross   = vol.tail(window).sum()
    flow_ratio = net_obv / gross if gross else 0.0
    s_obv = _score_band(flow_ratio, 0.20, 0.07)
    dims.append(("OBV Net Flow", s_obv,
                 f"{flow_ratio:+.0%} of {window}d volume net "
                 f"{'buying' if flow_ratio>0 else 'selling'}"))

    # ── 2. OBV / price divergence (absorption vs distribution) ────────────────
    price_chg = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window] * 100
    if price_chg < 1.0 and flow_ratio > 0.07:
        s_div = 2; dnote = f"price flat/down ({price_chg:+.1f}%) but accumulating — absorption"
    elif price_chg > 1.0 and flow_ratio < -0.07:
        s_div = -2; dnote = f"price up ({price_chg:+.1f}%) but distributing — bearish divergence"
    elif price_chg > 1.0 and flow_ratio > 0.07:
        s_div = 1; dnote = f"price up with buying — healthy confirmation"
    else:
        s_div = 0; dnote = f"price {price_chg:+.1f}%, flow aligned/neutral"
    dims.append(("Price/OBV Divergence", s_div, dnote))

    # ── 3. CMF level + slope ──────────────────────────────────────────────────
    cmf_now = float(cmf_s.iloc[-1])
    cmf_prev = float(cmf_s.iloc[-window]) if len(cmf_s) > window else cmf_now
    cmf_rising = cmf_now > cmf_prev
    s_cmf = _score_band(cmf_now, 0.10, 0.03)
    if s_cmf > 0 and not cmf_rising: s_cmf = max(0, s_cmf - 1)
    if s_cmf < 0 and cmf_rising:     s_cmf = min(0, s_cmf + 1)
    dims.append(("Chaikin Money Flow", s_cmf,
                 f"CMF {cmf_now:+.2f} {'rising' if cmf_rising else 'falling'}"))

    # ── 4. Accumulation persistence (consecutive up-flow days) ────────────────
    daily_flow = np.sign(obv_s.diff().tail(window).fillna(0))
    streak = 0
    for v in reversed(daily_flow.values):
        if v > 0: streak += 1
        elif v < 0: break
        else: break
    s_acc = 2 if streak >= 5 else 1 if streak >= 3 else 0
    dims.append(("Accumulation Streak", s_acc, f"{streak} consecutive up-flow days"))

    # ── 5. Premium flow (options, if available) ───────────────────────────────
    if opt_snapshot and opt_snapshot.get("call_pct") is not None:
        cp = opt_snapshot["call_pct"]
        s_prem = _score_band(cp - 50, 18, 8)   # 68%+ calls = +2
        dims.append(("Options Premium Flow", s_prem,
                     f"{cp:.0f}% of premium in calls"))
    else:
        dims.append(("Options Premium Flow", 0, "no options data (skipped)"))

    # ── Net + confidence ──────────────────────────────────────────────────────
    scores = [s for _, s, _ in dims]
    net = int(sum(scores))
    max_abs = 2 * len(scores)
    # Agreement: fraction of non-zero dims sharing the net's sign
    nz = [s for s in scores if s != 0]
    if nz and net != 0:
        agree = sum(1 for s in nz if np.sign(s) == np.sign(net)) / len(nz)
    else:
        agree = 0.0
    confidence = round(min(10, abs(net) / max_abs * 10 * (0.5 + 0.5 * agree) * 2))
    confidence = min(confidence, 10)

    direction = "Bullish" if net >= 2 else "Bearish" if net <= -2 else "Neutral"

    # ── Pattern label ─────────────────────────────────────────────────────────
    absorbing = (s_div == 2)
    if net >= 6:
        pattern = "Strong Accumulation"
    elif net >= 3:
        pattern = "Absorption / Quiet Accumulation" if absorbing else "Accumulation Building"
    elif net <= -6:
        pattern = "Strong Distribution"
    elif net <= -3:
        pattern = "Distribution into Strength" if s_div == -2 else "Distribution Building"
    else:
        pattern = "No Clear Pattern (chop)"

    # ── Context: GEX + RVOL (not summed; flavor the verdict) ──────────────────
    rvol_now = float(rvol_s.iloc[-1]) if not rvol_s.empty else 1.0
    gex_sign = opt_snapshot.get("gex_sign") if opt_snapshot else None
    ctx = []
    if rvol_now > 1.5: ctx.append(f"elevated participation (RVOL {rvol_now:.1f}×)")
    elif rvol_now < 0.7: ctx.append(f"thin participation (RVOL {rvol_now:.1f}×)")
    if gex_sign == "positive": ctx.append("positive GEX (dealers dampening vol / pinning)")
    elif gex_sign == "negative": ctx.append("negative GEX (dealers amplifying moves / squeeze risk)")

    # ── Verdict + confirm/negate ──────────────────────────────────────────────
    last_px = float(close.iloc[-1])
    hi20 = float(close.tail(window).max())
    lo20 = float(close.tail(window).min())
    if direction == "Bullish":
        confirm = f"volume-backed break above ${hi20:.2f} ({window}d high)"
        negate  = f"OBV rolling over or close below ${lo20:.2f}"
    elif direction == "Bearish":
        confirm = f"break below ${lo20:.2f} ({window}d low) on rising volume"
        negate  = f"OBV turning up or reclaim of ${hi20:.2f}"
    else:
        confirm = "a directional flow signal to emerge (currently mixed)"
        negate  = "n/a — no active thesis"

    ctx_txt = (" Context: " + "; ".join(ctx) + ".") if ctx else ""
    verdict = (f"{pattern} — {direction.lower()} flow confluence "
               f"{abs(net)}/{max_abs} ({confidence}/10 confidence).{ctx_txt}")

    out.update(dimensions=dims, net=net, confidence=confidence, direction=direction,
               pattern=pattern, verdict=verdict, confirm=confirm, negate=negate,
               series={"dates": df.index, "close": close, "obv": obv_s,
                       "cmf": cmf_s, "rvol": rvol_s})
    return out


DIRECTION_COLOR = {"Bullish": "#00cc66", "Bearish": "#ff4444", "Neutral": "#ffd700"}


def score_color(s: int) -> str:
    return ("#00cc66" if s >= 2 else "#4fc3f7" if s == 1 else
            "#ff4444" if s <= -2 else "#ff8c00" if s == -1 else "#888888")
