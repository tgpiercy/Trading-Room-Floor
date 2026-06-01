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

from utils.indicators import obv, cmf, relative_volume, mfi, force_index


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
    Score multi-day flow confluence across everything the Flow drilldown tracks,
    grouped into Price/Volume Flow and Options Flow (directional), with Dealer
    Gamma regime + RVOL participation as conviction modulators (not directional).

    daily        : daily OHLCV
    opt_snapshot : optional {call_pct, put_pct, prem_pcr, gex_sign, max_pain,
                   gamma_1wk (tilt), spot} from today's chain.

    dimensions entries are (group, name, score, note).
    """
    out = {"dimensions": [], "net": 0, "confidence": 0, "direction": "Neutral",
           "pattern": "Insufficient Data", "verdict": "", "confirm": "", "negate": "",
           "groups": {}, "series": {}}
    if daily.empty or len(daily) < window + 5:
        return out

    df = daily.copy()
    obv_s = obv(df)["OBV"]
    cmf_s = cmf(df)["CMF"]
    rvol_s = relative_volume(df)["RVOL"]
    mfi_s = mfi(df)["MFI"]
    fi_s = force_index(df)["FI"]
    close, vol = df["Close"], df["Volume"]

    pv = []   # price/volume flow (directional)

    # 1. OBV net flow
    net_obv = obv_s.iloc[-1] - obv_s.iloc[-window]
    gross = vol.tail(window).sum()
    flow_ratio = net_obv / gross if gross else 0.0
    pv.append(("Price/Volume", "OBV Net Flow", _score_band(flow_ratio, 0.20, 0.07),
               f"{flow_ratio:+.0%} of {window}d volume net "
               f"{'buying' if flow_ratio>0 else 'selling'}"))

    # 2. Price/OBV divergence (absorption)
    price_chg = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window] * 100
    if price_chg < 1.0 and flow_ratio > 0.07:
        s_div, dnote = 2, f"price flat/down ({price_chg:+.1f}%) but accumulating — absorption"
    elif price_chg > 1.0 and flow_ratio < -0.07:
        s_div, dnote = -2, f"price up ({price_chg:+.1f}%) but distributing — divergence"
    elif price_chg > 1.0 and flow_ratio > 0.07:
        s_div, dnote = 1, "price up with buying — healthy confirmation"
    else:
        s_div, dnote = 0, f"price {price_chg:+.1f}%, flow aligned/neutral"
    pv.append(("Price/Volume", "Price/OBV Divergence", s_div, dnote))

    # 3. CMF level + slope
    cmf_now = float(cmf_s.iloc[-1])
    cmf_prev = float(cmf_s.iloc[-window]) if len(cmf_s) > window else cmf_now
    cmf_rising = cmf_now > cmf_prev
    s_cmf = _score_band(cmf_now, 0.10, 0.03)
    if s_cmf > 0 and not cmf_rising: s_cmf = max(0, s_cmf - 1)
    if s_cmf < 0 and cmf_rising:     s_cmf = min(0, s_cmf + 1)
    pv.append(("Price/Volume", "Chaikin Money Flow", s_cmf,
               f"CMF {cmf_now:+.2f} {'rising' if cmf_rising else 'falling'}"))

    # 4. Money Flow Index (NEW) — level vs 50 + slope, overbought caution
    mfi_now = float(mfi_s.iloc[-1])
    mfi_prev = float(mfi_s.iloc[-window]) if len(mfi_s) > window else mfi_now
    s_mfi = _score_band(mfi_now - 50, 22, 8)
    if mfi_now > 80: s_mfi = min(s_mfi, 1)      # overbought caution
    if mfi_now < 20: s_mfi = max(s_mfi, -1)     # oversold caution
    if s_mfi > 0 and mfi_now < mfi_prev: s_mfi = max(0, s_mfi - 1)
    if s_mfi < 0 and mfi_now > mfi_prev: s_mfi = min(0, s_mfi + 1)
    pv.append(("Price/Volume", "Money Flow Index", s_mfi,
               f"MFI {mfi_now:.0f} {'rising' if mfi_now>=mfi_prev else 'falling'}"))

    # 5. Force Index (NEW) — normalized sign + slope
    fi_scale = fi_s.abs().tail(window).mean() + 1e-9
    fi_norm = float(fi_s.iloc[-1]) / fi_scale
    s_fi = _score_band(fi_norm, 1.0, 0.3)
    pv.append(("Price/Volume", "Force Index", s_fi,
               f"force {'positive' if fi_norm>=0 else 'negative'} "
               f"({fi_norm:+.1f}× avg)"))

    # 6. Accumulation streak
    daily_flow = np.sign(obv_s.diff().tail(window).fillna(0))
    streak = 0
    for v in reversed(daily_flow.values):
        if v > 0: streak += 1
        else: break
    s_acc = 2 if streak >= 5 else 1 if streak >= 3 else 0
    pv.append(("Price/Volume", "Accumulation Streak", s_acc,
               f"{streak} consecutive up-flow days"))

    opt = []   # options flow (directional)
    snap = opt_snapshot or {}
    if snap.get("call_pct") is not None:
        cp = snap["call_pct"]
        opt.append(("Options", "Premium Flow", _score_band(cp - 50, 18, 8),
                    f"{cp:.0f}% of premium in calls"))
        pcr = snap.get("prem_pcr")
        if pcr is not None:
            opt.append(("Options", "Put/Call (premium)", _score_band(1 - pcr, 0.30, 0.10),
                        f"premium PCR {pcr:.2f}"))
        mp, sp = snap.get("max_pain"), snap.get("spot")
        if mp and sp:
            pull = (mp - sp) / sp * 100
            s_mp = max(-1, min(1, _score_band(pull, 3.0, 1.0)))  # weak, capped ±1
            opt.append(("Options", "Max-Pain Pull", s_mp,
                        f"max pain ${mp:.2f} ({pull:+.1f}% vs spot)"))
    # DEX — net delta positioning lean (directional), independent of premium data
    if snap.get("dex_tilt") is not None:
        dxt = snap["dex_tilt"]
        opt.append(("Options", "Delta Exposure (DEX)", _score_band(dxt, 0.30, 0.10),
                    f"delta tilt {dxt:+.2f} — {'call' if dxt >= 0 else 'put'}-delta lean"))
        dxt = snap.get("dex_tilt")
        if dxt is not None:
            opt.append(("Options", "Delta Exposure (DEX)", _score_band(dxt, 0.40, 0.15),
                        f"net delta tilt {dxt:+.2f} "
                        f"({'call' if dxt >= 0 else 'put'}-delta lean)"))

    dims = pv + opt

    # ── Balanced net: each group normalized, then blended ─────────────────────
    pv_vals = [s for _, _, s, _ in pv]
    opt_vals = [s for _, _, s, _ in opt]
    pv_frac = sum(pv_vals) / (2 * len(pv_vals)) if pv_vals else 0.0
    opt_frac = (sum(opt_vals) / (2 * len(opt_vals))) if opt_vals else None
    blend = (0.6 * pv_frac + 0.4 * opt_frac) if opt_frac is not None else pv_frac
    net = int(round(blend * 10))   # −10..+10 flow score

    # Agreement across all directional non-zero signals
    all_vals = [s for _, _, s, _ in dims]
    nz = [s for s in all_vals if s != 0]
    agree = (sum(1 for s in nz if np.sign(s) == np.sign(net)) / len(nz)
             if nz and net != 0 else 0.0)

    # ── Modulators: RVOL participation + dealer gamma regime ──────────────────
    rvol_now = float(rvol_s.iloc[-1]) if not rvol_s.empty else 1.0
    gtilt = snap.get("gamma_1wk")
    gex_sign = snap.get("gex_sign")
    part_mod = 1.0 + max(-0.25, min(0.25, (rvol_now - 1.0) * 0.25))
    gamma_mod = 1.0
    gamma_note = None
    if gtilt is not None:
        if gtilt < -0.05:
            gamma_mod, gamma_note = 1.12, "short gamma (1wk) — moves likely to extend (squeeze fuel)"
        elif gtilt > 0.05:
            gamma_mod, gamma_note = 0.90, "long gamma (1wk) — expect pinning / mean reversion, fade extremes"
        else:
            gamma_note = "near-flat gamma (1wk)"
    elif gex_sign:
        gamma_note = ("negative GEX — dealers amplify moves" if gex_sign == "negative"
                      else "positive GEX — dealers dampen / pin")
        gamma_mod = 1.08 if gex_sign == "negative" else 0.94

    confidence = abs(blend) * (0.5 + 0.5 * agree) * part_mod * gamma_mod
    confidence = int(round(min(10, confidence * 11)))

    direction = "Bullish" if net >= 2 else "Bearish" if net <= -2 else "Neutral"

    absorbing = (s_div == 2)
    if net >= 6:   pattern = "Strong Accumulation"
    elif net >= 3: pattern = "Absorption / Quiet Accumulation" if absorbing else "Accumulation Building"
    elif net <= -6: pattern = "Strong Distribution"
    elif net <= -3: pattern = "Distribution into Strength" if s_div == -2 else "Distribution Building"
    else: pattern = "No Clear Pattern (chop)"

    # Context line
    ctx = []
    if rvol_now > 1.5: ctx.append(f"elevated participation (RVOL {rvol_now:.1f}×)")
    elif rvol_now < 0.7: ctx.append(f"thin participation (RVOL {rvol_now:.1f}×)")
    if gamma_note: ctx.append(gamma_note)
    charm = snap.get("charm")
    if charm is not None and abs(charm) > 1e4:
        ctx.append(f"charm drift {'up' if charm > 0 else 'down'} into expiry (OPEX pin)")
    charm_v = snap.get("charm")
    if charm_v is not None and abs(charm_v) > 5e5:
        ctx.append(f"charm pulls hedging {'up' if charm_v > 0 else 'down'} into expiry (pin pressure)")
    ctx_txt = (" Context: " + "; ".join(ctx) + ".") if ctx else ""

    last_px = float(close.iloc[-1])
    hi = float(close.tail(window).max()); lo = float(close.tail(window).min())
    short_gamma = (gtilt is not None and gtilt < -0.05) or gex_sign == "negative"
    if direction == "Bullish":
        confirm = (f"break above ${hi:.2f} ({window}d high)"
                   + (" — short gamma adds squeeze fuel" if short_gamma else " on rising volume"))
        negate = f"OBV rolling over or close below ${lo:.2f}"
    elif direction == "Bearish":
        confirm = f"break below ${lo:.2f} ({window}d low) on rising volume"
        negate = f"OBV turning up or reclaim of ${hi:.2f}"
    else:
        confirm = "a directional flow signal to emerge (currently mixed)"
        negate = "n/a — no active thesis"

    verdict = (f"{pattern} — {direction.lower()} flow {net:+d}/10 "
               f"({confidence}/10 confidence).{ctx_txt}")

    out.update(dimensions=dims, net=net, confidence=confidence, direction=direction,
               pattern=pattern, verdict=verdict, confirm=confirm, negate=negate,
               groups={"price_volume": round(pv_frac*10, 1),
                       "options": round(opt_frac*10, 1) if opt_frac is not None else None},
               modulators={"rvol": round(rvol_now, 2), "gamma_tilt_1wk": gtilt,
                           "gamma_note": gamma_note},
               series={"dates": df.index, "close": close, "obv": obv_s,
                       "cmf": cmf_s, "rvol": rvol_s, "mfi": mfi_s})
    return out


DIRECTION_COLOR = {"Bullish": "#00cc66", "Bearish": "#ff4444", "Neutral": "#ffd700"}


def score_color(s: int) -> str:
    return ("#00cc66" if s >= 2 else "#4fc3f7" if s == 1 else
            "#ff4444" if s <= -2 else "#ff8c00" if s == -1 else "#888888")


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED FLOW OUTLOOK — integrates price/volume flow + options premium + dealer
# positioning (DEX) + gamma regime into one directional read with key levels and
# a 'where money is moving' summary. Transparent weighted composite, NOT a
# black-box predictor. Reused by the single-ticker Summary and the sector Money Map.
# ══════════════════════════════════════════════════════════════════════════════
from utils.indicators import gamma_exposure, gamma_flip, max_pain, greek_exposures
from utils.order_flow import premium_flow, new_positioning


def flow_outlook(ticker: str, daily: pd.DataFrame, calls=None, puts=None,
                 spot: float = None, r: float = 0.045, expiry=None, window: int = 20) -> dict:
    """
    Returns a structured outlook:
      score (−10..+10), direction, conviction (0..10), pattern, regime (+read),
      levels {flip, max_pain, gex_resistance, gex_support, hi20, lo20},
      money_read (plain-language 'where money is moving'), positioning, base.
    Degrades gracefully to price/volume-only when options are unavailable.
    """
    out = {"ticker": ticker, "ok": False}
    if daily is None or daily.empty or len(daily) < window + 5:
        return out
    if spot is None or spot <= 0:
        spot = float(daily["Close"].iloc[-1])

    has_opts = (calls is not None and puts is not None
                and not calls.empty and not puts.empty)

    snap = None
    gex_net = flip = mp = dex_tilt = charm = vanna = call_pct = gtilt = None
    gx = None
    if has_opts:
        pf = premium_flow(calls, puts)
        call_pct = pf["call_pct"]
        gx = gamma_exposure(calls, puts, spot, expiry=expiry, r=r)
        if not gx.empty:
            gex_net = float(gx["net_gex"].sum())
            flip = gamma_flip(gx)
            if {"call_gex", "put_gex"}.issubset(gx.columns):
                gross = gx["call_gex"].abs().sum() + gx["put_gex"].abs().sum()
                gtilt = float(gex_net / gross) if gross else None
        mp = max_pain(calls, puts)
        ge = greek_exposures(calls, puts, spot, expiry=expiry, r=r)
        dex_tilt = ge.get("dex_tilt"); charm = ge.get("net_charm"); vanna = ge.get("net_vanna")
        snap = {"call_pct": call_pct, "prem_pcr": pf["prem_pcr"],
                "gex_sign": (("positive" if gex_net > 0 else "negative")
                             if gex_net is not None else None),
                "max_pain": round(mp, 2) if mp else None,
                "gamma_1wk": gtilt, "spot": float(spot)}

    base = analyze_flow(ticker, daily, window=window, opt_snapshot=snap)
    if base["pattern"] == "Insufficient Data":
        return out

    # New-positioning (vol > OI) directional skew
    newpos_bias, newpos_note = 0.0, "—"
    if has_opts:
        npd = new_positioning(calls, puts, min_premium=10_000)
        if not npd.empty and {"type", "premium"}.issubset(npd.columns):
            cp = npd[npd["type"].str.upper() == "CALL"]["premium"].sum()
            pp = npd[npd["type"].str.upper() == "PUT"]["premium"].sum()
            tot = cp + pp
            if tot > 0:
                newpos_bias = (cp - pp) / tot
                newpos_note = f"{cp / tot * 100:.0f}% of new (vol>OI) premium in calls"

    # Composite score (−10..+10): flow (incl. premium+gamma mod) + DEX + fresh money
    flow_c = base["net"]
    dex_c = (dex_tilt or 0.0) * 10
    np_c = newpos_bias * 10
    score = (0.55 * flow_c + 0.30 * dex_c + 0.15 * np_c) if has_opts else flow_c
    score = max(-10.0, min(10.0, score))
    direction = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"

    # Gamma regime
    if gtilt is None:
        regime, regime_read = "Unknown", "no options data"
    elif gtilt < -0.05:
        regime, regime_read = "Short gamma", "trending / squeezy — breakouts can run"
    elif gtilt > 0.05:
        regime, regime_read = "Long gamma", "pinned / mean-reverting — fade extremes"
    else:
        regime, regime_read = "Neutral gamma", "transitional"

    # Key levels
    hi20 = float(daily["Close"].tail(window).max())
    lo20 = float(daily["Close"].tail(window).min())
    gex_res = gex_sup = None
    if has_opts and gx is not None and not gx.empty:
        try:
            gex_res = float(gx.loc[gx["call_gex"].idxmax(), "strike"])
            gex_sup = float(gx.loc[gx["put_gex"].idxmin(), "strike"])
        except Exception:
            pass
    levels = {"flip": flip, "max_pain": round(mp, 2) if mp else None,
              "gex_resistance": gex_res, "gex_support": gex_sup,
              "hi20": round(hi20, 2), "lo20": round(lo20, 2)}

    # Money-movement read
    bits = []
    if call_pct is not None:
        bits.append(f"{call_pct:.0f}% of option premium in calls")
    if newpos_note != "—":
        bits.append(newpos_note)
    if dex_tilt is not None:
        bits.append(f"positioning {'call' if dex_tilt > 0 else 'put'}-delta heavy "
                    f"(DEX tilt {dex_tilt:+.2f})")
    money_read = "; ".join(bits) if bits else "price/volume flow only (no options)"

    out.update(ok=True, score=round(score, 1), direction=direction,
               conviction=base["confidence"], pattern=base["pattern"],
               regime=regime, regime_read=regime_read, levels=levels,
               money_read=money_read, base=base, last=round(float(daily["Close"].iloc[-1]), 2),
               positioning={"gex_net": gex_net, "dex_tilt": dex_tilt,
                            "charm": charm, "vanna": vanna, "newpos_bias": newpos_bias})
    return out
