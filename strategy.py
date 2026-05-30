"""
utils/strategy.py
GW2 v6.3 strategy signal engine — exact Pine Script logic.
All functions accept weekly OHLCV DataFrames (resampled from daily).
"""
import pandas as pd
import numpy as np

# ── Volume bucket definitions (from Volume Focus v2.2 Pine Script) ────────────
_B1 = set("SPY,QQQ,DIA,HXT,IWM,IWC,EEM,ACWX,FNDF,"
          "XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLV,XLU,XLY,GLD,SLV".split(","))
_B2 = set("BOTZ,XBI,REMX,ICLN,GDX".split(","))
_B3 = set("AIQ,IBB,ARKG,TAN,NLR,URA,GDXJ,SIL,COPX,AA,SLX,DD,MOO,"
          "XOP,OIH,MLPX,IEO,KBE,KRE,IAK,V,BLK,ITA,IYT,GRID,PAVE,"
          "SMCI,DELL,WDC,P,ANET,IGV,MSFT,SMH,NVDA,MU,ARM,CDNS,ASML,"
          "VRT,CRWD,CIBR,PPH,IHI,ITB,AMZN,TSLA,"
          "RY,BN,BAM,IFC,CNQ,ENB,FTS,CP,WSP,CAE,WCN,SHOP,ATD,CSU,"
          "CCO,AEM,NTR,TECK.B,TECK-B.TO".split(","))

def _vol_mult(ticker: str) -> float:
    t = ticker.upper()
    if t in _B1: return 1.25
    if t in _B2: return 1.40
    if t in _B3: return 1.50
    return 1.25  # default to B1 mult


# ── Price indicators ──────────────────────────────────────────────────────────
def calc_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA8/18/40 and ATR20 to a weekly OHLCV DataFrame."""
    df = df.copy()
    df["SMA8"]  = df["Close"].rolling(8).mean()
    df["SMA18"] = df["Close"].rolling(18).mean()
    df["SMA40"] = df["Close"].rolling(40).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR20"] = tr.rolling(20).mean()
    return df


# ── A/D line ──────────────────────────────────────────────────────────────────
def calc_ad(df: pd.DataFrame) -> pd.DataFrame:
    """Standard A/D line + SMA18 (GW2 simplified version, scaled /1M)."""
    hl  = df["High"] - df["Low"]
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl.replace(0, np.nan)
    clv = clv.fillna(0)
    ad  = (clv * df["Volume"]).cumsum() / 1_000_000
    return pd.DataFrame({
        "AD":      ad,
        "AD_SMA8": ad.rolling(8).mean(),
        "AD_SMA18":ad.rolling(18).mean(),
        "AD_SMA40":ad.rolling(40).mean(),
    }, index=df.index)


def calc_ad_state_gw2(ad_df: pd.DataFrame) -> tuple[bool, bool, str, str]:
    """
    GW2 simplified A/D state (scorecard inputs).
    Returns: (adAbove18, ad18Rising, adState, readComponent)
    Pine: ad18Rising = adSma18 >= adSma18[1]  (>= not >)
    """
    clean = ad_df.dropna(subset=["AD","AD_SMA18"])
    if len(clean) < 2:
        return False, False, "Weak", "WEAK"
    last, prev = clean.iloc[-1], clean.iloc[-2]
    above   = bool(last["AD"] > last["AD_SMA18"])
    rising  = bool(last["AD_SMA18"] >= prev["AD_SMA18"])
    state   = ("Strong" if above and rising else
               "Repair" if not above and rising else
               "Aging"  if above and not rising else "Weak")
    return above, rising, state, state


def calc_ad_weak_distrib(ad_df: pd.DataFrame) -> bool:
    """f_adWeakDistribSoft from GW2 Pine Script."""
    clean = ad_df.dropna(subset=["AD","AD_SMA18","AD_SMA40"])
    if len(clean) < 2:
        return False
    last, prev = clean.iloc[-1], clean.iloc[-2]
    ad, s18, s40 = last["AD"], last["AD_SMA18"], last["AD_SMA40"]
    s18_prev = prev["AD_SMA18"]
    cond1 = (ad < s18) and (s18 <= s18_prev)
    cond2 = (s18 < s40) and (s18 < s18_prev)
    return bool(cond1 or cond2)


def calc_ad_supportive(ad_df: pd.DataFrame) -> bool:
    """wAdSupportive: AD rising bar-over-bar AND AD_SMA18 rising."""
    clean = ad_df.dropna(subset=["AD","AD_SMA18"])
    if len(clean) < 2:
        return False
    last, prev = clean.iloc[-1], clean.iloc[-2]
    return bool(last["AD"] >= prev["AD"] and last["AD_SMA18"] >= prev["AD_SMA18"])


# ── GW2 7-point scorecard ─────────────────────────────────────────────────────
def calc_gw2_score(price_df: pd.DataFrame,
                   ad_df:    pd.DataFrame,
                   rs_df:    pd.DataFrame) -> dict:
    """
    GW2 v6.3 7-point scorecard. All Pine conditions matched exactly.
    Returns dict with score (0-7), individual flags, readTxt, adState, rsState.
    """
    result = dict(score=0, ad_above=False, ad_rising=False,
                  rs_above=False, rs_rising=False,
                  px_above=False, sma18_rising=False, sma18_gt_40=False,
                  read_txt="MIXED", ad_state="Weak", rs_state="Weak",
                  spread_atr=0.0, struct_strong=False)

    # ── Price flags ───────────────────────────────────────────────────────────
    p = price_df.dropna(subset=["SMA18","SMA40","ATR20"])
    if len(p) < 2:
        return result
    last_p, prev_p = p.iloc[-1], p.iloc[-2]
    px_above   = bool(last_p["Close"] > last_p["SMA18"])
    s18_rising = bool(last_p["SMA18"] >= prev_p["SMA18"])   # >= matches Pine
    s18_gt_40  = bool(last_p["SMA18"] > last_p["SMA40"])
    atr        = last_p["ATR20"]
    spread_atr = ((last_p["SMA18"] - last_p["SMA40"]) / atr) if atr > 0 else 0.0
    struct_strong = s18_rising and s18_gt_40 and spread_atr > 0.30

    # ── A/D flags ─────────────────────────────────────────────────────────────
    ad_above, ad_rising, ad_state, _ = calc_ad_state_gw2(ad_df)

    # ── RS flags ─────────────────────────────────────────────────────────────
    rs_clean = rs_df.dropna(subset=["RS","SMA18"])
    rs_above = rs_rising = False
    rs_state = "Weak"
    if len(rs_clean) >= 2:
        last_rs, prev_rs = rs_clean.iloc[-1], rs_clean.iloc[-2]
        rs_above  = bool(last_rs["RS"] > last_rs["SMA18"])
        rs_rising = bool(last_rs["SMA18"] >= prev_rs["SMA18"])   # >= matches Pine
        rs_state  = ("Strong" if rs_above and rs_rising else
                     "Repair" if not rs_above and rs_rising else
                     "Aging"  if rs_above and not rs_rising else "Weak")

    # ── Score ─────────────────────────────────────────────────────────────────
    score = sum([ad_above, ad_rising, rs_above, rs_rising,
                 px_above, s18_rising, s18_gt_40])

    # ── Read text (GW2 table header) ──────────────────────────────────────────
    if   ad_above and ad_rising and rs_above and rs_rising:   read = "LEADERSHIP"
    elif ad_above and ad_rising and not rs_above and rs_rising: read = "EARLY ROTATION"
    elif not ad_above and ad_rising and not rs_above and rs_rising: read = "REPAIR"
    elif ad_above and not ad_rising and rs_above and not rs_rising: read = "AGING LEADERSHIP"
    elif not ad_above and not ad_rising and not rs_above and not rs_rising: read = "DEAD / BROKEN"
    else: read = "MIXED"

    result.update(
        score=score, ad_above=ad_above, ad_rising=ad_rising,
        rs_above=rs_above, rs_rising=rs_rising,
        px_above=px_above, sma18_rising=s18_rising, sma18_gt_40=s18_gt_40,
        read_txt=read, ad_state=ad_state, rs_state=rs_state,
        spread_atr=round(spread_atr, 2), struct_strong=struct_strong,
        atr=atr,
    )
    return result


# ── Impulse state ─────────────────────────────────────────────────────────────
def calc_impulse_state(price_df:    pd.DataFrame,
                       ad_df:       pd.DataFrame,
                       rs_df:       pd.DataFrame,
                       gw2:         dict) -> str:
    """
    GW2 v6.3 Impulse state — weekly only (no daily data in screener).
    YES / IMPROVING / WEAKENING / SETUP / NO
    """
    p = price_df.dropna(subset=["SMA18","SMA40","ATR20"])
    if len(p) < 2:
        return "NO"

    last_p, prev_p = p.iloc[-1], p.iloc[-2]
    close   = last_p["Close"]
    sma18   = last_p["SMA18"]
    sma40   = last_p["SMA40"]
    atr     = last_p["ATR20"]

    w_px_above_18   = close > sma18
    w_px_above_40   = close > sma40
    w_18_above_40   = sma18 > sma40
    w_18_rising     = sma18 >= prev_p["SMA18"]
    w_struct_strong = gw2["struct_strong"]
    w_struct_improv = w_18_rising and w_18_above_40

    # Transition struct: above SMA40 but SMA18 still below SMA40, rising
    w_transition = (close > sma40 and w_18_rising and sma18 < sma40)

    # A/D flags
    ad_above, ad_rising, _, _ = calc_ad_state_gw2(ad_df)
    w_ad_strong     = ad_above and ad_rising
    w_ad_supportive = calc_ad_supportive(ad_df)
    w_ad_weak       = calc_ad_weak_distrib(ad_df)

    # RS flags
    rs_clean = rs_df.dropna(subset=["RS","SMA8","SMA18"])
    w_rs_strong = w_rs_supportive = w_rs_weakening = False
    if len(rs_clean) >= 2:
        last_rs, prev_rs = rs_clean.iloc[-1], rs_clean.iloc[-2]
        w_rs_strong     = (last_rs["RS"] > last_rs["SMA18"] and
                           last_rs["SMA18"] >= prev_rs["SMA18"])
        w_rs_supportive = last_rs["RS"] > last_rs["SMA18"]
        # Weakening: RS < SMA8 but SMA8 still above SMA18
        if "SMA8" in rs_clean.columns:
            w_rs_weakening = (last_rs["RS"] < last_rs["SMA8"] and
                              last_rs["SMA8"] > last_rs["SMA18"])

    # wImpulse (weekly full)
    w_impulse = w_ad_strong and w_rs_strong and w_px_above_18 and w_struct_strong

    # wSoftImproving
    w_soft_improving = (
        w_rs_supportive and
        (w_px_above_18 or w_transition) and
        (w_struct_improv or w_transition) and
        not w_ad_weak and
        not w_impulse
    )

    # wSetup
    w_setup = (
        w_px_above_40 and w_18_rising and
        not w_ad_weak and
        not w_soft_improving and
        not w_impulse
    )

    # wRsWeakening
    w_rs_weakening_state = (
        w_rs_weakening and
        not w_ad_weak and
        (w_px_above_18 or w_px_above_40)
    )

    # Final cascade (weekly only — dGood approximated as wImpulse)
    if   w_impulse:             return "YES"
    elif w_soft_improving:      return "IMPROVING"
    elif w_rs_weakening_state:  return "WEAKENING"
    elif w_setup:               return "SETUP"
    else:                       return "NO"


# ── Weekly state machine (0=OUT, 1=PILOT, 2=FULL) ────────────────────────────
def calc_weekly_state(price_df:  pd.DataFrame,
                      ad_df:     pd.DataFrame,
                      rs_df:     pd.DataFrame,
                      gw2:       dict,
                      impulse:   str) -> int:
    """
    GW2 weekly state: 0=OUT, 1=PILOT, 2=FULL.
    Evaluated as the current end-state (not a bar-by-bar simulation).
    """
    p = price_df.dropna(subset=["SMA18","SMA40","ATR20"])
    if len(p) < 2:
        return 0

    last_p, prev_p = p.iloc[-1], p.iloc[-2]
    close, sma18, sma40, atr = (last_p["Close"], last_p["SMA18"],
                                 last_p["SMA40"], last_p["ATR20"])
    atr_dist     = (close - sma18) / atr if atr > 0 else 0
    sma18_rising = sma18 >= prev_p["SMA18"]
    sma40_flatup = sma40 >= prev_p["SMA40"]
    px_above_8   = "SMA8" in last_p.index and close > last_p["SMA8"]
    sma8_rising  = False
    if "SMA8" in p.columns and len(p) >= 2:
        sma8_rising = last_p["SMA8"] >= prev_p["SMA8"]

    rs_strong = gw2["rs_above"] and gw2["rs_rising"]

    # earlyPilotOk
    early_pilot_ok = (impulse == "IMPROVING" and
                      gw2["score"] >= 3 and atr_dist < 2)

    # pilotStructOk
    pilot_ok = (impulse != "WEAKENING" and
                (early_pilot_ok or (sma40_flatup and px_above_8 and sma8_rising)))

    # addStructOk
    add_ok = (impulse == "YES" and rs_strong and
              gw2["score"] >= 4 and gw2["px_above"] and
              sma18_rising and atr_dist < 3)

    # fullStructOk
    full_ok = (impulse == "YES" and
               (add_ok or (pilot_ok and gw2["px_above"] and sma18_rising)))

    # Exit condition
    exit_now = close < sma18

    if exit_now:
        return 0
    elif full_ok:
        return 2
    elif pilot_ok:
        return 1
    else:
        return 0


# ── Stop levels ───────────────────────────────────────────────────────────────
def calc_stops(price_df: pd.DataFrame) -> dict:
    """
    GW2 stop logic.
    blueLine = structStop when ATR_dist < 1.0, else max(structStop, trailStop)
    """
    p = price_df.dropna(subset=["SMA18","ATR20"])
    if p.empty:
        return {}
    last = p.iloc[-1]
    close, sma18, atr = last["Close"], last["SMA18"], last["ATR20"]
    atr_dist    = (close - sma18) / atr if atr > 0 else 0
    struct_stop = sma18 - 0.50 * atr
    trail_stop  = close - 1.25 * atr
    blue_line   = struct_stop if atr_dist < 1.0 else max(struct_stop, trail_stop)
    green_line  = sma18 + 1.25 * atr
    red_line    = sma18 + 2.50 * atr
    return dict(
        struct_stop  = round(struct_stop, 2),
        trail_stop   = round(trail_stop,  2),
        program_stop = round(blue_line,   2),
        stop_limit   = round(blue_line - 0.15 * atr, 2),
        green_line   = round(green_line,  2),
        red_line     = round(red_line,    2),
        atr_dist     = round(atr_dist,    2),
        atr          = round(atr,         4),
        mgmt_mode    = "DAILY" if atr_dist >= 2.5 else "WEEKLY",
    )


# ── ATR regime ────────────────────────────────────────────────────────────────
def calc_atr_regime(price_df: pd.DataFrame) -> dict:
    """ATR v1.4 regime: Compression / Normal / Expansion."""
    p = price_df.dropna(subset=["ATR20"])
    if len(p) < 18:
        return {"state": "Unknown", "atr": 0, "ratio": 0}
    sma_atr = p["ATR20"].rolling(18).mean()
    last_atr = p["ATR20"].iloc[-1]
    last_sma = sma_atr.iloc[-1]
    ratio = last_atr / last_sma if last_sma > 0 else 1.0
    state = "Compression" if ratio < 1.0 else "Normal" if ratio < 1.1 else "Expansion"
    return {"state": state, "atr": round(last_atr, 4), "ratio": round(ratio, 3)}


# ── Dual RSI ──────────────────────────────────────────────────────────────────
def calc_dual_rsi(close: pd.Series) -> dict:
    """Dual RSI v1.2: RSI5 vs RSI20, four states."""
    def rsi(s, n):
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        l = (-d.clip(upper=0)).rolling(n).mean()
        return (100 - 100 / (1 + g / l)).dropna()

    r5  = rsi(close, 5)
    r20 = rsi(close, 20)
    if r5.empty or r20.empty:
        return {"state": "Red", "rsi5": 50, "rsi20": 50}

    v5, v20 = float(r5.iloc[-1]), float(r20.iloc[-1])

    if   v5 >= v20 and v20 > 50:  state = "Blue"
    elif v5 >  v20:                state = "Green"
    elif v5 <  v20 and v20 >= 50:  state = "Orange"
    else:                          state = "Red"

    return {"state": state, "rsi5": round(v5, 1), "rsi20": round(v20, 1)}


# ── Volume Focus ──────────────────────────────────────────────────────────────
def calc_volume_focus(df: pd.DataFrame, ticker: str) -> dict:
    """Volume Focus v2.2 — requires weekly OHLCV (with Open column)."""
    DIST_HOLD = 4
    if len(df) < 20 or "Volume" not in df.columns or "Open" not in df.columns:
        return {"state": "Neutral", "vol": 0, "vol_ma18": 0}

    d = df.copy()
    mult = _vol_mult(ticker)
    d["VolMA18"]  = d["Volume"].rolling(18).mean()
    d["VolMA5"]   = d["Volume"].rolling(5).mean()
    d["SMA18_p"]  = d["Close"].rolling(18).mean()
    clean = d.dropna(subset=["VolMA18","VolMA5","SMA18_p","Open"])
    if len(clean) < 2:
        return {"state": "Neutral", "vol": 0, "vol_ma18": 0}

    def _red(row, prev):
        thresh  = row["VolMA18"] * mult
        bearish = row["Close"] < row["Open"]
        block   = row["Close"] <= prev["Close"]    # blockDistIfUp
        gate    = row["Close"] > row["SMA18_p"]    # useSma18GateForDist
        return bool(row["Volume"] > thresh and bearish and block and gate)

    # Sticky RED — check last DIST_HOLD bars
    red_active = False
    tail = clean.tail(DIST_HOLD + 1)
    for i in range(1, len(tail)):
        if _red(tail.iloc[i], tail.iloc[i - 1]):
            bars_since = len(tail) - 1 - i
            if bars_since < DIST_HOLD:
                red_active = True
                break

    last, prev = clean.iloc[-1], clean.iloc[-2]
    thresh = last["VolMA18"] * mult
    g_thresh = last["VolMA5"] * 1.05

    is_major   = last["Volume"] > thresh
    is_green_v = last["Volume"] > g_thresh and last["Volume"] > prev["Volume"]
    bullish    = last["Close"] > last["Open"]
    bearish    = last["Close"] < last["Open"]
    above_s18  = last["Close"] > last["SMA18_p"]

    if _red(last, prev):
        red_active = True

    purple = is_major and bullish and not red_active
    green  = is_green_v and bullish and above_s18 and not red_active and not purple

    state = ("Distribution" if red_active else
             "Accumulation"  if purple     else
             "Early Accum"   if green      else "Neutral")

    return {
        "state":      state,
        "vol":        int(last["Volume"]),
        "vol_ma18":   round(last["VolMA18"], 0),
        "major_thresh": round(thresh, 0),
    }


# ── RS Momentum (SMA4) ────────────────────────────────────────────────────────
def calc_rs_momentum(rs_df: pd.DataFrame) -> str:
    """RS v6.2 momentum using SMA4 vs SMA8 vs SMA18."""
    if "SMA4" not in rs_df.columns:
        return "Unknown"
    clean = rs_df.dropna(subset=["SMA4","SMA8","SMA18"])
    if len(clean) < 2:
        return "Unknown"
    last, prev = clean.iloc[-1], clean.iloc[-2]
    sma4, sma8, sma18 = last["SMA4"], last["SMA8"], last["SMA18"]
    sma4_rising = sma4 > prev["SMA4"]
    if   sma4 > sma8 and sma8 > sma18 and sma4_rising: return "Strong"
    elif sma4 > sma8 and sma4_rising:                   return "Improving"
    elif sma4 < sma18 or not sma4_rising:               return "Deteriorating"
    else:                                                return "Flat"


# ── Parent RS state ───────────────────────────────────────────────────────────
def calc_parent_rs_state(parent_close: pd.Series,
                          bench_close:  pd.Series,
                          band_pct:     float = 0.75) -> str:
    """Parent RS v1.8 regime: Bullish / Neutral / Bearish."""
    from utils.rs_indicators import build_rs_df
    df = build_rs_df(parent_close, bench_close)
    clean = df.dropna(subset=["RS","SMA18"])
    if clean.empty:
        return "Unknown"
    last = clean.iloc[-1]
    rs, s18 = last["RS"], last["SMA18"]
    hi = s18 * (1 + band_pct / 100)
    lo = s18 * (1 - band_pct / 100)
    if   rs > hi: return "Bullish"
    elif rs < lo: return "Bearish"
    else:         return "Neutral"


# ── Composite entry signal ────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "Strong Entry":  "#00cc66",
    "Entry / Add":   "#00ff88",
    "Watch":         "#ffd700",
    "Neutral":       "#888888",
    "Reduce":        "#ff8c00",
    "Exit / Avoid":  "#ff4444",
}

def calc_entry_signal(impulse: str, weekly_state: int, gw2: dict,
                      rs_state: str, rsi_state: str, vol_state: str,
                      atr_state: str, parent_rs: str,
                      mh_pct: float) -> dict:
    """
    Composite entry/add/reduce/exit signal.
    Each condition scores a point; composite determines action zone.
    """
    pts = 0
    detail = {}

    # Market Health gate
    mh_ok = mh_pct > 40
    detail["Market Health > 40%"] = mh_ok
    if mh_ok: pts += 1

    # Impulse
    imp_ok = impulse in ("YES", "IMPROVING")
    detail[f"Impulse ({impulse})"] = imp_ok
    if imp_ok: pts += 1

    # GW2 score
    gw2_ok = gw2["score"] >= 5
    detail[f"GW2 Score ≥ 5 ({gw2['score']}/7)"] = gw2_ok
    if gw2_ok: pts += 1

    # RS state
    rs_ok = rs_state in ("Early Leadership", "Healthy Trend")
    detail[f"RS State ({rs_state})"] = rs_ok
    if rs_ok: pts += 1

    # RSI state
    rsi_ok = rsi_state in ("Blue", "Green")
    detail[f"RSI ({rsi_state})"] = rsi_ok
    if rsi_ok: pts += 1

    # Volume
    vol_ok = vol_state in ("Accumulation", "Early Accum")
    detail[f"Volume ({vol_state})"] = vol_ok
    if vol_ok: pts += 1

    # Parent RS
    par_ok = parent_rs in ("Bullish", "Neutral")
    detail[f"Parent RS ({parent_rs})"] = par_ok
    if par_ok: pts += 1

    # ATR not expansion
    atr_ok = atr_state != "Expansion"
    detail[f"ATR ({atr_state})"] = atr_ok
    if atr_ok: pts += 1

    # Weekly state
    ws_ok = weekly_state >= 1
    detail[f"Weekly State ({'PILOT' if weekly_state==1 else 'FULL' if weekly_state==2 else 'OUT'})"] = ws_ok
    if ws_ok: pts += 1

    # Exit triggers (hard overrides)
    exit_triggers = []
    if rs_state == "Broken Trend":        exit_triggers.append("RS Broken Trend")
    if impulse == "NO":                    exit_triggers.append("Impulse = NO")
    if gw2["score"] <= 2:                  exit_triggers.append(f"GW2 Score ≤ 2 ({gw2['score']})")
    if vol_state == "Distribution":        exit_triggers.append("Distribution signal")

    # Signal classification
    if exit_triggers:
        signal = "Exit / Avoid"
    elif pts >= 8:   signal = "Strong Entry"
    elif pts >= 6:   signal = "Entry / Add"
    elif pts >= 4:   signal = "Watch"
    elif pts >= 2:   signal = "Reduce"
    else:            signal = "Exit / Avoid"

    return {
        "signal":        signal,
        "pts":           pts,
        "max_pts":       9,
        "detail":        detail,
        "exit_triggers": exit_triggers,
        "color":         SIGNAL_COLORS.get(signal, "#888"),
    }
