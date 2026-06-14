"""
validation_finra_lab.py
FINRA SHORT-VOLUME LAB — standalone forward-edge event study.

The gate every candidate signal must clear before it earns an overlay test
(VSA and CMF both failed here). Question: does the FINRA short-volume ratio
carry CROSS-SECTIONAL forward information for this universe?

Each week, rank names by the signal, split into quintiles, and measure the
mean NEXT-WEEK return of each quintile. A real edge shows a monotonic gradient
across quintiles and a top-minus-bottom long-short spread whose Sharpe clears
DSR. Two signals are tested (multiple-testing aware):
  • LEVEL  — short-volume ratio itself (who is most shorted now)
  • CHANGE — ratio minus its 4-week mean (who is being shorted MORE)

Honest prior: market-maker hedging contaminates the ratio badly, and the
academic edge is thin — a clean negative is the expected, and still valuable,
outcome. Direction is read from the spread's sign, not assumed.

Data: one mid-week FINRA snapshot per week (tractable fetch; cached hard).
US-listed names only — Canadian (.TO) names have no FINRA data and drop out.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.finra_short import fetch_daily, daily_ratio_frame, to_weekly
from utils.significance import deflated_sharpe_ratio, verdict as sig_verdict
from stratflow_adapter import get_download_symbols, aux_frames, universe_label

ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="FINRA Short Lab", layout="wide")
st.title("FINRA Short-Volume Lab — forward-edge event study")
st.caption(f"**{universe_label()}** · does the short-volume ratio predict "
           "next-week cross-sectional returns? Standalone gate before any "
           "overlay.")

with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 2, 8, 3,
                      help="Short data fetched as one mid-week snapshot/week. "
                           "First run is slow (cached after).")
    q = st.slider("Quantiles", 3, 5, 5)
    min_names = st.slider("Min names/week", 15, 60, 30)
    run = st.button("Run FINRA Lab", type="primary", use_container_width=True)

if not run:
    st.info("Press **Run FINRA Lab**. Fetches weekly short-volume snapshots, "
            "then runs the event study. First fetch is slow; later runs cache.")
    st.stop()


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def _short_weekly(symbols, yrs):
    """One mid-week (Wed) FINRA snapshot per week → weekly short-ratio frame."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(yrs * 365.25))
    weds = pd.date_range(start, end, freq="W-WED")
    daily = {}
    for d in weds:
        df = fetch_daily(d.date())
        if df is not None:
            daily[d.date()] = df
    return to_weekly(daily_ratio_frame(symbols, daily), how="last")


def event_study(signal_wk, fwd1, q, min_names):
    """Per-week quintile sort → mean next-week return per basket + the weekly
    top-minus-bottom long-short series."""
    basket = {b: [] for b in range(q)}
    ls, weeks = [], 0
    for t in signal_wk.index:
        if t not in fwd1.index:
            continue
        s = signal_wk.loc[t].dropna()
        f = fwd1.loc[t].reindex(s.index)
        ok = f.notna()
        s, f = s[ok], f[ok]
        if len(s) < min_names:
            continue
        try:
            grp = pd.qcut(s.rank(method="first"), q, labels=False)
        except ValueError:
            continue
        means = f.groupby(grp).mean()
        for b in range(q):
            if b in means.index:
                basket[b].append(float(means[b]))
        if 0 in means.index and (q - 1) in means.index:
            ls.append(float(means[q - 1] - means[0]))
        weeks += 1
    grad = [round(float(np.mean(v)) * 100, 3) if v else None
            for b, v in basket.items()]            # mean next-wk % by quintile
    ls = pd.Series(ls, dtype=float)
    sh = float(ls.mean() / ls.std() * ANNUALISER) if ls.std() > 0 else 0.0
    return {"weeks": weeks, "gradient_pct": grad,
            "ls_sharpe": round(sh, 3),
            "ls_mean_bps": round(float(ls.mean()) * 1e4, 2)}, ls


# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner("Downloading prices…"):
    ohlcv = fetch_ohlcv_batch(get_download_symbols(), period=f"{years}y")
if not ohlcv:
    st.error("Price fetch failed. Retry shortly.")
    st.stop()
idx = next(iter(ohlcv.values()))["Close"].resample("W-FRI").last().dropna().index
aux = aux_frames(ohlcv, idx)
close = aux["close"]
fwd1 = close.pct_change().shift(-1)               # next-week return per name

syms = [c for c in close.columns]
with st.spinner(f"Fetching ~{years*52} weekly FINRA snapshots (cached after)…"):
    short_wk = _short_weekly(tuple(syms), years)

if short_wk.empty:
    st.error("No FINRA short-volume data returned. On Streamlit Cloud this "
             "fetches live; if it's empty, the CDN may be unreachable or the "
             "universe has no US-listed names with short data.")
    st.stop()

short_wk = short_wk.reindex(idx).reindex(columns=close.columns)
cover = int(short_wk.notna().any().sum())
st.caption(f"Short-volume coverage: **{cover}/{len(close.columns)}** names "
           f"(US-listed). {short_wk.notna().sum().sum():,} name-weeks of data.")

# signals
level = short_wk
change = short_wk - short_wk.rolling(4).mean()

# ── Event study (two arms) ────────────────────────────────────────────────────
res_level, ls_level = event_study(level, fwd1, q, min_names)
res_change, ls_change = event_study(change, fwd1, q, min_names)
arms = {"Level (short ratio)": res_level, "Change (Δ vs 4w mean)": res_change}

st.subheader("Quintile gradient — mean next-week return (%) by signal basket")
grad_df = pd.DataFrame({a: r["gradient_pct"] for a, r in arms.items()},
                       index=[f"Q{i+1}" for i in range(q)]).T
st.dataframe(grad_df, use_container_width=True)
st.caption("Q1 = lowest signal, "
           f"Q{q} = highest. A real edge is MONOTONIC across quintiles; a "
           "ragged row is noise. Long-short below uses top-minus-bottom.")

st.subheader("Long-short (top − bottom quintile, weekly)")
ls_df = pd.DataFrame({a: {"LS Sharpe (ann)": r["ls_sharpe"],
                          "LS mean (bps/wk)": r["ls_mean_bps"],
                          "weeks": r["weeks"]} for a, r in arms.items()}).T
st.dataframe(ls_df, use_container_width=True)

# ── Significance (best arm, both directions = sign of spread) ────────────────
allsr = [abs(res_level["ls_sharpe"]), abs(res_change["ls_sharpe"])]
best_arm = "Level (short ratio)" if allsr[0] >= allsr[1] else "Change (Δ vs 4w mean)"
best_sh = max(allsr)
n_obs = max(res_level["weeks"], res_change["weeks"])
sig = deflated_sharpe_ratio(best_sh, allsr, int(n_obs))
st.subheader("🎲 Significance")
st.markdown("- " + sig_verdict(sig))
st.caption("DSR on the better arm's |long-short Sharpe| across the 2 signals "
           "tested. Sign of the raw Sharpe gives direction (negative = high "
           "short ratio predicts LOWER forward returns).")

# ── Verdict ───────────────────────────────────────────────────────────────────
def _monotonic(g):
    g = [x for x in g if x is not None]
    if len(g) < 3:
        return False
    inc = all(g[i] <= g[i + 1] for i in range(len(g) - 1))
    dec = all(g[i] >= g[i + 1] for i in range(len(g) - 1))
    return inc or dec

mono = {a: _monotonic(r["gradient_pct"]) for a, r in arms.items()}
st.subheader("Verdict")
if sig["dsr"] >= 0.95 and sig["harvey_pass"] and mono.get(best_arm):
    direction = ("LOW short ratio → higher returns (go long the least-shorted)"
                 if (res_level["ls_sharpe"] if best_arm.startswith("Level")
                     else res_change["ls_sharpe"]) < 0
                 else "HIGH short ratio → higher returns (contrarian squeeze)")
    msg = (f"✅ **{best_arm}** shows a real, monotonic forward edge "
           f"(|LS Sharpe| {best_sh:.2f}, DSR {sig['dsr']:.3f}, Harvey pass). "
           f"Direction: {direction}. GRADUATES to an overlay test on the "
           "composite selector.")
elif sig["dsr"] >= 0.90 or any(mono.values()):
    msg = (f"🟠 **Marginal.** {best_arm} |LS Sharpe| {best_sh:.2f}, DSR "
           f"{sig['dsr']:.3f}; monotonic: {mono}. Borderline — not a clean "
           "pass. Would need a stronger gradient to justify an overlay.")
else:
    msg = (f"🛑 **No exploitable edge.** Best |LS Sharpe| {best_sh:.2f}, DSR "
           f"{sig['dsr']:.3f}, gradients ragged ({mono}). Consistent with the "
           "MM-hedging-contamination prior — short volume does not carry "
           "clean cross-sectional forward information here. Clean negative; "
           "document and move on (joins VSA / CMF).")
st.markdown("- " + msg)

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("📋 Results for Claude")
payload = {"stage": "finra_short_v1",
           "settings": {"years": years, "quantiles": q, "min_names": min_names,
                        "coverage": cover, "n_names": len(close.columns),
                        "n_weeks": int(n_obs)},
           "arms": arms,
           "monotonic": mono,
           "best_arm": best_arm,
           "significance": sig}
st.code(json.dumps(payload, indent=1, default=str), language="json")
