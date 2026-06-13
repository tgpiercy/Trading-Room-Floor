"""
pages/6_Market_Health.py
MARKET HEALTH — climate dashboard. Reframed (regime_lab_v1) from a magic
number into a transparent view of the ONE validated exposure gate and the
three inputs that drive it. Every decisional number on this page feeds a
validated decision; the old MH% score is demoted to a clearly-labelled
legacy expander (tested: credit and breadth confirmation do NOT improve the
gate — VIX already carries the stress signal).

Decisional core:
  • Headline = current_regime() — the canonical gate verdict
  • Component breakdown = the exact scoring of compute_regime_exposure
    (trend 40 + SPY/IEF risk-appetite 30 + VIX 30, cut 66/33), so you can
    see WHY the gate reads as it does and how close it is to flipping.
Diagnostic context (NOT decisional, shown for awareness, labelled as such):
  • Credit (HY OAS vs its 26w mean) and breadth (% sectors > 10w MA).
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils.market_health import current_regime, sector_breadth
from utils.data_fetcher import get_stock_data
from utils.strategy_backtest import compute_regime_exposure

st.title("🏥 Market Health")
st.caption("The one validated exposure gate (trend + risk-appetite + VIX), "
           "shown transparently. Regime Lab confirmed these three inputs span "
           "the climate — credit and breadth add nothing, so they appear here "
           "as context only, never as decisions.")


def _wk(sym):
    d = get_stock_data(sym, period="2y", interval="1d")
    return (d["Close"].resample("W-FRI").last().dropna()
            if (d is not None and not d.empty) else pd.Series(dtype=float))


with st.spinner("Reading the climate…"):
    spy, ief, vix = _wk("SPY"), _wk("IEF"), _wk("^VIX")

if spy.empty or ief.empty or vix.empty:
    st.error("Could not fetch SPY / IEF / VIX. Retry shortly.")
    st.stop()

cal = spy.index
ief = ief.reindex(cal).ffill()
vix = vix.reindex(cal).ffill()
exposure = compute_regime_exposure(spy, ief, vix)

# ── Headline: the canonical gate verdict ─────────────────────────────────────
reg = current_regime()
st.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;"
    f"background:{reg['color']}22;border:1px solid {reg['color']};"
    f"font-size:1.15rem;font-weight:700'>Exposure gate: {reg['label']}</div>",
    unsafe_allow_html=True)
st.caption("This single verdict gates capital on the Rebalance page and in "
           "the backtested strategy. Everything below explains it.")

# ── Component breakdown — exactly how the gate scores right now ───────────────
trend_ma = spy.rolling(40).mean()
ratio = spy / ief
ratio_ma = ratio.rolling(13).mean()

trend_on = bool(spy.iloc[-1] > trend_ma.iloc[-1])
ratio_on = bool(ratio.iloc[-1] > ratio_ma.iloc[-1])
vix_now = float(vix.iloc[-1])
vix_pts = 30 if vix_now < 20 else (15 if vix_now < 28 else 0)
trend_pts = 40 if trend_on else 0
ratio_pts = 30 if ratio_on else 0
score = trend_pts + ratio_pts + vix_pts

st.subheader("Why the gate reads this way")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trend (SPY vs 40w)", f"{trend_pts}/40",
          "above" if trend_on else "below")
c2.metric("Risk-appetite (SPY/IEF)", f"{ratio_pts}/30",
          "rising" if ratio_on else "falling")
c3.metric("VIX", f"{vix_pts}/30", f"{vix_now:.1f} "
          + ("calm" if vix_now < 20 else "elevated" if vix_now < 28 else "high"))
c4.metric("Gate score", f"{score}/100",
          "full exposure" if score >= 66 else
          "cash" if score < 33 else "half exposure")
st.caption("Scoring (frozen, validated): trend above its 40-week average = "
           "40 · SPY/IEF ratio above its 13-week average = 30 · VIX <20 = 30, "
           "<28 = 15, else 0. Total >=66 -> full exposure · <33 -> cash · "
           "between -> half. Watch a component near its line — that's where "
           "the gate is closest to flipping.")

# ── Component charts ──────────────────────────────────────────────────────────
look = 104
st.subheader("The inputs over time")
st.caption("SPY vs its 40-week trend line — the primary risk-on/off driver.")
st.line_chart(pd.DataFrame({"SPY": spy, "40-week MA": trend_ma}).iloc[-look:])
st.caption("SPY/IEF risk-appetite ratio vs its 13-week mean — stocks-over-"
           "bonds. Rising = risk appetite firm.")
st.line_chart(pd.DataFrame({"SPY/IEF": ratio, "13-week mean": ratio_ma}
                           ).iloc[-look:])
st.caption("VIX — reference levels 20 (calm/elevated) and 28 (elevated/high).")
st.line_chart(pd.DataFrame({"VIX": vix,
                            "20": pd.Series(20.0, index=cal),
                            "28": pd.Series(28.0, index=cal)}).iloc[-look:])

st.subheader("Exposure path (last 2 years)")
st.line_chart(exposure.iloc[-look:].rename("Gate exposure"))

# ── Diagnostic context — NOT decisional (tested, no gate edge) ────────────────
with st.expander("🔎 Diagnostic climate context — informational only "
                 "(tested: does NOT improve the gate)"):
    st.caption("Regime Lab (regime_lab_v1) tested both of these as gate "
               "confirmations across four folds. Neither helped — VIX already "
               "carries the credit stress signal, and breadth is a lagging, "
               "descriptive read. Shown here for awareness, not for decisions.")
    dc1, dc2 = st.columns(2)
    try:
        from utils.fred import fetch_fred_series, SERIES, fred_available
        if fred_available():
            oas = fetch_fred_series(SERIES["hy_spread"], days=400)
            ow = oas.resample("W-FRI").last().dropna()
            if len(ow) > 26:
                cur = float(ow.iloc[-1]); mn = float(ow.rolling(26).mean().iloc[-1])
                dc1.metric("HY credit spread", f"{cur:.2f}%",
                           ("above 26w mean — stress" if cur > mn
                            else "below 26w mean — calm"),
                           delta_color="inverse")
            else:
                dc1.caption("Credit: insufficient history.")
        else:
            dc1.caption("Credit: set FRED_API_KEY in secrets to display.")
    except Exception:
        dc1.caption("Credit: unavailable.")
    try:
        b = sector_breadth()
        dc2.metric("Sector breadth", f"{b:.0f}%",
                   ("broad" if b >= 60 else "narrow" if b < 40 else "mixed"),
                   help="% of sector ETFs above their 50-day average.")
    except Exception:
        dc2.caption("Breadth: unavailable.")

# ── Legacy MH% — retired as a decision metric ────────────────────────────────
with st.expander("📦 Legacy Market Health % — retired (not a decision input)"):
    st.caption("The old MH% score was a transcribed TradingView indicator: a "
               "hand-set matrix of the same SPY/IEF/VIX/breadth inputs, never "
               "validated. Regime Lab confirmed those inputs add nothing to "
               "the gate, so MH% is retained here only for reference and "
               "drives no decision. The gate verdict above is canonical.")
    try:
        from utils.market_health import calc_market_health
        mh = calc_market_health()
        st.metric("Legacy MH%", f"{mh['mh_pct']}%", "reference only — not used")
    except Exception:
        st.caption("Legacy reading unavailable.")
