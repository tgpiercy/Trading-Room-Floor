"""
pages/6_Market_Health.py
MARKET HEALTH — climate dashboard (landing page).

Headline: the VALIDATED regime gate (compute_regime_exposure) — the one
exposure ceiling that drives capital, used by Rebalance and the backtested
strategy. Below it, the three inputs behind that verdict, shown transparently.

Everything that influences a decision is validated and labeled. Credit and
breadth appear as DIAGNOSTIC context only — both were tested as gate inputs
and added no edge (regime_lab_v1: VIX already carries the credit-stress
signal; breadth is lagging and actively hurt F3). The old "Market Health %"
score is a transcribed, unvalidated Pine indicator; it now lives only in the
collapsed Legacy panel and computes nothing unless you open it.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.market_health import current_regime, gate_breakdown, sector_breadth

st.set_page_config(page_title="Market Health", layout="wide")
st.title("🏥 Market Health — climate dashboard")
st.caption("The validated regime gate and the inputs behind it. Everything "
           "that drives capital is here and validated; diagnostic context is "
           "labeled as such.")

# ── 1. Headline — the validated gate verdict ─────────────────────────────────
reg = current_regime()
st.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;background:{reg['color']}22;"
    f"border:1px solid {reg['color']};font-weight:700;font-size:1.15rem'>"
    f"Regime gate: {reg['label']}</div>", unsafe_allow_html=True)
st.caption("This is THE exposure ceiling used by Rebalance and the backtested "
           "strategy. Validated as a component (exit_stage2_v1 F) and confirmed "
           "well-specified — adding credit or breadth did not improve it "
           "(regime_lab_v1).")

gb = gate_breakdown()
if not gb.get("available"):
    st.warning("Gate data unavailable (could not fetch SPY/IEF/VIX). "
               "Retry shortly.")
    st.stop()

# ── 2. Why — the three validated inputs ──────────────────────────────────────
st.subheader("Why — the three gate inputs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trend · SPY vs 40w MA",
          "ON" if gb["trend_on"] else "OFF", f"{gb['trend_pts']} / 40 pts")
c2.metric("Risk appetite · SPY/IEF vs 13w",
          "ON" if gb["ratio_on"] else "OFF", f"{gb['ratio_pts']} / 30 pts")
c3.metric("Fear · VIX", f"{gb['vix']:.1f}", f"{gb['vix_pts']} / 30 pts")
c4.metric("Composite", f"{gb['total']} / 100",
          f"→ {gb['exposure']*100:.0f}% exposure")
st.caption("Score >= 66 -> full exposure · 33-65 -> half · < 33 -> cash. These "
           "three inputs span the climate information the gate needs; the "
           "regime lab confirmed nothing else improves the decision.")

_DARK = dict(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=20, b=0),
             paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
             legend=dict(orientation="h", y=1.02))

t1, t2, t3 = st.tabs(["📈 Trend (SPY vs 40w)", "⚖️ Risk appetite (SPY/IEF)",
                      "😨 Fear (VIX)"])
with t1:
    spy, ma = gb["spy"].dropna(), gb["spy_ma"].dropna()
    f = go.Figure()
    f.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY",
                           line=dict(color="#4fc3f7", width=2)))
    f.add_trace(go.Scatter(x=ma.index, y=ma, name="40-week MA",
                           line=dict(color="#ffd700", width=1.5, dash="dot")))
    f.update_layout(**_DARK)
    st.plotly_chart(f, use_container_width=True)
    st.caption("SPY above its 40-week average -> trend ON (40 pts). The single "
               "most reliable simple trend filter.")
with t2:
    r, rma = gb["ratio"].dropna(), gb["ratio_ma"].dropna()
    f = go.Figure()
    f.add_trace(go.Scatter(x=r.index, y=r, name="SPY/IEF",
                           line=dict(color="#9c59b0", width=2)))
    f.add_trace(go.Scatter(x=rma.index, y=rma, name="13-week MA",
                           line=dict(color="#ffd700", width=1.5, dash="dot")))
    f.update_layout(**_DARK)
    st.plotly_chart(f, use_container_width=True)
    st.caption("Stocks/bonds ratio above its 13-week average -> risk appetite "
               "ON (30 pts). Rising = capital favouring equities over safety.")
with t3:
    v = gb["vix_series"].dropna()
    f = go.Figure()
    f.add_trace(go.Scatter(x=v.index, y=v, name="VIX",
                           line=dict(color="#ff8c00", width=2),
                           fill="tozeroy", fillcolor="rgba(255,140,0,0.08)"))
    for lvl, col in [(20, "#00ff88"), (28, "#ff4444")]:
        f.add_hline(y=lvl, line_dash="dot", line_color=col, opacity=0.6,
                    annotation_text=str(lvl), annotation_position="right")
    f.update_layout(**_DARK)
    st.plotly_chart(f, use_container_width=True)
    st.caption("VIX < 20 -> 30 pts · < 28 -> 15 pts · >= 28 -> 0. Fast, "
               "well-proven stress gauge — already carries the credit signal.")

# ── 3. Diagnostic context (labeled — NOT decision inputs) ────────────────────
st.divider()
st.subheader("🔎 Diagnostic context — not decision inputs")
st.caption("Shown for situational awareness only. Both breadth and credit were "
           "tested as gate inputs and rejected (regime_lab_v1) — they add no "
           "edge over the three validated inputs above. Read them, don't trade "
           "them.")

try:
    br = sector_breadth()
    bcol = "#00cc66" if br > 60 else "#ff8c00" if br <= 40 else "#ffd700"
    st.markdown(
        f"<span style='color:{bcol};font-weight:600'>Breadth: {br:.0f}%</span> "
        "of sector ETFs above their 50-day average — "
        "<i>diagnostic; rejected as a gate input (lagging, deepened F3).</i>",
        unsafe_allow_html=True)
except Exception:
    st.caption("Breadth unavailable.")

try:
    from utils.fred import get_macro_data, fred_available
    if not fred_available():
        st.info("🔑 Add `FRED_API_KEY` in secrets to show credit / curve / "
                "stress context charts.")
    else:
        with st.spinner("Fetching macro context (FRED)…"):
            macro = get_macro_data()
        mt1, mt2, mt3 = st.tabs(["💳 Credit spreads", "📐 Yield curve",
                                 "😰 Financial stress"])
        with mt1:
            hy = macro.get("hy_spread", pd.Series(dtype=float))
            ig = macro.get("ig_spread", pd.Series(dtype=float))
            if not hy.empty:
                f = go.Figure()
                f.add_trace(go.Scatter(x=hy.index, y=hy, name="HY OAS",
                                       line=dict(color="#ff8c00", width=2)))
                if not ig.empty:
                    f.add_trace(go.Scatter(x=ig.index, y=ig, name="IG OAS",
                                           line=dict(color="#4fc3f7", width=1.5)))
                f.update_layout(**{**_DARK, "yaxis_title": "OAS %"})
                st.plotly_chart(f, use_container_width=True)
                st.caption("Credit context. Confirmation-tested against the "
                           "gate and added no edge — VIX already moves first.")
            else:
                st.caption("HY OAS unavailable.")
        with mt2:
            cv = macro.get("curve_10y2y", pd.Series(dtype=float))
            if not cv.empty:
                f = go.Figure()
                f.add_trace(go.Scatter(x=cv.index, y=cv, name="10y-2y",
                                       line=dict(color="#9c59b0", width=2),
                                       fill="tozeroy",
                                       fillcolor="rgba(156,89,176,0.08)"))
                f.add_hline(y=0, line_dash="dash", line_color="#ff4444",
                            annotation_text="inversion")
                f.update_layout(**{**_DARK, "yaxis_title": "%"})
                st.plotly_chart(f, use_container_width=True)
            else:
                st.caption("Yield-curve series unavailable.")
        with mt3:
            ss = macro.get("stress", pd.Series(dtype=float))
            if not ss.empty:
                f = go.Figure()
                f.add_trace(go.Scatter(x=ss.index, y=ss, name="STLFSI",
                                       line=dict(color="#ef5350", width=2)))
                f.add_hline(y=0, line_dash="dash", line_color="#888",
                            annotation_text="normal")
                f.update_layout(**_DARK)
                st.plotly_chart(f, use_container_width=True)
            else:
                st.caption("Stress index unavailable.")
except Exception as e:
    st.caption(f"Macro context unavailable: {e}")

# ── 4. Legacy panel (deprecated, computes only on request) ───────────────────
st.divider()
with st.expander("🗄️ Legacy: Market Health % (deprecated — not a decision input)"):
    st.caption("MH% is a transcribed TradingView Pine indicator (~15 hand-set "
               "constants, never validated). The regime lab showed even clean "
               "credit/breadth inputs add nothing to the validated gate, so a "
               "magic-number recombination of them is not a decision tool. "
               "Retained for reference only.")
    if st.checkbox("Compute legacy MH% (extra data fetch)", value=False):
        from utils.market_health import calc_market_health
        with st.spinner("Computing legacy MH%…"):
            mh = calc_market_health()
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("MH% (legacy)", f"{mh['mh_pct']}%")
        lc2.metric("RS score", f"{mh['rs_score']} / 2", mh["rs_regime"])
        lc3.metric("VIX", f"{mh['vix']:.1f}", mh["vix_regime"])
        st.caption("Not used by Rebalance, the backtest, or any decision. "
                   "The headline gate above is the canonical climate verdict.")
