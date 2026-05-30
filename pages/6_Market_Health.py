"""
pages/6_Market_Health.py
Market Health v2.5 dashboard — regime gate for all strategy decisions.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.market_health import calc_market_health
from utils.chart_utils import set_chart_window

st.set_page_config(page_title="Market Health · StratFlow",
                   page_icon="🏥", layout="wide")
st.title("🏥 Market Health")
st.caption("Market Health v2.5 · SPY/IEF RS + RSP rescue + VIX + S5FI breadth")

with st.sidebar:
    st.header("⚙️ Settings")
    target_risk_full = st.slider("Full Portfolio Risk % (when 100% ON)",
                                  2.0, 8.0, 4.5, step=0.25)
    st.divider()
    st.caption("Market Health % controls maximum allowable portfolio risk exposure. "
               "Target Risk = Full Risk × (MH%)^k where k is breadth-dependent.")

with st.spinner("Computing Market Health…"):
    mh = calc_market_health(target_risk_full)

if mh["mh_pct"] == 0 and mh["rs_score"] == 0 and mh["vix"] == 25:
    st.error("Could not fetch market data. Check internet connection.")
    st.stop()

# ── Colour helpers ────────────────────────────────────────────────────────────
MH_COLOR  = ("#00cc66" if mh["mh_pct"] >= 75 else
             "#ffd700" if mh["mh_pct"] >= 40 else "#ff4444")
VIX_COLOR = ("#4fc3f7" if mh["vix"] < 17 else
             "#00ff88" if mh["vix"] <= 20 else
             "#ff8c00" if mh["vix"] < 25 else "#ff4444")

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.subheader("📊 Current Reading")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Market Health",  f"{mh['mh_pct']}%",
          "Favorable" if mh["mh_pct"] >= 75 else
          "Moderate"  if mh["mh_pct"] >= 40 else "Defensive")
c2.metric("Target Risk",    f"{mh['target_risk']:.2f}%")
c3.metric("RS Score",       f"{mh['rs_score']} / 2",
          mh["rs_regime"])
c4.metric("VIX",            f"{mh['vix']:.1f}",
          mh["vix_regime"])
c5.metric("Breadth (S5FI)", f"{mh['s5fi']:.1f}%",
          f"k = {mh['k']}")
c6.metric("MH Raw",         f"{mh['mh_raw']}%",
          f"Base {mh['mh_base']}% "
          f"RSP {mh['rsp_adj']:+d}% "
          f"RS8 {mh['rs8_adj']:+d}%")

st.divider()

# ── Breakdown table ───────────────────────────────────────────────────────────
st.subheader("🔍 Component Breakdown")
rows = [
    ("b7 — SPY RS > RS18 OR RSP > RSP18",    "✅" if mh["b7"] else "❌",  mh["b7"]),
    ("b8 — RS18 Rising OR RSP18 Rising",       "✅" if mh["b8"] else "❌",  mh["b8"]),
    ("RSP above SMA18 (+10% adj)",             "✅" if mh["rsp_above_18"] else "❌", mh["rsp_above_18"]),
    ("RS SMA8 Rising (+5% adj)",               "✅" if mh["rs8_rising"] else "❌",   mh["rs8_rising"]),
]
breakdown = pd.DataFrame(rows, columns=["Condition","Status","Met"])
st.dataframe(breakdown[["Condition","Status"]], width="stretch", hide_index=True)

# ── MH Matrix ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Market Health Matrix (Base %)")
matrix_data = {
    "RS Score \\ VIX": ["RS=0","RS=1","RS=2"],
    "VIX ≥25 (0)":     [0,  0,  0],
    "VIX <25 (1)":     [0, 30, 56],
    "VIX ≤20 (2)":     [0, 55, 75],
    "VIX <17 (3)":     [0, 70,100],
}
mat_df = pd.DataFrame(matrix_data).set_index("RS Score \\ VIX")
st.dataframe(mat_df, width="stretch")
st.caption(f"Current: RS={mh['rs_score']}, VIX score={mh['vix_score']} "
           f"→ Base={mh['mh_base']}% → Raw={mh['mh_raw']}% → Final={mh['mh_pct']}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 SPY/IEF RS", "😨 VIX", "🫁 Breadth (S5FI)"])

with tab1:
    rs  = mh["rs_series"].dropna()
    s8  = mh["rs_sma8"].dropna()
    s18 = mh["rs_sma18"].dropna()
    if not rs.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs.index,  y=rs,  name="SPY/IEF RS",
                                 line=dict(color="#4fc3f7", width=2)))
        fig.add_trace(go.Scatter(x=s8.index,  y=s8,  name="RS SMA8",
                                 line=dict(color="#ffd700", width=1.5)))
        fig.add_trace(go.Scatter(x=s18.index, y=s18, name="RS SMA18",
                                 line=dict(color="#9c59b0", width=1.5)))
        # Band around SMA18
        fig.add_trace(go.Scatter(x=s18.index, y=s18*1.01, name="Band Hi",
                                 line=dict(color="rgba(150,90,176,0.3)", width=1, dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=s18.index, y=s18*0.99, name="Band Lo",
                                 line=dict(color="rgba(150,90,176,0.3)", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(150,90,176,0.05)",
                                 showlegend=False))
        fig.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=0,r=0,t=20,b=0),
                          legend=dict(orientation="h", y=1.02),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        set_chart_window(fig)
        st.plotly_chart(fig, width="stretch")

with tab2:
    vix = mh["vix_series"].dropna()
    if not vix.empty:
        vix_c = ["#4fc3f7" if v < 17 else
                 "#00ff88" if v <= 20 else
                 "#ff8c00" if v < 25 else "#ff4444" for v in vix]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vix.index, y=vix, name="VIX",
                                 line=dict(color="#ff8c00", width=2),
                                 fill="tozeroy", fillcolor="rgba(255,140,0,0.08)"))
        for lvl, c, lbl in [(17,"#4fc3f7","17"),(20,"#00ff88","20"),(25,"#ff4444","25")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=c, opacity=0.5,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(template="plotly_dark", height=280,
                          margin=dict(l=0,r=0,t=20,b=0),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        set_chart_window(fig)
        st.plotly_chart(fig, width="stretch")

with tab3:
    s5 = mh["s5fi_series"].dropna()
    if not s5.empty:
        # Normalize if in count form
        s5 = s5 / 5.0 if s5.max() > 100 else s5
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s5.index, y=s5, name="S5FI %",
                                 line=dict(color="#00ff88", width=2),
                                 fill="tozeroy", fillcolor="rgba(0,255,136,0.08)"))
        for lvl, c, lbl in [(60,"#00cc66","60% (Strong k=1.25)"),
                             (40,"#ff8c00","40% (Weak k=1.75)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=c, opacity=0.5,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(template="plotly_dark", height=280,
                          yaxis=dict(range=[0,100], title="% S&P 500 above 50d MA"),
                          margin=dict(l=0,r=0,t=20,b=0),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        set_chart_window(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("S5FI breadth data not available from Yahoo Finance (^SPXA50R). "
                "Using default k=1.50 (neutral breadth).")

# ── Regime interpretation ─────────────────────────────────────────────────────
st.divider()
if   mh["mh_pct"] >= 75:
    st.success(f"✅ **Market Health {mh['mh_pct']}%** — Favorable. "
               f"Target Risk: **{mh['target_risk']:.2f}%**. Full offensive posture permitted.")
elif mh["mh_pct"] >= 56:
    st.info(f"📊 **Market Health {mh['mh_pct']}%** — Moderate. "
            f"Target Risk: **{mh['target_risk']:.2f}%**. Selective exposure appropriate.")
elif mh["mh_pct"] > 0:
    st.warning(f"⚠️ **Market Health {mh['mh_pct']}%** — Cautious. "
               f"Target Risk: **{mh['target_risk']:.2f}%**. Pilot entries only.")
else:
    st.error("🛑 **Market Health 0%** — Defensive. Avoid new exposure. Protect capital.")
