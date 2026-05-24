"""
pages/4_Options_Positions.py
Open Interest analysis — Max Pain, Gamma Exposure, OI distribution.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_fetcher import get_options_chain, get_current_price
from utils.indicators import max_pain, gamma_exposure, pcr

st.set_page_config(page_title="Options Positions · StratFlow", page_icon="📊", layout="wide")
st.title("📊 Options Positions")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker       = st.text_input("Ticker", "SPY").upper().strip()
    strike_range = st.slider("Strike Range ± %", 5, 35, 15)
    top_n        = st.number_input("Top N strikes by OI", min_value=5, max_value=50, value=20)

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching options data for {ticker}…"):
    calls_raw, puts_raw, expirations = get_options_chain(ticker)
    spot = get_current_price(ticker)

if calls_raw.empty:
    st.error(f"No options data for **{ticker}**.")
    st.stop()

expiry = st.selectbox(
    "Expiration Date",
    expirations,
    format_func=lambda x: f"{x}  ({(pd.Timestamp(x) - pd.Timestamp.today()).days} DTE)"
)

with st.spinner("Loading selected expiry…"):
    calls, puts, _ = get_options_chain(ticker, expiry=expiry)

lo = spot * (1 - strike_range / 100)
hi = spot * (1 + strike_range / 100)

calls_f = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].copy()
puts_f  = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)].copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
mp    = max_pain(calls, puts)
r     = pcr(calls, puts)
itm_c = int(calls["inTheMoney"].sum()) if "inTheMoney" in calls.columns else 0
itm_p = int(puts["inTheMoney"].sum())  if "inTheMoney" in puts.columns  else 0

st.divider()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spot Price",  f"${spot:.2f}")
k2.metric("Max Pain",    f"${mp:.2f}",
          f"{((mp - spot)/spot*100):+.1f}% from spot")
k3.metric("P/C OI",      f"{r['pcr_oi']:.2f}",
          "Bearish bias" if r["pcr_oi"] > 1 else "Bullish bias")
k4.metric("Total Call OI", f"{r['call_oi']:,}")
k5.metric("Total Put OI",  f"{r['put_oi']:,}")
st.divider()

# ── OI by Strike ──────────────────────────────────────────────────────────────
st.subheader(f"📊 Open Interest by Strike  ·  ±{strike_range}% from Spot")

oi_merged = pd.merge(
    calls_f[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"}),
    puts_f[["strike",  "openInterest"]].rename(columns={"openInterest": "put_oi"}),
    on="strike", how="outer"
).fillna(0).sort_values("strike")

fig_oi = go.Figure()
fig_oi.add_trace(go.Bar(
    x=oi_merged["strike"], y=oi_merged["call_oi"],
    name="Call OI", marker_color="#00ff88", opacity=0.85,
    hovertemplate="Strike $%{x}<br>Call OI: %{y:,}<extra></extra>",
))
fig_oi.add_trace(go.Bar(
    x=oi_merged["strike"], y=-oi_merged["put_oi"],
    name="Put OI", marker_color="#ff4444", opacity=0.85,
    hovertemplate="Strike $%{x}<br>Put OI: %{y:,}<extra></extra>",
))
fig_oi.add_vline(x=spot, line_dash="dash", line_color="#ffd700",
                 annotation_text=f"Spot ${spot:.2f}", annotation_position="top right")
fig_oi.add_vline(x=mp, line_dash="dot", line_color="#00bfff",
                 annotation_text=f"Max Pain ${mp:.2f}", annotation_position="top left")
fig_oi.update_layout(
    template="plotly_dark", barmode="overlay", height=380,
    xaxis_title="Strike", yaxis_title="Open Interest (+ Calls  / − Puts)",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig_oi, width='stretch')

# ── Gamma Exposure ────────────────────────────────────────────────────────────
has_gamma = "gamma" in calls.columns and calls["gamma"].notna().any()
st.subheader("⚡ Gamma Exposure (GEX) by Strike")

if has_gamma:
    gex = gamma_exposure(calls_f, puts_f, spot)
    gex_bar_c = ["#00ff88" if v >= 0 else "#ff4444" for v in gex["net_gex"]]

    fig_gex = go.Figure()
    fig_gex.add_trace(go.Bar(
        x=gex["strike"], y=gex["call_gex"],
        name="Call GEX", marker_color="#00ff88", opacity=0.6,
        hovertemplate="Strike $%{x}<br>Call GEX: %{y:,.0f}<extra></extra>",
    ))
    fig_gex.add_trace(go.Bar(
        x=gex["strike"], y=gex["put_gex"],
        name="Put GEX", marker_color="#ff4444", opacity=0.6,
        hovertemplate="Strike $%{x}<br>Put GEX: %{y:,.0f}<extra></extra>",
    ))
    fig_gex.add_trace(go.Scatter(
        x=gex["strike"], y=gex["net_gex"],
        name="Net GEX", mode="lines+markers",
        line=dict(color="#ffd700", width=2),
        hovertemplate="Strike $%{x}<br>Net GEX: %{y:,.0f}<extra></extra>",
    ))
    fig_gex.add_vline(x=spot, line_dash="dash", line_color="#ffd700")
    fig_gex.add_hline(y=0, line_color="white", opacity=0.25)

    # Mark the zero-crossing flip point (gamma flip)
    net = gex["net_gex"].values
    strikes_ = gex["strike"].values
    for i in range(len(net) - 1):
        if net[i] * net[i+1] < 0:
            flip = (strikes_[i] + strikes_[i+1]) / 2
            fig_gex.add_vline(x=flip, line_dash="dot", line_color="#ff8c00",
                              annotation_text=f"Gamma Flip ~${flip:.0f}",
                              annotation_position="bottom right")
            break

    fig_gex.update_layout(
        template="plotly_dark", barmode="relative", height=380,
        xaxis_title="Strike", yaxis_title="Gamma Exposure ($)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    st.plotly_chart(fig_gex, width='stretch')

    total_gex = gex["net_gex"].sum()
    if total_gex > 0:
        st.success(f"**Net GEX: +${total_gex:,.0f}** — Positive gamma environment. "
                   "Dealers hedge by selling rallies / buying dips → dampens volatility.")
    else:
        st.warning(f"**Net GEX: ${total_gex:,.0f}** — Negative gamma environment. "
                   "Dealers hedge by buying rallies / selling dips → amplifies volatility.")
else:
    st.info("ℹ️  Greeks (gamma) not available for this ticker via Yahoo Finance. "
            "GEX chart requires gamma data from a paid provider (Polygon, Tradier, etc.).")

# ── Max Pain Breakdown ────────────────────────────────────────────────────────
st.divider()
st.subheader(f"🎯 Max Pain Analysis  ·  Max Pain: ${mp:.2f}")

st.caption(
    f"Max Pain is the strike price where options writers (who are short options) "
    f"experience the least financial loss at expiration. "
    f"Spot is **${spot:.2f}**; Max Pain is **${mp:.2f}** "
    f"({((mp - spot)/spot*100):+.1f}% away)."
)

# Pain curve
strikes_all = sorted(set(calls["strike"]) | set(puts["strike"]))
strikes_all = [s for s in strikes_all if lo <= s <= hi]
c_oi_map = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
p_oi_map = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))

pain_vals = []
for s in strikes_all:
    cp = sum(max(0, s - k) * v for k, v in c_oi_map.items() if s > k)
    pp = sum(max(0, k - s) * v for k, v in p_oi_map.items() if s < k)
    pain_vals.append(cp + pp)

fig_pain = go.Figure()
fig_pain.add_trace(go.Scatter(
    x=strikes_all, y=pain_vals,
    fill="tozeroy", name="Total Pain",
    line=dict(color="#7b68ee", width=2),
    fillcolor="rgba(123,104,238,0.12)",
    hovertemplate="Strike $%{x}<br>Pain: $%{y:,.0f}<extra></extra>",
))
fig_pain.add_vline(x=spot, line_dash="dash",  line_color="#ffd700",
                   annotation_text=f"Spot ${spot:.2f}")
fig_pain.add_vline(x=mp,   line_dash="solid", line_color="#00bfff",
                   annotation_text=f"Max Pain ${mp:.2f}",
                   annotation_position="bottom right")
fig_pain.update_layout(
    template="plotly_dark", height=300,
    xaxis_title="Strike", yaxis_title="Aggregate Pain ($)",
    margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig_pain, width='stretch')

# ── Top OI Strikes Table ──────────────────────────────────────────────────────
st.divider()
with st.expander(f"📋 Top {top_n} Strikes by Open Interest"):
    top_oi = oi_merged.copy()
    top_oi["total_oi"]  = top_oi["call_oi"] + top_oi["put_oi"]
    top_oi["pcr_strike"] = (top_oi["put_oi"] / top_oi["call_oi"].replace(0, np.nan)).round(2)
    top_oi = top_oi.nlargest(int(top_n), "total_oi")
    top_oi["strike"]    = top_oi["strike"].apply(lambda x: f"${x:.2f}")
    top_oi["call_oi"]   = top_oi["call_oi"].apply(lambda x: f"{int(x):,}")
    top_oi["put_oi"]    = top_oi["put_oi"].apply(lambda x: f"{int(x):,}")
    top_oi["total_oi"]  = top_oi["total_oi"].apply(lambda x: f"{int(x):,}")
    top_oi.rename(columns={
        "strike": "Strike", "call_oi": "Call OI", "put_oi": "Put OI",
        "total_oi": "Total OI", "pcr_strike": "P/C Ratio"
    }, inplace=True)
    st.dataframe(top_oi[["Strike","Call OI","Put OI","Total OI","P/C Ratio"]],
                 width='stretch', hide_index=True)
