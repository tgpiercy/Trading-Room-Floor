"""
pages/3_Options_Flow.py
Options chain scanner — unusual activity, put/call ratio, IV skew.
Data via yfinance (delayed ~15 min).
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.data_fetcher import get_options_chain, get_current_price, get_ticker_info
from utils.indicators import pcr
from utils.order_flow import premium_flow, new_positioning, detect_sweeps

st.set_page_config(page_title="Options Flow · StratFlow", page_icon="🎯", layout="wide")
st.title("🎯 Options Flow")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()

    st.subheader("Unusual Activity Filter")
    vol_oi_thresh  = st.slider("Min Vol/OI Ratio",  1.0, 10.0, 2.0, step=0.5)
    min_volume     = st.number_input("Min Volume",  min_value=0, value=100, step=50)
    min_oi         = st.number_input("Min Open Interest", min_value=0, value=50, step=50)
    contract_type  = st.selectbox("Contract Type", ["Both", "Calls Only", "Puts Only"])

    st.subheader("Chain Display")
    show_itm       = st.checkbox("Show ITM contracts", value=True)
    show_otm       = st.checkbox("Show OTM contracts", value=True)
    strike_range   = st.slider("Strike Range ± %", 5, 30, 15)

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching options chain for {ticker}…"):
    calls, puts, expirations = get_options_chain(ticker)
    spot = get_current_price(ticker)
    info = get_ticker_info(ticker)

if calls.empty or puts.empty:
    st.error(f"No options data for **{ticker}**. Verify the ticker and try again.")
    st.stop()

# ── Expiration Selector ───────────────────────────────────────────────────────
col_exp, col_price = st.columns([3, 1])
with col_exp:
    expiry = st.selectbox(
        "Expiration Date",
        expirations,
        format_func=lambda x: f"{x}  ({(pd.Timestamp(x) - pd.Timestamp.today()).days} DTE)"
    )
with col_price:
    st.metric("Current Price", f"${spot:.2f}")

# Re-fetch for selected expiry
with st.spinner("Loading selected expiry…"):
    calls, puts, _ = get_options_chain(ticker, expiry=expiry)

if calls.empty or puts.empty:
    st.error(f"No options data for **{ticker}** on expiry {expiry}. Try a different expiration.")
    st.stop()

# ── Put/Call Ratio ────────────────────────────────────────────────────────────
ratios = pcr(calls, puts)
st.divider()
st.subheader("📊 Put / Call Ratios")
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("P/C Vol",      f"{ratios['pcr_volume']:.2f}",
          "Bearish" if ratios["pcr_volume"] > 1.0 else "Bullish")
r2.metric("P/C OI",       f"{ratios['pcr_oi']:.2f}",
          "Bearish" if ratios["pcr_oi"] > 1.0 else "Bullish")
r3.metric("Call Volume",  f"{ratios['call_vol']:,}")
r4.metric("Put Volume",   f"{ratios['put_vol']:,}")
r5.metric("Call OI",      f"{ratios['call_oi']:,}")
r6.metric("Put OI",       f"{ratios['put_oi']:,}")
st.divider()

# ── Premium-Weighted Flow (dollar flow, not contract count) ───────────────────
pf = premium_flow(calls, puts)
st.subheader("💰 Premium-Weighted Flow")
st.caption("Where the **dollars** went (volume × price × 100) — not just contract count.")
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("Call Premium", f"${pf['call_premium']/1e6:.2f}M", f"{pf['call_pct']:.0f}% of flow")
pc2.metric("Put Premium",  f"${pf['put_premium']/1e6:.2f}M",  f"{pf['put_pct']:.0f}% of flow")
pc3.metric("Premium P/C",  f"{pf['prem_pcr']:.2f}",
           "Bearish $" if pf['prem_pcr'] > 1 else "Bullish $")
pc4.metric("Total Premium",f"${pf['total_premium']/1e6:.2f}M")

# Dollar flow bar
import plotly.graph_objects as _go
fig_prem = _go.Figure(_go.Bar(
    x=["Call $", "Put $"], y=[pf['call_premium'], pf['put_premium']],
    marker_color=["#00ff88", "#ff4444"],
    text=[f"${pf['call_premium']/1e6:.1f}M", f"${pf['put_premium']/1e6:.1f}M"],
    textposition="outside",
))
fig_prem.update_layout(template="plotly_dark", height=200, showlegend=False,
                       margin=dict(l=0,r=0,t=10,b=0),
                       paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig_prem, width="stretch")

# ── Sweep / cluster detection ─────────────────────────────────────────────────
sw = detect_sweeps(calls, puts, spot)
st.caption(f"**Sweep cluster:** {sw['bias']} · "
           f"{sw['call_strikes_hot']} hot call strikes (${sw['call_sweep_prem']/1e6:.1f}M) · "
           f"{sw['put_strikes_hot']} hot put strikes (${sw['put_sweep_prem']/1e6:.1f}M)")

# ── New positioning (Vol > OI) ────────────────────────────────────────────────
st.subheader("🆕 New Positioning  ·  Volume > Open Interest")
st.caption("Contracts opened today (volume exceeds existing OI), ranked by dollar premium.")
np_df = new_positioning(calls, puts, min_premium=25_000)
if np_df.empty:
    st.info("No significant new positioning (≥$25K premium) on this expiry.")
else:
    disp = np_df.copy()
    disp["premium"] = (disp["premium"]/1e3).round(0)
    if "impliedVolatility" in disp.columns:
        disp["impliedVolatility"] = (disp["impliedVolatility"]*100).round(1)
    disp.rename(columns={"type":"Type","strike":"Strike","lastPrice":"Last",
                         "volume":"Vol","openInterest":"OI","premium":"Premium $K",
                         "impliedVolatility":"IV%","inTheMoney":"ITM"}, inplace=True)
    def _np_style(r):
        c = "rgba(0,255,136,0.10)" if r.get("Type")=="CALL" else "rgba(255,68,68,0.10)"
        return [f"background-color:{c}"]*len(r)
    st.dataframe(disp.style.apply(_np_style, axis=1),
                 width="stretch", hide_index=True,
                 column_config={"Premium $K": st.column_config.NumberColumn(format="$%.0fK"),
                                "Last": st.column_config.NumberColumn(format="$%.2f")})
st.divider()

# ── IV Skew Chart ─────────────────────────────────────────────────────────────
lo   = spot * (1 - strike_range / 100)
hi   = spot * (1 + strike_range / 100)
c_iv = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
p_iv = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)]

st.subheader(f"📐 IV Skew  ·  ±{strike_range}% from spot")
fig_iv = go.Figure()
if not c_iv.empty:
    fig_iv.add_trace(go.Scatter(
        x=c_iv["strike"], y=c_iv["impliedVolatility"] * 100,
        mode="lines+markers", name="Calls IV",
        line=dict(color="#00ff88", width=2),
        marker=dict(size=6),
        hovertemplate="Strike $%{x}<br>IV %{y:.1f}%<extra>Calls</extra>",
    ))
if not p_iv.empty:
    fig_iv.add_trace(go.Scatter(
        x=p_iv["strike"], y=p_iv["impliedVolatility"] * 100,
        mode="lines+markers", name="Puts IV",
        line=dict(color="#ff4444", width=2),
        marker=dict(size=6),
        hovertemplate="Strike $%{x}<br>IV %{y:.1f}%<extra>Puts</extra>",
    ))
fig_iv.add_vline(x=spot, line_dash="dash", line_color="#ffd700",
                 annotation_text=f"Spot ${spot:.2f}", annotation_position="top right")
fig_iv.update_layout(
    template="plotly_dark", height=340,
    xaxis_title="Strike", yaxis_title="Implied Volatility (%)",
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", y=1.02),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig_iv, width='stretch')

# ── Unusual Activity Scanner ──────────────────────────────────────────────────
st.subheader(f"🔥 Unusual Activity Scanner  ·  Vol/OI ≥ {vol_oi_thresh}×")
st.caption("Contracts where volume significantly exceeds open interest — potential directional bets.")

def filter_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["volume"]       >= min_volume]
    out = out[out["openInterest"] >= min_oi]
    out = out[out["vol_oi_ratio"] >= vol_oi_thresh]
    if not show_itm:
        out = out[~out.get("inTheMoney", pd.Series(False, index=out.index))]
    if not show_otm:
        out = out[out.get("inTheMoney", pd.Series(True, index=out.index))]
    out = out[(out["strike"] >= lo) & (out["strike"] <= hi)]
    return out.sort_values("vol_oi_ratio", ascending=False)

unusual_calls = filter_chain(calls) if contract_type != "Puts Only"  else pd.DataFrame()
unusual_puts  = filter_chain(puts)  if contract_type != "Calls Only" else pd.DataFrame()
unusual_all   = pd.concat([unusual_calls, unusual_puts]).sort_values("vol_oi_ratio", ascending=False)

DISPLAY_COLS = [c for c in
    ["type", "strike", "lastPrice", "bid", "ask", "impliedVolatility",
     "volume", "openInterest", "vol_oi_ratio", "inTheMoney"]
    if c in unusual_all.columns]

if unusual_all.empty:
    st.info("No unusual activity matching the current filters. Try lowering the Vol/OI threshold.")
else:
    display = unusual_all[DISPLAY_COLS].copy()
    if "impliedVolatility" in display.columns:
        display["impliedVolatility"] = (display["impliedVolatility"] * 100).round(1).astype(str) + "%"
    display.rename(columns={
        "type": "Type", "strike": "Strike", "lastPrice": "Last",
        "bid": "Bid", "ask": "Ask", "impliedVolatility": "IV",
        "volume": "Volume", "openInterest": "OI", "vol_oi_ratio": "Vol/OI",
        "inTheMoney": "ITM"
    }, inplace=True)

    # Colour code by type
    def style_row(row):
        color = "background-color: rgba(0,255,136,0.08)" if row.get("Type") == "call" \
                else "background-color: rgba(255,68,68,0.08)"
        return [color] * len(row)

    st.dataframe(
        display.style.apply(style_row, axis=1),
        width='stretch', hide_index=True
    )
    st.caption(f"Showing {len(unusual_all)} unusual contracts  ·  "
               f"{len(unusual_calls)} calls  ·  {len(unusual_puts)} puts")

# ── Volume/OI Bar Chart ───────────────────────────────────────────────────────
st.divider()
tab1, tab2 = st.tabs(["📊 Call Flow by Strike", "📊 Put Flow by Strike"])

def flow_bar(df: pd.DataFrame, color: str, label: str) -> go.Figure:
    d = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    d = d.sort_values("strike")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["strike"], y=d["volume"],       name="Volume",
                         marker_color=color, opacity=0.85))
    fig.add_trace(go.Bar(x=d["strike"], y=d["openInterest"], name="Open Interest",
                         marker_color="rgba(255,255,255,0.25)", opacity=0.7))
    fig.add_vline(x=spot, line_dash="dash", line_color="#ffd700")
    fig.update_layout(
        template="plotly_dark", barmode="overlay", height=300,
        xaxis_title="Strike", yaxis_title="Contracts",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.05),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig

with tab1:
    st.plotly_chart(flow_bar(calls, "#00ff88", "Calls"), width='stretch')
with tab2:
    st.plotly_chart(flow_bar(puts,  "#ff4444", "Puts"),  width='stretch')
