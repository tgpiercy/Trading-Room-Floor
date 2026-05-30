"""
pages/8_Intraday_Flow.py
Intraday order-flow inference — volume-at-price profile, auction imbalance,
VWAP deviation. Reveals where institutions worked size during the session.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_fetcher import get_intraday_data, get_current_price
from utils.order_flow import volume_by_price, auction_analysis, vwap_deviation

st.set_page_config(page_title="Intraday Flow · StratFlow", page_icon="⏱️", layout="wide")
st.title("⏱️ Intraday Flow")
st.caption("Volume-at-price · auction imbalance · VWAP deviation — institutional footprints intraday")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", "SPY").upper().strip()
    interval = st.selectbox("Interval", ["5m","15m","30m","60m"], index=0)
    days     = st.slider("Days of History", 1, 10, 5)
    bins     = st.slider("Volume Profile Bins", 12, 40, 24)

with st.spinner(f"Loading {ticker} intraday…"):
    df = get_intraday_data(ticker, interval=interval, days=days)
    spot = get_current_price(ticker)

if df.empty:
    st.error(f"No intraday data for {ticker}. Note: yfinance limits intraday history "
             f"(5m ≈ 60 days max). Try a liquid ticker.")
    st.stop()

# ── Auction analysis ──────────────────────────────────────────────────────────
auc = auction_analysis(df)
vw  = vwap_deviation(df)

st.subheader("🔔 Latest Session Auction Profile")
if auc:
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Opening Vol %",  f"{auc['open_vol_pct']:.0f}%",
              "Heavy open" if auc['open_vol_pct'] > 30 else "Normal")
    a2.metric("Closing Vol %",  f"{auc['close_vol_pct']:.0f}%",
              "Heavy close" if auc['close_vol_pct'] > 25 else "Normal")
    a3.metric("Close Drift",    f"{auc['close_drift']:+.2f}%")
    a4.metric("Session Volume", f"{auc['total_vol']:,}")
    sig = auc["signal"]
    if "accumulation" in sig:
        st.success(f"🟢 {sig}")
    elif "distribution" in sig:
        st.error(f"🔴 {sig}")
    elif sig != "Neutral":
        st.info(f"📊 {sig}")
    st.caption(f"Session date: {auc['session_date']}")

if vw:
    st.caption(f"**VWAP:** ${vw['vwap']:.2f} · Price ${vw['price']:.2f} "
               f"({vw['dev_pct']:+.2f}% from VWAP) · "
               f"Held above VWAP {vw['pct_session_above_vwap']:.0f}% of session "
               f"{'(persistent bid — possible accumulation)' if vw['pct_session_above_vwap'] > 65 else ''}"
               f"{'(persistent offer — possible distribution)' if vw['pct_session_above_vwap'] < 35 else ''}")

st.divider()

# ── Volume-at-Price profile + price chart ─────────────────────────────────────
st.subheader("📊 Volume-at-Price Profile")
st.caption("High-volume price nodes = where size traded. POC (Point of Control) is the "
           "highest-volume price — a magnet/accumulation zone.")

prof = volume_by_price(df, bins=bins)
if prof.empty:
    st.info("Insufficient data for volume profile.")
else:
    poc = prof.attrs.get("poc")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28],
                        shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[f"{ticker} {interval}", "Vol@Price"])

    # Price line
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                             line=dict(color="#00ff88", width=1.5)), row=1, col=1)
    # POC line
    if poc:
        fig.add_hline(y=poc, line_dash="dash", line_color="#ffd700",
                      annotation_text=f"POC ${poc:.2f}", annotation_position="left",
                      row=1, col=1)

    # Horizontal volume profile
    prof_colors = ["#ffd700" if abs(p - (poc or 0)) < (prof["price"].iloc[1]-prof["price"].iloc[0])
                   else "#4fc3f7" for p in prof["price"]]
    fig.add_trace(go.Bar(y=prof["price"], x=prof["volume"], orientation="h",
                         marker_color=prof_colors, opacity=0.7, name="Vol",
                         showlegend=False), row=1, col=2)

    fig.update_layout(template="plotly_dark", height=480,
                      margin=dict(l=0,r=0,t=30,b=0),
                      showlegend=False,
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    st.plotly_chart(fig, width="stretch")

st.divider()

# ── Volume by time-of-day heatmap ─────────────────────────────────────────────
st.subheader("🕐 Volume by Time of Day")
st.caption("Average volume per intraday slot — reveals when size consistently trades "
           "(open drive, midday lull, close ramp).")

d = df.dropna(subset=["Volume"]).copy()
d["time"] = d.index.strftime("%H:%M")
tod = d.groupby("time")["Volume"].mean()
if not tod.empty:
    tod_colors = ["#ff8c00" if v > tod.mean()*1.3 else "#4fc3f7" for v in tod.values]
    fig_tod = go.Figure(go.Bar(x=list(tod.index), y=list(tod.values),
                               marker_color=tod_colors))
    fig_tod.update_layout(template="plotly_dark", height=240,
                          margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Time (exchange tz)", yaxis_title="Avg Volume",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig_tod, width="stretch")

st.divider()
st.caption("⚠️ Order-flow inference from free EOD/intraday data — not true tape or order-book. "
           "For real sweep/dark-pool data, a paid feed (Polygon, Unusual Whales) is required.")
