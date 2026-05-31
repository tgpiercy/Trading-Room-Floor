"""
app.py — StratFlow main dashboard
Market overview + module navigation.
"""
import streamlit as st
import plotly.graph_objects as go
from utils.data_fetcher import get_market_overview, get_stock_data
from utils.chart_utils import set_chart_window
from utils.chart_helpers import apply_default_range

st.set_page_config(
    page_title="StratFlow",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<h1 style='margin-bottom:0'>📊 StratFlow</h1>
<p style='color:#888;margin-top:4px;font-size:0.95rem'>
    Trading & Options Strategy Suite · Free data via Yahoo Finance · Most recent close shown when markets are closed
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Market Overview ───────────────────────────────────────────────────────────
st.subheader("🌐 Market Overview")
with st.spinner("Fetching market data…"):
    market = get_market_overview()

if not market:
    st.error("❌ Could not fetch any market data. Check your internet connection and refresh.")
    st.stop()

INVERT_DLT = {"VIX"}
cols = st.columns(len(market))
for col, (name, d) in zip(cols, market.items()):
    dcolor = "inverse" if name in INVERT_DLT else "normal"
    col.metric(
        label=name,
        value=f"{d['price']:,.2f}",
        delta=f"{d['change_pct']:+.2f}%",
        delta_color=dcolor,
        help=f"As of {d['date']} close"
    )

st.caption(f"Prices show most recent available close · As of {list(market.values())[0]['date']}")
st.divider()

# ── SPY Chart — 2y data, 6-month default window ───────────────────────────────
st.subheader("📈 S&P 500")
with st.spinner("Loading chart…"):
    spy = get_stock_data("^GSPC", period="2y", interval="1d")

if not spy.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy.index, y=spy["Close"],
        fill="tozeroy",
        line=dict(color="#00ff88", width=2),
        fillcolor="rgba(0,255,136,0.08)",
        name="S&P 500",
        hovertemplate="%{x|%b %d '%y}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    apply_default_range(fig, months_back=6)
    st.plotly_chart(fig, width='stretch')

st.divider()

# ── Module Cards ──────────────────────────────────────────────────────────────
st.subheader("🗺️ Modules")
st.caption("Use the **sidebar** to navigate between modules.")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""**🎯 Screener**

Unified RS Trend + GW2 engine. Top Opportunities, 7-point scorecard, Impulse states, Early Rotation, entry/exit signals across your portfolio.""")
with c2:
    st.markdown("""**🌊 Flow**

Money flow (OBV/MFI/CMF/RVOL), options flow (premium-weighted, sweeps, new positioning), positioning (Max Pain, GEX), intraday (vol-at-price, auction, VWAP). One ticker, all views.""")
with c3:
    st.markdown("""**🛰️ Rotation Radar**

RRG sector map + Early Rotation score. Catches money & momentum flow before the trend confirms.""")

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown("""**🏥 Market Health**

Regime gate — SPY/IEF RS, VIX, breadth → Market Health % and Target Risk. Check this first.""")
with c5:
    st.markdown("""**📈 Trend Analysis**

Candlestick + SMA/EMA, Bollinger Bands, RSI, MACD, ADX, Stochastic, RS panel. Per-ticker drill-down.""")
with c6:
    st.markdown("""**🧭 Flow Dashboard**

Multi-day flow interpretation — scored confluence table + plain verdict. Tells you what the pattern is, how strong, and what to watch.""")

c7, c8, c9 = st.columns(3)
with c7:
    st.markdown("""**🔬 Backtest**

Signal edge event study — proves which signals actually predict forward returns, vs a random-pick baseline. No lookahead.""")
with c8:
    st.markdown("""**⚙️ Strategy**

Phase 2 — RS-leadership momentum portfolio, regime-gated and risk-sized, backtested vs SPY. The signal turned into a system.""")
with c9:
    st.empty()

st.divider()
st.caption("StratFlow · Built with Streamlit + yfinance · Not financial advice.")
