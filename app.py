"""
app.py — StratFlow main dashboard
Market overview + module navigation.
"""
import streamlit as st
import plotly.graph_objects as go
from utils.data_fetcher import get_market_overview, get_stock_data

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
        label=f"{name}",
        value=f"{d['price']:,.2f}",
        delta=f"{d['change_pct']:+.2f}%",
        delta_color=dcolor,
        help=f"As of {d['date']} close"
    )

st.caption(f"Prices show most recent available close · As of {list(market.values())[0]['date']}")
st.divider()

# ── SPY Sparkline ─────────────────────────────────────────────────────────────
st.subheader("📈 S&P 500 — 6-Month")
with st.spinner("Loading chart…"):
    spy = get_stock_data("^GSPC", period="6mo", interval="1d")

if not spy.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy.index, y=spy["Close"],
        fill="tozeroy",
        line=dict(color="#00ff88", width=2),
        fillcolor="rgba(0,255,136,0.08)",
        name="S&P 500",
        hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Module Cards ──────────────────────────────────────────────────────────────
st.subheader("🗺️ Modules")
st.caption("Use the **sidebar** to navigate between modules.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    **📈 Trend Analysis**
    
    Candlestick + SMA/EMA overlays, Bollinger Bands, RSI, MACD, ADX, Stochastic. Auto signal summary.
    """)
with c2:
    st.markdown("""
    **🌊 Flow Indicators**
    
    Volume, Relative Volume (RVOL), OBV, Money Flow Index, Chaikin Money Flow, Force Index.
    """)
with c3:
    st.markdown("""
    **🎯 Options Flow**
    
    Options chain scanner, Put/Call ratio, unusual activity (Vol/OI spike), IV skew curve.
    """)
with c4:
    st.markdown("""
    **📊 Options Positions**
    
    Open Interest by strike, Max Pain calculator, Gamma Exposure (GEX) estimate, OI breakdown.
    """)

st.divider()
st.caption("StratFlow · Built with Streamlit + yfinance · Not financial advice.")
