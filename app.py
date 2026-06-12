"""
app.py — StratFlow entrypoint.
Grouped sidebar navigation (st.navigation) + cockpit home (regime + market overview).
The sidebar is organised by the decision workflow: Orient → Find → Execute → Research.
"""
import streamlit as st
import plotly.graph_objects as go
from utils.data_fetcher import get_market_overview, get_stock_data
from utils.chart_helpers import apply_default_range

st.set_page_config(page_title="StratFlow", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(ttl=120, show_spinner=False)
def _holdings_count():
    try:
        from utils.portfolio import load_holdings
        return len([h for h in load_holdings() if str(h.get("ticker", "")).strip()])
    except Exception:
        return 0


def home():
    st.markdown("""
    <h1 style='margin-bottom:0'>📊 StratFlow</h1>
    <p style='color:#888;margin-top:4px;font-size:0.95rem'>
        Trading & Options Strategy Suite · Free data · Most recent close shown when markets are closed
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    st.subheader("🌐 Market Overview")
    with st.spinner("Fetching market data…"):
        market = get_market_overview()
    if not market:
        st.error("❌ Could not fetch market data. Refresh to retry.")
        return
    INVERT = {"VIX"}
    cols = st.columns(len(market))
    for col, (name, d) in zip(cols, market.items()):
        col.metric(name, f"{d['price']:,.2f}", f"{d['change_pct']:+.2f}%",
                   delta_color="inverse" if name in INVERT else "normal",
                   help=f"As of {d['date']} close")
    st.caption(f"Most recent close · As of {list(market.values())[0]['date']}")
    st.divider()

    # Cockpit status — canonical regime + book at a glance
    try:
        from utils.market_health import current_regime
        reg = current_regime()
    except Exception:
        reg = {"label": "⚪ Regime unknown", "color": "#888"}
    s1, s2 = st.columns([3, 1])
    s1.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{reg['color']}22;"
                f"border:1px solid {reg['color']};font-weight:600'>{reg['label']}</div>",
                unsafe_allow_html=True)
    s2.metric("Holdings tracked", _holdings_count())
    st.divider()

    st.subheader("📈 S&P 500")
    with st.spinner("Loading chart…"):
        spy = get_stock_data("^GSPC", period="2y", interval="1d")
    if not spy.empty:
        fig = go.Figure(go.Scatter(
            x=spy.index, y=spy["Close"], fill="tozeroy",
            line=dict(color="#00ff88", width=2), fillcolor="rgba(0,255,136,0.08)",
            hovertemplate="%{x|%b %d '%y}<br>$%{y:,.2f}<extra></extra>"))
        fig.update_layout(template="plotly_dark", height=300,
                          margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#1e2130"), showlegend=False,
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        apply_default_range(fig, months_back=6)
        st.plotly_chart(fig, width="stretch")

    st.caption("Navigate via the grouped sidebar → **Orient · Find · Execute · Research**. "
               "Not financial advice.")


# ── Grouped sidebar navigation (workflow order) ───────────────────────────────
nav = st.navigation({
    "": [st.Page(home, title="Cockpit", icon="📊", default=True)],
    "Orient & Rotate": [
        st.Page("pages/6_Market_Health.py", title="Market Health", icon="🏥"),
        st.Page("pages/7_Rotation_Radar.py", title="Rotation Radar", icon="🛰️"),
    ],
    "Find & Confirm": [
        st.Page("pages/5_Screener.py", title="Screener", icon="🎯"),
        st.Page("pages/1_Trend_Analysis.py", title="Trend Analysis", icon="📈"),
        st.Page("pages/2_Flow.py", title="Flow", icon="🌊"),
    ],
    "Execute & Hold": [
        st.Page("pages/4_Rebalance.py", title="Rebalance", icon="⚖️"),
        st.Page("pages/3_Portfolio.py", title="Portfolio", icon="💼"),
    ],
    "Research": [
        st.Page("pages/9_Backtest.py", title="Backtest", icon="🔬"),
        st.Page("pages/10_Strategy.py", title="Strategy", icon="⚙️"),
        st.Page("pages/11_Validation.py", title="Validation", icon="🧪"),
        st.Page("validation_exit_sweep.py", title="Exit Sweep", icon="🚪"),
        st.Page("validation_selection_lab.py", title="Selection Lab", icon="🧬"),
    ],
})
nav.run()
