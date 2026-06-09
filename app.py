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


@st.cache_data(ttl=900, show_spinner=False)
def _regime_now():
    """Canonical regime exposure (the validated SPY/IEF/VIX signal) for the cockpit."""
    try:
        import pandas as pd
        from utils.strategy_backtest import compute_regime_exposure
        from utils.data_fetcher import get_stock_data

        def wk(sym):
            d = get_stock_data(sym, period="2y", interval="1d")
            return (d["Close"].resample("W-FRI").last().dropna()
                    if (d is not None and not d.empty) else pd.Series(dtype=float))
        spy, ief, vix = wk("SPY"), wk("IEF"), wk("^VIX")
        if spy.empty or ief.empty or vix.empty:
            return None
        exp = compute_regime_exposure(spy, ief, vix)
        return float(exp.iloc[-1]) if len(exp) else None
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _holdings_count():
    try:
        from utils.portfolio import load_holdings
        return len([h for h in load_holdings() if str(h.get("ticker", "")).strip()])
    except Exception:
        return 0

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

# ── Cockpit status — regime + book at a glance ────────────────────────────────
_exp = _regime_now()
if _exp is None:
    _rtxt, _rcol = "⚪ Regime: unknown (data unavailable)", "#888"
elif _exp >= 0.99:
    _rtxt, _rcol = "🟢 Risk-ON — model favors full exposure", "#00cc66"
elif _exp <= 0.01:
    _rtxt, _rcol = "🔴 Risk-OFF — model in cash", "#ff4444"
else:
    _rtxt, _rcol = f"🟡 Caution — model ~{_exp*100:.0f}% invested", "#ff9800"
_s1, _s2 = st.columns([3, 1])
_s1.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{_rcol}22;"
             f"border:1px solid {_rcol};font-weight:600'>{_rtxt}</div>", unsafe_allow_html=True)
_s2.metric("Holdings tracked", _holdings_count())
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

# ── Modules — grouped by the decision workflow ────────────────────────────────
st.subheader("🗺️ Workflow")
st.caption("Orient → Rotate → Find → Confirm → Execute → Hold. Navigate via the **sidebar**.")

st.markdown("**1 · Orient & Rotate** — regime, macro, breadth, and where leadership is moving")
o1, o2 = st.columns(2)
o1.markdown("""**🏥 Market Health**

Regime gate — SPY/IEF RS, VIX, breadth → Market Health % and Target Risk. *Start here.*""")
o2.markdown("""**🛰️ Rotation Radar**

Sector RRG + Early Rotation + the cross-asset Money Map. Where money is moving.""")

st.markdown("**2 · Find & Confirm** — screen candidates, then confirm one name's flow")
f1, f2, f3 = st.columns(3)
f1.markdown("""**🎯 Screener**

RS Trend + GW2 + Early Rotation. The one place candidates come from.""")
f2.markdown("""**📈 Trend Analysis**

Per-name technical drill-down — price, MAs, RSI/MACD/ADX, RS panel.""")
f3.markdown("""**🌊 Flow**

Single-name money flow, options premium, positioning (GEX/DEX), gamma, intraday.""")

st.markdown("**3 · Execute & Hold** — turn the model into orders, then manage the book")
e1, e2 = st.columns(2)
e1.markdown("""**⚖️ Rebalance**

Today's target book from the validated model → exact BUY/ADD/TRIM/SELL orders.""")
e2.markdown("""**💼 Portfolio**

Track holdings with balanced decisions (hold/add/trim/raise-stop/exit) + score history.""")

st.markdown("**4 · Validate** — the research pipeline (occasional)")
v1, v2, v3 = st.columns(3)
v1.markdown("""**🔬 Backtest**

Signal-edge event study vs a random baseline. No lookahead.""")
v2.markdown("""**⚙️ Strategy**

Regime-gated, risk-sized portfolio backtest vs SPY.""")
v3.markdown("""**🧪 Validation**

Walk-forward out-of-sample. OOS Sharpe + WFE is the truth.""")

st.divider()
st.caption("StratFlow · Built with Streamlit + yfinance · Not financial advice.")
