"""
pages/5_RS_Screener.py
RS Trend v1.8 — Relative Strength screener and drill-down.
Weekly RS scoring vs benchmark for a user-defined watchlist.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.chart_utils import set_chart_window
from utils.rs_indicators import (
    run_screener, build_rs_df, classify_state,
    STATE_CONFIG, EXT_BANDS,
)
from utils.data_fetcher import get_stock_data, get_current_price

st.set_page_config(
    page_title="RS Screener · StratFlow",
    page_icon="🏆", layout="wide"
)
st.title("🏆 RS Trend Screener")
st.caption("RS Trend v1.8 · Weekly relative strength vs benchmark · Scores −3 to +2")

# ── Sidebar ───────────────────────────────────────────────────────────────────
DEFAULT_LIST = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,AVGO,CRM,"
    "JPM,GS,BAC,V,MA,"
    "XLK,XLF,XLE,XLV,XLI,XLC,XLY,XLP,XLB,XLRE,"
    "QQQ,IWM,DIA,GLD,TLT"
)

with st.sidebar:
    st.header("⚙️ Screener Settings")

    raw_list  = st.text_area(
        "Watchlist (comma-separated)",
        value=DEFAULT_LIST, height=160,
        help="Add any valid Yahoo Finance ticker. Benchmark is excluded automatically."
    )
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)
    interval  = st.selectbox("Timeframe", ["1wk (Weekly)", "1d (Daily)"], index=0)
    period    = "2y"
    yf_interval = "1wk" if "wk" in interval else "1d"

    st.divider()
    score_filter = st.multiselect(
        "Filter by RS State",
        options=list(STATE_CONFIG.keys()),
        default=list(STATE_CONFIG.keys()),
    )
    run_btn = st.button("▶ Run Scan", type="primary", width='stretch')

# ── Parse Watchlist ───────────────────────────────────────────────────────────
watchlist = tuple(
    t.strip().upper() for t in raw_list.replace("\n", ",").split(",")
    if t.strip() and t.strip().upper() != benchmark
)

# ── State Legend ──────────────────────────────────────────────────────────────
with st.expander("📘 RS State Reference", expanded=False):
    leg_cols = st.columns(3)
    for i, (state, cfg) in enumerate(STATE_CONFIG.items()):
        with leg_cols[i % 3]:
            st.markdown(
                f"**{cfg['emoji']} {state}** · Score `{cfg['score']:+d}`  \n"
                f"*{cfg['description']}*  \n"
                f"➡ {cfg['action']}"
            )

st.divider()

# ── Run Screener ──────────────────────────────────────────────────────────────
if "rs_results" not in st.session_state or run_btn:
    with st.spinner(f"Scanning {len(watchlist)} tickers vs {benchmark}…"):
        st.session_state["rs_results"] = run_screener(
            watchlist, benchmark=benchmark,
            period=period, interval=yf_interval,
        )

results: pd.DataFrame = st.session_state.get("rs_results", pd.DataFrame())

if results.empty:
    st.warning("No results. Check tickers and try again.")
    st.stop()

# Apply state filter
filtered = results[results["RS State"].isin(score_filter)].copy()

# ── Summary Bar ───────────────────────────────────────────────────────────────
st.subheader("📊 State Distribution")
state_counts = results["RS State"].value_counts().reindex(STATE_CONFIG.keys(), fill_value=0)
bar_colors   = [STATE_CONFIG[s]["color"] for s in state_counts.index]

fig_dist = go.Figure(go.Bar(
    x=list(state_counts.index),
    y=list(state_counts.values),
    marker_color=bar_colors,
    text=list(state_counts.values),
    textposition="outside",
    hovertemplate="%{x}: %{y} tickers<extra></extra>",
))
fig_dist.update_layout(
    template="plotly_dark", height=200,
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False, yaxis=dict(showticklabels=False),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig_dist, width='stretch')
st.divider()

# ── Results Table ─────────────────────────────────────────────────────────────
st.subheader(f"📋 Results  ·  {len(filtered)} / {len(results)} tickers")

display = filtered[[
    "_emoji", "Ticker", "Price", "RS Score", "RS State", "Ext %", "RS Mom 4W", "Mansfield %", "Action"
]].copy().rename(columns={"_emoji": ""})

# Colour rows by state
def row_style(row):
    color = STATE_CONFIG.get(row["RS State"], {}).get("color", "#888")
    alpha = "18"
    bg    = f"background-color: {color}{alpha}"
    return [bg] * len(row)

st.dataframe(
    display.style.apply(row_style, axis=1),
    width='stretch', hide_index=True,
    column_config={
        "RS Score":  st.column_config.NumberColumn(format="%+d"),
        "Ext %":     st.column_config.NumberColumn(format="%.1f%%"),
        "RS Mom 4W":   st.column_config.NumberColumn("RS Mom 4W%",  format="%.1f%%"),
        "Mansfield %": st.column_config.NumberColumn("Mansfield %", format="%.1f%%",
                        help="(RS / SMA52 of RS) − 1 · Display only · Not used in scoring"),
        "Price":       st.column_config.NumberColumn(format="$%.2f"),
    }
)

# ── Drill-Down ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Drill-Down — RS Chart")

available = filtered["Ticker"].tolist()
if not available:
    st.info("No tickers match the current filter.")
    st.stop()

selected = st.selectbox("Select ticker to inspect", available)
row = filtered[filtered["Ticker"] == selected].iloc[0]
df_rs: pd.DataFrame = row["_df_rs"]

state   = row["RS State"]
score   = row["RS Score"]
ext_pct = row["Ext %"]
cfg     = STATE_CONFIG.get(state, {})

# State badge
c1, c2, c3, c4 = st.columns(4)
c1.metric("RS State",  f"{cfg.get('emoji','')} {state}")
c2.metric("RS Score",  f"{score:+d}")
c3.metric("Extension", f"{ext_pct:+.1f}%",
          "Extreme" if abs(ext_pct) >= 15 else ("High" if abs(ext_pct) >= 10 else "Normal"))
c4.metric("Action",    cfg.get("action","").split(";")[0])

# ── RS Chart ──────────────────────────────────────────────────────────────────
fig_rs = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.06, row_heights=[0.65, 0.35],
    subplot_titles=[f"{selected} / {benchmark}  RS Ratio", "Price"],
)

# — RS Ratio & SMAs —
fig_rs.add_trace(go.Scatter(
    x=df_rs.index, y=df_rs["RS"], name="RS",
    line=dict(color="#ffffff", width=2),
    hovertemplate="%{x|%b %d}<br>RS: %{y:.4f}<extra></extra>",
), row=1, col=1)

sma_styles = [
    ("SMA8",  "#ff8c00", "SMA8  (orange)"),
    ("SMA18", "#9c59b0", "SMA18 (purple)"),
    ("SMA40", "#c39bd3", "SMA40 (lt purple)"),
]
for col_name, color, label in sma_styles:
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=df_rs[col_name], name=label,
        line=dict(color=color, width=1.8),
    ), row=1, col=1)

# — Extension bands —
band_styles = [
    ("ext5",  "#4fc3f7", "+5%"),
    ("ext10", "#ef5350", "+10%"),
    ("ext15", "#ff8c00", "+15%"),
    ("ext20", "#ffd700", "+20%"),
]
for col_name, color, label in band_styles:
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=df_rs[col_name], name=label,
        line=dict(color=color, width=1, dash="dot"),
        opacity=0.6, showlegend=True,
    ), row=1, col=1)

# ── Mansfield overlay (secondary y-axis, display only) ────────────────────
if "Mansfield" in df_rs.columns:
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=(df_rs["Mansfield"] * 100).round(2),
        name="Mansfield RS %", yaxis="y3",
        line=dict(color="#4fc3f7", width=1.5, dash="dot"),
        opacity=0.7,
        hovertemplate="%{x|%b %d}<br>Mansfield: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=(df_rs["Mansfield_SMA18"] * 100).round(2),
        name="Mansfield SMA18", yaxis="y3",
        line=dict(color="#4fc3f7", width=1, dash="dot"),
        opacity=0.4, showlegend=False,
    ), row=1, col=1)

# Highlight where RS is above/below SMA18
above = df_rs["RS"] >= df_rs["SMA18"]
fig_rs.add_trace(go.Scatter(
    x=df_rs.index, y=df_rs["RS"].where(above),
    fill="tonexty", fillcolor="rgba(0,255,136,0.06)",
    line=dict(width=0), showlegend=False, name="Above SMA18",
), row=1, col=1)

# — Price panel —
price_data = get_stock_data(selected, period=period, interval=yf_interval)
if not price_data.empty:
    bar_c = ["#00ff88" if c >= o else "#ff4444"
             for c, o in zip(price_data["Close"], price_data["Open"])]
    fig_rs.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data["Open"], high=price_data["High"],
        low=price_data["Low"],   close=price_data["Close"],
        name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=2, col=1)

fig_rs.update_layout(
    template="plotly_dark",
    height=620,
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.03, x=0, font_size=11),
    margin=dict(l=0, r=60, t=40, b=0),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    yaxis3=dict(
        overlaying="y", side="right",
        showgrid=False, zeroline=True, zerolinecolor="#4fc3f780",
        tickformat=".0%", title="Mansfield %",
        title_font=dict(color="#4fc3f7", size=10),
        tickfont=dict(color="#4fc3f7", size=9),
    ),
)
set_chart_window(fig_rs)
st.plotly_chart(fig_rs, width='stretch')

# State detail card
st.info(
    f"**{cfg.get('emoji','')} {state}**  \n"
    f"📖 {cfg.get('description','')}  \n"
    f"➡ **Action:** {cfg.get('action','')}"
)
