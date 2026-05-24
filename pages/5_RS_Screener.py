"""
pages/5_RS_Screener.py
RS Trend v1.8 — grouped screener using the portfolio watchlist.
Order and grouping preserved exactly as defined in utils/watchlist.py.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf

from utils.rs_indicators import build_rs_df, classify_state, STATE_CONFIG
from utils.watchlist import PORTFOLIO, GROUP_ORDER, ALL_YF_SYMBOLS, yf_sym
from utils.data_fetcher import get_stock_data
from utils.chart_utils import set_chart_window

st.set_page_config(
    page_title="RS Screener · StratFlow",
    page_icon="\U0001f3c6", layout="wide"
)
st.title("\U0001f3c6 RS Trend Screener")
st.caption("RS Trend v1.8 · Weekly RS vs pair-specific benchmark · Scores −3 to +2")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    interval_label = st.selectbox("Timeframe", ["1wk (Weekly)", "1d (Daily)"], index=0)
    yf_interval = "1wk" if "wk" in interval_label else "1d"
    period      = "2y"

    st.subheader("Filter")
    score_filter = st.multiselect(
        "RS State",
        options=list(STATE_CONFIG.keys()),
        default=list(STATE_CONFIG.keys()),
    )
    group_filter = st.multiselect(
        "Groups",
        options=GROUP_ORDER,
        default=GROUP_ORDER,
    )
    run_btn = st.button("\u25b6 Run Scan", type="primary", use_container_width=True)

# ── State legend ──────────────────────────────────────────────────────────────
with st.expander("\U0001f4d8 RS State Reference", expanded=False):
    cols = st.columns(3)
    for i, (state, cfg) in enumerate(STATE_CONFIG.items()):
        with cols[i % 3]:
            st.markdown(
                f"**{cfg['emoji']} {state}** · Score `{cfg['score']:+d}`  \n"
                f"*{cfg['description']}*  \n\u27a1 {cfg['action']}"
            )
st.divider()


# ── Batch downloader ──────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_closes(symbols: tuple, period: str, interval: str) -> pd.DataFrame:
    """Download all closes in one batch call. Returns DataFrame[symbol → close]."""
    syms = list(symbols)
    try:
        raw = yf.download(syms, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": syms[0]})
        return closes.dropna(how="all")
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()


# ── Screener engine ───────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_screener(period: str, interval: str) -> pd.DataFrame:
    """Score every pair in PORTFOLIO using its own benchmark."""
    closes = fetch_all_closes(tuple(ALL_YF_SYMBOLS), period, interval)
    if closes.empty:
        return pd.DataFrame()

    rows = []
    for display_t, display_b, group in PORTFOLIO:
        yf_t = yf_sym(display_t)
        yf_b = yf_sym(display_b)

        if yf_t not in closes.columns or yf_b not in closes.columns:
            rows.append({
                "Group": group, "Pair": f"{display_t}/{display_b}",
                "Ticker": display_t, "Bench": display_b,
                "Price": None, "RS Score": None, "RS State": "No Data",
                "Ext %": None, "RS Mom 4W": None, "Mansfield %": None,
                "Action": "—", "_emoji": "❓", "_color": "#555",
                "_df_rs": pd.DataFrame(),
            })
            continue

        try:
            tk_close = closes[yf_t].dropna()
            bk_close = closes[yf_b].dropna()
            df_rs = build_rs_df(tk_close, bk_close)
            state, score, desc, action, ext = classify_state(df_rs)

            last_price = float(tk_close.iloc[-1]) if not tk_close.empty else None

            rs_mom = None
            if len(df_rs) >= 5:
                rs_now  = df_rs["RS"].iloc[-1]
                rs_4ago = df_rs["RS"].iloc[-5]
                rs_mom  = round((rs_now - rs_4ago) / rs_4ago * 100, 1) if rs_4ago else None

            mansfield_val = None
            if "Mansfield" in df_rs.columns:
                m = df_rs["Mansfield"].dropna()
                mansfield_val = round(float(m.iloc[-1]) * 100, 1) if not m.empty else None

            rows.append({
                "Group":       group,
                "Pair":        f"{display_t}/{display_b}",
                "Ticker":      display_t,
                "Bench":       display_b,
                "Price":       round(last_price, 2) if last_price else None,
                "RS Score":    score,
                "RS State":    state,
                "Ext %":       round(ext, 1),
                "RS Mom 4W":   rs_mom,
                "Mansfield %": mansfield_val,
                "Action":      action,
                "_emoji":      STATE_CONFIG.get(state, {}).get("emoji", "❓"),
                "_color":      STATE_CONFIG.get(state, {}).get("color", "#555"),
                "_df_rs":      df_rs,
            })
        except Exception:
            rows.append({
                "Group": group, "Pair": f"{display_t}/{display_b}",
                "Ticker": display_t, "Bench": display_b,
                "Price": None, "RS Score": None, "RS State": "Error",
                "Ext %": None, "RS Mom 4W": None, "Mansfield %": None,
                "Action": "—", "_emoji": "⚠️", "_color": "#555",
                "_df_rs": pd.DataFrame(),
            })

    return pd.DataFrame(rows)


# ── Run ───────────────────────────────────────────────────────────────────────
if "rs_results" not in st.session_state or run_btn:
    with st.spinner(f"Scanning {len(PORTFOLIO)} pairs…"):
        st.session_state["rs_results"] = run_screener(period, yf_interval)

results: pd.DataFrame = st.session_state.get("rs_results", pd.DataFrame())
if results.empty:
    st.warning("No results. Try clicking Run Scan.")
    st.stop()

# Apply filters
filtered = results[
    results["RS State"].isin(score_filter) &
    results["Group"].isin(group_filter)
].copy()

# ── State distribution ────────────────────────────────────────────────────────
st.subheader("\U0001f4ca State Distribution")
valid = results[results["RS Score"].notna()]
state_counts = valid["RS State"].value_counts().reindex(STATE_CONFIG.keys(), fill_value=0)
fig_dist = go.Figure(go.Bar(
    x=list(state_counts.index), y=list(state_counts.values),
    marker_color=[STATE_CONFIG[s]["color"] for s in state_counts.index],
    text=list(state_counts.values), textposition="outside",
    hovertemplate="%{x}: %{y} pairs<extra></extra>",
))
fig_dist.update_layout(
    template="plotly_dark", height=200,
    margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    yaxis=dict(showticklabels=False),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig_dist, width="stretch")
st.divider()

# ── Grouped results table ─────────────────────────────────────────────────────
st.subheader(f"\U0001f4cb Results  ·  {len(filtered)} pairs shown")

WANT_COLS = ["_emoji","Pair","Price","RS Score","RS State","Ext %","RS Mom 4W","Mansfield %"]
DISP_COLS = [c for c in WANT_COLS if c in filtered.columns]

def row_style(row):
    color = STATE_CONFIG.get(row.get("RS State",""), {}).get("color","#888")
    return [f"background-color: {color}18"] * len(row)

for group in GROUP_ORDER:
    if group not in group_filter:
        continue
    grp_df = filtered[filtered["Group"] == group]
    if grp_df.empty:
        continue

    st.markdown(f"**{group}**")
    display = grp_df[DISP_COLS].copy().rename(columns={"_emoji": ""})
    st.dataframe(
        display.style.apply(row_style, axis=1),
        width="stretch", hide_index=True,
        column_config={
            "RS Score":    st.column_config.NumberColumn(format="%+d"),
            "Ext %":       st.column_config.NumberColumn(format="%.1f%%"),
            "RS Mom 4W":   st.column_config.NumberColumn("Mom 4W%", format="%.1f%%"),
            "Mansfield %": st.column_config.NumberColumn("Mansfield%", format="%.1f%%",
                           help="(RS/SMA52)−1 · Display only"),
            "Price":       st.column_config.NumberColumn(format="$%.2f"),
        }
    )

# ── Drill-down ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("\U0001f50d Drill-Down — RS Chart")

available = filtered[filtered["RS Score"].notna()]["Pair"].tolist()
if not available:
    st.info("No valid pairs to drill into with current filters.")
    st.stop()

selected_pair = st.selectbox("Select pair", available)
row = filtered[filtered["Pair"] == selected_pair].iloc[0]
df_rs: pd.DataFrame = row["_df_rs"]
state   = row["RS State"]
score   = row["RS Score"]
ext_pct = row["Ext %"]
cfg     = STATE_CONFIG.get(state, {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pair",      selected_pair)
c2.metric("RS State",  f"{cfg.get('emoji','')} {state}")
c3.metric("RS Score",  f"{score:+.0f}" if score is not None else "—")
c4.metric("Extension", f"{ext_pct:+.1f}%" if ext_pct is not None else "—")

if df_rs.empty:
    st.warning("No chart data available for this pair.")
    st.stop()

# RS chart
fig_rs = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.06, row_heights=[0.65, 0.35],
    subplot_titles=[f"{selected_pair}  RS Ratio", f"{row['Ticker']} Price"],
)

fig_rs.add_trace(go.Scatter(
    x=df_rs.index, y=df_rs["RS"], name="RS",
    line=dict(color="#ffffff", width=2),
    hovertemplate="%{x|%b %d}<br>RS: %{y:.4f}<extra></extra>",
), row=1, col=1)

for col_, color_, lbl_ in [
    ("SMA8",  "#ff8c00", "SMA8"),
    ("SMA18", "#9c59b0", "SMA18"),
    ("SMA40", "#c39bd3", "SMA40"),
]:
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=df_rs[col_], name=lbl_,
        line=dict(color=color_, width=1.8),
    ), row=1, col=1)

for bc_, color_, lbl_ in [
    ("ext5",  "#4fc3f7", "+5%"),
    ("ext10", "#ef5350", "+10%"),
    ("ext15", "#ff8c00", "+15%"),
    ("ext20", "#ffd700", "+20%"),
]:
    fig_rs.add_trace(go.Scatter(
        x=df_rs.index, y=df_rs[bc_], name=lbl_,
        line=dict(color=color_, width=1, dash="dot"), opacity=0.6,
    ), row=1, col=1)

# Price panel
price_data = get_stock_data(yf_sym(row["Ticker"]), period=period, interval=yf_interval)
if not price_data.empty:
    fig_rs.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data["Open"], high=price_data["High"],
        low=price_data["Low"],   close=price_data["Close"],
        name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=2, col=1)

fig_rs.update_layout(
    template="plotly_dark", height=620,
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.03, x=0, font_size=11),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
set_chart_window(fig_rs)
st.plotly_chart(fig_rs, width="stretch")

st.info(
    f"**{cfg.get('emoji','')} {state}**  \n"
    f"\U0001f4d6 {cfg.get('description','')}  \n"
    f"\u27a1 **Action:** {cfg.get('action','')}"
)
