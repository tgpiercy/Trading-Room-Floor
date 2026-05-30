"""
pages/7_Strategy_Scanner.py
Full GW2 strategy scanner — all signals, grouped by portfolio section.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.watchlist import PORTFOLIO, GROUP_ORDER, ALL_YF_SYMBOLS, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch
from utils.rs_indicators import build_rs_df, classify_state, STATE_CONFIG
from utils.strategy import (
    calc_price_indicators, calc_ad, calc_ad_state_gw2, calc_ad_weak_distrib,
    calc_gw2_score, calc_impulse_state, calc_weekly_state, calc_stops,
    calc_atr_regime, calc_dual_rsi, calc_volume_focus, calc_rs_momentum,
    calc_parent_rs_state, calc_entry_signal, SIGNAL_COLORS,
)
from utils.market_health import calc_market_health
from utils.chart_utils import set_chart_window

st.set_page_config(page_title="Strategy Scanner · StratFlow",
                   page_icon="🎯", layout="wide")
st.title("🎯 Strategy Scanner")
st.caption("GW2 v6.3 · Full signal engine · Weekly primary timeframe")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    target_risk = st.slider("Full Portfolio Risk %", 2.0, 8.0, 4.5, step=0.25)
    period = "2y"

    st.subheader("Filters")
    signal_filter = st.multiselect(
        "Signal", options=list(SIGNAL_COLORS.keys()),
        default=list(SIGNAL_COLORS.keys())
    )
    group_filter = st.multiselect(
        "Groups", options=GROUP_ORDER, default=GROUP_ORDER
    )
    run_btn = st.button("▶ Run Full Scan", type="primary", use_container_width=True)
    st.caption("⏱ First run: ~30s for full OHLCV download")

# ── Market Health header ──────────────────────────────────────────────────────
with st.spinner("Loading Market Health…"):
    mh = calc_market_health(target_risk)

mh_col = ("#00cc66" if mh["mh_pct"] >= 75 else
          "#ffd700" if mh["mh_pct"] >= 40 else "#ff4444")
st.markdown(
    f"<div style='padding:8px;border-radius:6px;background:{mh_col}22;"
    f"border:1px solid {mh_col};margin-bottom:12px'>"
    f"🏥 <b>Market Health: {mh['mh_pct']}%</b> &nbsp;|&nbsp; "
    f"Target Risk: <b>{mh['target_risk']:.2f}%</b> &nbsp;|&nbsp; "
    f"VIX: <b>{mh['vix']:.1f}</b> ({mh['vix_regime']}) &nbsp;|&nbsp; "
    f"Breadth (S5FI): <b>{mh['s5fi']:.1f}%</b>"
    f"</div>", unsafe_allow_html=True
)

st.divider()


# ── Parent RS lookup (from watchlist sector/index rows) ───────────────────────
def _get_parent_rs(ticker: str, results: pd.DataFrame) -> str:
    """Find the parent's RS regime from already-computed screener results."""
    # Find rows where this ticker is the BENCH (i.e. it's the parent)
    # The parent of e.g. NVDA is XLK. XLK/SPY is already in results.
    for _, r in results.iterrows():
        if r["Ticker"] == ticker and r.get("RS State") not in (None, "No Data", "Error"):
            return r.get("Parent RS", "Unknown")
    return "Unknown"


# ── Full scan engine ──────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_strategy_scan(mh_pct: float, period: str) -> pd.DataFrame:
    """Download all OHLCV, compute all signals for every portfolio pair."""
    # Batch download
    all_syms = tuple(dict.fromkeys(
        yf_sym(t) for pair in PORTFOLIO for t in (pair[0], pair[1])
    ))
    ohlcv = fetch_ohlcv_batch(all_syms, period=period)
    if not ohlcv:
        return pd.DataFrame()

    rows = []
    # Build parent RS lookup: parent_ticker → closes
    parent_closes = {}
    for sym, df in ohlcv.items():
        parent_closes[sym] = df["Close"]

    for display_t, display_b, group in PORTFOLIO:
        yf_t = yf_sym(display_t)
        yf_b = yf_sym(display_b)

        if yf_t not in ohlcv or yf_b not in ohlcv:
            rows.append(_empty_row(display_t, display_b, group, "No Data"))
            continue

        try:
            df_t = ohlcv[yf_t]
            df_b = ohlcv[yf_b]

            # ── Price indicators ───────────────────────────────────────────────
            price_df = calc_price_indicators(df_t)

            # ── A/D ───────────────────────────────────────────────────────────
            ad_df = calc_ad(df_t)

            # ── RS (ticker/bench) ─────────────────────────────────────────────
            rs_df = build_rs_df(df_t["Close"], df_b["Close"])
            rs_state, rs_score_v, _, _, rs_ext = classify_state(rs_df)

            # ── GW2 Scorecard ─────────────────────────────────────────────────
            gw2 = calc_gw2_score(price_df, ad_df, rs_df)

            # ── Impulse ───────────────────────────────────────────────────────
            impulse = calc_impulse_state(price_df, ad_df, rs_df, gw2)

            # ── Weekly State ─────────────────────────────────────────────────
            wstate = calc_weekly_state(price_df, ad_df, rs_df, gw2, impulse)

            # ── Stops ────────────────────────────────────────────────────────
            stops = calc_stops(price_df)

            # ── ATR regime ───────────────────────────────────────────────────
            atr_r = calc_atr_regime(price_df)

            # ── Dual RSI ─────────────────────────────────────────────────────
            rsi_d = calc_dual_rsi(df_t["Close"])

            # ── Volume Focus ─────────────────────────────────────────────────
            vol_d = calc_volume_focus(df_t, display_t)

            # ── RS Momentum ──────────────────────────────────────────────────
            rs_mom = calc_rs_momentum(rs_df)

            # ── Parent RS ─────────────────────────────────────────────────────
            # Parent of this ticker is its benchmark; parent's parent is SPY/XBB
            # For sector/index: bench=SPY/IEF/XBB → these ARE the parent
            # For industry: bench=XLK/XLE etc. → look up XLK/SPY in PORTFOLIO
            parent_rs = "N/A"
            if display_b in parent_closes:
                # Find bench's bench (grandparent)
                for pt, pb, pg in PORTFOLIO:
                    if pt == display_b and pb in parent_closes:
                        parent_rs = calc_parent_rs_state(
                            parent_closes[yf_sym(display_b)],
                            parent_closes[yf_sym(pb)]
                        )
                        break
                if parent_rs == "N/A":
                    parent_rs = "SPY/IEF" if display_b in ("SPY","IEF") else "Top-Level"

            # ── Entry signal ─────────────────────────────────────────────────
            sig = calc_entry_signal(
                impulse, wstate, gw2, rs_state,
                rsi_d["state"], vol_d["state"],
                atr_r["state"], parent_rs, mh_pct
            )

            last_p = price_df.dropna(subset=["SMA18"]).iloc[-1]
            rows.append({
                "Group":       group,
                "Pair":        f"{display_t}/{display_b}",
                "Ticker":      display_t,
                "Bench":       display_b,
                "Price":       round(float(last_p["Close"]), 2),
                "Signal":      sig["signal"],
                "Sig Pts":     sig["pts"],
                "Impulse":     impulse,
                "W State":     {0:"OUT",1:"PILOT",2:"FULL"}.get(wstate,"—"),
                "GW2":         gw2["score"],
                "Read":        gw2["read_txt"],
                "RS State":    rs_state,
                "RS Ext%":     round(rs_ext, 1),
                "RS Mom":      rs_mom,
                "A/D":         gw2["ad_state"],
                "RSI":         rsi_d["state"],
                "Volume":      vol_d["state"],
                "ATR":         atr_r["state"],
                "Parent RS":   parent_rs,
                "Prog Stop":   stops.get("program_stop", None),
                "ATR Dist":    stops.get("atr_dist", None),
                "Mgmt":        stops.get("mgmt_mode","WEEKLY"),
                "_sig_color":  sig["color"],
                "_sig_detail": sig["detail"],
                "_exit_trigs": sig["exit_triggers"],
                "_price_df":   price_df,
                "_ad_df":      ad_df,
                "_rs_df":      rs_df,
                "_gw2":        gw2,
                "_stops":      stops,
                "_rsi":        rsi_d,
                "_vol":        vol_d,
            })
        except Exception as e:
            rows.append(_empty_row(display_t, display_b, group, "Error"))

    return pd.DataFrame(rows)


def _empty_row(t, b, grp, err):
    return {"Group": grp, "Pair": f"{t}/{b}", "Ticker": t, "Bench": b,
            "Price": None, "Signal": err, "Sig Pts": 0,
            "Impulse": "—", "W State": "—", "GW2": 0, "Read": "—",
            "RS State": err, "RS Ext%": None, "RS Mom": "—",
            "A/D": "—", "RSI": "—", "Volume": "—", "ATR": "—",
            "Parent RS": "—", "Prog Stop": None, "ATR Dist": None,
            "Mgmt": "—", "_sig_color": "#555", "_sig_detail": {},
            "_exit_trigs": [], "_price_df": pd.DataFrame(),
            "_ad_df": pd.DataFrame(), "_rs_df": pd.DataFrame(),
            "_gw2": {}, "_stops": {}, "_rsi": {}, "_vol": {}}


# ── Run ───────────────────────────────────────────────────────────────────────
if "strategy_results" not in st.session_state or run_btn:
    with st.spinner(f"Scanning {len(PORTFOLIO)} pairs — downloading OHLCV + computing all signals…"):
        st.session_state["strategy_results"] = run_strategy_scan(mh["mh_pct"], period)

results: pd.DataFrame = st.session_state.get("strategy_results", pd.DataFrame())
if results.empty:
    st.warning("No results. Click Run Full Scan.")
    st.stop()

filtered = results[
    results["Signal"].isin(signal_filter) &
    results["Group"].isin(group_filter)
].copy()

# ── Signal distribution ───────────────────────────────────────────────────────
st.subheader("📊 Signal Distribution")
sig_counts = results["Signal"].value_counts().reindex(
    SIGNAL_COLORS.keys(), fill_value=0)
fig_dist = go.Figure(go.Bar(
    x=list(sig_counts.index), y=list(sig_counts.values),
    marker_color=[SIGNAL_COLORS[s] for s in sig_counts.index],
    text=list(sig_counts.values), textposition="outside",
))
fig_dist.update_layout(template="plotly_dark", height=200, showlegend=False,
                        yaxis=dict(showticklabels=False),
                        margin=dict(l=0,r=0,t=10,b=0),
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig_dist, width="stretch")
st.divider()

# ── Grouped table ─────────────────────────────────────────────────────────────
WANT = ["Pair","Price","Signal","Sig Pts","Impulse","W State","GW2","Read",
        "RS State","RS Ext%","RS Mom","A/D","RSI","Volume","ATR","Prog Stop","Mgmt"]

def row_bg(row):
    c = row.get("_sig_color", "#888")
    return [f"background-color:{c}18"] * len(row)

st.subheader(f"📋 Results  ·  {len(filtered)} pairs")

for group in GROUP_ORDER:
    if group not in group_filter:
        continue
    grp = filtered[filtered["Group"] == group]
    if grp.empty:
        continue
    st.markdown(f"**{group}**")
    disp = grp[[c for c in WANT if c in grp.columns]].copy()
    st.dataframe(
        disp.style.apply(row_bg, axis=1),
        width="stretch", hide_index=True,
        column_config={
            "Price":    st.column_config.NumberColumn(format="$%.2f"),
            "GW2":      st.column_config.NumberColumn("GW2/7"),
            "Sig Pts":  st.column_config.NumberColumn("Pts/9"),
            "RS Ext%":  st.column_config.NumberColumn(format="%.1f%%"),
            "Prog Stop":st.column_config.NumberColumn("Stop", format="$%.2f"),
            "ATR Dist": st.column_config.NumberColumn("ATR Dist", format="%.2f"),
        }
    )

# ── Drill-Down ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Drill-Down")

valid = filtered[filtered["Sig Pts"].notna() & (filtered["Signal"] != "No Data")]
if valid.empty:
    st.info("No valid pairs to inspect with current filters.")
    st.stop()

sel = st.selectbox("Select pair", valid["Pair"].tolist())
row = valid[valid["Pair"] == sel].iloc[0]

# Signal card
sc = row["_sig_color"]
st.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{sc}22;border:1px solid {sc}'>"
    f"<b style='font-size:1.2rem'>{row['Signal']}</b>  "
    f"&nbsp;·&nbsp; {row['Sig Pts']}/9 points &nbsp;·&nbsp; "
    f"Impulse: <b>{row['Impulse']}</b> &nbsp;·&nbsp; "
    f"Weekly State: <b>{row['W State']}</b>"
    f"</div>", unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 GW2 Scorecard", "🎯 Signal Checklist",
                                    "📉 Stops & ATR", "📈 RS Chart"])

with tab1:
    gw2 = row["_gw2"]
    st.metric("GW2 Score", f"{gw2.get('score',0)} / 7",
              gw2.get("read_txt","—"))
    scorecard = [
        ("A/D > SMA18",     gw2.get("ad_above", False)),
        ("A/D SMA18 Rising",gw2.get("ad_rising",False)),
        ("RS > RS SMA18",   gw2.get("rs_above", False)),
        ("RS SMA18 Rising", gw2.get("rs_rising",False)),
        ("Price > SMA18",   gw2.get("px_above", False)),
        ("SMA18 Rising",    gw2.get("sma18_rising",False)),
        ("SMA18 > SMA40",   gw2.get("sma18_gt_40",False)),
    ]
    sc_df = pd.DataFrame(
        [("✅" if v else "❌", k, "1" if v else "0") for k, v in scorecard],
        columns=["","Condition","Pt"]
    )
    st.dataframe(sc_df, width="stretch", hide_index=True)
    st.caption(f"Struct ATR Spread: {gw2.get('spread_atr',0):.2f} "
               f"({'Strong' if gw2.get('struct_strong') else 'Not yet'})")

with tab2:
    detail = row["_sig_detail"]
    exit_t = row["_exit_trigs"]
    if exit_t:
        st.error("⚠️ Exit triggers active: " + " | ".join(exit_t))
    detail_rows = [(("✅" if v else "❌"), k) for k, v in detail.items()]
    st.dataframe(pd.DataFrame(detail_rows, columns=["","Condition"]),
                 width="stretch", hide_index=True)

with tab3:
    stops = row["_stops"]
    rsi_d = row["_rsi"]
    atr_d = row.get("ATR","—")
    if stops:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Program Stop",  f"${stops.get('program_stop',0):.2f}")
        c2.metric("Stop Limit",    f"${stops.get('stop_limit',0):.2f}")
        c3.metric("ATR Distance",  f"{stops.get('atr_dist',0):.2f}×",
                  stops.get("mgmt_mode","WEEKLY"))
        c4.metric("ATR Value",     f"${stops.get('atr',0):.2f}")
        st.caption(
            f"Structure Stop: ${stops.get('struct_stop',0):.2f} "
            f"(SMA18 − 0.50×ATR)  ·  "
            f"Trailing Stop: ${stops.get('trail_stop',0):.2f} "
            f"(Close − 1.25×ATR)  ·  "
            f"Green Extension: ${stops.get('green_line',0):.2f}  ·  "
            f"Red Warning: ${stops.get('red_line',0):.2f}"
        )
    st.metric("RSI State", rsi_d.get("state","—"),
              f"RSI5={rsi_d.get('rsi5',0)} / RSI20={rsi_d.get('rsi20',0)}")
    st.metric("ATR Regime", atr_d)

with tab4:
    rs_df: pd.DataFrame = row["_rs_df"]
    price_df: pd.DataFrame = row["_price_df"]
    if not rs_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.65, 0.35],
                            subplot_titles=[f"{sel} RS Ratio", f"{row['Ticker']} Price"])
        # RS lines
        fig.add_trace(go.Scatter(x=rs_df.index, y=rs_df["RS"],
                                 name="RS", line=dict(color="#fff",width=2)), row=1,col=1)
        for col_,clr,nm in [("SMA4","#ffd700","SMA4"),
                             ("SMA8","#ff8c00","SMA8"),
                             ("SMA18","#9c59b0","SMA18"),
                             ("SMA40","#c39bd3","SMA40")]:
            if col_ in rs_df.columns:
                fig.add_trace(go.Scatter(x=rs_df.index, y=rs_df[col_],
                                         name=nm, line=dict(color=clr,width=1.5)), row=1,col=1)
        for bc_,clr_,lbl_ in [("ext5","#4fc3f7","+5%"),("ext10","#ef5350","+10%"),
                               ("ext15","#ff8c00","+15%"),("ext20","#ffd700","+20%")]:
            fig.add_trace(go.Scatter(x=rs_df.index, y=rs_df[bc_], name=lbl_,
                                     line=dict(color=clr_,width=1,dash="dot"),
                                     opacity=0.5), row=1,col=1)
        # Price with stops
        if not price_df.empty:
            fig.add_trace(go.Candlestick(
                x=price_df.index, open=price_df["Open"], high=price_df["High"],
                low=price_df["Low"], close=price_df["Close"], name="Price",
                increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
            ), row=2,col=1)
            # MAs on price
            for col_,clr_ in [("SMA18","#9c59b0"),("SMA40","#c39bd3")]:
                if col_ in price_df.columns:
                    fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col_],
                                             name=col_, line=dict(color=clr_,width=1.5),
                                             showlegend=False), row=2,col=1)
            # Stop lines
            if stops:
                last_date = price_df.index[-1]
                for stop_key, clr_, nm_ in [
                    ("program_stop","#4fc3f7","Stop"),
                    ("green_line","#00ff88","Green Ext"),
                    ("red_line","#ff4444","Red Warn"),
                ]:
                    val = stops.get(stop_key)
                    if val:
                        fig.add_hline(y=val, line_dash="dot", line_color=clr_,
                                      opacity=0.6, row=2, col=1,
                                      annotation_text=f"{nm_} ${val:.2f}",
                                      annotation_position="right")
        fig.update_layout(template="plotly_dark", height=620,
                          xaxis_rangeslider_visible=False,
                          xaxis2_rangeslider_visible=False,
                          legend=dict(orientation="h",y=1.03,x=0,font_size=10),
                          margin=dict(l=0,r=0,t=40,b=0),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        set_chart_window(fig)
        st.plotly_chart(fig, width="stretch")
