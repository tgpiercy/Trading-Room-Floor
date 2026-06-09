"""
pages/5_Screener.py
Unified Screener — RS structure + full GW2 strategy signals in one place.
One OHLCV download, all signals computed once.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.watchlist import PORTFOLIO, GROUP_ORDER, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch, get_stock_data
from utils.rs_indicators import build_rs_df, classify_state, STATE_CONFIG
from utils.strategy import (
    calc_price_indicators, calc_ad, calc_gw2_score, calc_impulse_state,
    calc_weekly_state, calc_stops, calc_atr_regime, calc_dual_rsi,
    calc_volume_focus, calc_rs_momentum, calc_parent_rs_state,
    calc_entry_signal, SIGNAL_COLORS,
)
from utils.market_health import calc_market_health
from utils.rotation import (calc_rrg, rrg_quadrant, calc_rs_acceleration,
    calc_divergence, calc_accumulation_persistence, calc_rvol, rvol_zscore,
    calc_early_rotation_score, ROTATION_TIER_COLOR)
from utils.chart_utils import set_chart_window

st.title("🎯 Screener")
st.caption("RS Trend v1.8 + GW2 v6.3 · unified signal engine · weekly primary")

# ── View mode column presets ──────────────────────────────────────────────────
VIEW_MODES = {
    "Compact":        ["Pair","Signal","GW2","RS State"],
    "RS Focus":       ["Pair","Signal","RS Score","RS State","Ext %","RS Mom","Mansfield %"],
    "Strategy Focus": ["Pair","Signal","Sig Pts","Impulse","W State","GW2","RSI","Volume","A/D","ATR"],
    "Rotation Focus": ["Pair","Early Rot","Tier","Quadrant","RS Accel","RVOL z","Diverg","Accum wk"],
    "Full":           ["Pair","Price","Signal","Sig Pts","Impulse","W State","GW2","Read",
                       "RS State","Ext %","RS Mom","RSI","Volume","ATR","Early Rot","Quadrant","Prog Stop","Mgmt"],
}
SIGNAL_ORDER = list(SIGNAL_COLORS.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    view_mode = st.selectbox("View Mode", list(VIEW_MODES.keys()), index=1)
    grouping  = st.selectbox("Group By", ["Sector","Signal","RS State"], index=0)
    target_risk = st.slider("Full Portfolio Risk %", 2.0, 8.0, 4.5, step=0.25)

    st.subheader("Filters")
    group_filter = st.multiselect("Sectors", GROUP_ORDER, default=GROUP_ORDER)
    run_btn = st.button("▶ Run Scan", type="primary", width='stretch')
    st.caption("⏱ First run ~30s (downloads OHLCV for all pairs)")

# ── Market Health banner ──────────────────────────────────────────────────────
with st.spinner("Loading Market Health…"):
    mh = calc_market_health(target_risk)

mh_col = ("#00cc66" if mh["mh_pct"] >= 75 else
          "#ffd700" if mh["mh_pct"] >= 40 else "#ff4444")
st.markdown(
    f"<div style='padding:10px;border-radius:8px;background:{mh_col}22;"
    f"border:1px solid {mh_col};margin-bottom:10px'>"
    f"🏥 <b>Market Health: {mh['mh_pct']}%</b> &nbsp;|&nbsp; "
    f"Target Risk <b>{mh['target_risk']:.2f}%</b> &nbsp;|&nbsp; "
    f"VIX <b>{mh['vix']:.1f}</b> ({mh['vix_regime']}) &nbsp;|&nbsp; "
    f"Breadth <b>{mh['s5fi']:.0f}%</b> &nbsp;|&nbsp; "
    f"RS {mh['rs_score']}/2</div>", unsafe_allow_html=True)


# ── Scan engine ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_unified_scan(mh_pct: float, period: str = "2y"):
    """Download all OHLCV once, compute RS + strategy for every pair."""
    all_syms = tuple(dict.fromkeys(
        yf_sym(t) for pair in PORTFOLIO for t in (pair[0], pair[1])
    ))
    ohlcv = fetch_ohlcv_batch(all_syms, period=period)
    if not ohlcv:
        return pd.DataFrame(), {}

    # First pass: RVOL for all tickers (for cross-sectional z-scoring)
    rvol_map = {}
    for dt, db, g in PORTFOLIO:
        yt = yf_sym(dt)
        if yt in ohlcv:
            rvol_map[dt] = calc_rvol(ohlcv[yt])
    # Group membership for peer z-scoring
    group_members = {}
    for dt, db, g in PORTFOLIO:
        group_members.setdefault(g, []).append(dt)

    rows = []
    for display_t, display_b, group in PORTFOLIO:
        yf_t, yf_b = yf_sym(display_t), yf_sym(display_b)
        if yf_t not in ohlcv or yf_b not in ohlcv:
            rows.append(_blank(display_t, display_b, group, "No Data"))
            continue
        try:
            df_t, df_b = ohlcv[yf_t], ohlcv[yf_b]
            price_df = calc_price_indicators(df_t)
            ad_df    = calc_ad(df_t)
            rs_df    = build_rs_df(df_t["Close"], df_b["Close"])
            rs_state, rs_score_v, _, _, rs_ext = classify_state(rs_df)

            gw2     = calc_gw2_score(price_df, ad_df, rs_df)
            impulse = calc_impulse_state(price_df, ad_df, rs_df, gw2)
            wstate  = calc_weekly_state(price_df, ad_df, rs_df, gw2, impulse)
            stops   = calc_stops(price_df)
            atr_r   = calc_atr_regime(price_df)
            rsi_d   = calc_dual_rsi(df_t["Close"])
            vol_d   = calc_volume_focus(df_t, display_t)
            rs_mom  = calc_rs_momentum(rs_df)

            # Mansfield
            mansfield = None
            if "Mansfield" in rs_df.columns:
                m = rs_df["Mansfield"].dropna()
                mansfield = round(float(m.iloc[-1]) * 100, 1) if not m.empty else None

            # Parent RS (grandparent regime)
            parent_rs = "Top-Level"
            for pt, pb, pg in PORTFOLIO:
                if pt == display_b and yf_sym(pb) in ohlcv:
                    parent_rs = calc_parent_rs_state(ohlcv[yf_b]["Close"], ohlcv[yf_sym(pb)]["Close"])
                    break

            # ── Early Rotation signals ────────────────────────────────────
            rrg_ratio, rrg_mom = calc_rrg(df_t["Close"], df_b["Close"])
            if rrg_ratio is not None and not rrg_ratio.empty and not rrg_mom.empty:
                q_ratio = float(rrg_ratio.iloc[-1]); q_mom = float(rrg_mom.iloc[-1])
                quadrant = rrg_quadrant(q_ratio, q_mom)
            else:
                q_ratio = q_mom = 100.0; quadrant = "Unknown"
            rs_accel = calc_rs_acceleration(rs_df["RS"])
            diverg   = calc_divergence(df_t["Close"], ad_df["AD"])
            accum_p  = calc_accumulation_persistence(df_t, ad_df)
            peer_rvols = [rvol_map.get(t) for t in group_members.get(group, [])]
            rvol_z   = rvol_zscore(rvol_map.get(display_t), peer_rvols)
            early    = calc_early_rotation_score(rs_accel, rvol_z, diverg, accum_p, quadrant)

            sig = calc_entry_signal(impulse, wstate, gw2, rs_state,
                                    rsi_d["state"], vol_d["state"],
                                    atr_r["state"], parent_rs, mh_pct)

            last_p = price_df.dropna(subset=["SMA18"]).iloc[-1]
            rows.append({
                "Group": group, "Pair": f"{display_t}/{display_b}",
                "Ticker": display_t, "Bench": display_b,
                "Price": round(float(last_p["Close"]), 2),
                "Signal": sig["signal"], "Sig Pts": sig["pts"],
                "Impulse": impulse,
                "W State": {0:"OUT",1:"PILOT",2:"FULL"}.get(wstate,"—"),
                "GW2": gw2["score"], "Read": gw2["read_txt"],
                "RS Score": rs_score_v, "RS State": rs_state,
                "Ext %": round(rs_ext,1), "RS Mom": rs_mom,
                "Mansfield %": mansfield,
                "A/D": gw2["ad_state"], "RSI": rsi_d["state"],
                "Volume": vol_d["state"], "ATR": atr_r["state"],
                "Parent RS": parent_rs,
                "Prog Stop": stops.get("program_stop"),
                "ATR Dist": stops.get("atr_dist"),
                "Mgmt": stops.get("mgmt_mode","WEEKLY"),
                "Early Rot": early["score"], "Tier": early["tier"],
                "Quadrant": quadrant, "RS Accel": rs_accel["accel"],
                "RVOL z": rvol_z, "Diverg": diverg, "Accum wk": accum_p,
                "_rrg_ratio": q_ratio, "_rrg_mom": q_mom,
                "_early_detail": early["detail"],
                "_sig_color": sig["color"], "_sig_detail": sig["detail"],
                "_exit": sig["exit_triggers"], "_emoji": STATE_CONFIG.get(rs_state,{}).get("emoji","❓"),
                "_gw2": gw2, "_stops": stops, "_rsi": rsi_d, "_vol": vol_d,
                "_rs_df": rs_df, "_price_df": price_df,
            })
        except Exception:
            rows.append(_blank(display_t, display_b, group, "Error"))
    return pd.DataFrame(rows), ohlcv


def _blank(t,b,g,err):
    return {"Group":g,"Pair":f"{t}/{b}","Ticker":t,"Bench":b,"Price":None,
            "Signal":err,"Sig Pts":0,"Impulse":"—","W State":"—","GW2":0,"Read":"—",
            "RS Score":None,"RS State":err,"Ext %":None,"RS Mom":"—","Mansfield %":None,
            "A/D":"—","RSI":"—","Volume":"—","ATR":"—","Parent RS":"—",
            "Prog Stop":None,"ATR Dist":None,"Mgmt":"—",
            "Early Rot":0,"Tier":"💤 Quiet","Quadrant":"Unknown","RS Accel":0,
            "RVOL z":0,"Diverg":"None","Accum wk":0,
            "_rrg_ratio":100,"_rrg_mom":100,"_early_detail":{},
            "_sig_color":"#555","_sig_detail":{},"_exit":[],"_emoji":"❓",
            "_gw2":{},"_stops":{},"_rsi":{},"_vol":{},
            "_rs_df":pd.DataFrame(),"_price_df":pd.DataFrame()}


# ── Run ───────────────────────────────────────────────────────────────────────
if "screener_results" not in st.session_state or run_btn:
    with st.spinner(f"Scanning {len(PORTFOLIO)} pairs…"):
        res, _ = run_unified_scan(mh["mh_pct"])
        st.session_state["screener_results"] = res

results: pd.DataFrame = st.session_state.get("screener_results", pd.DataFrame())
if results.empty:
    st.warning("No results. Click Run Scan.")
    st.stop()

valid = results[results["Sig Pts"].notna() & ~results["Signal"].isin(["No Data","Error"])].copy()

# ── Quick filter ──────────────────────────────────────────────────────────────
qf = st.radio("Quick Filter",
              ["All","Strong Entry","Entry/Add+","GW2 ≥ 5","Impulse YES"],
              horizontal=True, label_visibility="collapsed")

filtered = valid[valid["Group"].isin(group_filter)].copy()
if   qf == "Strong Entry": filtered = filtered[filtered["Signal"]=="Strong Entry"]
elif qf == "Entry/Add+":   filtered = filtered[filtered["Signal"].isin(["Strong Entry","Entry / Add"])]
elif qf == "GW2 ≥ 5":      filtered = filtered[filtered["GW2"]>=5]
elif qf == "Impulse YES":  filtered = filtered[filtered["Impulse"]=="YES"]

# ── Top Opportunities ─────────────────────────────────────────────────────────
st.subheader("⭐ Top Opportunities")
top = valid[valid["Signal"].isin(["Strong Entry","Entry / Add"])].copy()
top = top.sort_values(["Sig Pts","GW2"], ascending=[False,False]).head(12)
if top.empty:
    st.info("No Strong Entry or Entry/Add setups in the current scan.")
else:
    top_disp = top[["Pair","Signal","Sig Pts","GW2","Impulse","RS State","RSI","Volume"]].copy()
    def _top_style(r):
        c = SIGNAL_COLORS.get(r["Signal"],"#888")
        return [f"background-color:{c}22"]*len(r)
    st.dataframe(top_disp.style.apply(_top_style,axis=1),
                 width="stretch", hide_index=True,
                 column_config={"Sig Pts":st.column_config.NumberColumn("Pts/9"),
                                "GW2":st.column_config.NumberColumn("GW2/7")})
st.divider()

# ── Distributions ─────────────────────────────────────────────────────────────
d1, d2 = st.columns(2)
with d1:
    st.caption("**Signal Distribution**")
    sc = valid["Signal"].value_counts().reindex(SIGNAL_ORDER, fill_value=0)
    f1 = go.Figure(go.Bar(x=list(sc.index), y=list(sc.values),
                          marker_color=[SIGNAL_COLORS[s] for s in sc.index],
                          text=list(sc.values), textposition="outside"))
    f1.update_layout(template="plotly_dark", height=180, showlegend=False,
                     yaxis=dict(showticklabels=False), margin=dict(l=0,r=0,t=10,b=0),
                     paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(f1, width="stretch")
with d2:
    st.caption("**RS State Distribution**")
    rc = valid["RS State"].value_counts().reindex(STATE_CONFIG.keys(), fill_value=0)
    f2 = go.Figure(go.Bar(x=list(rc.index), y=list(rc.values),
                          marker_color=[STATE_CONFIG[s]["color"] for s in rc.index],
                          text=list(rc.values), textposition="outside"))
    f2.update_layout(template="plotly_dark", height=180, showlegend=False,
                     yaxis=dict(showticklabels=False), margin=dict(l=0,r=0,t=10,b=0),
                     paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(f2, width="stretch")
st.divider()

# ── Results table ─────────────────────────────────────────────────────────────
cols = VIEW_MODES[view_mode]
st.subheader(f"📋 Results · {len(filtered)} pairs · {view_mode} view")

def row_bg(row):
    c = row.get("_sig_color","#888")
    return [f"background-color:{c}14"]*len(row)

COL_CFG = {
    "Price":     st.column_config.NumberColumn(format="$%.2f"),
    "GW2":       st.column_config.NumberColumn("GW2/7"),
    "Sig Pts":   st.column_config.NumberColumn("Pts/9"),
    "RS Score":  st.column_config.NumberColumn(format="%+d"),
    "Ext %":     st.column_config.NumberColumn(format="%.1f%%"),
    "Mansfield %":st.column_config.NumberColumn("Mans%", format="%.1f%%"),
    "Prog Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
    "Early Rot": st.column_config.NumberColumn("EarlyRot/10"),
    "RS Accel":  st.column_config.NumberColumn("RS Accel", format="%.2f"),
    "RVOL z":    st.column_config.NumberColumn("RVOL z", format="%.1f"),
    "Accum wk":  st.column_config.NumberColumn("Accum wk"),
}

def render_group(label, df):
    if df.empty: return
    st.markdown(f"**{label}**")
    show = df.copy()
    show["_sig_color"] = df["_sig_color"]
    disp = show[cols + ["_sig_color"]].copy()
    styled = disp.style.apply(row_bg, axis=1)
    disp_final = disp.drop(columns=["_sig_color"])
    st.dataframe(disp_final.style.apply(
        lambda r: [f"background-color:{df.loc[r.name,'_sig_color']}14"]*len(r), axis=1),
        width="stretch", hide_index=True,
        column_config={k:v for k,v in COL_CFG.items() if k in cols})

if grouping == "Sector":
    for g in GROUP_ORDER:
        if g in group_filter:
            render_group(g, filtered[filtered["Group"]==g])
elif grouping == "Signal":
    for s in SIGNAL_ORDER:
        render_group(s, filtered[filtered["Signal"]==s])
else:  # RS State
    for s in STATE_CONFIG.keys():
        render_group(s, filtered[filtered["RS State"]==s])

# ── Drill-down ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Drill-Down")
if filtered.empty:
    st.info("No pairs match filters.")
    st.stop()

sel = st.selectbox("Select pair", filtered["Pair"].tolist())
row = filtered[filtered["Pair"]==sel].iloc[0]
sc_ = row["_sig_color"]
st.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{sc_}22;border:1px solid {sc_}'>"
    f"<b style='font-size:1.2rem'>{row['Signal']}</b> &nbsp;·&nbsp; "
    f"{row['Sig Pts']}/9 pts &nbsp;·&nbsp; Impulse <b>{row['Impulse']}</b> &nbsp;·&nbsp; "
    f"W State <b>{row['W State']}</b> &nbsp;·&nbsp; GW2 <b>{row['GW2']}/7</b> &nbsp;·&nbsp; "
    f"RS <b>{row['RS State']}</b></div>", unsafe_allow_html=True)

t1,t2,t3,t4 = st.tabs(["📋 GW2 Scorecard","🎯 Signal Checklist","📉 Stops & ATR","📈 RS Chart"])

with t1:
    gw2 = row["_gw2"]
    st.metric("GW2 Score", f"{gw2.get('score',0)}/7", gw2.get("read_txt","—"))
    sc_rows = [("A/D > SMA18",gw2.get("ad_above")),("A/D SMA18 Rising",gw2.get("ad_rising")),
               ("RS > RS SMA18",gw2.get("rs_above")),("RS SMA18 Rising",gw2.get("rs_rising")),
               ("Price > SMA18",gw2.get("px_above")),("SMA18 Rising",gw2.get("sma18_rising")),
               ("SMA18 > SMA40",gw2.get("sma18_gt_40"))]
    st.dataframe(pd.DataFrame([("✅" if v else "❌",k) for k,v in sc_rows],
                 columns=["","Condition"]), width="stretch", hide_index=True)
    st.caption(f"Struct ATR spread {gw2.get('spread_atr',0):.2f} · "
               f"{'Strong separation' if gw2.get('struct_strong') else 'Not yet confirmed'}")

with t2:
    if row["_exit"]:
        st.error("⚠️ Exit triggers: " + " | ".join(row["_exit"]))
    st.dataframe(pd.DataFrame([("✅" if v else "❌",k) for k,v in row["_sig_detail"].items()],
                 columns=["","Condition"]), width="stretch", hide_index=True)

with t3:
    stops, rsi_d = row["_stops"], row["_rsi"]
    if stops:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Program Stop", f"${stops.get('program_stop',0):.2f}")
        c2.metric("Stop Limit",   f"${stops.get('stop_limit',0):.2f}")
        c3.metric("ATR Distance", f"{stops.get('atr_dist',0):.2f}×", stops.get("mgmt_mode"))
        c4.metric("ATR",          f"${stops.get('atr',0):.2f}")
        st.caption(f"Structure ${stops.get('struct_stop',0):.2f} (SMA18−0.5ATR) · "
                   f"Trailing ${stops.get('trail_stop',0):.2f} (Close−1.25ATR) · "
                   f"Green ${stops.get('green_line',0):.2f} · Red ${stops.get('red_line',0):.2f}")
    st.metric("RSI", rsi_d.get("state","—"),
              f"RSI5 {rsi_d.get('rsi5',0)} / RSI20 {rsi_d.get('rsi20',0)}")
    st.metric("Volume", row.get("Volume","—"))
    st.metric("ATR Regime", row.get("ATR","—"))

with t4:
    rs_df, price_df = row["_rs_df"], row["_price_df"]
    if not rs_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.62,0.38],
                            subplot_titles=[f"{sel} RS Ratio", f"{row['Ticker']} Price"])
        fig.add_trace(go.Scatter(x=rs_df.index,y=rs_df["RS"],name="RS",
                                 line=dict(color="#fff",width=2)),row=1,col=1)
        for c_,clr,nm in [("SMA4","#ffd700","SMA4"),("SMA8","#ff8c00","SMA8"),
                          ("SMA18","#9c59b0","SMA18"),("SMA40","#c39bd3","SMA40")]:
            if c_ in rs_df.columns:
                fig.add_trace(go.Scatter(x=rs_df.index,y=rs_df[c_],name=nm,
                                         line=dict(color=clr,width=1.4)),row=1,col=1)
        for bc,clr,lbl in [("ext5","#4fc3f7","+5%"),("ext10","#ef5350","+10%"),
                           ("ext15","#ff8c00","+15%"),("ext20","#ffd700","+20%")]:
            fig.add_trace(go.Scatter(x=rs_df.index,y=rs_df[bc],name=lbl,
                                     line=dict(color=clr,width=1,dash="dot"),
                                     opacity=0.5),row=1,col=1)
        if not price_df.empty:
            fig.add_trace(go.Candlestick(x=price_df.index,open=price_df["Open"],
                          high=price_df["High"],low=price_df["Low"],close=price_df["Close"],
                          name="Price",increasing_line_color="#00ff88",
                          decreasing_line_color="#ff4444"),row=2,col=1)
            for c_,clr in [("SMA18","#9c59b0"),("SMA40","#c39bd3")]:
                if c_ in price_df.columns:
                    fig.add_trace(go.Scatter(x=price_df.index,y=price_df[c_],name=c_,
                                  line=dict(color=clr,width=1.4),showlegend=False),row=2,col=1)
            stops = row["_stops"]
            for k_,clr,nm in [("program_stop","#4fc3f7","Stop"),
                              ("green_line","#00ff88","Green"),("red_line","#ff4444","Red")]:
                v = stops.get(k_)
                if v: fig.add_hline(y=v,line_dash="dot",line_color=clr,opacity=0.6,
                                    row=2,col=1,annotation_text=f"{nm} ${v:.2f}",
                                    annotation_position="right")
        fig.update_layout(template="plotly_dark",height=620,
                          xaxis_rangeslider_visible=False,xaxis2_rangeslider_visible=False,
                          legend=dict(orientation="h",y=1.03,x=0,font_size=10),
                          margin=dict(l=0,r=0,t=40,b=0),
                          paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
        set_chart_window(fig)
        st.plotly_chart(fig, width="stretch")
