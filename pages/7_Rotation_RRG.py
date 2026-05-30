"""
pages/7_Rotation_Radar.py
Rotation Radar — RRG sector/industry map + Early Rotation leaderboard.
Surfaces money & momentum flow before RS structure confirms.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.watchlist import PORTFOLIO, GROUP_ORDER, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch
from utils.rotation import (
    calc_rrg, rrg_quadrant, QUADRANT_COLOR, calc_rs_acceleration,
    calc_divergence, calc_accumulation_persistence, calc_rvol, rvol_zscore,
    calc_early_rotation_score, ROTATION_TIER_COLOR,
)
from utils.strategy import calc_ad
from utils.rs_indicators import build_rs_df

st.set_page_config(page_title="Rotation Radar · StratFlow", page_icon="🛰️", layout="wide")
st.title("🛰️ Rotation Radar")
st.caption("RRG map + Early Rotation score · catching money flow before the trend confirms")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    group_sel = st.selectbox("RRG Universe",
                             ["Sectors", "Themes", "Industries · XLK",
                              "Industries · HXT", "All Top-Level"], index=0)
    tail_len = st.slider("RRG Tail Length (weeks)", 3, 12, 6)
    rrg_window = st.slider("RRG Smoothing Window", 6, 16, 10)
    run_btn = st.button("▶ Run Radar", type="primary", width='stretch')
    st.caption("⏱ First run ~30s")

# ── Universe selection ────────────────────────────────────────────────────────
def _universe(sel):
    if sel == "Sectors":
        return [(t,b,g) for t,b,g in PORTFOLIO if g == "SECTORS"]
    if sel == "Themes":
        return [(t,b,g) for t,b,g in PORTFOLIO if g == "THEMES"]
    if sel == "Industries · XLK":
        return [(t,b,g) for t,b,g in PORTFOLIO if g == "INDUSTRIES · XLK"]
    if sel == "Industries · HXT":
        return [(t,b,g) for t,b,g in PORTFOLIO if g == "INDUSTRIES · HXT"]
    # All top-level: indices + sectors + themes
    return [(t,b,g) for t,b,g in PORTFOLIO
            if g in ("INDICES","SECTORS","THEMES","PRECIOUS METALS")]

universe = _universe(group_sel)


# ── Radar scan ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_radar(universe_key: tuple, window: int, tail: int):
    pairs = list(universe_key)
    syms = tuple(dict.fromkeys(
        yf_sym(t) for p in pairs for t in (p[0], p[1])
    ))
    ohlcv = fetch_ohlcv_batch(syms, period="2y")
    if not ohlcv:
        return pd.DataFrame(), {}

    # RVOL for peer z-scoring (within this universe)
    rvol_map = {p[0]: calc_rvol(ohlcv[yf_sym(p[0])])
                for p in pairs if yf_sym(p[0]) in ohlcv}
    peer_rvols = list(rvol_map.values())

    rows, tails = [], {}
    for dt, db, g in pairs:
        yt, yb = yf_sym(dt), yf_sym(db)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        try:
            df_t, df_b = ohlcv[yt], ohlcv[yb]
            ratio, mom = calc_rrg(df_t["Close"], df_b["Close"], window=window)
            if ratio is None or ratio.empty or mom.empty:
                continue

            # Align ratio & mom on common index, take tail
            common = ratio.index.intersection(mom.index)
            r_tail = ratio.loc[common].tail(tail)
            m_tail = mom.loc[common].tail(tail)
            tails[f"{dt}/{db}"] = (r_tail, m_tail)

            q_ratio, q_mom = float(r_tail.iloc[-1]), float(m_tail.iloc[-1])
            quadrant = rrg_quadrant(q_ratio, q_mom)

            rs_df    = build_rs_df(df_t["Close"], df_b["Close"])
            ad_df    = calc_ad(df_t)
            rs_accel = calc_rs_acceleration(rs_df["RS"])
            diverg   = calc_divergence(df_t["Close"], ad_df["AD"])
            accum_p  = calc_accumulation_persistence(df_t, ad_df)
            rvol_z   = rvol_zscore(rvol_map.get(dt), peer_rvols)
            early    = calc_early_rotation_score(rs_accel, rvol_z, diverg, accum_p, quadrant)

            rows.append({
                "Pair": f"{dt}/{db}", "Ticker": dt, "Group": g,
                "Quadrant": quadrant,
                "RS-Ratio": round(q_ratio,1), "RS-Mom": round(q_mom,1),
                "Early Rot": early["score"], "Tier": early["tier"],
                "RS Accel": rs_accel["accel"], "RoC": rs_accel["roc"],
                "RVOL z": rvol_z, "Diverg": diverg, "Accum wk": accum_p,
                "_detail": early["detail"],
                "_color": QUADRANT_COLOR.get(quadrant,"#888"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows), tails


# ── Run ───────────────────────────────────────────────────────────────────────
ukey = tuple(universe)
if "radar_results" not in st.session_state or run_btn or \
   st.session_state.get("radar_key") != (group_sel, rrg_window, tail_len):
    with st.spinner(f"Scanning {len(universe)} pairs for rotation…"):
        res, tails = run_radar(ukey, rrg_window, tail_len)
        st.session_state["radar_results"] = res
        st.session_state["radar_tails"]   = tails
        st.session_state["radar_key"]     = (group_sel, rrg_window, tail_len)

res:   pd.DataFrame = st.session_state.get("radar_results", pd.DataFrame())
tails: dict          = st.session_state.get("radar_tails", {})

if res.empty:
    st.warning("No rotation data. Click Run Radar.")
    st.stop()

# ── RRG Chart ─────────────────────────────────────────────────────────────────
st.subheader(f"📡 Relative Rotation Graph — {group_sel}")

fig = go.Figure()

# Quadrant background shading
fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115,
              fillcolor="rgba(0,204,102,0.07)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=100, y0=85, x1=115, y1=100,
              fillcolor="rgba(255,215,0,0.07)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=85, y0=85, x1=100, y1=100,
              fillcolor="rgba(255,68,68,0.07)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=85, y0=100, x1=100, y1=115,
              fillcolor="rgba(79,195,247,0.07)", line_width=0, layer="below")

# Quadrant labels
for x,y,txt,clr in [(112,113,"LEADING","#00cc66"),(112,87,"WEAKENING","#ffd700"),
                    (88,87,"LAGGING","#ff4444"),(88,113,"IMPROVING","#4fc3f7")]:
    fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                       font=dict(color=clr, size=11, family="monospace"))

# Center crosshair
fig.add_hline(y=100, line_color="#444", line_width=1)
fig.add_vline(x=100, line_color="#444", line_width=1)

# Tails + current points
for _, r in res.iterrows():
    pair = r["Pair"]
    if pair not in tails:
        continue
    r_tail, m_tail = tails[pair]
    clr = r["_color"]
    # Tail line
    fig.add_trace(go.Scatter(
        x=r_tail.values, y=m_tail.values, mode="lines",
        line=dict(color=clr, width=1), opacity=0.4,
        showlegend=False, hoverinfo="skip"))
    # Current marker + label
    fig.add_trace(go.Scatter(
        x=[r_tail.iloc[-1]], y=[m_tail.iloc[-1]], mode="markers+text",
        marker=dict(color=clr, size=11, line=dict(color="white", width=1)),
        text=[r["Ticker"]], textposition="top center",
        textfont=dict(size=9, color="#ddd"),
        name=r["Ticker"],
        hovertemplate=f"{pair}<br>RS-Ratio %{{x:.1f}}<br>RS-Mom %{{y:.1f}}"
                      f"<br>{r['Quadrant']}<br>Early Rot {r['Early Rot']}/10<extra></extra>",
        showlegend=False))

fig.update_layout(
    template="plotly_dark", height=560,
    xaxis=dict(title="RS-Ratio (relative strength)", range=[85,115], zeroline=False),
    yaxis=dict(title="RS-Momentum", range=[85,115], zeroline=False),
    margin=dict(l=0,r=0,t=10,b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
st.plotly_chart(fig, width="stretch")
st.caption("Money rotates clockwise: **Improving → Leading → Weakening → Lagging**. "
           "Names entering the **Improving** quadrant (top-left) with rising tails are the earliest rotation signals.")
st.divider()

# ── Early Rotation Leaderboard ────────────────────────────────────────────────
st.subheader("🔥 Early Rotation Leaderboard")
st.caption("High Early Rotation + not yet Leading = money moving in before the trend confirms.")

board = res.sort_values(["Early Rot","RS Accel"], ascending=[False,False]).copy()
disp = board[["Pair","Tier","Early Rot","Quadrant","RS Accel","RoC",
              "RVOL z","Diverg","Accum wk"]].copy()

def _tier_style(r):
    c = ROTATION_TIER_COLOR.get(r["Tier"],"#888")
    return [f"background-color:{c}1c"]*len(r)

st.dataframe(disp.style.apply(_tier_style, axis=1),
             width="stretch", hide_index=True,
             column_config={
                 "Early Rot": st.column_config.NumberColumn("Score/10"),
                 "RS Accel":  st.column_config.NumberColumn(format="%.2f"),
                 "RoC":       st.column_config.NumberColumn(format="%.1f%%"),
                 "RVOL z":    st.column_config.NumberColumn(format="%.1f"),
                 "Accum wk":  st.column_config.NumberColumn("Accum wk"),
             })

# ── Quadrant summary ──────────────────────────────────────────────────────────
st.divider()
qcols = st.columns(4)
for col, quad in zip(qcols, ["Improving","Leading","Weakening","Lagging"]):
    members = res[res["Quadrant"]==quad]["Ticker"].tolist()
    clr = QUADRANT_COLOR[quad]
    with col:
        st.markdown(f"<b style='color:{clr}'>{quad}</b> ({len(members)})",
                    unsafe_allow_html=True)
        st.caption(", ".join(members) if members else "—")

# ── Drill-down ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Early Rotation Detail")
sel = st.selectbox("Select pair", board["Pair"].tolist())
row = board[board["Pair"]==sel].iloc[0]
clr = row["_color"]
st.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{clr}22;border:1px solid {clr}'>"
    f"<b style='font-size:1.2rem'>{row['Tier']}</b> &nbsp;·&nbsp; "
    f"Early Rotation <b>{row['Early Rot']}/10</b> &nbsp;·&nbsp; "
    f"Quadrant <b>{row['Quadrant']}</b> &nbsp;·&nbsp; "
    f"RS-Ratio <b>{row['RS-Ratio']}</b> / RS-Mom <b>{row['RS-Mom']}</b>"
    f"</div>", unsafe_allow_html=True)
st.dataframe(pd.DataFrame([("✅" if v else "❌", k) for k,v in row["_detail"].items()],
             columns=["","Signal"]), width="stretch", hide_index=True)
