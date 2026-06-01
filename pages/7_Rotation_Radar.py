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
from utils.data_fetcher import fetch_ohlcv_batch, get_current_price
from utils.rotation import (
    calc_rrg, rrg_quadrant, QUADRANT_COLOR, calc_rs_acceleration,
    calc_divergence, calc_accumulation_persistence, calc_rvol, rvol_zscore,
    calc_early_rotation_score, ROTATION_TIER_COLOR,
)
# Detect which rotation engine is ACTUALLY running by inspecting calc_rrg's
# signature — no dependency on any importable name, so the page can never
# crash even if an old rotation.py is deployed. The stamp tells the truth.
import inspect as _inspect
_rrg_params = _inspect.signature(calc_rrg).parameters
if "presmooth" in _rrg_params:
    RRG_ENGINE_VERSION = "v2.1"
elif "smooth" in _rrg_params:
    RRG_ENGINE_VERSION = "v2.0"
else:
    RRG_ENGINE_VERSION = "v1 — rotation.py is STALE, update utils/rotation.py"
from utils.strategy import calc_ad
from utils.rs_indicators import build_rs_df

# Multi-horizon gamma (on-demand grid) — tolerate stale deploy
try:
    from utils.indicators import gamma_horizons, gamma_regime_label
    from utils.data_fetcher import get_options_chains_multi
    _GH_OK = True
except Exception:
    _GH_OK = False


# ── Heatmap cell colors (manual CSS — no matplotlib dependency) ────────────────
def _lerp(t, c0, c1):
    t = max(0.0, min(1.0, t))
    return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))

def _css_early(v):
    try: v = float(v)
    except Exception: return ""
    r, g, b = _lerp(v / 10.0, (200, 60, 60), (0, 200, 100))
    return f"background-color: rgba({r},{g},{b},0.40)"

def _css_mom(v):
    try: v = float(v)
    except Exception: return ""
    r, g, b = _lerp((v - 95) / 10.0, (200, 60, 60), (0, 200, 100))
    return f"background-color: rgba({r},{g},{b},0.32)"

def _css_accel(v):
    try: v = float(v)
    except Exception: return ""
    return ("background-color: rgba(0,200,100,0.32)" if v > 0
            else "background-color: rgba(200,60,60,0.28)")

def _css_diverg(v):
    return {"Bullish": "background-color: rgba(0,200,100,0.40)",
            "Bearish": "background-color: rgba(200,60,60,0.32)"}.get(v, "")

def _css_quad(v):
    return f"background-color: {QUADRANT_COLOR.get(v, '#888')}55"


def _build_gamma_grid(universe):
    """On-demand: per-ticker gamma tilt for 1wk/2wk/1mo. Rate-limit tolerant."""
    rows = []
    prog = st.progress(0.0, text="Fetching options chains…")
    n = len(universe)
    for i, (t, b, g) in enumerate(universe):
        prog.progress((i + 1) / n, text=f"Gamma {t} ({i+1}/{n})…")
        yt = yf_sym(t)
        rec = {"Ticker": t, "Group": g, "1wk": np.nan, "2wk": np.nan, "1mo": np.nan}
        try:
            chains = get_options_chains_multi(yt, max_days=35)
            if chains:
                spot = float(get_current_price(yt))
                tl = gamma_horizons(chains, spot)["bucket_tilt"]
                rec.update({"1wk": tl.get(7), "2wk": tl.get(14), "1mo": tl.get(30)})
        except Exception:
            pass
        rows.append(rec)
    prog.empty()
    return pd.DataFrame(rows)

st.set_page_config(page_title="Rotation Radar · StratFlow", page_icon="🛰️", layout="wide")
st.title("🛰️ Rotation Radar")
st.caption("RRG map + Early Rotation score · catching money flow before the trend confirms")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    group_sel = st.selectbox("RRG Universe",
                             ["Sectors", "Themes", "Industries · XLK",
                              "Industries · HXT", "All Top-Level"], index=0)
    tail_len = st.slider("Tail Length (weeks)", 3, 12, 5)
    rrg_window = st.slider("Normalization Window", 8, 20, 14,
                           help="Longer = smoother, slower to react")
    rrg_smooth = st.slider("Smoothing", 1, 8, 4,
                           help="Higher = smoother tails, more lag")
    show_tails = st.checkbox("Show rotation tails", value=True)
    spline = st.checkbox("Curved (spline) tails", value=True)
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
def run_radar(universe_key: tuple, window: int, tail: int, smooth: int):
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
            # Call adapts to whichever rotation.py is deployed (old sig has no smooth)
            if "smooth" in _rrg_params:
                ratio, mom = calc_rrg(df_t["Close"], df_b["Close"],
                                      window=window, smooth=smooth)
            else:
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR ROTATION DASHBOARD — top-down big picture (always on combined universe)
# ══════════════════════════════════════════════════════════════════════════════
st.header("📊 Sector Rotation Dashboard")
st.caption("Big-picture rotation + flow across indices, sectors & themes. Spot early "
           "rotation here → **Screener** for entries → **Flow Dashboard** to confirm one name.")

_dash_universe = tuple((t, b, g) for t, b, g in PORTFOLIO
                       if g in ("INDICES", "SECTORS", "THEMES"))
with st.spinner("Scanning rotation + flow…"):
    dash_df, _ = run_radar(_dash_universe, rrg_window, tail_len, rrg_smooth)

if dash_df.empty:
    st.info("Rotation scan unavailable (data or rate limit). Try again shortly.")
else:
    hot = dash_df.sort_values("Early Rot", ascending=False)
    cols = ["Ticker", "Group", "Quadrant", "RS-Mom", "RS Accel",
            "RVOL z", "Diverg", "Accum wk", "Early Rot"]
    show = hot[cols].copy()
    try:
        # pandas ≥2.1 uses Styler.map; older uses .applymap. Prefer the modern one.
        sty = show.style
        _cell = sty.map if hasattr(sty, "map") else sty.applymap
        _cell(_css_quad, subset=["Quadrant"])
        _cell(_css_mom, subset=["RS-Mom"])
        _cell(_css_accel, subset=["RS Accel"])
        _cell(_css_diverg, subset=["Diverg"])
        _cell(_css_early, subset=["Early Rot"])
        st.dataframe(sty, width="stretch", hide_index=True, height=460)
    except Exception:
        st.dataframe(show, width="stretch", hide_index=True, height=460)
    st.caption("Read for early rotation: **Improving** quadrant + **RS Accel > 0** + "
               "**Bullish** divergence + rising **Early Rot** = money moving in before the "
               "trend confirms. Leading + Weakening with negative accel = rotating out.")

    # ── On-demand gamma grid ──────────────────────────────────────────────────
    st.subheader("⚡ Gamma Regime Grid")
    if not _GH_OK:
        st.caption("Needs updated utils/indicators.py + utils/data_fetcher.py "
                   "(gamma_horizons + get_options_chains_multi).")
    else:
        st.caption("Dealer gamma tilt per name over 1wk / 2wk / 1mo. Loads on demand — "
                   "fetches options chains, so it's slower and can be partially "
                   "rate-limited (blank cells = unavailable this run; cached 15 min).")
        if st.button("⚡ Load gamma grid", width="stretch"):
            st.session_state["load_gamma_grid"] = True
        if st.session_state.get("load_gamma_grid"):
            gg = _build_gamma_grid(_dash_universe)
            gg2 = gg.dropna(how="all", subset=["1wk", "2wk", "1mo"])
            if gg2.empty:
                st.warning("No gamma data returned (likely rate-limited). Wait ~1 min and retry.")
            else:
                z = gg2[["1wk", "2wk", "1mo"]].astype(float).values
                emoji = [[gamma_regime_label(v)[0] if pd.notna(v) else "·" for v in row]
                         for row in z]
                figg = go.Figure(go.Heatmap(
                    z=z, x=["1wk", "2wk", "1mo"], y=list(gg2["Ticker"]),
                    colorscale=[[0, "#ff4444"], [0.5, "#2a2a2a"], [1, "#00cc66"]],
                    zmid=0, zmin=-0.3, zmax=0.3, text=emoji, texttemplate="%{text}",
                    colorbar=dict(title="tilt"), xgap=2, ygap=2))
                figg.update_layout(template="plotly_dark",
                                   height=max(320, 24 * len(gg2)),
                                   margin=dict(l=0, r=0, t=10, b=0),
                                   paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
                st.plotly_chart(figg, width="stretch")
                st.caption("🟢 long gamma (pinning / mean-revert) · 🔴 short gamma "
                           "(squeezy / trending) · 🟡 near-flat. Near-dated (1wk) carries "
                           "the most hedging force. Estimate from Yahoo IV/OI under the "
                           "standard dealer convention — landscape, not the confirmed book.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# RRG EXPLORER (sidebar-driven, single group)
# ══════════════════════════════════════════════════════════════════════════════


# ── Run ───────────────────────────────────────────────────────────────────────
ukey = tuple(universe)
_key = (group_sel, rrg_window, tail_len, rrg_smooth)
if "radar_results" not in st.session_state or run_btn or \
   st.session_state.get("radar_key") != _key:
    with st.spinner(f"Scanning {len(universe)} pairs for rotation…"):
        res, tails = run_radar(ukey, rrg_window, tail_len, rrg_smooth)
        st.session_state["radar_results"] = res
        st.session_state["radar_tails"]   = tails
        st.session_state["radar_key"]     = _key

res:   pd.DataFrame = st.session_state.get("radar_results", pd.DataFrame())
tails: dict          = st.session_state.get("radar_tails", {})

if res.empty:
    st.warning("No rotation data. Click Run Radar.")
    st.stop()

# ── RRG Chart ─────────────────────────────────────────────────────────────────
st.subheader(f"📡 Relative Rotation Graph — {group_sel}")
st.caption(f"⚙️ RRG engine **{RRG_ENGINE_VERSION}** · window {rrg_window} · "
           f"smoothing {rrg_smooth} · tail {tail_len}w · "
           f"{'spline' if spline else 'straight'} tails — "
           f"if this stamp didn't change after deploying, hit **Run Radar** to clear the cache")

fig = go.Figure()

# ── Auto-range: symmetric around 100, fit to data with padding ────────────────
_all_vals = []
for pr in res["Pair"]:
    if pr in tails:
        rt, mt = tails[pr]
        _all_vals.extend(rt.values.tolist())
        _all_vals.extend(mt.values.tolist())
if _all_vals:
    _max_dev = max(abs(v - 100) for v in _all_vals)
    _max_dev = max(_max_dev * 1.18, 1.5)   # padding + floor so quadrants show
else:
    _max_dev = 5
_lo, _hi = 100 - _max_dev, 100 + _max_dev

# Quadrant background shading (full visible area)
fig.add_shape(type="rect", x0=100, y0=100, x1=_hi, y1=_hi,
              fillcolor="rgba(0,204,102,0.06)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=100, y0=_lo, x1=_hi, y1=100,
              fillcolor="rgba(255,215,0,0.06)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=_lo, y0=_lo, x1=100, y1=100,
              fillcolor="rgba(255,68,68,0.06)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=_lo, y0=100, x1=100, y1=_hi,
              fillcolor="rgba(79,195,247,0.06)", line_width=0, layer="below")

# Quadrant labels (corners)
_pad = _max_dev * 0.08
for x,y,txt,clr in [(_hi-_pad,_hi-_pad,"LEADING","#00cc66"),
                    (_hi-_pad,_lo+_pad,"WEAKENING","#ffd700"),
                    (_lo+_pad,_lo+_pad,"LAGGING","#ff4444"),
                    (_lo+_pad,_hi-_pad,"IMPROVING","#4fc3f7")]:
    fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                       font=dict(color=clr, size=11, family="monospace"))

# Center crosshair
fig.add_hline(y=100, line_color="#555", line_width=1)
fig.add_vline(x=100, line_color="#555", line_width=1)

_shape = "spline" if spline else "linear"

# Tails + current points
for _, r in res.iterrows():
    pair = r["Pair"]
    if pair not in tails:
        continue
    r_tail, m_tail = tails[pair]
    clr = r["_color"]
    if show_tails and len(r_tail) > 1:
        # Smooth tail line
        fig.add_trace(go.Scatter(
            x=r_tail.values, y=m_tail.values, mode="lines",
            line=dict(color=clr, width=1.5, shape=_shape), opacity=0.45,
            showlegend=False, hoverinfo="skip"))
        # Small fading dots along the tail (oldest faint → newest solid)
        n = len(r_tail)
        fig.add_trace(go.Scatter(
            x=r_tail.values[:-1], y=m_tail.values[:-1], mode="markers",
            marker=dict(color=clr, size=4,
                        opacity=[0.25 + 0.5*i/max(n-1,1) for i in range(n-1)]),
            showlegend=False, hoverinfo="skip"))
    # Current marker + label
    fig.add_trace(go.Scatter(
        x=[r_tail.iloc[-1]], y=[m_tail.iloc[-1]], mode="markers+text",
        marker=dict(color=clr, size=13, line=dict(color="white", width=1.2)),
        text=[r["Ticker"]], textposition="top center",
        textfont=dict(size=9, color="#ddd"),
        name=r["Ticker"],
        hovertemplate=f"{pair}<br>RS-Ratio %{{x:.1f}}<br>RS-Mom %{{y:.1f}}"
                      f"<br>{r['Quadrant']}<br>Early Rot {r['Early Rot']}/10<extra></extra>",
        showlegend=False))

fig.update_layout(
    template="plotly_dark", height=580,
    xaxis=dict(title="RS-Ratio (relative strength)", range=[_lo,_hi], zeroline=False),
    yaxis=dict(title="RS-Momentum", range=[_lo,_hi], zeroline=False,
               scaleanchor="x", scaleratio=1),
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
