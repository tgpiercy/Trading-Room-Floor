"""
pages/8_Flow_Dashboard.py
Multi-day Flow Dashboard — the interpretation layer.
Trend charts + scored confluence table + plain-language verdict.

Live price-derived history (OBV/CMF/RVOL) is always available; options-derived
flow (premium, GEX) is snapshotted daily and accumulates going forward.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_fetcher import get_stock_data, get_current_price, get_options_chain
from utils.flow_analysis import (analyze_flow, save_snapshot, get_persisted,
                                 DIRECTION_COLOR, score_color)

# Defensive imports (new-ish utils; tolerate partial deploys)
try:
    from utils.order_flow import premium_flow
except Exception:
    premium_flow = None
try:
    from utils.indicators import max_pain, gamma_exposure
except Exception:
    max_pain = gamma_exposure = None
try:
    from utils.fred import get_risk_free_rate
except Exception:
    def get_risk_free_rate(default=0.045): return default

st.set_page_config(page_title="Flow Dashboard · StratFlow", page_icon="🧭", layout="wide")
st.title("🧭 Flow Dashboard")
st.caption("Multi-day flow interpretation — confluence score + verdict. "
           "Tells you what the pattern is, how strong, and what to watch.")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    window = st.slider("Analysis Window (days)", 10, 40, 20)
    st.caption("Price-flow history is live. Options-flow history builds each day "
               "you open this page for a ticker.")

with st.spinner(f"Analyzing {ticker} flow…"):
    daily = get_stock_data(ticker, period="1y", interval="1d")
    spot  = get_current_price(ticker)

if daily.empty:
    st.error(f"No data for {ticker}.")
    st.stop()

# ── Options snapshot (today) → persist for forward history ────────────────────
opt_snap = None
if premium_flow is not None:
    try:
        calls, puts, expirations = get_options_chain(ticker)
        if not calls.empty and not puts.empty:
            pf = premium_flow(calls, puts)
            gex_sign = None
            if gamma_exposure is not None and expirations:
                try:
                    g = gamma_exposure(calls, puts, spot,
                                       expiry=expirations[0], r=get_risk_free_rate())
                    if not g.empty:
                        gex_sign = "positive" if g["net_gex"].sum() > 0 else "negative"
                except Exception:
                    pass
            mp = max_pain(calls, puts) if max_pain is not None else None
            opt_snap = {"call_pct": pf["call_pct"], "put_pct": pf["put_pct"],
                        "prem_pcr": pf["prem_pcr"], "gex_sign": gex_sign,
                        "max_pain": round(mp, 2) if mp else None}
            save_snapshot(ticker, opt_snap)   # persist today's reading
    except Exception:
        opt_snap = None

# ── Run analysis ──────────────────────────────────────────────────────────────
res = analyze_flow(ticker, daily, window=window, opt_snapshot=opt_snap)
if res["pattern"] == "Insufficient Data":
    st.warning("Not enough history to analyze. Try a longer-listed ticker.")
    st.stop()

# ── Verdict header ────────────────────────────────────────────────────────────
dcol = DIRECTION_COLOR.get(res["direction"], "#888")
chg = (daily["Close"].iloc[-1]-daily["Close"].iloc[-2])/daily["Close"].iloc[-2]*100
hk1, hk2 = st.columns([1, 4])
hk1.metric(ticker, f"${spot:.2f}", f"{chg:+.2f}%")
with hk2:
    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background:{dcol}22;"
        f"border:1px solid {dcol}'>"
        f"<span style='font-size:1.3rem;font-weight:700'>{res['pattern']}</span>"
        f" &nbsp;·&nbsp; {res['direction']} &nbsp;·&nbsp; "
        f"Confidence <b>{res['confidence']}/10</b><br>"
        f"<span style='font-size:0.92rem'>{res['verdict']}</span></div>",
        unsafe_allow_html=True)

# Confidence meter
st.progress(res["confidence"] / 10)

c1, c2 = st.columns(2)
c1.success(f"**✓ Confirms thesis:** {res['confirm']}")
c2.error(f"**✗ Negates thesis:** {res['negate']}")
st.divider()

# ── Confluence table ──────────────────────────────────────────────────────────
st.subheader("📊 Confluence Scorecard")
rows = []
for name, s, note in res["dimensions"]:
    arrow = "🟢▲" if s >= 2 else "🔵△" if s == 1 else "🔴▼" if s <= -2 else "🟠▽" if s == -1 else "⚪–"
    rows.append({"Signal": name, "Read": arrow, "Score": f"{s:+d}", "Detail": note})
tbl = pd.DataFrame(rows)

def _row_color(r):
    sc = int(r["Score"])
    return [f"background-color:{score_color(sc)}1c"]*len(r)

st.dataframe(tbl.style.apply(_row_color, axis=1), width="stretch", hide_index=True)
st.caption(f"Net confluence: **{res['net']:+d}** across {len(res['dimensions'])} signals "
           f"(max ±{2*len(res['dimensions'])}). More signals agreeing = higher confidence.")
st.divider()

# ── Trend charts ──────────────────────────────────────────────────────────────
st.subheader("📈 Multi-Day Trends")
s = res["series"]
tail = window + 10
dates = s["dates"][-tail:]

# Price + OBV (absorption visual)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.5, 0.5],
                    subplot_titles=[f"{ticker} Price vs OBV (divergence = absorption)",
                                    "Chaikin Money Flow"])
fig.add_trace(go.Scatter(x=dates, y=s["close"][-tail:], name="Price",
                         line=dict(color="#e0e0e0", width=2)), row=1, col=1)
# OBV on secondary axis (normalized to price range for visual overlay)
obv_t = s["obv"][-tail:]
obv_norm = (obv_t - obv_t.min()) / (obv_t.max() - obv_t.min() + 1e-9)
px_t = s["close"][-tail:]
obv_scaled = obv_norm * (px_t.max() - px_t.min()) + px_t.min()
fig.add_trace(go.Scatter(x=dates, y=obv_scaled, name="OBV (scaled)",
                         line=dict(color="#4fc3f7", width=1.5, dash="dot")), row=1, col=1)
# CMF
cmf_t = s["cmf"][-tail:]
fig.add_trace(go.Bar(x=dates, y=cmf_t, name="CMF",
                     marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in cmf_t]),
              row=2, col=1)
fig.add_hline(y=0, line_color="white", opacity=0.25, row=2, col=1)
fig.update_layout(template="plotly_dark", height=480, showlegend=True,
                  legend=dict(orientation="h", y=1.06),
                  margin=dict(l=0, r=0, t=40, b=0),
                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig, width="stretch")
st.caption("When the blue OBV line rises while price stays flat, buyers are absorbing "
           "supply — the footprint of quiet accumulation.")

# ── Options-flow history (persisted, builds forward) ──────────────────────────
st.subheader("🎯 Options Premium Flow — History")
hist = get_persisted(ticker)
if hist.empty or "call_pct" not in hist.columns or len(hist) < 2:
    if opt_snap:
        st.info(f"📌 Today's reading saved: **{opt_snap['call_pct']:.0f}% calls**. "
                "Premium-flow *trend* needs ≥2 days — revisit this ticker daily and the "
                "history line will build automatically.")
    else:
        st.info("No options data available for this ticker to track.")
else:
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=hist.index, y=hist["call_pct"], name="% Premium in Calls",
                              line=dict(color="#00ff88", width=2), mode="lines+markers"))
    figp.add_hline(y=50, line_dash="dash", line_color="#888",
                   annotation_text="Balanced")
    figp.update_layout(template="plotly_dark", height=260,
                       yaxis=dict(title="% Calls", range=[0, 100]),
                       margin=dict(l=0, r=0, t=20, b=0),
                       paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(figp, width="stretch")
    trend = hist["call_pct"].iloc[-1] - hist["call_pct"].iloc[0]
    st.caption(f"Call-premium share has {'risen' if trend>0 else 'fallen'} "
               f"{abs(trend):.0f} pts over {len(hist)} tracked sessions "
               f"(rising = building bullish positioning).")

st.divider()
st.caption("⚠️ Interpretive estimate from free data. Options-flow history is best-effort "
           "and resets if the app redeploys; price-flow history is always rebuilt live.")
