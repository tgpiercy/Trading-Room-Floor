"""
pages/2_Flow.py
Unified Flow hub — money flow, options flow, positioning, intraday in one place.
One ticker drives all views. View-mode selector + Flow Summary synthesis strip.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_fetcher import (get_stock_data, get_ticker_info, get_options_chain,
                                 get_current_price, get_intraday_data)
from utils.indicators import (obv, mfi, cmf, relative_volume, force_index, atr,
                              pcr, max_pain, gamma_exposure)
# Defensive: gamma_flip only exists in the updated indicators.py. If an older
# copy is deployed (e.g. page pushed before the util), fall back to an inline
# version so the page still loads instead of hard-crashing.
try:
    from utils.indicators import gamma_flip
except ImportError:
    import numpy as _np
    def gamma_flip(gex_df):
        if gex_df.empty or "net_gex" not in gex_df.columns or len(gex_df) < 2:
            return None
        g = gex_df.sort_values("strike").reset_index(drop=True)
        strikes, net = g["strike"].values, g["net_gex"].values
        crossings = []
        for i in range(1, len(net)):
            if (net[i-1] < 0 <= net[i]) or (net[i-1] > 0 >= net[i]):
                x0, x1 = strikes[i-1], strikes[i]
                y0, y1 = net[i-1], net[i]
                x = x0 - y0*(x1-x0)/(y1-y0) if y1 != y0 else x1
                crossings.append(float(x))
        if not crossings:
            return None
        mid = strikes[len(strikes)//2]
        return round(min(crossings, key=lambda x: abs(x-mid)), 2)
from utils.order_flow import (premium_flow, new_positioning, detect_sweeps,
                              volume_by_price, auction_analysis, vwap_deviation)
# Multi-horizon gamma (new): tolerate stale deploy
try:
    from utils.indicators import gamma_horizons, gamma_regime_label, greek_exposures
    from utils.data_fetcher import get_options_chains_multi
    _GH_OK = True
except Exception:
    _GH_OK = False
# Dealer greeks (DEX / Charm / Vanna) — new in indicators.py
try:
    from utils.indicators import greek_exposures
    _GREEKS_OK = True
except Exception:
    _GREEKS_OK = False
from utils.chart_utils import set_chart_window
# Flow Dashboard interpretation (merged in): verdict + confluence synthesis
try:
    from utils.flow_analysis import (analyze_flow, save_snapshot, get_persisted,
                                     DIRECTION_COLOR, score_color)
    _FA_OK = True
except Exception:
    _FA_OK = False
# Defensive: fred.py is a new file; if not yet deployed, GEX uses a default rate.
try:
    from utils.fred import get_risk_free_rate
except ImportError:
    def get_risk_free_rate(default=0.045):
        return default
# Detect whether the deployed gamma_exposure supports the new IV-based signature
import inspect as _inspect
_GEX_NEW = "expiry" in _inspect.signature(gamma_exposure).parameters

st.set_page_config(page_title="Flow · StratFlow", page_icon="🌊", layout="wide")
st.title("🌊 Flow")
st.caption("Unified money flow · options flow · positioning · intraday — one ticker, all views")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    view = st.selectbox("View", ["Summary", "Money Flow", "Options Flow",
                                  "Positioning", "Gamma", "Intraday"], index=0)
    st.divider()
    if view == "Money Flow":
        period   = st.selectbox("History", ["1y","2y","5y"], index=1)
        interval = st.selectbox("Interval", ["1d","1wk"], index=0)
        mfi_p  = st.slider("MFI Period", 7, 30, 14)
        cmf_p  = st.slider("CMF Period", 10, 40, 20)
        rvol_p = st.slider("RVOL Lookback", 10, 40, 20)
    elif view in ("Options Flow","Positioning"):
        strike_range = st.slider("Strike Range ± %", 5, 35, 15)
        if view == "Options Flow":
            vol_oi_thresh = st.slider("Min Vol/OI", 1.0, 10.0, 2.0, step=0.5)
            min_vol       = st.number_input("Min Volume", 0, value=100, step=50)
            min_prem      = st.number_input("Min New-Position Premium $", 0, value=25000, step=5000)
    elif view == "Intraday":
        intra_int = st.selectbox("Interval", ["5m","15m","30m","60m"], index=0)
        intra_days= st.slider("Days", 1, 10, 5)
        vp_bins   = st.slider("Profile Bins", 12, 40, 24)

# ── Shared data fetch (all cached) ────────────────────────────────────────────
with st.spinner(f"Loading {ticker}…"):
    daily = get_stock_data(ticker, period="2y", interval="1d")
    spot  = get_current_price(ticker)
    info  = get_ticker_info(ticker)

if daily.empty:
    st.error(f"No data for {ticker}. Check the symbol.")
    st.stop()

# ── Flow Summary synthesis strip ──────────────────────────────────────────────
def build_summary():
    parts = []
    # Money flow (from daily)
    try:
        d = obv(daily.copy()); d = mfi(d); d = cmf(d); d = relative_volume(d)
        last = d.dropna().iloc[-1]
        obv_trend = d["OBV"].iloc[-1] - d["OBV"].iloc[-10]
        parts.append("🟢 OBV accumulating" if obv_trend > 0 else "🔴 OBV distributing")
        parts.append(f"MFI {last['MFI']:.0f}")
        parts.append(f"CMF {'+' if last['CMF']>=0 else ''}{last['CMF']:.2f}")
        if last["RVOL"] > 1.5:
            parts.append(f"RVOL {last['RVOL']:.1f}× 🔥")
    except Exception:
        pass
    # Options premium flow
    try:
        c, p, _ = get_options_chain(ticker)
        if not c.empty and not p.empty:
            pf = premium_flow(c, p)
            parts.append(f"Premium {pf['call_pct']:.0f}% calls")
            mp = max_pain(c, p)
            if mp:
                parts.append(f"Max Pain ${mp:.0f}")
    except Exception:
        pass
    # Intraday auction
    try:
        intra = get_intraday_data(ticker, interval="5m", days=3)
        auc = auction_analysis(intra)
        if auc and auc.get("signal","Neutral") != "Neutral":
            short = (auc["signal"].split("(")[0]).strip()
            parts.append(f"⏱ {short}")
    except Exception:
        pass
    return parts

with st.spinner("Synthesizing flow…"):
    summary = build_summary()

chg = (daily["Close"].iloc[-1] - daily["Close"].iloc[-2]) / daily["Close"].iloc[-2] * 100
hk1, hk2 = st.columns([1, 4])
hk1.metric(ticker, f"${spot:.2f}", f"{chg:+.2f}%")
with hk2:
    st.caption(f"**{info.get('longName', ticker)}**")
    if summary:
        st.markdown(
            "<div style='padding:8px;border-radius:6px;background:#1e2130;"
            "font-size:0.9rem;line-height:1.8'>🌊 <b>Flow Read:</b> &nbsp;"
            + " &nbsp;·&nbsp; ".join(summary) + "</div>", unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: SUMMARY (merged Flow Dashboard — verdict + confluence + multi-day trend)
# ══════════════════════════════════════════════════════════════════════════════
if view == "Summary":
    st.subheader("🧭 Flow Verdict")
    st.caption("What the flow says, how strong, and what to watch — a confluence "
               "synthesis across price, OBV, CMF, accumulation and options premium.")
    if not _FA_OK:
        st.warning("⚠️ Needs **utils/flow_analysis.py**. Push that util, then retry.")
        st.stop()

    # Options snapshot today → persist forward for premium-flow history
    opt_snap = None
    try:
        calls_s, puts_s, expirations_s = get_options_chain(ticker)
        if not calls_s.empty and not puts_s.empty:
            pf = premium_flow(calls_s, puts_s)
            gex_sign = None; gtilt = None
            if expirations_s:
                try:
                    g = gamma_exposure(calls_s, puts_s, spot,
                                       expiry=expirations_s[0], r=get_risk_free_rate()) \
                        if _GEX_NEW else gamma_exposure(calls_s, puts_s, spot)
                    if not g.empty:
                        net_g = g["net_gex"].sum()
                        gex_sign = "positive" if net_g > 0 else "negative"
                        if {"call_gex", "put_gex"}.issubset(g.columns):
                            gross = g["call_gex"].abs().sum() + g["put_gex"].abs().sum()
                            gtilt = float(net_g / gross) if gross else None
                except Exception:
                    pass
            mp = max_pain(calls_s, puts_s)
            opt_snap = {"call_pct": pf["call_pct"], "put_pct": pf["put_pct"],
                        "prem_pcr": pf["prem_pcr"], "gex_sign": gex_sign,
                        "max_pain": round(mp, 2) if mp else None,
                        "gamma_1wk": gtilt, "spot": float(spot)}
            # Dealer greeks → DEX tilt (directional) + charm (pin context)
            if _GREEKS_OK and expirations_s:
                try:
                    ge = greek_exposures(calls_s, puts_s, spot,
                                         expiry=expirations_s[0], r=get_risk_free_rate())
                    opt_snap["dex_tilt"] = ge.get("dex_tilt")
                    opt_snap["charm"] = ge.get("net_charm")
                except Exception:
                    pass
            save_snapshot(ticker, opt_snap)
    except Exception:
        opt_snap = None

    res = analyze_flow(ticker, daily, window=20, opt_snapshot=opt_snap)
    if res["pattern"] == "Insufficient Data":
        st.warning("Not enough history to analyze this ticker.")
        st.stop()

    # Verdict header
    dcol = DIRECTION_COLOR.get(res["direction"], "#888")
    chg = (daily["Close"].iloc[-1] - daily["Close"].iloc[-2]) / daily["Close"].iloc[-2] * 100
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
    st.progress(res["confidence"] / 10)
    c1, c2 = st.columns(2)
    c1.success(f"**✓ Confirms:** {res['confirm']}")
    c2.error(f"**✗ Negates:** {res['negate']}")
    st.divider()

    # Confluence scorecard (grouped: Price/Volume + Options)
    st.subheader("📊 Confluence Scorecard")
    g = res.get("groups", {})
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Price/Volume Flow", f"{g.get('price_volume', 0):+.1f}/10")
    gc2.metric("Options Flow",
               f"{g['options']:+.1f}/10" if g.get("options") is not None else "n/a")
    mod = res.get("modulators", {})
    gc3.metric("Participation (RVOL)", f"{mod.get('rvol', 1):.1f}×")
    rows = []
    for group, name, s, note in res["dimensions"]:
        arrow = ("🟢▲" if s >= 2 else "🔵△" if s == 1 else "🔴▼" if s <= -2
                 else "🟠▽" if s == -1 else "⚪–")
        rows.append({"Group": group, "Signal": name, "Read": arrow,
                     "Score": f"{s:+d}", "Detail": note})
    tbl = pd.DataFrame(rows)
    try:
        st.dataframe(tbl.style.apply(
            lambda r: [f"background-color:{score_color(int(r['Score']))}1c"] * len(r),
            axis=1), width="stretch", hide_index=True)
    except Exception:
        st.dataframe(tbl, width="stretch", hide_index=True)
    gnote = mod.get("gamma_note")
    st.caption(f"Net flow score **{res['net']:+d}/10**. Price/Volume and Options scored "
               "separately then blended (60/40). "
               + (f"Dealer gamma modulates conviction: _{gnote}_." if gnote else
                  "Gamma/RVOL modulate conviction when available."))
    st.divider()

    # Multi-day trend (price vs OBV absorption + CMF)
    st.subheader("📈 Multi-Day Trends")
    s = res["series"]; tail = 30
    dates = s["dates"][-tail:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{ticker} Price vs OBV (divergence = absorption)",
                                        "Chaikin Money Flow"])
    fig.add_trace(go.Scatter(x=dates, y=s["close"][-tail:], name="Price",
                             line=dict(color="#e0e0e0", width=2)), row=1, col=1)
    obv_t = s["obv"][-tail:]
    obv_norm = (obv_t - obv_t.min()) / (obv_t.max() - obv_t.min() + 1e-9)
    px_t = s["close"][-tail:]
    obv_scaled = obv_norm * (px_t.max() - px_t.min()) + px_t.min()
    fig.add_trace(go.Scatter(x=dates, y=obv_scaled, name="OBV (scaled)",
                             line=dict(color="#4fc3f7", width=1.5, dash="dot")), row=1, col=1)
    cmf_t = s["cmf"][-tail:]
    fig.add_trace(go.Bar(x=dates, y=cmf_t, name="CMF",
                         marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in cmf_t]),
                  row=2, col=1)
    fig.add_hline(y=0, line_color="white", opacity=0.25, row=2, col=1)
    fig.update_layout(template="plotly_dark", height=480,
                      legend=dict(orientation="h", y=1.06),
                      margin=dict(l=0, r=0, t=40, b=0),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue OBV rising while price stays flat = buyers absorbing supply "
               "(quiet accumulation).")

    # Persisted options premium-flow history
    st.subheader("🎯 Options Premium Flow — History")
    hist = get_persisted(ticker)
    if hist.empty or "call_pct" not in hist.columns or len(hist) < 2:
        if opt_snap:
            st.info(f"📌 Today saved: **{opt_snap['call_pct']:.0f}% calls**. Trend needs "
                    "≥2 days — revisit daily and the line builds automatically.")
        else:
            st.info("No options data available to track for this ticker.")
    else:
        figp = go.Figure(go.Scatter(x=hist.index, y=hist["call_pct"],
                         name="% Premium in Calls", line=dict(color="#00ff88", width=2),
                         mode="lines+markers"))
        figp.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="Balanced")
        figp.update_layout(template="plotly_dark", height=260,
                           yaxis=dict(title="% Calls", range=[0, 100]),
                           margin=dict(l=0, r=0, t=20, b=0),
                           paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(figp, width="stretch")
        trend = hist["call_pct"].iloc[-1] - hist["call_pct"].iloc[0]
        st.caption(f"Call-premium share has {'risen' if trend>0 else 'fallen'} "
                   f"{abs(trend):.0f} pts over {len(hist)} sessions.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: MONEY FLOW
# ══════════════════════════════════════════════════════════════════════════════
if view == "Money Flow":
    df = get_stock_data(ticker, period=period, interval=interval)
    df = obv(df); df = mfi(df, period=mfi_p); df = cmf(df, period=cmf_p)
    df = relative_volume(df, period=rvol_p); df = force_index(df); df = atr(df)
    last, prev = df.iloc[-1], df.iloc[-2]

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("MFI", f"{last['MFI']:.1f}",
              "Overbought" if last["MFI"]>80 else ("Oversold" if last["MFI"]<20 else "Neutral"))
    k2.metric("CMF", f"{last['CMF']:.3f}", "Bullish" if last["CMF"]>0 else "Bearish")
    k3.metric("RVOL", f"{last['RVOL']:.2f}×",
              "High" if last["RVOL"]>1.5 else ("Low" if last["RVOL"]<0.7 else "Normal"))
    k4.metric("Force Idx", f"{last['FI']:,.0f}")
    k5.metric("Volume", f"{int(last['Volume']):,}")

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.32,0.17,0.17,0.17,0.17],
                        subplot_titles=["Price & Volume","OBV","MFI","CMF","RVOL"])
    bar_c = ["#00ff88" if c>=o else "#ff4444" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Price",
                             line=dict(color="#00ff88",width=2)),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=bar_c,opacity=0.3,
                         yaxis="y2",showlegend=False),row=1,col=1)
    fig.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False,showticklabels=False))
    oc = "#00ff88" if df["OBV"].iloc[-1]>df["OBV"].iloc[-20] else "#ff4444"
    fig.add_trace(go.Scatter(x=df.index,y=df["OBV"],name="OBV",line=dict(color=oc,width=1.5)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MFI"],name="MFI",line=dict(color="#ffd700",width=1.5)),row=3,col=1)
    for lvl,c in [(80,"red"),(20,"lime")]:
        fig.add_hline(y=lvl,line_dash="dash",line_color=c,opacity=0.35,row=3,col=1)
    fig.update_yaxes(range=[0,100],row=3,col=1)
    cmf_c=["#00ff88" if v>=0 else "#ff4444" for v in df["CMF"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index,y=df["CMF"].fillna(0),marker_color=cmf_c,opacity=0.8,name="CMF"),row=4,col=1)
    fig.add_hline(y=0,line_color="white",opacity=0.3,row=4,col=1)
    rv_c=["#ff8c00" if v>2 else ("#00ff88" if v>1.2 else "#888") for v in df["RVOL"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index,y=df["RVOL"].fillna(0),marker_color=rv_c,opacity=0.85,name="RVOL"),row=5,col=1)
    fig.add_hline(y=1.0,line_color="white",opacity=0.3,row=5,col=1)
    fig.update_layout(template="plotly_dark",height=820,xaxis_rangeslider_visible=False,
                      showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
    set_chart_window(fig)
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: OPTIONS FLOW
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Options Flow":
    calls, puts, expirations = get_options_chain(ticker)
    if calls.empty or puts.empty:
        st.error(f"No options data for {ticker}.")
        st.stop()
    expiry = st.selectbox("Expiration",
        expirations, format_func=lambda x: f"{x} ({(pd.Timestamp(x)-pd.Timestamp.today()).days}d)")
    calls, puts, _ = get_options_chain(ticker, expiry=expiry)

    ratios = pcr(calls, puts)
    pf = premium_flow(calls, puts)
    sw = detect_sweeps(calls, puts, spot)

    st.subheader("💰 Premium-Weighted Flow")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Call Premium", f"${pf['call_premium']/1e6:.2f}M", f"{pf['call_pct']:.0f}%")
    c2.metric("Put Premium",  f"${pf['put_premium']/1e6:.2f}M",  f"{pf['put_pct']:.0f}%")
    c3.metric("Premium P/C",  f"{pf['prem_pcr']:.2f}", "Bearish $" if pf['prem_pcr']>1 else "Bullish $")
    c4.metric("P/C Vol",      f"{ratios['pcr_volume']:.2f}", "Bearish" if ratios['pcr_volume']>1 else "Bullish")

    fig_p = go.Figure(go.Bar(x=["Call $","Put $"],y=[pf['call_premium'],pf['put_premium']],
                             marker_color=["#00ff88","#ff4444"],
                             text=[f"${pf['call_premium']/1e6:.1f}M",f"${pf['put_premium']/1e6:.1f}M"],
                             textposition="outside"))
    fig_p.update_layout(template="plotly_dark",height=200,showlegend=False,
                        margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
    st.plotly_chart(fig_p, width="stretch")
    st.caption(f"**Sweep cluster:** {sw['bias']} · {sw['call_strikes_hot']} hot call / "
               f"{sw['put_strikes_hot']} hot put strikes")

    st.subheader("🆕 New Positioning · Vol > OI")
    np_df = new_positioning(calls, puts, min_premium=min_prem)
    if np_df.empty:
        st.info(f"No new positioning ≥${min_prem:,} premium on this expiry.")
    else:
        disp = np_df.copy()
        disp["premium"]=(disp["premium"]/1e3).round(0)
        if "impliedVolatility" in disp.columns:
            disp["impliedVolatility"]=(disp["impliedVolatility"]*100).round(1)
        disp.rename(columns={"type":"Type","strike":"Strike","lastPrice":"Last","volume":"Vol",
                             "openInterest":"OI","premium":"Prem$K","impliedVolatility":"IV%",
                             "inTheMoney":"ITM"},inplace=True)
        def _s(r):
            c="rgba(0,255,136,0.10)" if r.get("Type")=="CALL" else "rgba(255,68,68,0.10)"
            return [f"background-color:{c}"]*len(r)
        st.dataframe(disp.style.apply(_s,axis=1),width="stretch",hide_index=True,
                     column_config={"Prem$K":st.column_config.NumberColumn(format="$%.0fK"),
                                    "Last":st.column_config.NumberColumn(format="$%.2f")})

    # IV skew
    st.subheader(f"📐 IV Skew · ±{strike_range}%")
    lo,hi = spot*(1-strike_range/100), spot*(1+strike_range/100)
    c_iv=calls[(calls["strike"]>=lo)&(calls["strike"]<=hi)]
    p_iv=puts[(puts["strike"]>=lo)&(puts["strike"]<=hi)]
    fig_iv=go.Figure()
    if not c_iv.empty:
        fig_iv.add_trace(go.Scatter(x=c_iv["strike"],y=c_iv["impliedVolatility"]*100,
                         mode="lines+markers",name="Calls",line=dict(color="#00ff88",width=2)))
    if not p_iv.empty:
        fig_iv.add_trace(go.Scatter(x=p_iv["strike"],y=p_iv["impliedVolatility"]*100,
                         mode="lines+markers",name="Puts",line=dict(color="#ff4444",width=2)))
    fig_iv.add_vline(x=spot,line_dash="dash",line_color="#ffd700",
                     annotation_text=f"${spot:.2f}")
    fig_iv.update_layout(template="plotly_dark",height=300,xaxis_title="Strike",
                         yaxis_title="IV %",legend=dict(orientation="h",y=1.02),
                         margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
    st.plotly_chart(fig_iv, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: POSITIONING
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Positioning":
    calls, puts, expirations = get_options_chain(ticker)
    if calls.empty:
        st.error(f"No options data for {ticker}.")
        st.stop()
    expiry = st.selectbox("Expiration",
        expirations, format_func=lambda x: f"{x} ({(pd.Timestamp(x)-pd.Timestamp.today()).days}d)")
    calls, puts, _ = get_options_chain(ticker, expiry=expiry)

    lo,hi = spot*(1-strike_range/100), spot*(1+strike_range/100)
    cf=calls[(calls["strike"]>=lo)&(calls["strike"]<=hi)].copy()
    pf2=puts[(puts["strike"]>=lo)&(puts["strike"]<=hi)].copy()
    mp = max_pain(calls, puts)
    r  = pcr(calls, puts)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Spot", f"${spot:.2f}")
    k2.metric("Max Pain", f"${mp:.2f}", f"{((mp-spot)/spot*100):+.1f}%")
    k3.metric("P/C OI", f"{r['pcr_oi']:.2f}", "Bearish" if r['pcr_oi']>1 else "Bullish")
    k4.metric("Total OI", f"{r['call_oi']+r['put_oi']:,}")

    # OI by strike
    st.subheader(f"📊 Open Interest by Strike · ±{strike_range}%")
    oi=pd.merge(cf[["strike","openInterest"]].rename(columns={"openInterest":"call_oi"}),
                pf2[["strike","openInterest"]].rename(columns={"openInterest":"put_oi"}),
                on="strike",how="outer").fillna(0).sort_values("strike")
    fig_oi=go.Figure()
    fig_oi.add_trace(go.Bar(x=oi["strike"],y=oi["call_oi"],name="Call OI",marker_color="#00ff88",opacity=0.85))
    fig_oi.add_trace(go.Bar(x=oi["strike"],y=-oi["put_oi"],name="Put OI",marker_color="#ff4444",opacity=0.85))
    fig_oi.add_vline(x=spot,line_dash="dash",line_color="#ffd700",annotation_text=f"Spot ${spot:.2f}")
    fig_oi.add_vline(x=mp,line_dash="dot",line_color="#00bfff",annotation_text=f"Max Pain ${mp:.2f}")
    fig_oi.update_layout(template="plotly_dark",barmode="overlay",height=360,
                         xaxis_title="Strike",yaxis_title="OI (+Calls/−Puts)",
                         legend=dict(orientation="h",y=1.02),margin=dict(l=0,r=0,t=30,b=0),
                         paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
    st.plotly_chart(fig_oi, width="stretch")

    # GEX — gamma computed from implied volatility via Black-Scholes
    st.subheader("⚡ Gamma Exposure (GEX)")
    if not _GEX_NEW:
        st.warning("⚠️ GEX needs the updated **utils/indicators.py** (the deployed copy "
                   "is the old version). Push indicators.py to enable gamma exposure.")
    else:
        st.caption("Gamma reconstructed from implied volatility (Yahoo has no native Greeks). "
                   "Calls +γ, puts −γ; expressed as $ per 1% spot move.")
        st.caption(f"Risk-free rate (3mo T-bill via FRED): {get_risk_free_rate()*100:.2f}%")
        rf = get_risk_free_rate()
        gex = gamma_exposure(cf, pf2, spot, expiry=expiry, r=rf)
        if gex.empty:
            st.info("Not enough valid IV/OI data on this expiry to compute GEX. "
                    "Try a nearer expiry or a more liquid underlying.")
        else:
            flip = gamma_flip(gamma_exposure(calls, puts, spot, expiry=expiry, r=rf))
            fig_g=go.Figure()
            fig_g.add_trace(go.Bar(x=gex["strike"],y=gex["call_gex"],name="Call GEX",marker_color="#00ff88",opacity=0.6))
            fig_g.add_trace(go.Bar(x=gex["strike"],y=gex["put_gex"],name="Put GEX",marker_color="#ff4444",opacity=0.6))
            fig_g.add_trace(go.Scatter(x=gex["strike"],y=gex["net_gex"],name="Net GEX",
                            line=dict(color="#ffd700",width=2)))
            fig_g.add_vline(x=spot,line_dash="dash",line_color="#ffd700",
                            annotation_text=f"Spot ${spot:.2f}")
            if flip:
                fig_g.add_vline(x=flip,line_dash="dot",line_color="#4fc3f7",
                                annotation_text=f"γ-Flip ${flip:.2f}")
            fig_g.add_hline(y=0,line_color="white",opacity=0.25)
            fig_g.update_layout(template="plotly_dark",barmode="relative",height=340,
                                xaxis_title="Strike",yaxis_title="GEX $ / 1% move",
                                legend=dict(orientation="h",y=1.02),margin=dict(l=0,r=0,t=30,b=0),
                                paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
            st.plotly_chart(fig_g, width="stretch")
            total_gex=gex["net_gex"].sum()
            flip_txt = f" · γ-flip near **${flip:.2f}**" if flip else ""
            if total_gex>0:
                st.success(f"**Net GEX +${total_gex:,.0f}/1%** — positive gamma, dealers dampen "
                           f"volatility (price tends to pin){flip_txt}.")
            else:
                st.warning(f"**Net GEX ${total_gex:,.0f}/1%** — negative gamma, dealers amplify "
                           f"moves (trend/squeeze risk){flip_txt}.")
            st.caption("⚠️ Estimate only — assumes standard dealer positioning (short calls / "
                       "long puts) and BS gamma from Yahoo's IV, which can be noisy on illiquid strikes.")

    # ── Dealer Greeks: DEX · Charm · Vanna ────────────────────────────────────
    if _GREEKS_OK:
        st.subheader("🧮 Dealer Greeks — DEX · Charm · Vanna")
        rf2 = get_risk_free_rate()
        ge = greek_exposures(calls, puts, spot, expiry=expiry, r=rf2)
        bs = ge["by_strike"]
        if bs.empty:
            st.info("Not enough OI to compute dealer greeks on this expiry.")
        else:
            dx, ch, vn = ge["net_dex"], ge["net_charm"], ge["net_vanna"]
            dtilt = ge["dex_tilt"]
            g1, g2, g3 = st.columns(3)
            g1.metric("Net DEX", f"${dx/1e6:+,.0f}M",
                      "call-delta lean" if dx > 0 else "put-delta lean")
            g2.metric("Net Charm", f"${ch/1e6:+,.2f}M/day",
                      help="Per-day delta-hedge drift from time decay — pins into expiry")
            g3.metric("Net Vanna", f"${vn/1e6:+,.1f}M/volpt",
                      help="Delta-hedge flow per 1 vol-point move")

            # DEX by strike (where directional pressure sits)
            fig_d = go.Figure(go.Bar(
                x=bs["strike"], y=bs["dex"] / 1e6,
                marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in bs["dex"]],
                name="DEX $M"))
            fig_d.add_vline(x=spot, line_dash="dash", line_color="#ffd700",
                            annotation_text=f"Spot ${spot:.2f}")
            fig_d.add_hline(y=0, line_color="white", opacity=0.25)
            fig_d.update_layout(template="plotly_dark", height=300,
                                xaxis_title="Strike", yaxis_title="Delta exposure $M",
                                margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_d, width="stretch")

            dexr = (f"**DEX {('+'if dx>=0 else '')}{dx/1e6:,.0f}M** (tilt {dtilt:+.2f}) — "
                    + ("call-delta dominant: positioning leans **bullish**; dealers short "
                       "that delta buy dips/sell rips." if dx > 0 else
                       "put-delta dominant: positioning leans **bearish/hedged**."))
            charmr = ("Charm pulls hedging **toward higher strikes** as time passes "
                      if ch > 0 else "Charm pulls hedging **toward lower strikes** as time passes ")
            vannar = ("Positive vanna: **falling** IV → dealer **buying** (melt-up fuel); "
                      "rising IV → selling." if vn > 0 else
                      "Negative vanna: **rising** IV → dealer **buying**; falling IV → selling.")
            st.markdown(f"- {dexr}")
            st.markdown(f"- {charmr}— strongest into expiry (the OPEX pin).")
            st.markdown(f"- {vannar}")
            st.caption("⚠️ DEX uses native delta when available (CBOE); charm & vanna are "
                       "Black-Scholes from IV. Dealer convention assumed — estimate, not the book.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: GAMMA (multi-horizon dealer gamma: long vs short over 1wk / 2wk / 1mo)
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Gamma":
    st.subheader("⚡ Dealer Gamma by Horizon")
    st.caption("Long vs short gamma over 1-week / 2-week / 1-month windows. "
               "Gamma concentrates in near-dated options, so the weekly read is the "
               "most forcing; the monthly is the structural backdrop.")
    if not _GH_OK:
        st.warning("⚠️ Needs the updated **utils/indicators.py** and **utils/data_fetcher.py** "
                   "(gamma_horizons + get_options_chains_multi). Push those utils, then retry.")
        st.stop()

    rf = get_risk_free_rate()
    with st.spinner("Pulling multi-expiration chains…"):
        chains = get_options_chains_multi(ticker, max_days=35)
    if not chains:
        st.info("No options within ~35 days for this ticker (or rate-limited). "
                "Try a liquid underlying like SPY/QQQ.")
        st.stop()

    gh = gamma_horizons(chains, spot, r=rf)
    if not gh["per_expiry"]:
        st.info("Not enough valid IV/OI to compute gamma. Try a more liquid underlying.")
        st.stop()

    # ── Horizon buckets up top ────────────────────────────────────────────────
    hlabels = {7: "1 Week", 14: "2 Weeks", 30: "1 Month"}
    cols = st.columns(3)
    for col, h in zip(cols, (7, 14, 30)):
        net = gh["buckets"][h]; tilt = gh["bucket_tilt"][h]; flip = gh["bucket_flip"][h]
        emoji, label, read = gamma_regime_label(tilt)
        side = ""
        if flip:
            side = ("spot **above** flip → stable side" if spot > flip
                    else "spot **below** flip → unstable side")
        clr = "#00cc66" if "Long" in label else "#ff4444" if "Short" in label else "#ffd700"
        with col:
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{clr}1c;"
                f"border:1px solid {clr}'>"
                f"<div style='font-size:0.8rem;color:#aaa'>{hlabels[h]}</div>"
                f"<div style='font-size:1.15rem;font-weight:700'>{emoji} {label}</div>"
                f"<div style='font-size:0.95rem'>${net/1e6:+,.0f}M / 1%</div>"
                f"<div style='font-size:0.8rem;color:#bbb'>flip "
                f"{('$'+format(flip,'.2f')) if flip else 'n/a'}</div></div>",
                unsafe_allow_html=True)
            st.caption(read + (f" · {side}" if side else ""))

    # ── Net GEX by expiration (chart) ─────────────────────────────────────────
    st.subheader("📊 Net Gamma by Expiration")
    pe = gh["per_expiry"]
    fig = go.Figure(go.Bar(
        x=[f"{e['expiry']}\n({e['days']}d)" for e in pe],
        y=[e["net_gex"] / 1e6 for e in pe],
        marker_color=["#00ff88" if e["net_gex"] >= 0 else "#ff4444" for e in pe],
        text=[f"${e['net_gex']/1e6:+.0f}M" for e in pe], textposition="outside"))
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(template="plotly_dark", height=300,
                      yaxis_title="Net GEX $M / 1% move",
                      margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green = dealers long gamma that expiry (dampening) · red = short "
               "(amplifying). Near-dated bars carry the most hedging force.")

    # ── Per-expiration detail ─────────────────────────────────────────────────
    st.subheader("🔍 Per-Expiration Detail")
    det = pd.DataFrame([{
        "Expiry": e["expiry"], "Days": e["days"],
        "Net GEX $M": round(e["net_gex"] / 1e6, 1),
        "Tilt": round(e["net_gex"] / e["gross_gex"], 2) if e["gross_gex"] else 0,
        "γ-Flip": e["flip"],
        "Regime": gamma_regime_label(
            e["net_gex"] / e["gross_gex"] if e["gross_gex"] else 0)[1],
    } for e in pe])
    st.dataframe(det, width="stretch", hide_index=True,
                 column_config={"Net GEX $M": st.column_config.NumberColumn(format="$%.1fM"),
                                "γ-Flip": st.column_config.NumberColumn(format="$%.2f")})
    st.caption(f"Spot ${spot:.2f} · risk-free {rf*100:.2f}% · {len(pe)} expirations ≤35d. "
               "⚠️ Estimate: assumes standard dealer convention (short calls / long puts) "
               "and BS gamma from Yahoo IV. Maps the gamma landscape, not the confirmed book.")

    # ── Dealer Greek Exposure: DEX / Charm / Vanna ────────────────────────────
    try:
        ge = greek_exposures(chains, spot, r=rf)
    except Exception:
        ge = None
    if ge:
        st.divider()
        st.subheader("📐 Dealer Greek Exposure")
        st.caption("Beyond gamma: directional (DEX), time-decay (Charm) and vol-driven "
                   "(Vanna) hedging pressure — the full second-order positioning picture.")
        d = ge["net_dex"] / 1e6; ch = ge["net_charm"] / 1e6; vn = ge["net_vanna"] / 1e6
        gg1, gg2, gg3 = st.columns(3)
        gg1.metric("Net Delta (DEX)", f"${d:+,.0f}M",
                   help="Net OI delta. >0 = call-delta-heavy positioning; dealers hold "
                        "the inverse (short delta → buy dips / sell rips).")
        gg2.metric("Charm $/day", f"${ch:+,.2f}M",
                   help="Delta that decays per day. Near expiry this pulls hedging toward "
                        "high-OI strikes — the mechanical OPEX pin.")
        gg3.metric("Vanna $/1% IV", f"${vn:+,.2f}M",
                   help="Delta shift per 1% IV move. Drives vol-triggered hedging: "
                        "IV down → dealers sell; IV up → vanna rally.")
        reads = []
        reads.append("Positioning leans **call-delta heavy** (bullish OI) — dealers net short "
                     "delta, a dip-buying / rally-selling hedging bias." if d > 0 else
                     "Positioning leans **put-delta heavy** (hedged/bearish OI) — dealers net "
                     "long delta.")
        if abs(ch) > 0:
            reads.append(f"Charm is **{'pulling toward pin' if ch < 0 else 'pushing off pin'}** "
                         "as expiry nears (strongest in the front expiry).")
        reads.append("Vanna is **positive** — a vol decline mechanically pressures price "
                     "(dealers sell); a vol pop fuels a vanna rally." if vn > 0 else
                     "Vanna is **negative** — a vol decline is supportive; a vol spike pressures price.")
        st.caption(" ".join(reads))
        st.caption("⚠️ DEX uses native (CBOE) delta; charm/vanna are Black-Scholes from IV. "
                   "Same dealer-convention caveat as gamma — directional inference, not the book.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: INTRADAY
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Intraday":
    intra = get_intraday_data(ticker, interval=intra_int, days=intra_days)
    if intra.empty:
        st.error(f"No intraday data for {ticker} (yfinance limits intraday history).")
        st.stop()

    auc = auction_analysis(intra)
    vw  = vwap_deviation(intra)

    st.subheader("🔔 Latest Session Auction")
    if auc:
        a1,a2,a3,a4=st.columns(4)
        a1.metric("Open Vol %",f"{auc['open_vol_pct']:.0f}%","Heavy" if auc['open_vol_pct']>30 else "Normal")
        a2.metric("Close Vol %",f"{auc['close_vol_pct']:.0f}%","Heavy" if auc['close_vol_pct']>25 else "Normal")
        a3.metric("Close Drift",f"{auc['close_drift']:+.2f}%")
        a4.metric("Session Vol",f"{auc['total_vol']:,}")
        sig=auc["signal"]
        if "accumulation" in sig: st.success(f"🟢 {sig}")
        elif "distribution" in sig: st.error(f"🔴 {sig}")
        elif sig!="Neutral": st.info(f"📊 {sig}")
    if vw:
        st.caption(f"**VWAP** ${vw['vwap']:.2f} · Price ${vw['price']:.2f} "
                   f"({vw['dev_pct']:+.2f}%) · Above VWAP {vw['pct_session_above_vwap']:.0f}% of session")

    st.subheader("📊 Volume-at-Price")
    prof=volume_by_price(intra,bins=vp_bins)
    if not prof.empty:
        poc=prof.attrs.get("poc")
        fig=make_subplots(rows=1,cols=2,column_widths=[0.72,0.28],shared_yaxes=True,
                          horizontal_spacing=0.02,subplot_titles=[f"{ticker} {intra_int}","Vol@Price"])
        fig.add_trace(go.Scatter(x=intra.index,y=intra["Close"],name="Price",
                                 line=dict(color="#00ff88",width=1.5)),row=1,col=1)
        if poc:
            fig.add_hline(y=poc,line_dash="dash",line_color="#ffd700",
                          annotation_text=f"POC ${poc:.2f}",row=1,col=1)
        fig.add_trace(go.Bar(y=prof["price"],x=prof["volume"],orientation="h",
                             marker_color="#4fc3f7",opacity=0.7,showlegend=False),row=1,col=2)
        fig.update_layout(template="plotly_dark",height=460,showlegend=False,
                          margin=dict(l=0,r=0,t=30,b=0),paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
        fig.update_xaxes(showticklabels=False,row=1,col=2)
        st.plotly_chart(fig, width="stretch")

    st.subheader("🕐 Volume by Time of Day")
    d=intra.dropna(subset=["Volume"]).copy()
    d["time"]=d.index.strftime("%H:%M")
    tod=d.groupby("time")["Volume"].mean()
    if not tod.empty:
        tc=["#ff8c00" if v>tod.mean()*1.3 else "#4fc3f7" for v in tod.values]
        fig_t=go.Figure(go.Bar(x=list(tod.index),y=list(tod.values),marker_color=tc))
        fig_t.update_layout(template="plotly_dark",height=240,margin=dict(l=0,r=0,t=10,b=0),
                            xaxis_title="Time",yaxis_title="Avg Vol",
                            paper_bgcolor="#0e1117",plot_bgcolor="#0e1117")
        st.plotly_chart(fig_t, width="stretch")

st.divider()
st.caption("Flow inference from free EOD/intraday + delayed options data — not true tape/order-book.")
