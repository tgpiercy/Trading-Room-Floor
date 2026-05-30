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
                              pcr, max_pain, gamma_exposure, gamma_flip)
from utils.order_flow import (premium_flow, new_positioning, detect_sweeps,
                              volume_by_price, auction_analysis, vwap_deviation)
from utils.fred import get_risk_free_rate
from utils.chart_utils import set_chart_window

st.set_page_config(page_title="Flow · StratFlow", page_icon="🌊", layout="wide")
st.title("🌊 Flow")
st.caption("Unified money flow · options flow · positioning · intraday — one ticker, all views")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    view = st.selectbox("View", ["Money Flow", "Options Flow",
                                  "Positioning", "Intraday"], index=0)
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
