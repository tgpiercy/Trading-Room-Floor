"""
pages/2_Flow_Indicators.py
Volume & money-flow indicators. Default view: last 6 months.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_fetcher import get_stock_data, get_ticker_info
from utils.indicators import obv, mfi, cmf, relative_volume, force_index, atr
from utils.chart_utils import set_chart_window

st.set_page_config(page_title="Flow Indicators · StratFlow", page_icon="🌊", layout="wide")
st.title("🌊 Flow Indicators")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", "SPY").upper().strip()
    period   = st.selectbox("Data History", ["1y","2y","5y"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)
    st.subheader("Indicators")
    show_obv  = st.checkbox("OBV",          value=True)
    show_mfi  = st.checkbox("MFI",          value=True)
    show_cmf  = st.checkbox("CMF",          value=True)
    show_rvol = st.checkbox("RVOL",         value=True)
    show_fi   = st.checkbox("Force Index",  value=False)
    mfi_p  = st.slider("MFI Period",   7, 30, 14)
    cmf_p  = st.slider("CMF Period",  10, 40, 20)
    rvol_p = st.slider("RVOL Lookback",10, 40, 20)

with st.spinner(f"Loading {ticker}…"):
    df = get_stock_data(ticker, period=period, interval=interval)

if df.empty:
    st.error("No data. Check the ticker symbol.")
    st.stop()

df = obv(df); df = mfi(df, period=mfi_p); df = cmf(df, period=cmf_p)
df = relative_volume(df, period=rvol_p); df = force_index(df); df = atr(df)

last = df.iloc[-1]; prev = df.iloc[-2]
chgp = (last["Close"] - prev["Close"]) / prev["Close"] * 100

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Price",  f"${last['Close']:.2f}", f"{chgp:+.2f}%")
k2.metric("MFI",    f"{last['MFI']:.1f}",
          "Overbought" if last["MFI"]>80 else ("Oversold" if last["MFI"]<20 else "Neutral"))
k3.metric("CMF",    f"{last['CMF']:.3f}",
          "Bullish" if last["CMF"]>0 else "Bearish")
k4.metric("RVOL",   f"{last['RVOL']:.2f}×",
          "High" if last["RVOL"]>1.5 else ("Low" if last["RVOL"]<0.7 else "Normal"))
k5.metric("Volume", f"{int(last['Volume']):,}")
st.divider()

active = [(n,f) for n,f in [("OBV",show_obv),("MFI",show_mfi),("CMF",show_cmf),
                              ("RVOL",show_rvol),("Force Index",show_fi)] if f]
rows    = 1 + len(active)
heights = [0.40] + [0.60/max(len(active),1)] * len(active)

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, row_heights=heights,
                    subplot_titles=[f"{ticker} Price & Volume"]+[a[0] for a in active])

bar_c = ["#00ff88" if c>=o else "#ff4444" for c,o in zip(df["Close"],df["Open"])]
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                         line=dict(color="#00ff88",width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                     marker_color=bar_c, opacity=0.35,
                     yaxis="y2", showlegend=False), row=1, col=1)
fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                               showgrid=False, showticklabels=False))

row_cursor = 2
for name_, _ in active:
    r = row_cursor
    if name_ == "OBV":
        oc = "#00ff88" if df["OBV"].iloc[-1]>df["OBV"].iloc[-20] else "#ff4444"
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV",
                                 line=dict(color=oc,width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"].rolling(20).mean(),
                                 name="OBV SMA20", line=dict(color="#ffd700",width=1.2,dash="dot"),
                                 showlegend=False), row=r, col=1)
    elif name_ == "MFI":
        fig.add_trace(go.Scatter(x=df.index, y=df["MFI"], name="MFI",
                                 line=dict(color="#ffd700",width=1.5)), row=r, col=1)
        for lvl,c in [(80,"red"),(20,"lime"),(50,"gray")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c, opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0,100], row=r, col=1)
    elif name_ == "CMF":
        cmf_c = ["#00ff88" if v>=0 else "#ff4444" for v in df["CMF"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["CMF"].fillna(0), name="CMF",
                             marker_color=cmf_c, opacity=0.8), row=r, col=1)
        fig.add_hline(y=0,    line_color="white",  opacity=0.3,  row=r, col=1)
        fig.add_hline(y=0.05, line_dash="dot", line_color="lime", opacity=0.3, row=r, col=1)
        fig.add_hline(y=-0.05,line_dash="dot", line_color="red",  opacity=0.3, row=r, col=1)
    elif name_ == "RVOL":
        rv_c = ["#ff8c00" if v>2 else ("#00ff88" if v>1.2 else "#888")
                for v in df["RVOL"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["RVOL"].fillna(0), name="RVOL",
                             marker_color=rv_c, opacity=0.85), row=r, col=1)
        fig.add_hline(y=1.0, line_color="white",  opacity=0.3,  row=r, col=1)
        fig.add_hline(y=1.5, line_dash="dot", line_color="orange", opacity=0.3, row=r, col=1)
        fig.add_hline(y=2.0, line_dash="dot", line_color="red",    opacity=0.3, row=r, col=1)
    elif name_ == "Force Index":
        fi_c = ["#00ff88" if v>=0 else "#ff4444" for v in df["FI"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["FI"].fillna(0), name="Force Index",
                             marker_color=fi_c, opacity=0.7), row=r, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=r, col=1)
    row_cursor += 1

fig.update_layout(
    template="plotly_dark", xaxis_rangeslider_visible=False,
    height=480 + len(active)*170,
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
set_chart_window(fig)
st.plotly_chart(fig, width='stretch')

st.subheader("💡 Flow Analysis")
signals = []
obv_trend = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]
signals.append(("🟢","OBV rising — accumulation","Accumulation") if obv_trend>0
               else ("🔴","OBV falling — distribution","Distribution"))
m = last["MFI"]
if m>80:   signals.append(("⚠️",f"MFI {m:.1f} — Overbought","Caution"))
elif m<20: signals.append(("⚠️",f"MFI {m:.1f} — Oversold","Caution"))
elif m>50: signals.append(("🟢",f"MFI {m:.1f} — Positive flow","Bullish"))
else:      signals.append(("🔴",f"MFI {m:.1f} — Negative flow","Bearish"))
c_ = last["CMF"]
if c_>0.05:   signals.append(("🟢",f"CMF {c_:.3f} — Buying pressure","Bullish"))
elif c_<-0.05:signals.append(("🔴",f"CMF {c_:.3f} — Selling pressure","Bearish"))
else:          signals.append(("🟡",f"CMF {c_:.3f} — Neutral","Neutral"))
rv = last["RVOL"]
if rv>2.0:   signals.append(("🔥",f"RVOL {rv:.2f}× — Very high volume!","High"))
elif rv>1.4: signals.append(("📈",f"RVOL {rv:.2f}× — Above-average volume","Elevated"))
elif rv<0.7: signals.append(("😴",f"RVOL {rv:.2f}× — Low volume","Thin"))
st.dataframe(pd.DataFrame(signals, columns=["","Signal","Classification"]),
             width='stretch', hide_index=True)

with st.expander("📋 Recent Volume Profile (Last 30 Sessions)"):
    t30 = df.tail(30)[["Close","Volume","RVOL","MFI","CMF"]].copy()
    t30.index = t30.index.strftime("%Y-%m-%d")
    t30["Volume"] = t30["Volume"].apply(lambda x: f"{int(x):,}")
    t30["RVOL"]   = t30["RVOL"].apply(lambda x: f"{x:.2f}×")
    t30["MFI"]    = t30["MFI"].apply(lambda x: f"{x:.1f}")
    t30["CMF"]    = t30["CMF"].apply(lambda x: f"{x:.3f}")
    st.dataframe(t30[::-1], width='stretch')
