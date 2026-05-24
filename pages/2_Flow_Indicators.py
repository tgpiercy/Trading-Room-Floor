"""
pages/2_Flow_Indicators.py
Volume & money-flow indicators: OBV, MFI, CMF, RVOL, Force Index.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_fetcher import get_stock_data, get_ticker_info
from utils.indicators import obv, mfi, cmf, relative_volume, force_index, atr

st.set_page_config(page_title="Flow Indicators · StratFlow", page_icon="🌊", layout="wide")
st.title("🌊 Flow Indicators")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", "SPY").upper().strip()
    period   = st.selectbox("Period",   ["1mo","3mo","6mo","1y","2y"], index=2)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)

    st.subheader("Indicators")
    show_obv  = st.checkbox("OBV — On-Balance Volume",   value=True)
    show_mfi  = st.checkbox("MFI — Money Flow Index",    value=True)
    show_cmf  = st.checkbox("CMF — Chaikin Money Flow",  value=True)
    show_rvol = st.checkbox("RVOL — Relative Volume",    value=True)
    show_fi   = st.checkbox("Force Index",               value=False)

    mfi_period  = st.slider("MFI Period",  7, 30, 14)
    cmf_period  = st.slider("CMF Period", 10, 40, 20)
    rvol_period = st.slider("RVOL Lookback", 10, 40, 20)

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker}…"):
    df = get_stock_data(ticker, period=period, interval=interval)

if df.empty:
    st.error("No data. Check the ticker symbol.")
    st.stop()

df = obv(df)
df = mfi(df, period=mfi_period)
df = cmf(df, period=cmf_period)
df = relative_volume(df, period=rvol_period)
df = force_index(df)
df = atr(df)

last = df.iloc[-1]
prev = df.iloc[-2]
chgp = (last["Close"] - prev["Close"]) / prev["Close"] * 100

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Price",  f"${last['Close']:.2f}", f"{chgp:+.2f}%")
k2.metric("MFI",    f"{last['MFI']:.1f}",
          "Overbought" if last["MFI"] > 80 else ("Oversold" if last["MFI"] < 20 else "Neutral"))
k3.metric("CMF",    f"{last['CMF']:.3f}",
          "Bullish Flow" if last["CMF"] > 0 else "Bearish Flow")
k4.metric("RVOL",   f"{last['RVOL']:.2f}×",
          "High Vol" if last["RVOL"] > 1.5 else ("Low Vol" if last["RVOL"] < 0.7 else "Normal"))
k5.metric("Volume", f"{int(last['Volume']):,}")
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
active = [(n, f) for n, f in [
    ("OBV",    show_obv),
    ("MFI",    show_mfi),
    ("CMF",    show_cmf),
    ("RVOL",   show_rvol),
    ("Force Index", show_fi),
] if f]

rows    = 1 + len(active)
heights = [0.40] + [0.60 / max(len(active), 1)] * len(active)
fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.04, row_heights=heights,
    subplot_titles=[f"{ticker} Price & Volume"] + [a[0] for a in active],
)

# — Price + Volume —
bar_c = ["#00ff88" if c >= o else "#ff4444"
         for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                         line=dict(color="#00ff88", width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                     marker_color=bar_c, opacity=0.35,
                     yaxis="y2", showlegend=False), row=1, col=1)

# dual y-axis for volume on price panel
fig.update_layout(
    yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
)

row_cursor = 2
for subplot_name, _ in active:
    r = row_cursor

    if subplot_name == "OBV":
        obv_c = "#00ff88" if df["OBV"].iloc[-1] > df["OBV"].iloc[-20] else "#ff4444"
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV",
                                 line=dict(color=obv_c, width=1.5)), row=r, col=1)
        # OBV 20-day SMA
        obv_ma = df["OBV"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=obv_ma, name="OBV SMA20",
                                 line=dict(color="#ffd700", width=1.2, dash="dot"),
                                 showlegend=False), row=r, col=1)

    elif subplot_name == "MFI":
        mfi_c = ["#ff4444" if v > 80 else ("#00ff88" if v < 20 else "#ffd700")
                 for v in df["MFI"]]
        fig.add_trace(go.Scatter(x=df.index, y=df["MFI"], name="MFI",
                                 line=dict(color="#ffd700", width=1.5)), row=r, col=1)
        for lvl, c, label in [(80, "red", "OB"), (20, "lime", "OS"), (50, "gray", "")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c,
                          opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    elif subplot_name == "CMF":
        cmf_vals = df["CMF"].fillna(0)
        cmf_bar_c = ["#00ff88" if v >= 0 else "#ff4444" for v in cmf_vals]
        fig.add_trace(go.Bar(x=df.index, y=cmf_vals, name="CMF",
                             marker_color=cmf_bar_c, opacity=0.8), row=r, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=r, col=1)
        fig.add_hline(y=0.05,  line_dash="dot", line_color="lime",  opacity=0.3, row=r, col=1)
        fig.add_hline(y=-0.05, line_dash="dot", line_color="red",   opacity=0.3, row=r, col=1)

    elif subplot_name == "RVOL":
        rvol_c = ["#ff8c00" if v > 2.0 else ("#00ff88" if v > 1.2 else "#888888")
                  for v in df["RVOL"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["RVOL"], name="RVOL",
                             marker_color=rvol_c, opacity=0.85), row=r, col=1)
        fig.add_hline(y=1.0, line_color="white", opacity=0.3, row=r, col=1)
        fig.add_hline(y=1.5, line_dash="dot", line_color="orange", opacity=0.3, row=r, col=1)
        fig.add_hline(y=2.0, line_dash="dot", line_color="red",    opacity=0.3, row=r, col=1)

    elif subplot_name == "Force Index":
        fi_c = ["#00ff88" if v >= 0 else "#ff4444" for v in df["FI"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["FI"], name="Force Index",
                             marker_color=fi_c, opacity=0.7), row=r, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=r, col=1)

    row_cursor += 1

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=480 + len(active) * 170,
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
)
st.plotly_chart(fig, width='stretch')

# ── Flow Interpretation ───────────────────────────────────────────────────────
st.subheader("💡 Flow Analysis")

signals = []
# OBV trend
obv_trend = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]
if obv_trend > 0:
    signals.append(("🟢", "OBV rising — buyers absorbing supply", "Accumulation"))
else:
    signals.append(("🔴", "OBV falling — sellers in control", "Distribution"))

# MFI
m = last["MFI"]
if m > 80:
    signals.append(("⚠️", f"MFI {m:.1f} — Overbought, potential reversal zone", "Caution"))
elif m < 20:
    signals.append(("⚠️", f"MFI {m:.1f} — Oversold, watch for bounce", "Caution"))
elif m > 50:
    signals.append(("🟢", f"MFI {m:.1f} — Positive money flow", "Bullish"))
else:
    signals.append(("🔴", f"MFI {m:.1f} — Negative money flow", "Bearish"))

# CMF
c_ = last["CMF"]
if c_ > 0.05:
    signals.append(("🟢", f"CMF {c_:.3f} — Strong buying pressure", "Bullish"))
elif c_ < -0.05:
    signals.append(("🔴", f"CMF {c_:.3f} — Strong selling pressure", "Bearish"))
else:
    signals.append(("🟡", f"CMF {c_:.3f} — Neutral flow", "Neutral"))

# RVOL
rv = last["RVOL"]
if rv > 2.0:
    signals.append(("🔥", f"RVOL {rv:.2f}× — Very high relative volume!", "High Activity"))
elif rv > 1.4:
    signals.append(("📈", f"RVOL {rv:.2f}× — Above-average volume", "Elevated"))
elif rv < 0.7:
    signals.append(("😴", f"RVOL {rv:.2f}× — Low volume, low conviction", "Thin"))

sig_df = pd.DataFrame(signals, columns=["", "Signal", "Classification"])
st.dataframe(sig_df, width='stretch', hide_index=True)

# ── Volume Profile Table ──────────────────────────────────────────────────────
with st.expander("📋 Recent Volume Profile (Last 30 Sessions)"):
    last30 = df.tail(30)[["Close", "Volume", "RVOL", "MFI", "CMF"]].copy()
    last30.index = last30.index.strftime("%Y-%m-%d")
    last30.columns = ["Close", "Volume", "RVOL", "MFI", "CMF"]
    last30["Volume"] = last30["Volume"].apply(lambda x: f"{int(x):,}")
    last30["RVOL"]   = last30["RVOL"].apply(lambda x: f"{x:.2f}×")
    last30["MFI"]    = last30["MFI"].apply(lambda x: f"{x:.1f}")
    last30["CMF"]    = last30["CMF"].apply(lambda x: f"{x:.3f}")
    st.dataframe(last30[::-1], width='stretch')
