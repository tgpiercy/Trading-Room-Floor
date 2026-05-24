"""
pages/1_Trend_Analysis.py
Candlestick + MA overlays + oscillator subplots + signal summary.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_fetcher import get_stock_data, get_ticker_info
from utils.indicators import (
    sma, ema, bollinger_bands, rsi, macd, adx, stochastic, atr
)

st.set_page_config(page_title="Trend Analysis · StratFlow", page_icon="📈", layout="wide")
st.title("📈 Trend Analysis")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", "SPY").upper().strip()
    period   = st.selectbox("Period",   ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)

    st.subheader("Overlays")
    show_sma = st.checkbox("SMA 20 / 50 / 200",  value=True)
    show_ema = st.checkbox("EMA 9 / 21",          value=False)
    show_bb  = st.checkbox("Bollinger Bands",      value=True)

    st.subheader("Subplots")
    show_vol   = st.checkbox("Volume",      value=True)
    show_rsi   = st.checkbox("RSI",         value=True)
    show_macd  = st.checkbox("MACD",        value=True)
    show_adx   = st.checkbox("ADX",         value=False)
    show_stoch = st.checkbox("Stochastic",  value=False)

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker}…"):
    df   = get_stock_data(ticker, period=period, interval=interval)
    info = get_ticker_info(ticker)

if df.empty:
    st.error("No data returned. Check the ticker symbol and try again.")
    st.stop()

# Calculate all indicators
df = sma(df);  df = ema(df);  df = bollinger_bands(df)
df = rsi(df);  df = macd(df); df = adx(df)
df = stochastic(df); df = atr(df)

last  = df.iloc[-1]
prev  = df.iloc[-2]
chg   = last["Close"] - prev["Close"]
chgp  = chg / prev["Close"] * 100
name  = info.get("longName", ticker)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.caption(f"**{name}**  ·  {info.get('exchange','')}")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Price",    f"${last['Close']:.2f}",    f"{chgp:+.2f}%")
k2.metric("RSI (14)", f"{last['RSI']:.1f}",
          "Overbought" if last["RSI"] > 70 else ("Oversold" if last["RSI"] < 30 else "Neutral"))
k3.metric("ATR (14)", f"${last['ATR']:.2f}")
k4.metric("MACD",     f"{last['MACD']:.3f}",
          "Bullish" if last["MACD"] > last["MACD_Sig"] else "Bearish")
k5.metric("ADX",      f"{last['ADX']:.1f}",
          "Strong" if last["ADX"] > 25 else "Weak")
k6.metric("BB %B",    f"{last['BB_PctB']:.2f}",
          "Extended" if last["BB_PctB"] > 1 or last["BB_PctB"] < 0 else "Inside")
st.divider()

# ── Build Chart ───────────────────────────────────────────────────────────────
subplot_flags = [show_vol, show_rsi, show_macd, show_adx, show_stoch]
subplot_names = ["Volume", "RSI", "MACD", "ADX", "Stochastic"]
active = [(n, f) for n, f in zip(subplot_names, subplot_flags) if f]
n_sub  = len(active)
rows   = 1 + n_sub
heights = [0.55] + [0.45 / max(n_sub, 1)] * n_sub

titles = [f"{ticker}"] + [a[0] for a in active]
fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=heights,
    subplot_titles=titles,
)

# — Candlestick —
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name=ticker,
    increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    increasing_fillcolor="#00ff88",  decreasing_fillcolor="#ff4444",
), row=1, col=1)

# — MA Overlays —
if show_sma:
    clrs = ["#ffd700", "#ff8c00", "#ff69b4"]
    for p, c in zip([20, 50, 200], clrs):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{p}"], name=f"SMA {p}",
                                 line=dict(color=c, width=1.4)), row=1, col=1)
if show_ema:
    for p, c in zip([9, 21], ["#00bfff", "#7b68ee"]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{p}"], name=f"EMA {p}",
                                 line=dict(color=c, width=1.4, dash="dot")), row=1, col=1)
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(150,130,255,0.5)", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(150,130,255,0.5)", width=1),
                             fill="tonexty", fillcolor="rgba(150,130,255,0.06)",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid",
                             line=dict(color="rgba(150,130,255,0.4)", width=1, dash="dot"),
                             showlegend=False), row=1, col=1)

# — Subplots —
row_cursor = 2
bar_colors = ["#00ff88" if c >= o else "#ff4444"
              for c, o in zip(df["Close"], df["Open"])]

for subplot_name, _ in active:
    r = row_cursor

    if subplot_name == "Volume":
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=bar_colors, opacity=0.6), row=r, col=1)

    elif subplot_name == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(color="#ffd700", width=1.5)), row=r, col=1)
        for lvl, c in [(70, "red"), (30, "lime"), (50, "gray")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c,
                          opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    elif subplot_name == "MACD":
        hist_c = ["#00ff88" if v >= 0 else "#ff4444" for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                             marker_color=hist_c, opacity=0.7), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],     name="MACD",
                                 line=dict(color="#00bfff", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Sig"], name="Signal",
                                 line=dict(color="#ff8c00", width=1.5)), row=r, col=1)

    elif subplot_name == "ADX":
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"],  name="ADX",
                                 line=dict(color="#ffd700", width=2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["+DI"],  name="+DI",
                                 line=dict(color="#00ff88", width=1.2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["-DI"],  name="-DI",
                                 line=dict(color="#ff4444", width=1.2)), row=r, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="white",
                      opacity=0.25, row=r, col=1)

    elif subplot_name == "Stochastic":
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="%K",
                                 line=dict(color="#00bfff", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="%D",
                                 line=dict(color="#ff8c00", width=1.5)), row=r, col=1)
        for lvl, c in [(80, "red"), (20, "lime")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c,
                          opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    row_cursor += 1

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=560 + n_sub * 160,
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
)
st.plotly_chart(fig, width='stretch')

# ── Signal Summary ────────────────────────────────────────────────────────────
st.subheader("🎯 Signal Summary")
signals = []

# Price vs MAs
p = last["Close"]
if p > last["SMA_20"] > last["SMA_50"] > last["SMA_200"]:
    signals.append(("🟢", "Price > SMA20 > SMA50 > SMA200", "Strong Bullish"))
elif p < last["SMA_20"] < last["SMA_50"] < last["SMA_200"]:
    signals.append(("🔴", "Price < SMA20 < SMA50 < SMA200", "Strong Bearish"))
elif p > last["SMA_50"]:
    signals.append(("🟡", "Price above SMA50 — medium-term uptrend", "Bullish"))
else:
    signals.append(("🟡", "Price below SMA50 — medium-term downtrend", "Bearish"))

# RSI
r_val = last["RSI"]
if r_val > 70:
    signals.append(("⚠️", f"RSI {r_val:.1f} — Overbought territory", "Caution"))
elif r_val < 30:
    signals.append(("⚠️", f"RSI {r_val:.1f} — Oversold territory", "Caution"))
else:
    signals.append(("✅", f"RSI {r_val:.1f} — Normal range", "Neutral"))

# MACD crossover
if last["MACD"] > last["MACD_Sig"] and prev["MACD"] <= prev["MACD_Sig"]:
    signals.append(("🔀", "MACD bullish crossover (signal line cross)", "Bullish"))
elif last["MACD"] < last["MACD_Sig"] and prev["MACD"] >= prev["MACD_Sig"]:
    signals.append(("🔀", "MACD bearish crossover (signal line cross)", "Bearish"))
elif last["MACD"] > last["MACD_Sig"]:
    signals.append(("🟢", "MACD above signal line", "Bullish"))
else:
    signals.append(("🔴", "MACD below signal line", "Bearish"))

# ADX
if last["ADX"] > 25:
    td = "Uptrend" if last["+DI"] > last["-DI"] else "Downtrend"
    signals.append(("💪", f"ADX {last['ADX']:.1f} — Strong {td} confirmed", td))
else:
    signals.append(("😴", f"ADX {last['ADX']:.1f} — Weak or no trend", "Ranging"))

# Bollinger
if last["BB_PctB"] > 1.0:
    signals.append(("📊", "Price above upper Bollinger Band — extended", "Caution"))
elif last["BB_PctB"] < 0.0:
    signals.append(("📊", "Price below lower Bollinger Band — extended", "Caution"))

sig_df = pd.DataFrame(signals, columns=["", "Signal", "Bias"])
st.dataframe(sig_df, width='stretch', hide_index=True)
