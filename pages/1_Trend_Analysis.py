"""
pages/1_Trend_Analysis.py
Candlestick + MA overlays + oscillator subplots + RS panel + signal summary.
Default view: last 6 months. Pan left or use buttons for full history.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_fetcher import get_stock_data, get_ticker_info
from utils.indicators import sma, ema, bollinger_bands, rsi, macd, adx, stochastic, atr
from utils.rs_indicators import build_rs_df, classify_state, STATE_CONFIG
from utils.chart_utils import set_chart_window

st.set_page_config(page_title="Trend Analysis · StratFlow", page_icon="📈", layout="wide")
st.title("📈 Trend Analysis")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", "AAPL").upper().strip()
    period   = st.selectbox("Data History", ["1y","2y","5y"], index=1,
                             help="More history = more to scroll back through")
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)

    st.subheader("Overlays")
    show_sma = st.checkbox("SMA 20 / 50 / 200", value=True)
    show_ema = st.checkbox("EMA 9 / 21",         value=False)
    show_bb  = st.checkbox("Bollinger Bands",     value=True)

    st.subheader("Oscillators")
    show_vol   = st.checkbox("Volume",     value=True)
    show_rsi   = st.checkbox("RSI",        value=True)
    show_macd  = st.checkbox("MACD",       value=True)
    show_adx   = st.checkbox("ADX",        value=False)
    show_stoch = st.checkbox("Stochastic", value=False)

    st.subheader("RS Panel")
    show_rs  = st.checkbox("RS vs Benchmark", value=True)
    rs_bench = st.selectbox("Benchmark", ["SPY","QQQ","IWM"], index=0)

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker}…"):
    df   = get_stock_data(ticker, period=period, interval=interval)
    info = get_ticker_info(ticker)

if df.empty:
    st.error("No data returned. Check the ticker symbol and try again.")
    st.stop()

df = sma(df);  df = ema(df);  df = bollinger_bands(df)
df = rsi(df);  df = macd(df); df = adx(df)
df = stochastic(df); df = atr(df)

last = df.iloc[-1]
prev = df.iloc[-2]
chgp = (last["Close"] - prev["Close"]) / prev["Close"] * 100

# RS
rs_state_label = rs_score = rs_ext = None
df_rs = pd.DataFrame()
if show_rs:
    bench_df = get_stock_data(rs_bench, period=period, interval=interval)
    tick_df  = get_stock_data(ticker,   period=period, interval=interval)
    if not bench_df.empty and not tick_df.empty:
        df_rs = build_rs_df(tick_df["Close"], bench_df["Close"])
        rs_state_label, rs_score, rs_desc, rs_action, rs_ext = classify_state(df_rs)

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.caption(f"**{info.get('longName', ticker)}**  ·  {info.get('exchange','')}")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Price",    f"${last['Close']:.2f}", f"{chgp:+.2f}%")
k2.metric("RSI (14)", f"{last['RSI']:.1f}",
          "Overbought" if last["RSI"] > 70 else ("Oversold" if last["RSI"] < 30 else "Neutral"))
k3.metric("ATR (14)", f"${last['ATR']:.2f}")
k4.metric("MACD",     f"{last['MACD']:.3f}",
          "Bullish" if last["MACD"] > last["MACD_Sig"] else "Bearish")
k5.metric("ADX",      f"{last['ADX']:.1f}",
          "Strong" if last["ADX"] > 25 else "Weak")
if rs_state_label and rs_state_label != "Insufficient Data":
    cfg = STATE_CONFIG.get(rs_state_label, {})
    k6.metric("RS State", f"{cfg.get('emoji','')} {rs_state_label}", f"{rs_ext:+.1f}% ext")
else:
    k6.metric("BB %B", f"{last['BB_PctB']:.2f}")
st.divider()

# ── Build Chart ───────────────────────────────────────────────────────────────
subplot_flags  = [show_vol, show_rsi, show_macd, show_adx, show_stoch,
                  show_rs and not df_rs.empty]
subplot_labels = ["Volume","RSI","MACD","ADX","Stochastic", f"RS vs {rs_bench}"]
active   = [(n, f) for n, f in zip(subplot_labels, subplot_flags) if f]
n_sub    = len(active)
heights  = [0.50] + [0.50 / max(n_sub, 1)] * n_sub

fig = make_subplots(
    rows=1 + n_sub, cols=1, shared_xaxes=True,
    vertical_spacing=0.04, row_heights=heights,
    subplot_titles=[ticker] + [a[0] for a in active],
)

# — Candlestick —
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name=ticker,
    increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    increasing_fillcolor="#00ff88",  decreasing_fillcolor="#ff4444",
), row=1, col=1)

# — MA Overlays —
if show_sma:
    for p, c in zip([20,50,200], ["#ffd700","#ff8c00","#ff69b4"]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{p}"], name=f"SMA {p}",
                                 line=dict(color=c, width=1.4)), row=1, col=1)
if show_ema:
    for p, c in zip([9,21], ["#00bfff","#7b68ee"]):
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
                             line=dict(color="rgba(150,130,255,0.35)", width=1, dash="dot"),
                             showlegend=False), row=1, col=1)

# — Subplots —
bar_c = ["#00ff88" if c >= o else "#ff4444" for c, o in zip(df["Close"], df["Open"])]
row_cursor = 2

for name_, _ in active:
    r = row_cursor
    if name_ == "Volume":
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=bar_c, opacity=0.6), row=r, col=1)
    elif name_ == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(color="#ffd700", width=1.5)), row=r, col=1)
        for lvl, c in [(70,"red"),(30,"lime"),(50,"gray")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c, opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0,100], row=r, col=1)
    elif name_ == "MACD":
        hc = ["#00ff88" if v >= 0 else "#ff4444" for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                             marker_color=hc, opacity=0.7), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],     name="MACD",
                                 line=dict(color="#00bfff", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Sig"], name="Signal",
                                 line=dict(color="#ff8c00", width=1.5)), row=r, col=1)
    elif name_ == "ADX":
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"],  name="ADX",
                                 line=dict(color="#ffd700", width=2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["+DI"],  name="+DI",
                                 line=dict(color="#00ff88", width=1.2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["-DI"],  name="-DI",
                                 line=dict(color="#ff4444", width=1.2)), row=r, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="white", opacity=0.25, row=r, col=1)
    elif name_ == "Stochastic":
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="%K",
                                 line=dict(color="#00bfff", width=1.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="%D",
                                 line=dict(color="#ff8c00", width=1.5)), row=r, col=1)
        for lvl, c in [(80,"red"),(20,"lime")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=c, opacity=0.35, row=r, col=1)
        fig.update_yaxes(range=[0,100], row=r, col=1)
    elif "RS vs" in name_ and not df_rs.empty:
        fig.add_trace(go.Scatter(x=df_rs.index, y=df_rs["RS"], name="RS Ratio",
                                 line=dict(color="#ffffff", width=2)), row=r, col=1)
        for col_, color_, lbl_ in [("SMA8","#ff8c00","RS SMA8"),
                                    ("SMA18","#9c59b0","RS SMA18"),
                                    ("SMA40","#c39bd3","RS SMA40")]:
            fig.add_trace(go.Scatter(x=df_rs.index, y=df_rs[col_], name=lbl_,
                                     line=dict(color=color_, width=1.4)), row=r, col=1)
        for bc_, color_, lbl_ in [("ext5","#4fc3f7","+5%"),("ext10","#ef5350","+10%"),
                                   ("ext15","#ff8c00","+15%"),("ext20","#ffd700","+20%")]:
            fig.add_trace(go.Scatter(x=df_rs.index, y=df_rs[bc_], name=lbl_,
                                     line=dict(color=color_, width=1, dash="dot"),
                                     opacity=0.5), row=r, col=1)
    row_cursor += 1

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=540 + n_sub * 160,
    legend=dict(orientation="h", y=1.02, x=0, font_size=10),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
)
set_chart_window(fig)   # ← default 6-month view, pan/zoom freely
st.plotly_chart(fig, width='stretch')

# ── Signal Tabs ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Price Signals", "🏆 RS Signal"])

with tab1:
    signals = []
    p = last["Close"]
    if   p > last["SMA_20"] > last["SMA_50"] > last["SMA_200"]:
        signals.append(("🟢","Price > SMA20 > SMA50 > SMA200","Strong Bullish"))
    elif p < last["SMA_20"] < last["SMA_50"] < last["SMA_200"]:
        signals.append(("🔴","Price < SMA20 < SMA50 < SMA200","Strong Bearish"))
    elif p > last["SMA_50"]:
        signals.append(("🟡","Price above SMA50","Bullish"))
    else:
        signals.append(("🟡","Price below SMA50","Bearish"))
    rv = last["RSI"]
    if rv > 70: signals.append(("⚠️",f"RSI {rv:.1f} — Overbought","Caution"))
    elif rv < 30: signals.append(("⚠️",f"RSI {rv:.1f} — Oversold","Caution"))
    else: signals.append(("✅",f"RSI {rv:.1f} — Normal","Neutral"))
    if last["MACD"] > last["MACD_Sig"] and prev["MACD"] <= prev["MACD_Sig"]:
        signals.append(("🔀","MACD bullish crossover","Bullish"))
    elif last["MACD"] < last["MACD_Sig"] and prev["MACD"] >= prev["MACD_Sig"]:
        signals.append(("🔀","MACD bearish crossover","Bearish"))
    elif last["MACD"] > last["MACD_Sig"]: signals.append(("🟢","MACD above signal","Bullish"))
    else: signals.append(("🔴","MACD below signal","Bearish"))
    td = "Uptrend" if last["+DI"] > last["-DI"] else "Downtrend"
    if last["ADX"] > 25: signals.append(("💪",f"ADX {last['ADX']:.1f} — Strong {td}",td))
    else: signals.append(("😴",f"ADX {last['ADX']:.1f} — Weak / ranging","Ranging"))
    if last["BB_PctB"] > 1.0: signals.append(("📊","Above upper Bollinger Band","Caution"))
    elif last["BB_PctB"] < 0.0: signals.append(("📊","Below lower Bollinger Band","Caution"))
    st.dataframe(pd.DataFrame(signals, columns=["","Signal","Bias"]),
                 width='stretch', hide_index=True)

with tab2:
    if rs_state_label and rs_state_label != "Insufficient Data":
        cfg_ = STATE_CONFIG[rs_state_label]
        st.metric("RS State", f"{cfg_['emoji']} {rs_state_label}",
                  f"Score {rs_score:+d}  ·  {rs_ext:+.1f}% extension")
        st.info(f"📖 **{rs_state_label}:** {cfg_['description']}  \n"
                f"➡ **Action:** {cfg_['action']}")
        rs_rows = [
            ("RS vs SMA18", "Above ✅" if df_rs["RS"].iloc[-1] > df_rs["SMA18"].iloc[-1] else "Below ❌"),
            ("RS vs SMA40", "Above ✅" if df_rs["RS"].iloc[-1] > df_rs["SMA40"].iloc[-1] else "Below ❌"),
            ("SMA8 vs SMA18","Above ✅" if df_rs["SMA8"].iloc[-1] > df_rs["SMA18"].iloc[-1] else "Below ⚠️"),
            ("SMA18 Rising", "Yes 📈"  if df_rs["SMA18"].iloc[-1] > df_rs["SMA18"].iloc[-2] else "No 📉"),
            ("Extension",    f"{rs_ext:+.1f}%"),
        ]
        st.dataframe(pd.DataFrame(rs_rows, columns=["Condition","Status"]),
                     width='stretch', hide_index=True)
    else:
        st.info("RS needs 40+ bars. Use 1y+ period with weekly interval.")
