"""
pages/4_Swing_Screen.py
Multi-factor swing/position screen for 1wk–1mo+ holds. Clean-sheet, parallel to
the existing Screener. Momentum (Clenow slope×R²) + pullback entry + low-vol,
gated by market regime and a per-name trend/liquidity filter.
"""
import streamlit as st
import pandas as pd

from utils.data_fetcher import fetch_daily_batch, get_stock_data
from utils.watchlist import ALL_YF_SYMBOLS

try:
    from utils.swing_screen import (run_swing_screen, market_regime, hold_exit_note,
                                    get_sp500_tickers, LOOKBACK, DEFAULT_WEIGHTS)
    _OK, _ERR = True, None
except Exception as e:
    _OK, _ERR = False, f"{type(e).__name__}: {e}"

st.set_page_config(page_title="Swing Screen · StratFlow", page_icon="🎚️", layout="wide")
st.title("🎚️ Swing Screen")
st.caption("Multi-factor screen for 1-week to 1-month+ holds: intermediate momentum "
           "(Clenow slope×R²) + pullback entry timing + low-vol, gated by market regime "
           "and a per-name trend/liquidity filter. Cross-sectional ranking.")

if not _OK:
    st.error(f"**Import failed:** `{_ERR}`")
    st.warning("Push **utils/swing_screen.py** and ensure **utils/data_fetcher.py** "
               "(fetch_daily_batch) is deployed.")
    st.stop()

c1, c2, c3 = st.columns([1.2, 1.2, 1])
universe_choice = c1.radio("Universe", ["Watchlist (~90)", "S&P 500"], horizontal=True)
horizon = c2.radio("Holding emphasis",
                   ["Short ~1–2wk", "Medium ~3–6wk", "Long 1mo+"], index=1, horizontal=True)
top_n = c3.slider("Show top", 10, 50, 25, step=5)
lb_key = {"Short ~1–2wk": "short", "Medium ~3–6wk": "medium", "Long 1mo+": "long"}[horizon]
lookback = LOOKBACK[lb_key]

with st.expander("⚙️ Factor weights (advanced — defaults are validated priors, not curve-fit)"):
    wc1, wc2, wc3 = st.columns(3)
    w_mom = wc1.slider("Momentum", 0.0, 1.0, DEFAULT_WEIGHTS["momentum"], 0.05)
    w_ent = wc2.slider("Entry (pullback)", 0.0, 1.0, DEFAULT_WEIGHTS["entry"], 0.05)
    w_vol = wc3.slider("Low-vol", 0.0, 1.0, DEFAULT_WEIGHTS["lowvol"], 0.05)
    tot = (w_mom + w_ent + w_vol) or 1.0
    weights = {"momentum": w_mom / tot, "entry": w_ent / tot, "lowvol": w_vol / tot}
    st.caption(f"Normalized → momentum {weights['momentum']:.2f} · entry {weights['entry']:.2f} "
               f"· low-vol {weights['lowvol']:.2f}")

# ── Market regime banner ──────────────────────────────────────────────────────
spy = get_stock_data("SPY", period="2y", interval="1d")
reg = market_regime(spy)
if reg["risk_on"]:
    st.success(f"🟢 **{reg['label']}**" + (f" ( +{reg['spy_vs_200']}% vs 200d )"
               if reg["spy_vs_200"] is not None else "") + " — new longs favored.")
else:
    st.error(f"🔴 **{reg['label']}**" + (f" ( {reg['spy_vs_200']}% vs 200d )"
             if reg["spy_vs_200"] is not None else "") +
             " — system advises caution: stand aside or size down. Candidates shown for context.")

if st.button("🎚️ Run screen", type="primary", width="stretch"):
    st.session_state["swing_run"] = {"u": universe_choice, "lb": lookback,
                                     "w": weights, "n": top_n}

run = st.session_state.get("swing_run")
if run:
    if "S&P 500" in run["u"]:
        syms = get_sp500_tickers()
        st.caption(f"Scanning {len(syms)} S&P 500 names — first run is slow "
                   "(cached 15 min afterward).")
    else:
        syms = list(dict.fromkeys(ALL_YF_SYMBOLS))
    with st.spinner(f"Fetching daily data for {len(syms)} names…"):
        data = fetch_daily_batch(tuple(syms), period="2y")
    if not data:
        st.warning("No data returned (Yahoo + Stooq both unavailable right now). Retry shortly.")
        st.stop()

    res = run_swing_screen(data, lookback=run["lb"], weights=run["w"], top_n=run["n"])
    if res.empty:
        st.info("No names passed the trend + liquidity gate in this universe right now. "
                "In a risk-off tape that's expected.")
        st.stop()

    st.subheader(f"🏁 Top {len(res)} candidates")
    show = res.copy()
    show["Entry"] = show.apply(lambda r: f"${r['entry_low']:.2f}–${r['entry_high']:.2f}", axis=1)
    table = show[["ticker", "score", "ann_ret_pct", "r2", "rsi", "vol_pct",
                  "Entry", "stop", "weight_pct"]].rename(columns={
        "ticker": "Ticker", "score": "Score", "ann_ret_pct": "Ann %", "r2": "R²",
        "rsi": "RSI", "vol_pct": "Vol %", "stop": "Stop", "weight_pct": "Wt %"})

    def _score_css(v):
        try:
            t = float(v) / 100
        except Exception:
            return ""
        r, g, b = int(200 - 200 * t), int(80 + 120 * t), int(80)
        return f"background-color: rgba({r},{g},{b},0.40); font-weight:700"
    try:
        sty = table.style
        (sty.map if hasattr(sty, "map") else sty.applymap)(_score_css, subset=["Score"])
        st.dataframe(sty, width="stretch", hide_index=True, height=min(720, 60 + 30 * len(table)),
                     column_config={"Stop": st.column_config.NumberColumn(format="$%.2f"),
                                    "Ann %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Vol %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Wt %": st.column_config.NumberColumn(format="%.1f%%")})
    except Exception:
        st.dataframe(table, width="stretch", hide_index=True)

    st.caption("**Score** = momentum + entry + low-vol percentiles (your weights). "
               "**Ann %** annualized regression return · **R²** trend smoothness · "
               "**RSI** lower = more pulled-back (better entry) · **Wt %** inverse-vol risk-parity size. "
               "**Entry** = pullback zone toward the 20-day; **Stop** = 2.5×ATR.")

    with st.expander("📋 Hold / exit rules per candidate"):
        for _, r in res.iterrows():
            st.markdown(f"**{r['ticker']}** — {hold_exit_note(r)}")

st.divider()
st.caption("⚠️ Factor weights are evidence-based priors, **not yet validated out-of-sample on "
           "your universe**. Treat as a research screen; confirm edge in Validation before sizing up. "
           "Signal-based decision support, not financial advice.")
