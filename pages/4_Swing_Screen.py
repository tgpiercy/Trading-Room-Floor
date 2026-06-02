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
                                    get_sp500_map, sector_strength, LOOKBACK, DEFAULT_WEIGHTS)
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

active_only = st.checkbox("Active growth only — exclude sideways chop "
                          "(deteriorating trends are always excluded)", value=True)

with st.expander("⚙️ Factor weights (advanced — defaults are validated priors, not curve-fit)"):
    wc1, wc2, wc3, wc4 = st.columns(4)
    w_mom = wc1.slider("Momentum", 0.0, 1.0, DEFAULT_WEIGHTS["momentum"], 0.05)
    w_ent = wc2.slider("Entry (pullback)", 0.0, 1.0, DEFAULT_WEIGHTS["entry"], 0.05)
    w_sec = wc3.slider("Sector bias", 0.0, 1.0, DEFAULT_WEIGHTS["sector"], 0.05)
    w_vol = wc4.slider("Low-vol", 0.0, 1.0, DEFAULT_WEIGHTS["lowvol"], 0.05)
    tot = (w_mom + w_ent + w_sec + w_vol) or 1.0
    weights = {"momentum": w_mom / tot, "entry": w_ent / tot,
               "sector": w_sec / tot, "lowvol": w_vol / tot}
    st.caption(f"Normalized → momentum {weights['momentum']:.2f} · entry {weights['entry']:.2f} "
               f"· sector {weights['sector']:.2f} · low-vol {weights['lowvol']:.2f}")

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
                                     "w": weights, "n": top_n, "ag": active_only}

run = st.session_state.get("swing_run")
if run:
    sp_map = get_sp500_map()                      # {ticker: sector ETF}
    sec_pct = sector_strength(run["lb"])          # {ETF: 0-100 strength}
    if "S&P 500" in run["u"]:
        syms = list(sp_map.keys()) or []
        if not syms:
            from utils.swing_screen import _SP500_FALLBACK
            syms = _SP500_FALLBACK
        st.caption(f"Scanning {len(syms)} S&P 500 names — first run is slow "
                   "(cached 15 min afterward).")
    else:
        syms = list(dict.fromkeys(ALL_YF_SYMBOLS))
    if sec_pct:
        lead = sorted(sec_pct.items(), key=lambda kv: -kv[1])[:3]
        st.caption("Sector leaders: " + " · ".join(f"{e} ({int(p)})" for e, p in lead))

    with st.spinner(f"Fetching daily data for {len(syms)} names…"):
        data = fetch_daily_batch(tuple(syms), period="2y")
    if not data:
        st.warning("No data returned (Yahoo + Stooq both unavailable right now). Retry shortly.")
        st.stop()

    res = run_swing_screen(data, lookback=run["lb"], weights=run["w"], top_n=run["n"],
                           sector_of=sp_map, sector_pct_by_etf=sec_pct,
                           active_growth_only=run["ag"])
    if res.empty:
        st.info("No names passed the trend + liquidity + state gate in this universe right now. "
                "In a risk-off or choppy tape that's expected — try unchecking 'Active growth only'.")
        st.stop()

    st.subheader(f"🏁 Top {len(res)} candidates")
    show = res.copy()
    show["Entry"] = show.apply(lambda r: f"${r['entry_low']:.2f}–${r['entry_high']:.2f}", axis=1)
    table = show[["ticker", "score", "trend_state", "sector", "ann_ret_pct", "r2", "rsi",
                  "vol_pct", "Entry", "stop", "weight_pct"]].rename(columns={
        "ticker": "Ticker", "score": "Score", "trend_state": "State", "sector": "Sector",
        "ann_ret_pct": "Ann %", "r2": "R²", "rsi": "RSI", "vol_pct": "Vol %",
        "stop": "Stop", "weight_pct": "Wt %"})

    def _score_css(v):
        try:
            t = float(v) / 100
        except Exception:
            return ""
        r, g, b = int(200 - 200 * t), int(80 + 120 * t), int(80)
        return f"background-color: rgba({r},{g},{b},0.40); font-weight:700"

    def _state_css(v):
        c = {"Active Growth": "#00cc66", "Early": "#4fc3f7", "Chop": "#ff9800"}.get(v, "#888")
        return f"color:{c}; font-weight:600"
    try:
        sty = table.style
        _m = sty.map if hasattr(sty, "map") else sty.applymap
        _m(_score_css, subset=["Score"])
        _m(_state_css, subset=["State"])
        st.dataframe(sty, width="stretch", hide_index=True, height=min(720, 60 + 30 * len(table)),
                     column_config={"Stop": st.column_config.NumberColumn(format="$%.2f"),
                                    "Ann %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Vol %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Wt %": st.column_config.NumberColumn(format="%.1f%%")})
    except Exception:
        st.dataframe(table, width="stretch", hide_index=True)

    st.caption("**State** = Active Growth / Early / Chop (deteriorating excluded). "
               "**Sector** ETF + its momentum rank feeds the score. **Score** = momentum + "
               "entry + sector + low-vol percentiles (your weights). **Ann %** regression return · "
               "**R²** trend smoothness · **RSI** lower = more pulled-back · **Wt %** inverse-vol size. "
               "**Entry** = pullback zone toward the 20-day; **Stop** = 2.5×ATR.")

    with st.expander("📋 Hold / exit rules per candidate"):
        for _, r in res.iterrows():
            st.markdown(f"**{r['ticker']}** — {hold_exit_note(r)}")

st.divider()
st.caption("⚠️ Factor weights are evidence-based priors, **not yet validated out-of-sample on "
           "your universe**. Treat as a research screen; confirm edge in Validation before sizing up. "
           "Signal-based decision support, not financial advice.")
