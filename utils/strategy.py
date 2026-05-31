"""
pages/10_Strategy.py
Phase 2 — Combined Strategy backtest. Ranks by RS leadership, gates by regime,
sizes two ways, compares to SPY. The validated signal turned into a system.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.watchlist import PORTFOLIO, GROUP_ORDER, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch

try:
    from utils.strategy_backtest import run_portfolio_backtest
    _OK, _ERR = True, None
except Exception as e:
    _OK, _ERR = False, f"{type(e).__name__}: {e}"

st.set_page_config(page_title="Strategy · StratFlow", page_icon="⚙️", layout="wide")
st.title("⚙️ Combined Strategy")
st.caption("Phase 2 — rank by RS leadership · gate by regime (scaled) · size 2 ways · "
           "vs SPY. The Phase-1 signal, turned into a system.")

if not _OK:
    st.error(f"**Import failed:** `{_ERR}`")
    st.warning("If this names `run_portfolio_backtest` as missing, the running "
               "strategy_backtest.py is a stale cached copy — **Reboot** the app and delete any "
               "committed `utils/__pycache__/`. If it's a different error, send me that line.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    period   = st.selectbox("History", ["2y", "3y", "5y"], index=2)
    top_n    = st.slider("Hold top N names", 3, 25, 10)
    cadence  = st.slider("Rebalance every (weeks)", 1, 12, 4)
    vol_lb   = st.slider("Vol lookback (weeks)", 8, 26, 13)
    groups   = st.multiselect("Universe", GROUP_ORDER, default=GROUP_ORDER)
    run_btn  = st.button("▶ Run Strategy", type="primary", width="stretch")
    st.caption("⏱ First run pulls history; ~1–2 min.")

pairs = [(t, b, g) for t, b, g in PORTFOLIO if g in groups]

_key = (period, top_n, cadence, vol_lb, tuple(groups))
if run_btn or st.session_state.get("strat_key") != _key:
    # regime inputs always included
    syms = set(yf_sym(t) for p in pairs for t in (p[0], p[1]))
    syms |= {"SPY", "IEF", "^VIX"}
    with st.spinner("Downloading history…"):
        ohlcv = fetch_ohlcv_batch(tuple(syms), period=period)
    if not ohlcv:
        st.error("Download failed (possibly rate-limited). Wait and rerun.")
        st.stop()
    with st.spinner("Running portfolio backtest…"):
        res = run_portfolio_backtest(ohlcv, pairs, top_n=top_n,
                                     cadence=cadence, vol_lookback=vol_lb)
    st.session_state["strat_res"] = res
    st.session_state["strat_key"] = _key

res = st.session_state.get("strat_res", {})
if not res:
    st.info("Click ▶ Run Strategy to begin.")
    st.stop()
if "error" in res:
    st.error(res["error"])
    st.stop()

# ── Equity curves ─────────────────────────────────────────────────────────────
st.subheader("📈 Equity Curves")
colors = {"Equal-Weight": "#00ff88", "Vol-Targeted": "#4fc3f7", "SPY (benchmark)": "#888888"}
fig = go.Figure()
for name, eq in res["equity"].items():
    fig.add_trace(go.Scatter(x=res["dates"], y=eq, name=name,
                  line=dict(color=colors.get(name), width=2,
                            dash="dot" if "SPY" in name else "solid")))
fig.update_layout(template="plotly_dark", height=420,
                  yaxis_title="Growth of $1", legend=dict(orientation="h", y=1.04),
                  margin=dict(l=0, r=0, t=30, b=0),
                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig, width="stretch")
st.caption(f"{res['n_rebalances']} rebalances · top {res['top_n']} · "
           f"every {res['cadence']}w. Both sizing methods shown vs SPY.")

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("📊 Performance vs Benchmark")
mt = pd.DataFrame(res["metrics"]).T
def _hl(col):
    return ["font-weight:700" if i < 2 else "" for i in range(len(col))]
st.dataframe(mt, width="stretch")

ew = res["metrics"]["Equal-Weight"]; sp = res["metrics"]["SPY (benchmark)"]
beat = ew.get("CAGR %", 0) - sp.get("CAGR %", 0)
if beat > 0:
    st.success(f"✅ Equal-weight strategy CAGR beat SPY by **{beat:+.1f}%/yr** "
               f"with {'lower' if ew.get('Max Drawdown %',0) > sp.get('Max Drawdown %',0) else 'comparable'} "
               f"drawdown. Sharpe {ew.get('Sharpe')} vs {sp.get('Sharpe')}.")
else:
    st.warning(f"⚠️ Strategy did not beat SPY on CAGR over this window "
               f"({beat:+.1f}%/yr). Risk-adjusted: Sharpe {ew.get('Sharpe')} vs {sp.get('Sharpe')}.")

# ── Regime exposure path ──────────────────────────────────────────────────────
st.subheader("🚦 Regime Exposure Over Time")
st.caption("When the regime gate cut exposure to 0.5 (caution) or 0.0 (risk-off). "
           "This is what protects against momentum crashes like 2022.")
exp_df = pd.DataFrame(res["exposure_path"], columns=["date", "exposure"])
figx = go.Figure(go.Scatter(x=exp_df["date"], y=exp_df["exposure"], fill="tozeroy",
                            line=dict(color="#ffd700", width=1.5), mode="lines"))
figx.update_layout(template="plotly_dark", height=200,
                   yaxis=dict(title="Exposure", range=[-0.05, 1.05],
                              tickvals=[0, 0.5, 1.0]),
                   margin=dict(l=0, r=0, t=10, b=0),
                   paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(figx, width="stretch")

# ── Current holdings ──────────────────────────────────────────────────────────
st.subheader("🎯 Current Top Holdings")
st.caption("What the strategy would hold right now, ranked by RS extension.")
if res["current_holdings"]:
    hold_df = pd.DataFrame(res["current_holdings"], columns=["Ticker", "RS Extension %"])
    st.dataframe(hold_df, width="stretch", hide_index=True,
                 column_config={"RS Extension %": st.column_config.NumberColumn(format="%.2f%%")})

st.divider()
st.caption("⚠️ **In-sample backtest.** This optimises nothing yet, but it's still one "
           "historical path with survivorship bias (delisted names absent) inflating "
           "returns. Cash earns 0% here (conservative). The real test is **Phase 3**: "
           "walk-forward out-of-sample + edge-decay monitoring. Don't simultaneously "
           "tune top-N/cadence to maximise this curve — that's overfitting.")
