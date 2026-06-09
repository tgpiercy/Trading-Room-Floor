"""
pages/11_Validation.py
Phase 3 — Walk-forward validation. The honest test: pick parameters on each
2y train window, trade the next 6mo unseen, stitch into one out-of-sample curve.
Out-of-sample Sharpe vs in-sample Sharpe is the verdict.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.watchlist import PORTFOLIO, GROUP_ORDER, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch

try:
    from utils.strategy_backtest import walk_forward
    _OK, _ERR = True, None
except Exception as e:
    _OK, _ERR = False, f"{type(e).__name__}: {e}"

st.title("🧪 Walk-Forward Validation")
st.caption("Phase 3 — pick params on each train window, trade the next UNSEEN window, "
           "roll forward. Out-of-sample vs in-sample is the truth.")

if not _OK:
    st.error(f"**Import failed:** `{_ERR}`")
    if "walk_forward" in str(_ERR):
        st.warning("The running `strategy_backtest.py` lacks `walk_forward` — even if GitHub "
                   "shows it. That's a **stale deploy**: Streamlit is running a cached copy. "
                   "Fix: **Manage app → Reboot**, and check for a committed `utils/__pycache__/` "
                   "folder (delete it + add `__pycache__/` to `.gitignore`) — stale `.pyc` files "
                   "shadow the real source.")
    else:
        st.warning("This isn't a missing `walk_forward` — it's a different import error "
                   "(shown above), likely a dependency. Tell me that error line and I'll trace it.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    period = st.selectbox("History", ["5y", "max"], index=0,
                          help="Walk-forward needs long history (≥3.5y).")
    train_w = st.slider("Train window (weeks)", 78, 156, 104, step=13)
    test_w  = st.slider("Test window (weeks)", 13, 52, 26, step=13)
    groups  = st.multiselect("Universe", GROUP_ORDER, default=GROUP_ORDER)
    signal_choice = st.radio("System to validate",
                             ["RS Extension", "RS + trend filter", "Compare both"],
                             index=2,
                             help="RS Extension = the validated ExtPct ranking. "
                                  "RS + trend filter = same ranking, but skip names that are "
                                  "chopping or rolling over (the trend-state filter).")
    run_btn = st.button("▶ Run Validation", type="primary", width="stretch")
    st.caption("⏱ Grid-searches each train window — 2–4 min (×2 for Compare).")

pairs = [(t, b, g) for t, b, g in PORTFOLIO if g in groups]

_SIG = {"RS Extension": "extpct", "RS + trend filter": "extpct_filtered"}
_key = (period, train_w, test_w, tuple(groups), signal_choice)
if run_btn or st.session_state.get("wf_key") != _key:
    syms = set(yf_sym(t) for p in pairs for t in (p[0], p[1])) | {"SPY", "IEF", "^VIX"}
    with st.spinner("Downloading history…"):
        ohlcv = fetch_ohlcv_batch(tuple(syms), period=period)
    if not ohlcv:
        st.error("Download failed (possibly rate-limited). Wait and rerun.")
        st.stop()

    def _runwf(sig, label):
        prog = st.progress(0.0, text=f"Walk-forward ({label})…")
        r = walk_forward(ohlcv, pairs, train_weeks=train_w, test_weeks=test_w, signal=sig,
                         progress_cb=lambda f: prog.progress(min(f, 1.0),
                                                             text=f"Walk-forward ({label})…"))
        prog.empty()
        return r

    if signal_choice == "Compare both":
        base = _runwf("extpct", "RS Extension")
        filt = _runwf("extpct_filtered", "RS + trend filter")
        st.session_state["wf_compare"] = {"RS Extension": base, "RS + trend filter": filt}
        st.session_state["wf_res"] = base if "error" not in base else filt
    else:
        st.session_state["wf_compare"] = None
        st.session_state["wf_res"] = _runwf(_SIG[signal_choice], signal_choice)
    st.session_state["wf_key"] = _key

res = st.session_state.get("wf_res", {})
if not res:
    st.info("Click ▶ Run Validation to begin.")
    st.stop()

# ── Head-to-head comparison (if Compare both) ─────────────────────────────────
cmp = st.session_state.get("wf_compare")
if cmp:
    st.subheader("⚔️ Head-to-head — out-of-sample")
    rows, spy_cagr = [], None
    for name, r in cmp.items():
        if "error" in r:
            rows.append({"System": name, "OOS Sharpe": "—", "OOS CAGR %": "—",
                         "OOS MaxDD %": "—", "WFE": "—"})
            continue
        m = r["oos_metrics"]
        rows.append({"System": name, "OOS Sharpe": r["oos_sharpe"],
                     "OOS CAGR %": m.get("CAGR %"), "OOS MaxDD %": m.get("Max Drawdown %"),
                     "WFE": r["wfe"]})
        spy_cagr = spy_cagr or r["spy_metrics"].get("CAGR %")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if spy_cagr is not None:
        st.caption(f"SPY over the same out-of-sample windows: **{spy_cagr}% CAGR**. "
                   "WFE = out-of-sample ÷ in-sample Sharpe (>0.5 is respectable; "
                   "near/above 1.0 means little overfitting). Detail below = RS Extension.")
    st.divider()

if "error" in res:
    st.error(res["error"])
    st.stop()

# ── Verdict ───────────────────────────────────────────────────────────────────
wfe = res["wfe"]; oos_sh = res["oos_sharpe"]; is_sh = res["is_sharpe"]
st.subheader("🧭 Verdict")
v1, v2, v3 = st.columns(3)
v1.metric("In-Sample Sharpe", is_sh, help="Avg Sharpe the optimizer saw on train windows")
v2.metric("Out-of-Sample Sharpe", oos_sh, help="Stitched test windows — the honest number")
v3.metric("Walk-Forward Efficiency", wfe, help="OOS ÷ IS. Near 1 = robust; near 0 = overfit")

if oos_sh > 0.3 and wfe >= 0.5:
    st.success(f"✅ **Edge survives out-of-sample.** OOS Sharpe {oos_sh} holds at {wfe:.0%} "
               f"of in-sample — the strategy generalises to data it never saw. This is the "
               f"strongest evidence so far that the edge is real.")
elif oos_sh > 0 and wfe >= 0.25:
    st.warning(f"🟠 **Degraded but alive.** OOS Sharpe {oos_sh} is positive but only {wfe:.0%} "
               f"of in-sample. Marginal — real but fragile; treat size conservatively.")
else:
    st.error(f"🛑 **Does not survive.** OOS Sharpe {oos_sh} (WFE {wfe}). The in-sample "
             f"result was largely curve-fit / regime-dependent. Do not trade this as-is.")

# vs benchmark, out-of-sample
sm = res["spy_metrics"]; om = res["oos_metrics"]
st.caption(f"Out-of-sample: strategy CAGR **{om.get('CAGR %')}%** (Sharpe {oos_sh}, "
           f"maxDD {om.get('Max Drawdown %')}%) vs SPY **{sm.get('CAGR %')}%** "
           f"(Sharpe {sm.get('Sharpe')}, maxDD {sm.get('Max Drawdown %')}%) over the same window.")
st.divider()

# ── Stitched OOS equity ───────────────────────────────────────────────────────
st.subheader("📈 Out-of-Sample Equity (stitched test windows)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["oos_dates"], y=res["oos_equity"], name="Strategy (OOS)",
                         line=dict(color="#00ff88", width=2)))
fig.add_trace(go.Scatter(x=res["oos_dates"], y=res["spy_equity"], name="SPY (same window)",
                         line=dict(color="#888", width=1.5, dash="dot")))
fig.update_layout(template="plotly_dark", height=400, yaxis_title="Growth of $1",
                  legend=dict(orientation="h", y=1.04), margin=dict(l=0, r=0, t=30, b=0),
                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig, width="stretch")
st.caption("This curve is the honest one — every point was traded with parameters chosen "
           "*before* that data was seen.")
st.divider()

# ── IS vs OOS table ───────────────────────────────────────────────────────────
st.subheader("📊 In-Sample vs Out-of-Sample")
cmp = pd.DataFrame({"Out-of-Sample": res["oos_metrics"], "SPY (OOS window)": res["spy_metrics"]}).T
st.dataframe(cmp, width="stretch")
st.divider()

# ── Per-window detail / param stability ───────────────────────────────────────
st.subheader("🔍 Per-Window: Chosen Params & Train→Test Carryover")
st.caption("If chosen Top-N/Cadence cluster tightly and Test Sharpe tracks Train Sharpe, "
           "the edge is stable. Wild jumps = fragility.")
rolls = pd.DataFrame(res["rolls"])
st.dataframe(rolls, width="stretch", hide_index=True)

# Param stability read
tn_var = rolls["Top N"].nunique(); cad_var = rolls["Cadence"].nunique()
if tn_var <= 2 and cad_var <= 2:
    st.success("✅ Parameters clustered tightly across windows — stable, not fragile.")
elif tn_var >= 4 or cad_var >= 3:
    st.warning("🟠 Parameters jumped around between windows — the 'best' settings are "
               "unstable, a sign the edge is partly noise.")

st.divider()
st.caption("⚠️ Walk-forward is the gold standard but not infallible: still one historical "
           "path, still survivorship-biased (delisted names absent), still **no transaction "
           "costs** (you chose to validate raw edge first — costs are the next layer). "
           "A surviving edge here earns a small live/paper allocation, not the farm.")
