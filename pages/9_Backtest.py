"""
pages/9_Backtest.py
Signal Edge Backtest — rigorous forward-return event study across the watchlist.
Phase 1 of the systematic loop: prove which signals actually predict, before
combining anything. No lookahead; every signal measured against a random-pick
universe baseline.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.watchlist import PORTFOLIO, GROUP_ORDER, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch
from utils.backtest import (build_signal_panel, universe_baseline, edge_table,
                            signal_edge_ranking, FWD_WEEKS,
                            multi_horizon_table, concentration_check, subperiod_split)

st.set_page_config(page_title="Backtest · StratFlow", page_icon="🔬", layout="wide")
st.title("🔬 Signal Edge Backtest")
st.caption("Forward-return event study · no lookahead · each signal vs a "
           "random-pick baseline. Phase 1: prove what works before combining.")

with st.sidebar:
    st.header("⚙️ Settings")
    period = st.selectbox("History", ["2y", "3y", "5y"], index=1)
    primary_w = st.selectbox("Primary horizon (weeks)", list(FWD_WEEKS), index=1)
    groups = st.multiselect("Universe", GROUP_ORDER, default=GROUP_ORDER)
    run_btn = st.button("▶ Run Backtest", type="primary", width="stretch")
    st.caption("⏱ Builds the full panel — 1–3 min on first run for 5y. Cached after.")

pairs = [(t, b, g) for t, b, g in PORTFOLIO if g in groups]

# ── Run ───────────────────────────────────────────────────────────────────────
_key = (period, tuple(groups))
if run_btn or st.session_state.get("bt_key") != _key:
    syms = tuple(dict.fromkeys(yf_sym(t) for p in pairs for t in (p[0], p[1])))
    with st.spinner("Downloading OHLCV…"):
        ohlcv = fetch_ohlcv_batch(syms, period=period)
    if not ohlcv:
        st.error("Download failed (possibly rate-limited). Wait a moment and rerun.")
        st.stop()
    prog = st.progress(0.0, text="Building signal panel…")
    def _cb(frac, name):
        prog.progress(min(frac, 1.0), text=f"Computing signals… {name}")
    panel = build_signal_panel(ohlcv, pairs, progress_cb=_cb)
    prog.empty()
    st.session_state["bt_panel"] = panel
    st.session_state["bt_key"] = _key

panel: pd.DataFrame = st.session_state.get("bt_panel", pd.DataFrame())
if panel.empty:
    st.info("Click ▶ Run Backtest to begin.")
    st.stop()

base = universe_baseline(panel)
st.caption(f"Panel: **{len(panel):,}** signal observations across "
           f"{panel['ticker'].nunique()} names, {period} history.")

# ── Universe baseline ─────────────────────────────────────────────────────────
st.subheader("🎯 Universe Baseline")
st.caption("The 'random pick from your watchlist' benchmark. A signal only has "
           "edge if it beats THIS, not zero.")
bc = st.columns(len(FWD_WEEKS))
for col, w in zip(bc, FWD_WEEKS):
    if w in base:
        col.metric(f"{w}-week forward", f"{base[w]['mean']:+.2f}%",
                   f"{base[w]['win']:.0f}% positive")
st.divider()

# ── Discrimination ranking ────────────────────────────────────────────────────
st.subheader("🏅 Which Signals Separate Winners from Losers?")
st.caption(f"Spread = best bucket − worst bucket forward return at {primary_w}w. "
           "Higher = the signal discriminates returns more.")
rank = signal_edge_ranking(panel, base, primary_w)
if not rank.empty:
    fig = go.Figure(go.Bar(
        x=rank["Spread (discrimination)"], y=rank["Signal"], orientation="h",
        marker_color="#4fc3f7",
        text=[f"{v:+.2f}%" for v in rank["Spread (discrimination)"]],
        textposition="outside"))
    fig.update_layout(template="plotly_dark", height=240,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title=f"Best−Worst spread at {primary_w}w (%)",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
st.divider()

# ── Per-signal edge tables ────────────────────────────────────────────────────
st.subheader("📋 Signal-by-Signal Edge")
SIGNALS = {
    "RS State": "rs_state", "GW2 Score": "gw2_bucket", "Impulse": "impulse",
    "RS Momentum": "rs_mom", "RRG Quadrant": "quadrant",
}
base_mean = base.get(primary_w, {}).get("mean", 0.0)

def _style(df):
    def _row(r):
        e = r.get("Edge vs Univ", 0)
        c = "#00cc66" if e > 0.3 else "#ff4444" if e < -0.3 else "#888888"
        return [f"background-color:{c}1c"] * len(r)
    return df.style.apply(_row, axis=1)

tabs = st.tabs(list(SIGNALS.keys()))
verdicts = []
for tab, (label, col) in zip(tabs, SIGNALS.items()):
    with tab:
        et = edge_table(panel, col, base, primary_w)
        if et.empty:
            st.info("Insufficient sample.")
            continue
        st.dataframe(_style(et), width="stretch", hide_index=True)
        # Verdict for this signal: best bucket vs baseline
        best = et.iloc[0]
        edge = best["Edge vs Univ"]
        win  = best["Win %"]
        n    = int(best["N"])
        real = edge > 0.5 and win > 50 and n >= 50
        verdicts.append((label, best["Bucket"], edge, win, n, real))
        if real:
            st.success(f"✅ **{label} = {best['Bucket']}** beats the universe by "
                       f"**{edge:+.2f}%** at {primary_w}w "
                       f"({win:.0f}% win, n={n}). Looks like real edge.")
        else:
            st.warning(f"⚠️ **{label}**'s best bucket ({best['Bucket']}) shows "
                       f"{edge:+.2f}% edge ({win:.0f}% win, n={n}) — "
                       f"weak or unreliable on this sample.")

# ── Overall verdict ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🧭 Verdict — What to Build On")
real_ones = [v for v in verdicts if v[5]]
if real_ones:
    st.markdown("**Signals with measurable standalone edge** (worth combining in Phase 2):")
    for label, bucket, edge, win, n, _ in sorted(real_ones, key=lambda x: -x[2]):
        st.markdown(f"- **{label} = {bucket}** → {edge:+.2f}% vs universe at "
                    f"{primary_w}w, {win:.0f}% win rate (n={n})")
else:
    st.warning("No signal cleared the edge bar on this sample/period. Try a longer "
               "history, a different horizon, or a broader universe before concluding.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — is an edge real, or a horizon / concentration / regime mirage?
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔬 Diagnostics — Stress-Test an Edge")
st.caption("Before trusting any bucket, check three things: does it hold across "
           "horizons, is it broad or one-name-driven, and does it persist across years?")

diag_signal = st.selectbox("Signal to inspect", list(SIGNALS.keys()))
dcol = SIGNALS[diag_signal]

# ── 1. Multi-horizon ──────────────────────────────────────────────────────────
st.subheader("1️⃣ Across Horizons")
st.caption("If a 'strong' bucket wins at 1w but loses at 12w, your signal is a "
           "short-horizon momentum tool and the 12w table was measuring mean reversion.")
mh = multi_horizon_table(panel, dcol)
if not mh.empty:
    def _mh_style(df):
        def _row(r):
            v = r.get(f"{FWD_WEEKS[0]}w %", 0) or 0
            c = "#00cc66" if v > 0.3 else "#ff4444" if v < -0.3 else "#888"
            return [f"background-color:{c}14"]*len(r)
        return df.style.apply(_row, axis=1)
    st.dataframe(_mh_style(mh), width="stretch", hide_index=True)

# ── 2 & 3: pick a bucket ──────────────────────────────────────────────────────
buckets = [b for b in panel[dcol].astype(str).unique()]
diag_bucket = st.selectbox("Bucket to stress-test", sorted(buckets))

cc = concentration_check(panel, dcol, diag_bucket, primary_w)
if cc:
    st.subheader("2️⃣ Breadth & Concentration")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bucket mean", f"{cc['mean']:+.2f}%")
    k2.metric("Breadth", f"{cc['breadth']:.0f}%", "names positive")
    k3.metric("Ex–top name", f"{cc['mean_ex_top1']:+.2f}%",
              f"removed {cc['top_name']}")
    k4.metric("Distinct names", f"{cc['n_names']} / {cc['n_months']}mo")

    # Interpretation
    collapse = (cc["mean"] > 0 and cc["mean_ex_top1"] is not None
                and cc["mean_ex_top1"] < cc["mean"] * 0.5)
    if cc["breadth"] >= 60 and not collapse:
        st.success(f"✅ **Broad edge** — {cc['breadth']:.0f}% of names positive and the "
                   f"edge survives removing {cc['top_name']}. Trustworthy.")
    elif collapse or cc["breadth"] < 45:
        st.error(f"⚠️ **Concentrated** — edge leans on a few names "
                 f"({cc['breadth']:.0f}% breadth; drops to {cc['mean_ex_top1']:+.2f}% "
                 f"without {cc['top_name']}). Likely an artifact, not a tradeable edge.")
    else:
        st.warning(f"🟠 **Mixed** — moderate breadth ({cc['breadth']:.0f}%). Treat with caution.")

    st.caption("Contribution to bucket mean (percentage points) by name:")
    st.dataframe(cc["top_names"], width="stretch", hide_index=True,
                 column_config={"mean": st.column_config.NumberColumn("Name mean %", format="%.2f"),
                                "contrib_pp": st.column_config.NumberColumn("Contrib (pp)", format="%.3f")})

    # ── 3. Sub-period ─────────────────────────────────────────────────────────
    st.subheader("3️⃣ Across Years (regime persistence)")
    sp = subperiod_split(panel, dcol, diag_bucket, primary_w)
    if not sp.empty:
        fig = go.Figure(go.Bar(
            x=sp["Year"].astype(str), y=sp[f"Mean {primary_w}w %"],
            marker_color=["#00cc66" if v > 0 else "#ff4444" for v in sp[f"Mean {primary_w}w %"]],
            text=[f"{v:+.1f}%<br>n={n}" for v, n in zip(sp[f"Mean {primary_w}w %"], sp["N"])],
            textposition="outside"))
        fig.update_layout(template="plotly_dark", height=260,
                          yaxis_title=f"Mean {primary_w}w return %",
                          margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, width="stretch")
        pos_years = (sp[f"Mean {primary_w}w %"] > 0).sum()
        tot_years = len(sp)
        if pos_years == tot_years:
            st.success(f"✅ Positive in **all {tot_years} years** — persistent across regimes.")
        elif pos_years <= 1:
            st.error(f"⚠️ Positive in only **{pos_years}/{tot_years} years** — "
                     f"this is a one-regime effect, not a durable edge.")
        else:
            st.warning(f"🟠 Positive in **{pos_years}/{tot_years} years** — partially regime-dependent.")

st.divider()
st.caption("⚠️ **Read with discipline.** (1) Survivorship bias — yfinance omits "
           "delisted names, so results skew optimistic. (2) Small buckets (low N) are "
           "noise, not edge. (3) An edge here is necessary but not sufficient — Phase 2 "
           "(combining) and out-of-sample validation come next. (4) Past edge can decay; "
           "Phase 3 will monitor that.")
