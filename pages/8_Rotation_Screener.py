"""
pages/8_Rotation_Screener.py
CAPITAL ROTATION SCREENER — sleeve-level rotation detection built on the
validated composite machinery, with its OWN validation tab. Price-based
(backtestable today), unlike flow-based rotation which waits on the tape.

Per sleeve (watchlist group, ≥3 members), three causal components, each a
cross-sleeve percentile, averaged into a 0-100 Rotation Score:
  1. RANK MIGRATION  — 4-week improvement in the sleeve's mean composite
                       rank (capital moving IN shows up here first)
  2. BREADTH         — % of members with positive 26w RS momentum
                       (rotation is broad; one hot name is not rotation)
  3. RS ACCELERATION — sleeve mean RS 4w-change minus the prior 4w-change
                       (second derivative: the turn, before the trend)

VALIDATION TAB: event study — does this week's Rotation Score quartile
predict the sleeve's forward 4-week return vs SPY? Run it BEFORE trusting
the live screen. Diagnostic surface: names still enter ONLY via the
selector; this tells you which sleeves to expect them from next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.watchlist import PORTFOLIO, GROUP_ORDER
from stratflow_adapter import get_download_symbols, prepare, aux_frames
from utils.selection import composite_components, MOM_WEEKS

MIG_WEEKS = 4
FWD_WEEKS = 4
MIN_MEMBERS = 3
N_FOLDS = 4
ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="Rotation Screener", layout="wide")
st.title("🔄 Rotation Screener — sleeve-level capital rotation")
st.caption("Rank migration + breadth + RS acceleration per sleeve, on the "
           "validated composite machinery. Diagnostic: tells you WHERE the "
           "selector's next entries are likely to come from. Validate in "
           "the second tab before leaning on it.")

with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 3, 15, 10)
    run = st.button("Run", type="primary", use_container_width=True)

if not run and "rot_data" not in st.session_state:
    st.info("Press **Run** to compute the live screen and the event study.")
    st.stop()

if run or "rot_data" not in st.session_state:
    with st.spinner("Downloading universe + benchmarks…"):
        ohlcv = fetch_ohlcv_batch(get_download_symbols(), period=f"{years}y")
    if not ohlcv:
        st.error("Data fetch failed (possibly rate-limited). Retry shortly.")
        st.stop()
    try:
        P = prepare(ohlcv)
    except ValueError as e:
        st.error(f"Signal engine: {e}")
        st.stop()
    idx = next(iter(P["data"].values())).index
    aux = aux_frames(ohlcv, idx)
    mom_pct, extadj_pct, rank_df = composite_components(P["ext"], aux["rs"])
    spy = ohlcv["SPY"]["Close"].reindex(idx).ffill()
    st.session_state["rot_data"] = (idx, aux, rank_df, spy)

idx, aux, rank_df, spy = st.session_state["rot_data"]
rs, close_df = aux["rs"], aux["close"]
mom26 = rs.pct_change(MOM_WEEKS)

# ── sleeve structures ─────────────────────────────────────────────────────────
members = {}
for tk, _b, g in PORTFOLIO:
    if tk in rank_df.columns:
        members.setdefault(g, []).append(tk)
members = {g: m for g, m in members.items() if len(set(m)) >= MIN_MEMBERS}

def sleeve_frame(df, agg="mean"):
    out = {}
    for g, m in members.items():
        sub = df[[t for t in dict.fromkeys(m)]]
        out[g] = sub.mean(axis=1) if agg == "mean" else sub
    return pd.DataFrame(out, index=df.index)

sleeve_rank = sleeve_frame(rank_df)                    # lower = stronger
mig = -sleeve_rank.diff(MIG_WEEKS)                     # + = improving rank
breadth = pd.DataFrame({g: (mom26[list(dict.fromkeys(m))] > 0).mean(axis=1)
                        for g, m in members.items()}, index=idx)
srs = sleeve_frame(rs)
accel = srs.pct_change(MIG_WEEKS) - srs.pct_change(MIG_WEEKS).shift(MIG_WEEKS)

score = (mig.rank(axis=1, pct=True)
         + breadth.rank(axis=1, pct=True)
         + accel.rank(axis=1, pct=True)) / 3 * 100     # 0-100 rotation score

# forward sleeve return vs SPY (for the event study)
rel = close_df.div(spy, axis=0)
fwd = rel.shift(-FWD_WEEKS) / rel - 1
sleeve_fwd = pd.DataFrame({g: fwd[list(dict.fromkeys(m))].mean(axis=1)
                           for g, m in members.items()}, index=idx) * 100

tab_live, tab_val = st.tabs(["🔄 Live screen", "🔬 Event study (validate first)"])

with tab_live:
    last = pd.DataFrame({
        "Rotation score": score.iloc[-1].round(0),
        "Δ4w score": (score.iloc[-1] - score.iloc[-1 - MIG_WEEKS]).round(0),
        "Mean rank": sleeve_rank.iloc[-1].round(1),
        "Rank Δ4w": mig.iloc[-1].round(1),
        "Breadth %": (breadth.iloc[-1] * 100).round(0),
        "Members": pd.Series({g: len(set(m)) for g, m in members.items()}),
    }).sort_values("Rotation score", ascending=False)
    st.dataframe(last, use_container_width=True)
    st.caption("**Rotation score** = cross-sleeve percentile of rank "
               "migration + breadth + RS acceleration. **Δ4w score** rising "
               "fast = rotation igniting. **Rank Δ4w** positive = members "
               "climbing the composite ladder toward the entry zone. Watch "
               "the top 2-3 sleeves for the selector's next entries.")
    pick = st.selectbox("Sleeve detail", list(last.index))
    mtab = pd.DataFrame({
        "Composite rank": rank_df[list(dict.fromkeys(members[pick]))].iloc[-1],
        "Rank Δ4w": -rank_df[list(dict.fromkeys(members[pick]))].diff(MIG_WEEKS).iloc[-1],
        "Mom26 +": mom26[list(dict.fromkeys(members[pick]))].iloc[-1] > 0,
    }).sort_values("Composite rank")
    st.dataframe(mtab, use_container_width=True)

with tab_val:
    st.caption(f"Does this week's Rotation Score predict the sleeve's "
               f"forward {FWD_WEEKS}-week return vs SPY? Quartile event "
               f"study across all (sleeve, week) observations.")
    long = pd.DataFrame({"score": score.stack(), "fwd": sleeve_fwd.stack()}
                        ).dropna()
    long["q"] = pd.qcut(long["score"], 4,
                        labels=["Q1 (cold)", "Q2", "Q3", "Q4 (hot)"])
    et = long.groupby("q", observed=True)["fwd"].agg(
        **{"Mean fwd %": "mean", "Win %": lambda s: (s > 0).mean() * 100,
           "N": "count"}).round(2)
    spread = float(et["Mean fwd %"].iloc[-1] - et["Mean fwd %"].iloc[0])
    st.dataframe(et, use_container_width=True)

    # fold persistence of the Q4-Q1 spread
    dates = long.index.get_level_values(0)
    edges = pd.Series(np.linspace(0, 1, N_FOLDS + 1)).quantile
    qt = dates.unique().sort_values()
    fold_rows = {}
    cuts = np.linspace(0, len(qt), N_FOLDS + 1).astype(int)
    for fi in range(N_FOLDS):
        sub = long[(dates >= qt[cuts[fi]]) & (dates < qt[min(cuts[fi + 1],
                                                             len(qt) - 1)])]
        if len(sub) < 40:
            continue
        g = sub.groupby("q", observed=True)["fwd"].mean()
        fold_rows[f"F{fi+1} ({qt[cuts[fi]].year}-{qt[min(cuts[fi+1],len(qt)-1)].year})"] = \
            round(float(g.iloc[-1] - g.iloc[0]), 2)
    st.subheader("Q4−Q1 spread per fold (pp, fwd 4w vs SPY)")
    st.dataframe(pd.Series(fold_rows, name="Spread").to_frame().T,
                 use_container_width=True)
    if spread > 0.3 and all(v > 0 for v in fold_rows.values()):
        st.success(f"✅ Hot sleeves beat cold by {spread:+.2f}pp at "
                   f"{FWD_WEEKS}w, positive in every fold — the screen "
                   "carries signal. Still diagnostic-only until it earns a "
                   "lab arm.")
    elif spread > 0:
        st.warning(f"🟠 Positive but inconsistent spread ({spread:+.2f}pp). "
                   "Treat as weak context.")
    else:
        st.error(f"🛑 No predictive spread ({spread:+.2f}pp) — use the "
                 "screen as description, not prediction.")

    import json as _json
    payload = {"stage": "rotation_screen_v1",
               "settings": {"years": years, "mig_weeks": MIG_WEEKS,
                            "fwd_weeks": FWD_WEEKS,
                            "n_sleeves": len(members),
                            "n_obs": int(len(long))},
               "quartiles": et.reset_index().astype(str).to_dict("records"),
               "spread_q4_q1": round(spread, 2),
               "folds": fold_rows}
    st.subheader("📋 Results for Claude")
    st.code(_json.dumps(payload, indent=1, default=str), language="json")
