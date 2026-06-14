"""
pages/7_Performance.py
PERFORMANCE — the loop-closer. Validation tells you the edge SHOULD be ~1.67
Sharpe; this page measures whether it's actually showing up in your account.

It tracks TOTAL equity (holdings mark-to-market + cash), because the regime
gate's cash periods are part of the strategy and the backtest equity curve
includes them — so comparing invested-value-only to the backtest would flatter
or punish the system unfairly.

Sections:
  1. This week's snapshot — live MtM + your cash → total equity, logged one
     per day to the equity_curve sheet (the realized series everything runs on)
  2. Equity curve vs SPY (rebased) + underwater drawdown
  3. Realized metrics (return / CAGR / vol / Sharpe / Sortino / drawdown) and
     benchmark comparison (excess, tracking error, beta, up/down capture)
  4. Realized-vs-1.67 verdict, with honest short-track-record humility and the
     survivorship caveat on the backtest number itself

Early on this will mostly say "track record too short to judge" — that's
honest, not broken. The point is to start the series and watch it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.journal import log_equity, equity_total_series, storage_status
from utils.portfolio import load_holdings
from utils.data_fetcher import get_current_price, fetch_ohlcv_batch
from utils.watchlist import yf_sym
from utils.performance import (equity_metrics, drawdown_series,
                               benchmark_compare, vs_expectation,
                               _to_weekly, EXPECTED_SHARPE)

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
st.title("📈 Performance — realized vs validated")
st.caption(f"Storage: {storage_status()}")

# ── 1. This week's snapshot ──────────────────────────────────────────────────
st.subheader("This week's snapshot")

hl = [h for h in load_holdings()
      if str(h.get("ticker", "")).strip() and float(h.get("shares") or 0) > 0]
mtm, priced, missed = 0.0, 0, []
for h in hl:
    try:
        p = get_current_price(yf_sym(str(h["ticker"]).upper().strip()))
        if p:
            mtm += float(h["shares"]) * float(p)
            priced += 1
        else:
            missed.append(str(h["ticker"]).upper())
    except Exception:
        missed.append(str(h.get("ticker", "?")).upper())

cash = st.number_input(
    "Cash / dry powder ($) — total equity = holdings MtM + cash",
    min_value=0.0, value=0.0, step=1000.0,
    help="Include the regime-gate cash sleeve so the live curve is comparable "
         "to the backtest, which holds cash in risk-off periods.")
total_equity = mtm + cash

c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdings MtM", f"${mtm:,.0f}", f"{priced}/{len(hl)} priced")
c2.metric("Cash", f"${cash:,.0f}")
c3.metric("Total equity", f"${total_equity:,.0f}")
c4.metric("Positions", f"{priced}")
if missed:
    st.caption("⚠️ Couldn't price: " + ", ".join(missed)
               + " — MtM excludes these. Persistent misses usually mean a "
                 "ticker change or delisting to fix in the universe.")

if st.button("📸 Log this snapshot", type="primary",
             disabled=(priced == 0 and cash == 0)):
    fresh = log_equity(mtm, priced, exposure=None, note=f"{priced}/{len(hl)} priced",
                       total_equity=total_equity, cash=cash)
    if fresh:
        st.success(f"Logged total equity ${total_equity:,.0f} for today.")
    else:
        st.info("Already snapshotted today — one snapshot per day.")

st.divider()

# ── 2/3/4. Realized analytics ────────────────────────────────────────────────
total = equity_total_series()
if total.empty or len(total) < 2:
    st.info("Not enough history yet. Log a snapshot each week (or each "
            "rebalance) and the curve, metrics, and the realized-vs-1.67 "
            "verdict will populate here. Two points start the curve; ~8+ "
            "weeks before any metric is worth reading.")
    st.stop()

# Benchmark (SPY weekly), best-effort
spy_close = None
try:
    spy_px = fetch_ohlcv_batch(("SPY",), period="2y").get("SPY")
    if spy_px is not None and not spy_px.empty:
        spy_close = spy_px["Close"]
except Exception:
    spy_close = None

m = equity_metrics(total)

# 2. Equity curve vs rebased SPY
st.subheader("Equity curve")
tw = _to_weekly(total)
chart = pd.DataFrame({"Total equity": tw})
if spy_close is not None:
    spy_w = _to_weekly(spy_close)
    common = tw.index.intersection(spy_w.index)
    if len(common) >= 2:
        base = float(tw.loc[common].iloc[0])
        spy_reb = base * spy_w.loc[common] / float(spy_w.loc[common].iloc[0])
        chart = pd.DataFrame({"Total equity": tw.loc[common],
                              "SPY (rebased)": spy_reb})
st.line_chart(chart)
st.caption("SPY rebased to your starting equity for a like-for-like shape "
           "comparison — not a claim about dollars.")

# Underwater drawdown
dd = drawdown_series(total)
if not dd.empty:
    st.subheader("Underwater (drawdown)")
    st.area_chart(dd.rename("Drawdown"))

# 3. Realized metrics
st.subheader("Realized metrics")
if "sharpe" not in m:
    st.info("Need at least 2 snapshots for metrics.")
    st.stop()

g1, g2, g3, g4 = st.columns(4)
g1.metric("Total return", f"{m['total_return']*100:.1f}%")
g2.metric("CAGR", f"{m['cagr']*100:.1f}%" if m['cagr'] == m['cagr'] else "—")
g3.metric("Realized Sharpe", f"{m['sharpe']:.2f}")
g4.metric("Sortino", f"{m['sortino']:.2f}" if m.get("sortino") else "—")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Volatility (ann)", f"{m['vol_ann']*100:.1f}%")
h2.metric("Max drawdown", f"{m['max_dd']*100:.1f}%")
h3.metric("Current drawdown", f"{m['current_dd']*100:.1f}%")
h4.metric("Weeks tracked", f"{m['n_weeks']}")
st.caption(f"Span {m['start']} → {m['end']} · "
           f"${m['start_equity']:,.0f} → ${m['end_equity']:,.0f}")

# Benchmark comparison
bc = benchmark_compare(total, spy_close) if spy_close is not None else {"insufficient": True}
if not bc.get("insufficient"):
    st.markdown("**Versus SPY**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Excess CAGR", f"{bc['excess_cagr']*100:.1f}%")
    b2.metric("Tracking error", f"{bc['tracking_error']*100:.1f}%")
    b3.metric("Beta", f"{bc['beta']:.2f}")
    b4.metric("Correlation", f"{bc['corr']:.2f}")
    if bc.get("up_capture") is not None and bc.get("down_capture") is not None:
        st.caption(f"Up capture {bc['up_capture']:.2f} · "
                   f"down capture {bc['down_capture']:.2f} "
                   "(below 1.0 down-capture = you fall less than SPY in "
                   "down weeks — the regime gate's job).")

st.divider()

# 4. Realized vs validated expectation
st.subheader("Is the edge showing up?")
ve = vs_expectation(m["sharpe"], m["n_weeks"])
if ve["enough_data"] and "✅" in ve["verdict"]:
    st.success(ve["verdict"])
elif ve["enough_data"]:
    st.error(ve["verdict"])
else:
    st.warning(ve["verdict"])

st.caption(
    f"Reference: the validated backtest Sharpe is {EXPECTED_SHARPE:.2f} — but "
    "that number is **survivorship-optimistic**: it was computed on today's "
    "surviving universe, so delisted names (e.g. LYTE) never dragged on it. "
    "Expect realized Sharpe to land *somewhat below* the backtest even when "
    "everything is working. In-line-or-slightly-below is healthy; well-below, "
    "once the track record is long enough to trust, is the signal to "
    "investigate slippage, timing, or tax drag.")
