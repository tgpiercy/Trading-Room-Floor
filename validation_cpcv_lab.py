"""
validation_cpcv_lab.py
CPCV LAB — empirical Probability of Backtest Overfitting on the selection
config family. The empirical complement to the analytic DSR: DSR said the
chosen selector clears the multiple-testing bar; PBO asks, combinatorially,
whether picking the in-sample-best (mom, corr) cell is itself an overfit act.

It reconstructs each config's full weekly return series through the frozen
stack (composite selector → redundancy filter → hold-band/decay + 4×ATR trail
→ regime → equal-weight sizing, the labs' canonical convention), assembles the
T × N returns matrix, and runs combinatorially-symmetric cross-validation
(utils/cpcv) over the mom∈{13,26,39} × corr∈{0.80,0.85,0.90} grid.

Reports: PBO, the in-sample-best distribution across the grid (a flat spread =
plateau, no single cell dominates = robust), and the production config's
OOS-Sharpe distribution across all combinatorial splits (a confidence interval
on the frozen strategy). Expectation: LOW PBO — the Gate-3 plateau and DSR 1.0
already implied the cells are interchangeable, so IS-best should hold up OOS.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.exits import decay_matrix, EXIT_RANK, CONFIRM_WEEKS, TRAIL_K
from utils.cpcv import cscv_pbo, verdict as pbo_verdict
from stratflow_adapter import (get_download_symbols, prepare, aux_frames,
                               universe_label, ENTRY_TOP_N)

ATR_PERIOD = 14
VOL_WEEKS = 26
CORR_MAX = 0.85
CORR_WIN = 26
CAND_POOL = 25
SWEEP_MOM = [13, 26, 39]
SWEEP_CORR = [0.80, 0.85, 0.90]
PROD = (26, 0.85)
ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="CPCV Lab", layout="wide")
st.title("CPCV Lab — Probability of Backtest Overfitting")
st.caption(f"**{universe_label()}** · empirical PBO over the "
           f"{len(SWEEP_MOM)}×{len(SWEEP_CORR)} selection grid through the "
           "frozen stack. Complements the analytic DSR.")

with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 5, 15, 10)
    cost_bps = st.slider("Cost (bps/turn)", 0, 30, 10)
    S = st.select_slider("CSCV blocks (S)", [8, 10, 12, 14, 16], value=16,
                         help="More blocks = more splits but smaller test "
                              "windows. C(S, S/2) combinations.")
    run = st.button("Run CPCV Lab", type="primary", use_container_width=True)

if not run:
    st.info("Press **Run CPCV Lab**. Builds 9 config return series, then runs "
            "combinatorial cross-validation for the empirical PBO.")
    st.stop()


# ── Engine (mirrors the validated selection-lab engine) ──────────────────────
def wilder_atr(high, low, close, n=ATR_PERIOD):
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = np.full_like(tr, np.nan)
    if len(tr) <= n:
        return atr
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def simulate_ticker(close, atr, entry, decay, k):
    n = len(close); pos = np.zeros(n, dtype=np.int8)
    in_pos = False; peak = 0.0
    for t in range(n - 1):
        if not in_pos:
            if entry[t] and not np.isnan(atr[t]):
                in_pos = True; peak = close[t]; pos[t + 1] = 1
        else:
            peak = max(peak, close[t])
            exit_now = bool(decay[t])
            if not exit_now and not np.isnan(atr[t]) and close[t] < peak - k * atr[t]:
                exit_now = True
            pos[t + 1] = 0 if exit_now else 1
            in_pos = not exit_now
    return pos


def _pct(df):
    return df.rank(axis=1, pct=True)


def rank_from_score(score):
    return score.rank(axis=1, ascending=False)


def redundancy_filtered_entries(rank, rets, top_n=ENTRY_TOP_N, pool=CAND_POOL,
                                corr_max=CORR_MAX, window=CORR_WIN):
    out = pd.DataFrame(False, index=rank.index, columns=rank.columns)
    R = rets[rank.columns]
    for ti in range(window, len(rank)):
        row = rank.iloc[ti].dropna().sort_values()
        cands = list(row[row <= pool].index)
        if not cands:
            continue
        seg = R.iloc[ti - window:ti]
        accepted = []
        for tk in cands:
            s1 = seg[tk]
            if s1.std() == 0 or s1.isna().all():
                continue
            if any((s1.corr(seg[a]) or 0) > corr_max for a in accepted):
                continue
            accepted.append(tk)
            if len(accepted) >= top_n:
                break
        out.iloc[ti, [out.columns.get_loc(a) for a in accepted]] = True
    return out


def config_port(rank, corr_max, data, rets_df, exposure, idx, cost_bps):
    """Weekly portfolio return series for one config (equal-weight, regime,
    costed) — the labs' canonical sizing convention."""
    entry = redundancy_filtered_entries(rank, rets_df, corr_max=corr_max)
    decay = decay_matrix(rank, EXIT_RANK, CONFIRM_WEEKS)
    cols = list(data.keys())
    pos_mat = np.zeros((len(idx), len(cols)))
    for j, t in enumerate(cols):
        d = data[t].reindex(idx)
        close = d["close"].to_numpy()
        valid = ~np.isnan(close)
        if valid.sum() < 40:
            continue
        atr = np.full(len(idx), np.nan)
        atr[valid] = wilder_atr(d["high"].to_numpy()[valid],
                                d["low"].to_numpy()[valid], close[valid])
        e = entry[t].to_numpy() & valid if t in entry.columns else np.zeros(len(idx), bool)
        x = decay[t].to_numpy() | ~valid if t in decay.columns else np.ones(len(idx), bool)
        c2 = pd.Series(np.where(valid, close, np.nan)).ffill().to_numpy()
        pos_mat[:, j] = simulate_ticker(c2, atr, e, x, TRAIL_K) * valid
    rmat = rets_df[cols].to_numpy()
    n_active = pos_mat.sum(axis=1)
    w = np.divide(pos_mat, n_active[:, None],
                  out=np.zeros_like(pos_mat), where=n_active[:, None] > 0)
    ex = exposure.reindex(idx).ffill().fillna(0.5).to_numpy()
    w = w * ex[:, None]
    port = np.nansum(w * rmat, axis=1)
    if cost_bps > 0:
        dw = np.abs(np.diff(w, axis=0, prepend=w[:1] * 0)).sum(axis=1)
        port = port - dw * (cost_bps / 1e4)
    return pd.Series(port, index=idx)


# ── Build config family return matrix ────────────────────────────────────────
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

data, exposure, ext = P["data"], P["exposure"], P["ext"]
idx = next(iter(data.values())).index
aux = aux_frames(ohlcv, idx)
rs, close_df = aux["rs"], aux["close"]
rets_df = close_df.pct_change()
rs_vol = rs.pct_change().rolling(VOL_WEEKS).std() * 100
ext_adj = ext / rs_vol.replace(0, np.nan)

labels, series, prod_idx = [], {}, 0
prog = st.progress(0.0, text="Building config return series…")
grid = [(m, c) for m in SWEEP_MOM for c in SWEEP_CORR]
for k, (mw, cm) in enumerate(grid):
    rank = rank_from_score(_pct(rs.pct_change(mw)) + _pct(ext_adj))
    lab = f"mom{mw}/corr{cm:.2f}"
    labels.append(lab)
    if (mw, cm) == PROD:
        prod_idx = k
    series[lab] = config_port(rank, cm, data, rets_df, exposure, idx, cost_bps)
    prog.progress((k + 1) / len(grid), text=f"Done: {lab}")
prog.empty()

mat = pd.DataFrame(series).dropna()
st.caption(f"Return matrix: {mat.shape[0]} weeks × {mat.shape[1]} configs · "
           f"production = **{labels[prod_idx]}** (tracked for the OOS "
           "distribution).")

# ── Run CSCV / PBO ────────────────────────────────────────────────────────────
try:
    res = cscv_pbo(mat.to_numpy(), S=int(S), prod_idx=prod_idx)
except ValueError as e:
    st.error(f"CSCV could not run: {e}")
    st.stop()

st.subheader("Probability of Backtest Overfitting")
st.markdown("- " + pbo_verdict(res))
st.caption("PBO ≤ 0.10 = robust selection (in-sample winners stay winners "
           "out-of-sample) · → 0.5 = overfit / no real edge. This is the "
           "empirical complement to the analytic Deflated Sharpe Ratio.")

c1, c2, c3 = st.columns(3)
c1.metric("PBO", f"{res['pbo']:.3f}")
c2.metric("Combinatorial splits", f"{res['n_combinations']:,}")
if "prod_oos_sharpe" in res:
    d = res["prod_oos_sharpe"]
    c3.metric("Prod OOS Sharpe (median)", f"{d['median']:.2f}",
              f"5–95%: {d['p05']:.2f}–{d['p95']:.2f}")

st.subheader("In-sample-best distribution across the grid")
ibd = pd.DataFrame({"config": labels, "IS-best wins": res["is_best_distribution"]})
st.dataframe(ibd.set_index("config"), use_container_width=True)
st.caption("A flat spread across cells = a true plateau (no single config "
           "dominates in-sample) = the selection is not knife-edge. A single "
           "cell hogging the wins would itself be an overfitting warning.")

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("📋 Results for Claude")
payload = {"stage": "cpcv_lab_v1",
           "settings": {"years": years, "cost_bps": cost_bps, "S": int(S),
                        "n_weeks": int(mat.shape[0]), "grid": labels,
                        "production": labels[prod_idx]},
           "pbo": res}
st.code(json.dumps(payload, indent=1, default=str), language="json")
