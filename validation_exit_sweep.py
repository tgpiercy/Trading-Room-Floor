"""
StratFlow Validation — Exit Layer Stage 1: Chandelier Trail-Width Sweep
==================================================================
Purpose
-------
Sweep the chandelier trail multiplier k across a frozen grid and locate the
performance plateau. The frozen k chosen from this sweep feeds Stage 2
(layered exit comparison: decay-only vs trail-only vs decay+trail).

Methodology notes
-----------------
* Weekly cadence (W-FRI resample of daily data), one-bar execution lag —
  decisions made on bar t take effect for the return earned over bar t+1.
  No lookahead anywhere.
* Trail level = (highest close since entry) - k * ATR. ATR is Wilder's,
  computed on weekly bars, frozen at 14 periods.
* Signals are the VALIDATED StratFlow chain: precompute_series → causal
  ExtPct vs each name's own benchmark → entry on top-10 rank, decay on
  rank>20 or ExtPct<0 (see stratflow_adapter.py for the frozen mapping).
  Data via utils.data_fetcher.fetch_ohlcv_batch (Stooq-resilient).
* IS/OOS split (70/30 chronological) reported per k as a stability check.
  Final frozen k still gets confirmed in the full walk-forward harness.
* Position handling uses numpy arrays throughout (no pandas chained
  assignment — see project learnings).

What "good" looks like
----------------------
A plateau: Sharpe rising from k=2 toward ~3, flat through ~4-4.5, then a
slow decline. Pick a frozen k from the middle of the plateau. If instead
there is a sharp peak at one k, treat that as overfit and prefer the
flattest neighbourhood. Watch the giveback ratio: if median giveback
exceeds ~0.5 the trail is too wide relative to the trend lengths in the
universe.

Registering this page in StratFlow (st.navigation)
-------------------------------------
    st.Page("validation_exit_sweep.py", title="Exit Sweep",
            icon=":material/exit_to_app:")
under the Validation group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# StratFlow integration — everything flows through stratflow_adapter.py.
# That file is the only place to wire universe / signals / regime.
# ---------------------------------------------------------------------------
from utils.data_fetcher import fetch_ohlcv_batch
from stratflow_adapter import (
    get_download_symbols, prepare, build_decay, universe_label,
    ENTRY_TOP_N, EXIT_RANK, SIGNALS_ARE_STANDIN,
)

UNIVERSE_SOURCE = universe_label()

# ---------------------------------------------------------------------------
# Frozen configuration — deliberately NOT exposed as UI knobs except where
# the sweep itself requires it. Changing these means re-running validation.
# ---------------------------------------------------------------------------
ATR_PERIOD = 14            # weekly Wilder ATR
TREND_MA = 30              # weeks — min-history check only
IS_FRACTION = 0.70         # chronological IS/OOS split
K_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
K_FROZEN = 4.0             # Stage-1 plateau candidate — Stage 2 backstop
ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="Exit Lab", layout="wide")
st.title("Exit Validation — Stage 1 Sweep · Stage 2 Layered Comparison")
st.caption(
    f"**{UNIVERSE_SOURCE}** · entry: ExtPct rank ≤ {ENTRY_TOP_N} · "
    f"decay: rank > {EXIT_RANK} or ExtPct < 0 · weekly cadence · "
    f"one-bar lag · Wilder ATR({ATR_PERIOD})"
)
if SIGNALS_ARE_STANDIN:
    st.warning("Adapter reports stand-in signals — do not freeze k from "
               "this run.", icon="⚠️")

with st.sidebar:
    st.header("Settings")
    stage = st.radio("Stage", ["Stage 2 — layered exits", "Stage 1 — trail sweep"])
    years = st.slider("History (years)", 5, 15, 10)
    cost_bps = st.slider("Cost (bps/side)", 0, 30, 10, step=5,
                         help="Applied to weight turnover each week.")
    exit_mode = "decay_plus_trail"
    if "Stage 1" in stage:
        exit_mode = st.radio("Stage-1 exit mode",
                             ["trail_only", "decay_plus_trail"])
    run = st.button("Run", type="primary", use_container_width=True)


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               n: int) -> np.ndarray:
    """Wilder-smoothed ATR. Returns array aligned to input, NaN warmup."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    atr = np.full_like(tr, np.nan)
    if len(tr) <= n:
        return atr
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


# ---------------------------------------------------------------------------
# Per-ticker trade simulator — pure numpy state machine
# ---------------------------------------------------------------------------
def simulate_ticker(close: np.ndarray, atr: np.ndarray,
                    entry: np.ndarray, decay: np.ndarray,
                    k: float | None, mode: str):
    """Returns (position array, trade records).

    position[t] = 1 means the position is held over the return
    close[t-1] -> close[t]. Decisions at bar t affect position[t+1].
    A trade record: (entry_idx, exit_idx, ret, peak_open_ret, exit_layer).
    """
    n = len(close)
    pos = np.zeros(n, dtype=np.int8)
    trades = []
    in_pos = False
    entry_px = peak_px = 0.0
    entry_i = -1

    for t in range(n - 1):
        if not in_pos:
            if entry[t] and not np.isnan(atr[t]):
                in_pos = True
                entry_i = t + 1
                entry_px = close[t]      # executed at bar-t close, earns t+1
                peak_px = close[t]
                pos[t + 1] = 1
        else:
            peak_px = max(peak_px, close[t])
            exit_now, layer = False, ""
            if mode in ("decay_plus_trail", "decay_only") and decay[t]:
                exit_now, layer = True, "decay"
            if (not exit_now and k is not None
                    and mode in ("trail_only", "decay_plus_trail")
                    and not np.isnan(atr[t])
                    and close[t] < peak_px - k * atr[t]):
                exit_now, layer = True, "trail"
            if exit_now:
                ret = close[t] / entry_px - 1.0
                peak_ret = peak_px / entry_px - 1.0
                trades.append((entry_i, t, ret, peak_ret, layer))
                in_pos = False
            else:
                pos[t + 1] = 1

    if in_pos:  # mark open position to last bar
        ret = close[-1] / entry_px - 1.0
        peak_ret = peak_px / entry_px - 1.0
        trades.append((entry_i, n - 1, ret, peak_ret, "open"))
        pos[-1] = 1
    return pos, trades


# ---------------------------------------------------------------------------
# Portfolio aggregation + metrics
# ---------------------------------------------------------------------------
def run_config(data: dict, entry: pd.DataFrame, decay: pd.DataFrame,
               k: float | None, mode: str, idx: pd.DatetimeIndex,
               cost_bps: float = 0.0, exposure: pd.Series | None = None):
    rets = pd.DataFrame({t: d["close"].pct_change() for t, d in data.items()},
                        index=idx)
    pos_mat = np.zeros(rets.shape)
    all_trades = []
    cols = list(rets.columns)
    for j, t in enumerate(cols):
        d = data[t].reindex(idx)
        close = d["close"].to_numpy()
        valid = ~np.isnan(close)
        if valid.sum() < TREND_MA + 10:
            continue
        atr = np.full(len(idx), np.nan)
        atr[valid] = wilder_atr(d["high"].to_numpy()[valid],
                                d["low"].to_numpy()[valid],
                                close[valid], ATR_PERIOD)
        e = entry[t].to_numpy() & valid
        x = decay[t].to_numpy() | ~valid
        # run state machine only over the valid span
        c2 = np.where(valid, close, np.nan)
        c2 = pd.Series(c2).ffill().to_numpy()  # carry price through gaps
        pos, trades = simulate_ticker(c2, atr, e, x, k, mode)
        pos_mat[:, j] = pos * valid  # no exposure where price missing
        all_trades.extend(
            dict(ticker=t, ret=r, peak=p, layer=lay, bars=ei2 - ei1)
            for ei1, ei2, r, p, lay in trades
        )

    n_active = pos_mat.sum(axis=1)
    w = np.divide(pos_mat, n_active[:, None],
                  out=np.zeros_like(pos_mat), where=n_active[:, None] > 0)
    if exposure is not None:                      # graduated regime layer
        e = exposure.reindex(idx).ffill().fillna(0.5).to_numpy()
        w = w * e[:, None]
    port = np.nansum(w * rets.to_numpy(), axis=1)
    if cost_bps > 0:                              # cost on weight turnover
        dw = np.abs(np.diff(w, axis=0, prepend=w[:1] * 0)).sum(axis=1)
        port = port - dw * (cost_bps / 1e4)
    port = pd.Series(port, index=idx).iloc[1:]
    return port, pd.DataFrame(all_trades), n_active


def metrics(port: pd.Series, trades: pd.DataFrame, n_active: np.ndarray,
            split: int) -> dict:
    def sharpe(x):
        return float(x.mean() / x.std() * ANNUALISER) if x.std() > 0 else 0.0

    eq = (1 + port).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(port) / 52
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0
    closed = trades[trades["layer"] != "open"] if len(trades) else trades
    give = np.nan
    if len(closed):
        winners = closed[closed["peak"] > 0.02]
        if len(winners):
            give = float(((winners["peak"] - winners["ret"])
                          / winners["peak"]).median())
    return {
        "Sharpe": sharpe(port),
        "Sharpe IS": sharpe(port.iloc[:split]),
        "Sharpe OOS": sharpe(port.iloc[split:]),
        "CAGR": cagr,
        "MaxDD": dd,
        "Trades": int(len(closed)),
        "Win %": float((closed["ret"] > 0).mean()) if len(closed) else np.nan,
        "Skew": float(closed["ret"].skew()) if len(closed) > 2 else np.nan,
        "Giveback (med)": give,
        "Time in mkt": float((n_active > 0).mean()),
        "Avg # pos": float(n_active[n_active > 0].mean())
        if (n_active > 0).any() else 0.0,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if not run:
    st.info("Configure in the sidebar and press **Run sweep**. The decay-only "
            "baseline is always included for comparison.")
    st.stop()

with st.spinner("Downloading universe + benchmarks (Stooq-resilient)…"):
    ohlcv = fetch_ohlcv_batch(get_download_symbols(), period=f"{years}y")
if not ohlcv:
    st.error("Data fetch failed (possibly rate-limited). Wait and retry.")
    st.stop()
try:
    P = prepare(ohlcv)
    data, entry_sig, decay_sig = P["data"], P["entry"], P["decay"]
    rank_f, ext_f, exposure = P["rank"], P["ext"], P["exposure"]
except ValueError as e:
    st.error(f"Signal engine: {e}")
    st.stop()
if len(data) < 10:
    st.error("Too few names with sufficient history — check the fetch.")
    st.stop()

idx = next(iter(data.values())).index
split = int(len(idx) * IS_FRACTION)

rows, dists = {}, {}
prog = st.progress(0.0, text="Running…")
if "Stage 1" in stage:
    configs = [("decay only (baseline)", None, "decay_only", decay_sig, None)] + [
        (f"k = {k:.1f}", k, exit_mode, decay_sig, None) for k in K_GRID
    ]
    stage_tag = "exit_sweep_v1"
else:
    d_fast   = decay_sig                                        # current rules
    d_conf2  = build_decay(rank_f, ext_f, EXIT_RANK, 2, True)   # 2wk confirm
    d_r30c2  = build_decay(rank_f, ext_f, 30, 2, True)          # wider + confirm
    d_rankonly = build_decay(rank_f, ext_f, 30, 2, False)       # drop ext<0
    configs = [
        ("A · decay now (baseline)",      None,     "decay_only",       d_fast,     None),
        ("B · decay now + trail",         K_FROZEN, "decay_plus_trail", d_fast,     None),
        ("C · 2wk-confirm + trail",       K_FROZEN, "decay_plus_trail", d_conf2,    None),
        ("D · rank30/2wk + trail",        K_FROZEN, "decay_plus_trail", d_r30c2,    None),
        ("E · rank-only 30/2wk + trail",  K_FROZEN, "decay_plus_trail", d_rankonly, None),
        ("F · D + regime layer",          K_FROZEN, "decay_plus_trail", d_r30c2,    exposure),
        ("G · trail-only (reference)",    K_FROZEN, "trail_only",       d_fast,     None),
    ]
    stage_tag = "exit_stage2_v1"

ports = {}
for i, (label, k, mode, dvar, expo) in enumerate(configs):
    port, trades, n_act = run_config(data, entry_sig, dvar, k, mode, idx,
                                     cost_bps=cost_bps, exposure=expo)
    rows[label] = metrics(port, trades, n_act, split)
    dists[label] = trades
    ports[label] = port
    prog.progress((i + 1) / len(configs), text=f"Done: {label}")
prog.empty()

# ── Subperiod robustness: Sharpe per quarter-of-sample fold ──────────────────
N_FOLDS = 4
fold_rows = {}
for label, port in ports.items():
    edges = np.linspace(0, len(port), N_FOLDS + 1).astype(int)
    fr = {}
    for fi in range(N_FOLDS):
        seg = port.iloc[edges[fi]:edges[fi + 1]]
        d0, d1 = seg.index[0].year, seg.index[-1].year
        fr[f"F{fi+1} ({d0}-{d1})"] = (round(float(seg.mean() / seg.std()
                                     * ANNUALISER), 2) if seg.std() > 0 else 0.0)
    fold_rows[label] = fr
folds_df = pd.DataFrame(fold_rows).T

res = pd.DataFrame(rows).T
st.subheader("Sweep results")
st.dataframe(
    res.style.format({
        "Sharpe": "{:.2f}", "Sharpe IS": "{:.2f}", "Sharpe OOS": "{:.2f}",
        "CAGR": "{:.1%}", "MaxDD": "{:.1%}", "Win %": "{:.0%}",
        "Skew": "{:.2f}", "Giveback (med)": "{:.0%}",
        "Time in mkt": "{:.0%}", "Avg # pos": "{:.1f}",
    }),
    use_container_width=True,
)

if "Stage 1" in stage:
    sweep_only = res.iloc[1:].copy()
    sweep_only.index = K_GRID
    st.subheader("Plateau check — Sharpe vs trail width")
    st.line_chart(sweep_only[["Sharpe", "Sharpe IS", "Sharpe OOS"]])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Giveback vs k")
        st.line_chart(sweep_only[["Giveback (med)"]])
    with c2:
        st.subheader("Max drawdown vs k")
        st.line_chart(sweep_only[["MaxDD"]])

# Trade distribution for an inspected k
st.subheader("Subperiod robustness — Sharpe per fold")
st.caption("A real improvement should hold (or at least not invert) across "
           "folds, not live in one era.")
st.dataframe(folds_df, use_container_width=True)

st.subheader("Trade-level return distribution")
pick = st.selectbox("Inspect configuration", list(rows.keys()),
                    index=min(4, len(rows) - 1))
tr = dists[pick]
if len(tr) and len(tr[tr["layer"] != "open"]):
    closed = tr[tr["layer"] != "open"]
    hist = np.histogram(closed["ret"].clip(-0.5, 1.5), bins=40)
    st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 2)))
    lc = closed["layer"].value_counts(normalize=True)
    st.caption(
        f"{len(closed)} closed trades · exit attribution: "
        + " · ".join(f"{k}: {v:.0%}" for k, v in lc.items())
        + f" · best trade {closed['ret'].max():.0%}"
        + f" · worst {closed['ret'].min():.0%}"
    )

# ── Copy-paste results block for analysis ───────────────────────────────────
st.subheader("📋 Results for Claude")
st.caption("Tap the copy icon on this block and paste it back into the chat.")
import json as _json
_attr = {}
for _lbl, _tr in dists.items():
    if len(_tr):
        _cl = _tr[_tr["layer"] != "open"]
        if len(_cl):
            _attr[_lbl] = {
                **_cl["layer"].value_counts(normalize=True).round(2).to_dict(),
                "med_bars": float(_cl["bars"].median()),
                "best": round(float(_cl["ret"].max()), 3),
                "worst": round(float(_cl["ret"].min()), 3),
            }
_payload = {
    "stage": stage_tag,
    "settings": {"years": years, "mode": exit_mode, "cost_bps": cost_bps,
                 "k_frozen": K_FROZEN, "entry_top_n": ENTRY_TOP_N,
                 "exit_rank": EXIT_RANK, "n_names": len(data),
                 "n_weeks": int(len(idx))},
    "results": res.round(3).reset_index()
                  .rename(columns={"index": "config"}).to_dict("records"),
    "attribution": _attr,
    "folds": fold_rows,
}
st.code(_json.dumps(_payload, indent=1, default=str), language="json")

with st.expander("How to read this (decision criteria)"):
    st.markdown(
        """
**Accept a frozen k when all of these hold:**
1. **Plateau, not peak** — Sharpe within ~0.1 across at least three adjacent
   k values. Choose from the middle of the flat region.
2. **IS/OOS agreement** — OOS Sharpe at the chosen k within ~30% of IS.
   Divergence at one k but not its neighbours = noise; prefer neighbours.
3. **Skew preserved** — trade-return skew should *increase* with k. If skew
   is flat across the grid, the trail isn't binding and decay is doing all
   the work (also a valid finding — it argues for a wide backstop only).
4. **Giveback in band** — median giveback between ~25% and ~50% on winners.
   Below 25% with low Sharpe → trail too tight, choking winners. Above 55%
   → too wide for this universe's typical trend length.
5. **Trail should fire rarely** in `decay_plus_trail` mode (≲20% of exits).
   If it dominates, the decay exit is too slow, which is a Stage 2 question.

**Then:** freeze k, and confirm in the walk-forward harness before it
touches production. The number chosen here is a candidate, not a result.
        """
    )
