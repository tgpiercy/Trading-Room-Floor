"""
StratFlow Validation — Selection Lab
====================================
Tests SELECTION-rank variants while holding everything validated constant:
the exit stack (enter rank≤10 · hold band ≤30 · 2wk-confirm release · 4×ATR
chandelier), the regime layer (graduated exposure), and costs on turnover.
One variable moves per arm: HOW names are ranked.

Arms (all frozen — no per-arm tuning):
  A · raw ExtPct (production today)
  B · vol-adjusted ExtPct        ext / 26w RS-return vol  (signal-to-noise)
  C · composite 50/50            rank(26w RS momentum) ⊕ rank(ExtPct)
  D · composite vol-adjusted     rank(26w RS momentum) ⊕ rank(vol-adj ExtPct)
  E · single-benchmark control   ExtPct vs SPY for every name
  F · D + redundancy filter      skip candidates >0.85 corr (26w) with an
                                 already-accepted stronger name
  G · D + defensive overlay      regime-freed cash → top defensive names by
                                 26w absolute momentum (TLT/IEF/GLD/DBMF/
                                 USMV/TIP, ≤3 names, long only if mom>0)

Methodology: weekly W-FRI, one-bar lag, costs on weight turnover, IS/OOS
70/30 + 4-fold subperiod Sharpes, exit attribution. JSON block at the bottom
to paste back. Engine math is the verified Exit Lab engine (duplicated by
design — research pages stay single-file for the upload workflow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.exits import decay_matrix, EXIT_RANK, CONFIRM_WEEKS, TRAIL_K
from stratflow_adapter import (get_download_symbols, prepare, aux_frames,
                               universe_label, ENTRY_TOP_N)

# ── Frozen lab constants ─────────────────────────────────────────────────────
ATR_PERIOD = 14
IS_FRACTION = 0.70
N_FOLDS = 4
MOM_WEEKS = 26               # slow relative-momentum lookback
VOL_WEEKS = 26               # RS-return vol window for adjustment
CORR_MAX = 0.85              # redundancy filter ceiling
CORR_WIN = 26                # weeks for the correlation window
CAND_POOL = 25               # filter walks this many top candidates
DEFENSIVE = ["TLT", "IEF", "GLD", "DBMF", "USMV", "TIP"]
DEF_MAX = 3
SWEEP_MOM = [13, 26, 39]          # momentum-lookback neighborhood (weeks)
SWEEP_CORR = [0.80, 0.85, 0.90]   # redundancy-ceiling neighborhood
ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="Selection Lab", layout="wide")
st.title("Selection Lab — ranking variants through the validated stack")
st.caption(f"**{universe_label()}** · exit stack frozen "
           f"(≤{ENTRY_TOP_N} / band ≤{EXIT_RANK} / {CONFIRM_WEEKS}wk / "
           f"{TRAIL_K}×ATR) · regime layer ON for all arms")

with st.sidebar:
    st.header("Settings")
    lab_mode = st.radio("Mode", ["Arms A-G", "Confirmation sweep (F neighborhood)"],
                        help="Confirmation sweep: Gate-3 parameter-insensitivity "
                             "check on the F architecture — mom weeks × corr "
                             "ceiling grid, frozen, no winner-picking.")
    years = st.slider("History (years)", 5, 15, 10)
    cost_bps = st.slider("Cost (bps/side)", 0, 30, 10, step=5)
    run = st.button("Run Selection Lab", type="primary",
                    use_container_width=True)


# ═════════════════════════ engine (verified, Exit Lab) ══════════════════════
def wilder_atr(high, low, close, n=ATR_PERIOD):
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close),
                                           np.abs(low - prev_close)))
    atr = np.full_like(tr, np.nan)
    if len(tr) <= n:
        return atr
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def simulate_ticker(close, atr, entry, decay, k, mode):
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
                entry_px = close[t]
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
                trades.append((entry_i, t, close[t] / entry_px - 1.0,
                               peak_px / entry_px - 1.0, layer))
                in_pos = False
            else:
                pos[t + 1] = 1
    if in_pos:
        trades.append((entry_i, n - 1, close[-1] / entry_px - 1.0,
                       peak_px / entry_px - 1.0, "open"))
        pos[-1] = 1
    return pos, trades


def run_config(data, entry, decay, k, mode, idx, cost_bps=0.0,
               exposure=None, extra_w=None):
    rets = pd.DataFrame({t: d["close"].pct_change() for t, d in data.items()},
                        index=idx)
    pos_mat = np.zeros(rets.shape)
    all_trades = []
    for j, t in enumerate(rets.columns):
        d = data[t].reindex(idx)
        close = d["close"].to_numpy()
        valid = ~np.isnan(close)
        if valid.sum() < 40:
            continue
        atr = np.full(len(idx), np.nan)
        atr[valid] = wilder_atr(d["high"].to_numpy()[valid],
                                d["low"].to_numpy()[valid],
                                close[valid], ATR_PERIOD)
        e = entry[t].to_numpy() & valid if t in entry.columns else np.zeros(len(idx), bool)
        x = decay[t].to_numpy() | ~valid if t in decay.columns else np.ones(len(idx), bool)
        c2 = pd.Series(np.where(valid, close, np.nan)).ffill().to_numpy()
        pos, trades = simulate_ticker(c2, atr, e, x, k, mode)
        pos_mat[:, j] = pos * valid
        all_trades.extend(dict(ticker=t, ret=r, peak=p, layer=lay, bars=b - a)
                          for a, b, r, p, lay in trades)
    n_active = pos_mat.sum(axis=1)
    w = np.divide(pos_mat, n_active[:, None],
                  out=np.zeros_like(pos_mat), where=n_active[:, None] > 0)
    if exposure is not None:
        ex = exposure.reindex(idx).ffill().fillna(0.5).to_numpy()
        w = w * ex[:, None]
    if extra_w is not None:                       # defensive overlay weights
        w = w + extra_w
    port = np.nansum(w * rets.to_numpy(), axis=1)
    if cost_bps > 0:
        dw = np.abs(np.diff(w, axis=0, prepend=w[:1] * 0)).sum(axis=1)
        port = port - dw * (cost_bps / 1e4)
    return (pd.Series(port, index=idx).iloc[1:],
            pd.DataFrame(all_trades), n_active)


def metrics(port, trades, n_active, split):
    def sharpe(x):
        return float(x.mean() / x.std() * ANNUALISER) if x.std() > 0 else 0.0
    eq = (1 + port).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(port) / 52
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0
    closed = trades[trades["layer"] != "open"] if len(trades) else trades
    return {"Sharpe": sharpe(port), "Sharpe IS": sharpe(port.iloc[:split]),
            "Sharpe OOS": sharpe(port.iloc[split:]), "CAGR": cagr,
            "MaxDD": dd, "Trades": int(len(closed)),
            "Win %": float((closed["ret"] > 0).mean()) if len(closed) else np.nan,
            "Med hold (wk)": float(closed["bars"].median()) if len(closed) else np.nan,
            "Avg # pos": float(n_active[n_active > 0].mean())
                         if (n_active > 0).any() else 0.0}


# ═════════════════════════ arm construction ═════════════════════════════════
def _pct_strength(df):
    """Cross-sectional percentile, 1.0 = strongest."""
    return df.rank(axis=1, pct=True)


def rank_from_score(score):
    """Strength score → ordinal rank frame (1 = best)."""
    return score.rank(axis=1, ascending=False)


def entries_from_rank(rank):
    return (rank <= ENTRY_TOP_N).fillna(False).astype(bool)


def redundancy_filtered_entries(rank, rets, top_n=ENTRY_TOP_N,
                                pool=CAND_POOL, corr_max=CORR_MAX,
                                window=CORR_WIN):
    """Walk candidates strongest-first each week; accept unless >corr_max
    correlated (trailing `window` weekly returns) with an accepted name."""
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
            ok = True
            s1 = seg[tk]
            if s1.std() == 0 or s1.isna().all():
                ok = False
            else:
                for a in accepted:
                    c = s1.corr(seg[a])
                    if c == c and c > corr_max:
                        ok = False
                        break
            if ok:
                accepted.append(tk)
                if len(accepted) >= top_n:
                    break
        out.iloc[ti, [out.columns.get_loc(a) for a in accepted]] = True
    return out


def defensive_overlay(close_df, exposure, rets_cols, idx):
    """(1-exposure) allocated equally across ≤DEF_MAX defensive names with
    positive 26w momentum (decided at t-1, held over t). Returns weight
    matrix aligned to rets columns."""
    mom = close_df[ [c for c in DEFENSIVE if c in close_df.columns] ]\
              .pct_change(MOM_WEEKS)
    W = np.zeros((len(idx), len(rets_cols)))
    col_ix = {c: i for i, c in enumerate(rets_cols)}
    ex = exposure.reindex(idx).ffill().fillna(0.5).to_numpy()
    for ti in range(1, len(idx)):
        free = 1.0 - ex[ti]
        if free <= 0:
            continue
        row = mom.iloc[ti - 1].dropna()          # decided at t-1
        qual = row[row > 0].sort_values(ascending=False).head(DEF_MAX)
        if len(qual) == 0:
            continue
        wq = free / len(qual)
        for tk in qual.index:
            if tk in col_ix:
                W[ti, col_ix[tk]] = wq
    return W


# ═════════════════════════ run ═══════════════════════════════════════════════
if not run:
    st.info("Press **Run Selection Lab**. Seven arms, one ranking change "
            "each, identical exit/regime/cost stack.")
    st.stop()

with st.spinner("Downloading universe + benchmarks…"):
    ohlcv = fetch_ohlcv_batch(get_download_symbols(), period=f"{years}y")
if not ohlcv:
    st.error("Data fetch failed (possibly rate-limited). Wait and retry.")
    st.stop()
try:
    P = prepare(ohlcv)
except ValueError as e:
    st.error(f"Signal engine: {e}")
    st.stop()
data, exposure = P["data"], P["exposure"]
ext = P["ext"]
idx = next(iter(data.values())).index
aux = aux_frames(ohlcv, idx)
rs, ext_spy, close_df = aux["rs"], aux["ext_spy"], aux["close"]
rets_df = close_df.pct_change()

# component scores (all causal)
mom26 = rs.pct_change(MOM_WEEKS)                          # slow RS momentum
rs_vol = rs.pct_change().rolling(VOL_WEEKS).std() * 100   # weekly RS vol, %
ext_adj = ext / rs_vol.replace(0, np.nan)                 # signal-to-noise

def composite_rank(mom_w):
    m = rs.pct_change(mom_w)
    return rank_from_score(_pct_strength(m) + _pct_strength(ext_adj))


if "Confirmation" in lab_mode:
    ARMS = [("A · raw ExtPct (production)", rank_from_score(ext), None, False)]
    for mw in SWEEP_MOM:
        rk = composite_rank(mw)
        for cm in SWEEP_CORR:
            ARMS.append((f"F(mom={mw}, corr={cm:.2f})", rk,
                         ("filter", cm), False))
    stage_tag = "selection_confirm_v1"
else:
    score_A = ext
    score_B = ext_adj
    score_C = _pct_strength(mom26) + _pct_strength(ext)
    score_D = _pct_strength(mom26) + _pct_strength(ext_adj)
    score_E = ext_spy
    ranks = {k: rank_from_score(s) for k, s in
             [("A", score_A), ("B", score_B), ("C", score_C),
              ("D", score_D), ("E", score_E)]}
    ARMS = [
        ("A · raw ExtPct (production)",  ranks["A"], None,  False),
        ("B · vol-adjusted ExtPct",      ranks["B"], None,  False),
        ("C · composite mom26⊕ext",      ranks["C"], None,  False),
        ("D · composite vol-adjusted",   ranks["D"], None,  False),
        ("E · single-benchmark (SPY)",   ranks["E"], None,  False),
        ("F · D + redundancy filter",    ranks["D"], ("filter", CORR_MAX), False),
        ("G · D + defensive overlay",    ranks["D"], None,  True),
    ]
    stage_tag = "selection_lab_v1"

split = int(len(idx) * IS_FRACTION)
rows, dists, ports, overlap = {}, {}, {}, {}
entry_A = entries_from_rank(ARMS[0][1])
prog = st.progress(0.0, text="Running arms…")
for i, (label, rk, special, defensive) in enumerate(ARMS):
    if isinstance(special, tuple) and special[0] == "filter":
        entry = redundancy_filtered_entries(rk, rets_df, corr_max=special[1])
    else:
        entry = entries_from_rank(rk)
    decay = decay_matrix(rk, EXIT_RANK, CONFIRM_WEEKS)
    extra = (defensive_overlay(close_df, exposure, list(data.keys()), idx)
             if defensive else None)
    port, trades, n_act = run_config(data, entry, decay, TRAIL_K,
                                     "decay_plus_trail", idx,
                                     cost_bps=cost_bps, exposure=exposure,
                                     extra_w=extra)
    rows[label] = metrics(port, trades, n_act, split)
    dists[label] = trades
    ports[label] = port
    inter = (entry & entry_A).sum(axis=1)
    union = (entry | entry_A).sum(axis=1)
    jac = (inter / union.replace(0, np.nan)).mean()
    overlap[label] = round(float(jac), 2) if jac == jac else None
    prog.progress((i + 1) / len(ARMS), text=f"Done: {label}")
prog.empty()

res = pd.DataFrame(rows).T
res["Book overlap vs A"] = pd.Series(overlap)
st.subheader("Arm results (costed, regime ON, exit stack frozen)")
st.dataframe(res.style.format({
    "Sharpe": "{:.2f}", "Sharpe IS": "{:.2f}", "Sharpe OOS": "{:.2f}",
    "CAGR": "{:.1%}", "MaxDD": "{:.1%}", "Win %": "{:.0%}",
    "Med hold (wk)": "{:.0f}", "Avg # pos": "{:.1f}",
    "Book overlap vs A": "{:.2f}"}), use_container_width=True)

# folds
fold_rows = {}
for label, port in ports.items():
    edges = np.linspace(0, len(port), N_FOLDS + 1).astype(int)
    fr = {}
    for fi in range(N_FOLDS):
        seg = port.iloc[edges[fi]:edges[fi + 1]]
        fr[f"F{fi+1} ({seg.index[0].year}-{seg.index[-1].year})"] = (
            round(float(seg.mean() / seg.std() * ANNUALISER), 2)
            if seg.std() > 0 else 0.0)
    fold_rows[label] = fr
st.subheader("Subperiod robustness — Sharpe per fold")
st.dataframe(pd.DataFrame(fold_rows).T, use_container_width=True)

# attribution + export
import json as _json
_attr = {}
for _lbl, _tr in dists.items():
    if len(_tr):
        _cl = _tr[_tr["layer"] != "open"]
        if len(_cl):
            _attr[_lbl] = {**_cl["layer"].value_counts(normalize=True)
                           .round(2).to_dict(),
                           "med_bars": float(_cl["bars"].median())}
st.subheader("📋 Results for Claude")
st.caption("Tap the copy icon and paste the block back into the chat.")
payload = {
    "stage": stage_tag,
    "settings": {"years": years, "cost_bps": cost_bps,
                 "mom_weeks": MOM_WEEKS, "vol_weeks": VOL_WEEKS,
                 "corr_max": CORR_MAX, "defensive": DEFENSIVE,
                 "sweep_mom": SWEEP_MOM, "sweep_corr": SWEEP_CORR,
                 "n_names": len(data), "n_weeks": int(len(idx))},
    "results": res.round(3).reset_index()
                  .rename(columns={"index": "config"}).to_dict("records"),
    "folds": fold_rows,
    "attribution": _attr,
}
st.code(_json.dumps(payload, indent=1, default=str), language="json")

with st.expander("How to read this (decision criteria)"):
    st.markdown(f"""
1. **B vs A** isolates vol-adjustment; **C vs A** isolates slow momentum;
   **D** combines them. A real selection gain shows in Sharpe *and* fold
   consistency, not one era.
2. **E vs A**: if the SPY-only control matches A, the per-name benchmark
   structure isn't earning its complexity.
3. **F** should show lower book overlap with A and help most in the folds
   where everything else struggles (concentration regimes). If it only
   costs Sharpe in strong folds and saves nothing in weak ones, drop it.
4. **G** only differs from D when exposure < 1; judge it on CAGR/MaxDD in
   the 2021-2023 fold specifically.
5. Mind **Book overlap vs A** — an arm with 0.9 overlap and +0.3 Sharpe is
   noise; an arm with 0.5 overlap and +0.3 Sharpe is a different (better)
   selector.
6. Winner ≠ production yet: the winning arm gets the walk-forward harness
   before touching the Rebalance model — same rule as the exit stack.
""")
