"""
validation_hrp_lab.py
HRP LAB — does correlation-aware sizing beat naive inverse-vol?

This isolates the SIZING layer. The held book each week is reconstructed with
the identical validated stack (composite-26 selector → redundancy filter →
hold-band 11-30 / 2wk decay + 4×ATR trail → regime gate), so the ONLY thing
that varies across arms is how weight is distributed across the held names.

Arms:
  • Inverse-vol + filter (production baseline) — 1/vol, normalised
  • HRP + filter                              — Hierarchical Risk Parity
  • Equal-weight + filter                     — reference (hard to beat)
  • HRP no-filter                             — does HRP's correlation-aware
                                                sizing make the entry filter
                                                redundant?

Sizing is causal: weights at week t use the trailing CORR_WIN returns ending
at t-1. Reported: Sharpe (full/IS/OOS), CAGR, MaxDD, TURNOVER (one-way weekly,
the cost HRP must justify), avg #positions, per-fold Sharpe, and DSR.

Adoption criterion: HRP + filter must beat inverse-vol + filter on risk-
adjusted return AND/OR drawdown across folds, net of its turnover, with DSR
support. Honest hypothesis: the redundancy filter already removes the worst
correlation clusters at entry, so HRP may find little left to fix (the
literature's ~50% gain was on UNFILTERED books — the no-filter arm probes that).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.exits import decay_matrix, EXIT_RANK, CONFIRM_WEEKS, TRAIL_K
from utils.hrp import hrp_from_returns
from utils.significance import deflated_sharpe_ratio, verdict as sig_verdict
from stratflow_adapter import (get_download_symbols, prepare, aux_frames,
                               universe_label, ENTRY_TOP_N)

ATR_PERIOD = 14
IS_FRACTION = 0.70
N_FOLDS = 4
MOM_WEEKS = 26
VOL_WEEKS = 26
CORR_MAX = 0.85
CORR_WIN = 26
CAND_POOL = 25
ANNUALISER = np.sqrt(52)

st.set_page_config(page_title="HRP Lab", layout="wide")
st.title("HRP Lab — correlation-aware sizing vs inverse-vol")
st.caption(f"**{universe_label()}** · identical validated book across arms "
           f"(composite-26 → filter → band ≤{EXIT_RANK}/{CONFIRM_WEEKS}wk → "
           f"4×ATR trail → regime); only the sizing changes.")

with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 5, 15, 10)
    cost_bps = st.slider("Cost (bps/turn)", 0, 30, 10)
    run = st.button("Run HRP Lab", type="primary", use_container_width=True)

if not run:
    st.info("Press **Run HRP Lab**. Four sizing arms on one frozen book.")
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
    n = len(close)
    pos = np.zeros(n, dtype=np.int8)
    in_pos = False
    peak = 0.0
    for t in range(n - 1):
        if not in_pos:
            if entry[t] and not np.isnan(atr[t]):
                in_pos = True
                peak = close[t]
                pos[t + 1] = 1
        else:
            peak = max(peak, close[t])
            exit_now = bool(decay[t])
            if (not exit_now and not np.isnan(atr[t])
                    and close[t] < peak - k * atr[t]):
                exit_now = True
            pos[t + 1] = 0 if exit_now else 1
            in_pos = not exit_now
    return pos


def _pct(df):
    return df.rank(axis=1, pct=True)


def rank_from_score(score):
    return score.rank(axis=1, ascending=False)


def entries_from_rank(rank):
    return (rank <= ENTRY_TOP_N).fillna(False).astype(bool)


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


def build_pos_mat(entry, decay, data, idx):
    """Held 0/1 matrix from the validated entry/decay/trail logic."""
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
    return pos_mat, cols


def run_sizing(pos_mat, cols, rets_df, idx, exposure, sizing, cost_bps,
               window=CORR_WIN):
    """Apply a sizing scheme to the held book. Weights at t use trailing
    returns ending at t-1 (causal). Returns (port, turnover, n_active)."""
    R = rets_df[cols]
    rmat = R.to_numpy()
    T, N = pos_mat.shape
    W = np.zeros((T, N))
    ex = exposure.reindex(idx).ffill().fillna(0.5).to_numpy()
    for t in range(window, T):
        active = np.where(pos_mat[t] > 0)[0]
        if active.size == 0:
            continue
        if active.size == 1:
            w = np.array([1.0])
        elif sizing == "equal":
            w = np.full(active.size, 1.0 / active.size)
        elif sizing == "invvol":
            seg = rmat[t - window:t, active]
            vol = np.nanstd(seg, axis=0)
            vol = np.where(vol <= 0, np.nan, vol)
            iv = 1.0 / vol
            iv = np.where(np.isnan(iv), 0.0, iv)
            w = iv / iv.sum() if iv.sum() > 0 else np.full(active.size, 1.0 / active.size)
        else:  # hrp
            names = [cols[a] for a in active]
            seg = pd.DataFrame(rmat[t - window:t, active], columns=names)
            wser = hrp_from_returns(seg)
            w = wser.reindex(names).fillna(0.0).to_numpy()
            if w.sum() <= 0:
                w = np.full(active.size, 1.0 / active.size)
            else:
                w = w / w.sum()
        W[t, active] = w * ex[t]
    port = np.nansum(W * rmat, axis=1)
    dw = np.abs(np.diff(W, axis=0, prepend=W[:1] * 0)).sum(axis=1)
    turn = float(dw[window:].mean())
    if cost_bps > 0:
        port = port - dw * (cost_bps / 1e4)
    n_active = (pos_mat > 0).sum(axis=1)
    return pd.Series(port, index=idx).iloc[1:], turn, n_active


def metrics(port, turn, n_active, split):
    def sh(x):
        return float(x.mean() / x.std() * ANNUALISER) if x.std() > 0 else 0.0
    eq = (1 + port).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(port) / 52
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 and eq.iloc[-1] > 0 else 0.0
    return {"Sharpe": round(sh(port), 3), "Sharpe IS": round(sh(port.iloc[:split]), 3),
            "Sharpe OOS": round(sh(port.iloc[split:]), 3), "CAGR": round(cagr, 3),
            "MaxDD": round(dd, 3), "Turnover/wk": round(turn, 3),
            "Avg # pos": round(float(n_active[n_active > 0].mean()), 1)
                         if (n_active > 0).any() else 0.0}


# ── Build the books and run the sizing arms ──────────────────────────────────
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
split = int(len(idx) * IS_FRACTION)

mom = rs.pct_change(MOM_WEEKS)
rs_vol = rs.pct_change().rolling(VOL_WEEKS).std() * 100
ext_adj = ext / rs_vol.replace(0, np.nan)
rank = rank_from_score(_pct(mom) + _pct(ext_adj))          # composite-26
decay = decay_matrix(rank, EXIT_RANK, CONFIRM_WEEKS)

entry_filt = redundancy_filtered_entries(rank, rets_df, corr_max=CORR_MAX)
entry_raw = entries_from_rank(rank)

with st.spinner("Reconstructing the validated book…"):
    pos_filt, cols = build_pos_mat(entry_filt, decay, data, idx)
    pos_raw, _ = build_pos_mat(entry_raw, decay, data, idx)

ARMS = [
    ("Inverse-vol + filter (production)", pos_filt, "invvol"),
    ("HRP + filter", pos_filt, "hrp"),
    ("Equal-weight + filter", pos_filt, "equal"),
    ("HRP no-filter", pos_raw, "hrp"),
]
rows, ports = {}, {}
prog = st.progress(0.0, text="Running sizing arms…")
for i, (label, pm, sizing) in enumerate(ARMS):
    port, turn, n_act = run_sizing(pm, cols, rets_df, idx, exposure, sizing, cost_bps)
    rows[label] = metrics(port, turn, n_act, split)
    ports[label] = port
    prog.progress((i + 1) / len(ARMS), text=f"Done: {label}")
prog.empty()

res = pd.DataFrame(rows).T
st.subheader("Sizing arm results (costed, regime ON, book frozen)")
st.dataframe(res, use_container_width=True)
st.caption("Book (who's held) is identical across the first three arms; only "
           "the weighting differs. Turnover/wk is the one-way weekly weight "
           "churn — HRP must justify any extra turnover with better "
           "risk-adjusted return or drawdown.")

# ── Folds ─────────────────────────────────────────────────────────────────────
st.subheader("Per-fold Sharpe")
edges = np.linspace(0, len(idx), N_FOLDS + 1).astype(int)
fold_sh = {}
for label, port in ports.items():
    p = port.reindex(idx)
    row = {}
    for fi in range(N_FOLDS):
        seg = p.iloc[edges[fi]:edges[fi + 1]].dropna()
        row[f"F{fi+1} ({idx[edges[fi]].year}-{idx[min(edges[fi+1],len(idx)-1)].year})"] = (
            round(float(seg.mean() / seg.std() * ANNUALISER), 2)
            if seg.std() > 0 else None)
    fold_sh[label] = row
st.dataframe(pd.DataFrame(fold_sh).T, use_container_width=True)

# ── Significance + verdict ────────────────────────────────────────────────────
_allsr = [r["Sharpe"] for r in rows.values()]
_sig = deflated_sharpe_ratio(float(max(_allsr)), _allsr, int(len(idx)))
st.subheader("🎲 Significance")
st.markdown("- " + sig_verdict(_sig))

st.subheader("Verdict")
base = res.loc["Inverse-vol + filter (production)"]
hrp = res.loc["HRP + filter"]
d_sh = hrp["Sharpe"] - base["Sharpe"]
d_dd = hrp["MaxDD"] - base["MaxDD"]            # + = shallower (better)
d_turn = hrp["Turnover/wk"] - base["Turnover/wk"]
folds_better = sum(1 for f in fold_sh["HRP + filter"]
                   if fold_sh["HRP + filter"][f] is not None
                   and fold_sh["Inverse-vol + filter (production)"][f] is not None
                   and fold_sh["HRP + filter"][f] >= fold_sh["Inverse-vol + filter (production)"][f] - 1e-9)
if (d_sh > 0.05 or d_dd > 0.01) and folds_better >= 3:
    msg = (f"✅ **HRP + filter** beats inverse-vol (Sharpe {d_sh:+.2f}, MaxDD "
           f"{d_dd*100:+.1f}pp, better/equal in {folds_better}/4 folds, "
           f"turnover {d_turn:+.3f}/wk) — candidate for adoption pending the "
           "turnover trade-off.")
elif abs(d_sh) <= 0.05 and abs(d_dd) <= 0.01:
    msg = (f"🟠 **HRP + filter** is at parity (Sharpe {d_sh:+.2f}, MaxDD "
           f"{d_dd*100:+.1f}pp, turnover {d_turn:+.3f}/wk). The filter likely "
           "already removed the clusters HRP would fix — keep inverse-vol "
           "unless HRP also lowers turnover.")
else:
    msg = (f"🛑 **HRP + filter** does not beat inverse-vol (Sharpe {d_sh:+.2f}, "
           f"MaxDD {d_dd*100:+.1f}pp, {folds_better}/4 folds) — inverse-vol "
           "stays frozen.")
st.markdown("- " + msg)
nofilt = res.loc["HRP no-filter"]
st.markdown(f"- HRP no-filter vs production: Sharpe {nofilt['Sharpe']-base['Sharpe']:+.2f}, "
            f"MaxDD {(nofilt['MaxDD']-base['MaxDD'])*100:+.1f}pp — tests whether "
            "HRP sizing can replace the entry filter (it should NOT beat "
            "filter+invvol if the filter is doing real work).")
eqw = res.loc["Equal-weight + filter"]
st.markdown(f"- Equal-weight reference: Sharpe {eqw['Sharpe']:.3f} — if neither "
            "inverse-vol nor HRP clears this by much, sizing barely matters on "
            "this book.")

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("📋 Results for Claude")
payload = {"stage": "hrp_lab_v1",
           "settings": {"years": years, "cost_bps": cost_bps,
                        "corr_win": CORR_WIN, "n_weeks": int(len(idx)),
                        "n_names": len(cols)},
           "results": res.reset_index().to_dict("records"),
           "fold_sharpe": fold_sh,
           "significance": _sig}
st.code(json.dumps(payload, indent=1, default=str), language="json")
