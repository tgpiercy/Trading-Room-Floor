"""
validation_regime_lab.py
REGIME LAB — brings the exposure gate through the gauntlet for the first
time, and tests whether NEW climate information improves it.

Design discipline (this is the whole point):
  • The validated 3-input gate (trend 40 + SPY/IEF 30 + VIX 30, cut 66/33)
    is the BASELINE and is left completely untouched — no re-tuning of its
    weights or thresholds (we proved per-window optimization is fatal).
  • Each new input enters ONLY as a de-risking CONFIRMATION overlay: it can
    knock exposure DOWN one notch (1.0→0.5→0.0) when it flags stress, and can
    never add exposure. This tests whether the information helps, a priori,
    without disturbing the frozen baseline.

Variants:
  Baseline                — current gate
  +Credit                 — also de-risk a notch when HY OAS > its 26w mean
  +Breadth                — also de-risk a notch when <40% of sectors > 10w MA
  +Both                   — both overlays

Evaluation: each exposure path is applied CAUSALLY to SPY weekly returns
(exposure decided at t drives the t→t+1 return; cash earns 0). We compare
Sharpe, MaxDD, CAGR, and average exposure (the cost of de-risking) over the
full sample and across 4 contiguous folds. A confirmation overlay EARNS its
place only if it improves Sharpe and/or cuts MaxDD consistently across folds
for an acceptable give-up in return. Buy-&-hold SPY is shown for context.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import fetch_ohlcv_batch
from utils.strategy_backtest import compute_regime_exposure

# ── Frozen, a-priori overlay constants (NOT tuning knobs) ─────────────────────
CREDIT_MA_W = 26       # HY OAS trend window — "spreads elevated vs recent norm"
BREADTH_W = 10         # weekly SMA ≈ 50-day for the sector-breadth basket
BREADTH_WEAK = 40.0    # <40% participation = weak breadth (standard threshold)
N_FOLDS = 4
ANN = np.sqrt(52)
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
               "XLP", "XLU", "XLRE", "XLB", "XLC"]

st.set_page_config(page_title="Regime Lab", layout="wide")
st.title("🧭 Regime Lab — does new climate information improve the gate?")
st.caption("Baseline = the validated 3-input gate, untouched. Credit and "
           "breadth enter only as de-risking confirmation overlays (knock "
           "exposure down a notch on stress, never up). A priori constants: "
           f"credit OAS vs {CREDIT_MA_W}w mean · breadth <{BREADTH_WEAK:.0f}% "
           f"of sectors > {BREADTH_W}w MA. No tuning knobs by design.")

with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 4, 15, 10)
    run = st.button("Run regime lab", type="primary", use_container_width=True)

if not run:
    st.info("Press **Run regime lab**. Credit variants need FRED_API_KEY in "
            "secrets; breadth needs the sector ETFs. Missing inputs degrade "
            "gracefully — the variants that can run, will.")
    st.stop()


def _notch_down(exp: pd.Series, stress: pd.Series) -> pd.Series:
    """Reduce exposure by one 0.5 notch wherever stress is True."""
    s = stress.reindex(exp.index).fillna(False).astype(bool)
    return (exp - 0.5 * s).clip(lower=0.0)


def _metrics(exp: pd.Series, spy_ret: pd.Series) -> dict:
    port = exp.shift(1).reindex(spy_ret.index).fillna(0.0) * spy_ret
    port = port.dropna()
    if len(port) < 10:
        return {}
    eq = (1 + port).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(port) / 52
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else np.nan
    sharpe = port.mean() / port.std() * ANN if port.std() > 0 else np.nan
    return {"Sharpe": round(float(sharpe), 3),
            "CAGR": round(float(cagr), 3),
            "MaxDD": round(float(dd), 3),
            "Avg exp": round(float(exp.mean()), 3),
            "Time in mkt %": round(float((exp > 0).mean() * 100), 1)}


with st.spinner("Downloading SPY / IEF / VIX + sector basket…"):
    core = fetch_ohlcv_batch(("SPY", "IEF", "^VIX"), period=f"{years}y")
    sect = fetch_ohlcv_batch(tuple(SECTOR_ETFS), period=f"{years}y")

if not core or any(s not in core for s in ("SPY", "IEF", "^VIX")):
    st.error("Could not fetch SPY/IEF/VIX — cannot run. Retry shortly.")
    st.stop()

spy = core["SPY"]["Close"].dropna()
ief = core["IEF"]["Close"].reindex(spy.index).ffill()
vix = core["^VIX"]["Close"].reindex(spy.index).ffill()
cal = spy.index
spy_ret = spy.pct_change()

baseline = compute_regime_exposure(spy, ief, vix).reindex(cal).ffill()

# ── Breadth series (historical % of sectors above their 10w MA) ──────────────
breadth = pd.Series(index=cal, dtype=float)
have_breadth = False
try:
    above = pd.DataFrame(index=cal)
    for s in SECTOR_ETFS:
        if s in sect and not sect[s].empty:
            c = sect[s]["Close"].reindex(cal).ffill()
            above[s] = (c > c.rolling(BREADTH_W).mean()).where(
                c.rolling(BREADTH_W).mean().notna())
    if above.shape[1] >= 6:
        breadth = above.mean(axis=1) * 100
        have_breadth = breadth.notna().sum() > 52
except Exception:
    have_breadth = False

# ── Credit series (HY OAS, weekly) ───────────────────────────────────────────
oas_w = pd.Series(dtype=float)
have_credit = False
try:
    from utils.fred import fetch_fred_series, SERIES, fred_available
    if fred_available():
        oas = fetch_fred_series(SERIES["hy_spread"], days=int(years * 372))
        if not oas.empty:
            oas_w = oas.resample("W-FRI").last().reindex(cal).ffill()
            have_credit = oas_w.notna().sum() > 52
except Exception:
    have_credit = False

# ── Build variants ────────────────────────────────────────────────────────────
variants = {"Baseline": baseline}
credit_stress = breadth_weak = None
if have_credit:
    credit_stress = oas_w > oas_w.rolling(CREDIT_MA_W).mean()
    variants["+Credit"] = _notch_down(baseline, credit_stress)
if have_breadth:
    breadth_weak = breadth < BREADTH_WEAK
    variants["+Breadth"] = _notch_down(baseline, breadth_weak)
if have_credit and have_breadth:
    both = _notch_down(_notch_down(baseline, credit_stress), breadth_weak)
    variants["+Both"] = both

if not have_credit:
    st.warning("⚠️ Credit overlay skipped — FRED_API_KEY missing or HY OAS "
               "unavailable. Add the key in secrets to test it.")
if not have_breadth:
    st.warning("⚠️ Breadth overlay skipped — sector ETF history insufficient.")

# ── Full-sample comparison ────────────────────────────────────────────────────
st.subheader("Full-sample results")
rows = {"SPY buy & hold": _metrics(pd.Series(1.0, index=cal), spy_ret)}
for name, exp in variants.items():
    rows[name] = _metrics(exp, spy_ret)
full = pd.DataFrame(rows).T
st.dataframe(full, use_container_width=True)
st.caption("Lower (more negative) MaxDD is worse. A confirmation overlay "
           "should cut MaxDD and hold or raise Sharpe; falling Avg exp shows "
           "the cost of de-risking. Baseline vs SPY confirms the gate itself "
           "still earns its keep.")

# ── Fold breakdown ────────────────────────────────────────────────────────────
st.subheader("Per-fold Sharpe and MaxDD")
edges = np.linspace(0, len(cal), N_FOLDS + 1).astype(int)
fold_sh, fold_dd = {}, {}
for name, exp in variants.items():
    sh_row, dd_row = {}, {}
    for fi in range(N_FOLDS):
        sl = slice(edges[fi], edges[fi + 1])
        seg_exp = exp.iloc[sl]
        seg_ret = spy_ret.iloc[sl]
        m = _metrics(seg_exp, seg_ret)
        lbl = f"F{fi+1} ({cal[edges[fi]].year}-{cal[min(edges[fi+1], len(cal)-1)].year})"
        sh_row[lbl] = m.get("Sharpe")
        dd_row[lbl] = m.get("MaxDD")
    fold_sh[name] = sh_row
    fold_dd[name] = dd_row
st.markdown("**Sharpe by fold**")
st.dataframe(pd.DataFrame(fold_sh).T, use_container_width=True)
st.markdown("**MaxDD by fold**")
st.dataframe(pd.DataFrame(fold_dd).T, use_container_width=True)

# ── Exposure paths ────────────────────────────────────────────────────────────
st.subheader("Exposure paths")
st.line_chart(pd.DataFrame({k: v for k, v in variants.items()}, index=cal))

# ── Verdict ───────────────────────────────────────────────────────────────────
st.subheader("Verdict")
base = full.loc["Baseline"]


def _judge(name):
    v = full.loc[name]
    d_sh = v["Sharpe"] - base["Sharpe"]
    d_dd = v["MaxDD"] - base["MaxDD"]            # + = shallower (better)
    folds_better = sum(1 for f in fold_dd[name]
                       if fold_dd[name][f] is not None
                       and base_dd_map[f] is not None
                       and fold_dd[name][f] >= base_dd_map[f] - 1e-9)
    give_up = base["CAGR"] - v["CAGR"]
    if d_dd > 0.01 and d_sh >= -0.05 and folds_better >= 3:
        return (f"✅ **{name}** cuts MaxDD by {d_dd*100:+.1f}pp "
                f"(Sharpe {d_sh:+.2f}, CAGR give-up {give_up*100:.1f}pp, "
                f"DD better in {folds_better}/4 folds) — earns adoption.")
    if d_sh > 0.05 and d_dd >= -0.005:
        return (f"✅ **{name}** lifts Sharpe {d_sh:+.2f} without worsening "
                f"drawdown — earns adoption.")
    if d_dd > 0 or d_sh > 0:
        return (f"🟠 **{name}** helps weakly (Sharpe {d_sh:+.2f}, MaxDD "
                f"{d_dd*100:+.1f}pp, {folds_better}/4 folds) — not decisive.")
    return (f"🛑 **{name}** does not improve the gate (Sharpe {d_sh:+.2f}, "
            f"MaxDD {d_dd*100:+.1f}pp) — baseline stands.")


base_dd_map = fold_dd["Baseline"]
spy_sh = full.loc["SPY buy & hold", "Sharpe"]
st.markdown(f"- Gate vs passive: Baseline Sharpe **{base['Sharpe']}** vs "
            f"SPY **{spy_sh}**, MaxDD **{base['MaxDD']*100:.1f}%** vs "
            f"**{full.loc['SPY buy & hold','MaxDD']*100:.1f}%** — "
            + ("gate earns its keep ✅" if base["Sharpe"] >= spy_sh
               and base["MaxDD"] >= full.loc["SPY buy & hold", "MaxDD"]
               else "gate's edge is risk-reduction, read MaxDD"))
for name in variants:
    if name != "Baseline":
        st.markdown("- " + _judge(name))

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("📋 Results for Claude")
from utils.significance import deflated_sharpe_ratio, verdict as _sigv
_var_sr = [float(full.loc[k, "Sharpe"]) for k in variants
           if k in full.index and full.loc[k, "Sharpe"] == full.loc[k, "Sharpe"]]
_sig = deflated_sharpe_ratio(float(max(_var_sr)), _var_sr, int(len(cal))) \
       if len(_var_sr) >= 2 else {"dsr": None, "n_trials": len(_var_sr),
                                  "sr0_ann": None, "t_stat": None,
                                  "harvey_pass": None, "psr0": None}
if _sig.get("dsr") is not None:
    st.subheader("🎲 Significance (multiple-testing corrected)")
    st.markdown("- " + _sigv(_sig))
    st.caption("Gate variants are few, so DSR is informational here; its main "
               "home is the selection and exit sweeps.")
payload = {"stage": "regime_lab_v1",
           "significance": _sig,
           "settings": {"years": years, "credit_ma_w": CREDIT_MA_W,
                        "breadth_w": BREADTH_W, "breadth_weak": BREADTH_WEAK,
                        "n_weeks": int(len(cal)),
                        "have_credit": have_credit,
                        "have_breadth": have_breadth},
           "full_sample": full.round(3).reset_index().to_dict("records"),
           "fold_sharpe": fold_sh, "fold_maxdd": fold_dd}
st.code(json.dumps(payload, indent=1, default=str), language="json")
