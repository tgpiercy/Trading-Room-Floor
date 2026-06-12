"""
pages/0_Playbook.py
THE PLAYBOOK — how to read every surface in StratFlow, what is decisional
vs diagnostic, the frozen constants, and the decision vocabulary captured
in the trade journal. This page is documentation: it changes only when a
validation result changes the system.
"""
import streamlit as st

st.title("📖 Playbook — how to read this system")
st.caption("Validated stack: selection_lab_v1 + Gate-3 sweep · "
           "exit_stage2_v1 + folds · risk layer · regime gate. "
           "Updated with each validation cycle.")

st.markdown("""
## The decision flow (capital moves through these gates, in order)

**1 · REGIME (Layer 1)** — the validated SPY/IEF/VIX gate sets gross
exposure: 1.0 / 0.5 / 0.0. Nothing below can add exposure past this cap.
Shown on the **Cockpit** badge and the **Rebalance** banner — that number
is THE regime. (Market Health % is a second opinion — see below.)

**2 · SELECTOR** — composite_v1: cross-sectional rank of
*26-week RS momentum* ⊕ *vol-adjusted ExtPct*, each vs the name's own
designated benchmark. Enter at **rank ≤ 10**.

**3 · REDUNDANCY FILTER** — walking candidates strongest-first, a name is
skipped if its 26-week return correlation with an already-accepted stronger
name exceeds **0.85**. Caps hidden concentration the sleeve labels miss.

**4 · HOLD BAND + EXIT LAYERS** — owned names persist while rank ≤ 30;
released only after rank > 30 for **2 consecutive weeks** (Layer 2, ~97% of
exits). The **4×ATR chandelier** (Layer 3, ~3%) is disaster insurance, with
stops anchored at your entry date. Re-entry is automatic via the selector.

**5 · RISK LAYER** — per-trade risk budget → 20% position cap → 40% sleeve
cap → vol target (≈18%). Reshapes weights; predicts nothing.

**6 · ORDERS** — the Rebalance diff. You place them. Then **📓 log the
journal** and **📡 log the flow snapshot**.

---

## Decision vocabulary (what the journal records)

| Decision | Gate | Meaning |
|---|---|---|
| ENTER | selector | rank ≤ 10 and passed redundancy |
| HELD | selector | owned, still rank ≤ 10 |
| HOLD-BAND | band | owned, rank 11–30, decay unconfirmed (weeks_breach shows progress toward release) |
| RELEASE-DECAY | layer 2 | rank > 30 for 2+ consecutive weeks |
| EXIT-TRAIL | layer 3 | close below the chandelier |
| SKIP-REDUNDANT | filter | >0.85 corr with an accepted stronger name |
| REGIME | layer 1 | run-level exposure context |

`mom_pct` / `extadj_pct` are the composite components (1.00 = strongest in
the universe). A name entering on mom_pct 0.95 / extadj_pct 0.60 is a slow
trend leader; 0.60 / 0.95 is a fresh thrust — both valid, different risks.

---

## How to read each page

**Decisional (the system):**
- **Rebalance** — the canonical book, decision matrix, orders. If any page
  disagrees with Rebalance, Rebalance wins.
- **Cockpit** — regime + book at a glance.

**Diagnostic (context — informs judgement, never overrides the matrix):**
- **Screener** — RS *states* (Early Leadership → Broken Trend): the
  validated narrative around a name. Use to understand WHAT KIND of
  strength the selector found, not to pick differently.
- **Trend Analysis** — the chart drill-down for any name the matrix
  surfaced. Confirm structure; check the chandelier visually.
- **Flow** — options positioning context. A research dataset is accruing
  (📡 button); flow becomes a candidate selection input only after it can
  be validated (~6–12 months of snapshots).
- **Market Health** — a parallel, price+breadth regime READ (MH%). When it
  agrees with the gate, conviction; when it disagrees, caution and smaller
  discretion — but the GATE controls exposure, not MH%.


**Research (harnesses, not trading surfaces):**
- **Backtest** — Phase-1 event studies (which raw signals have edge).
- **Exit Lab / Selection Lab** — the validation harnesses that froze the
  current stack. Any proposed change re-runs here first.
- **Strategy** — LEGACY Phase-2 backtest (predates the composite selector
  and hold band). Historical reference only.
- **Swing Screen** — EXPERIMENTAL parallel screen; its validated job
  (momentum + entry quality) is now done by the composite selector.

**Retired (validated negative — removed from navigation):**
- **Rotation Radar / Rotation Screener** — the sleeve-rotation event study
  (rotation_screen_v1) found NO forward spread: hot sleeves did not beat
  cold at 4w, in any fold. Rotation scoring is description of where
  strength already is — information the selector acts on directly. Files
  remain in the repo for reference; the predictive claim is dead.

---

## Frozen constants (changing any = re-validation)

| Constant | Value | Source |
|---|---|---|
| Entry rank | ≤ 10 | validated fixed top-N |
| Hold band | 11–30 | exit_stage2_v1 D |
| Decay confirmation | 2 weeks | exit_stage2_v1 D |
| Chandelier | 4.0 × ATR(14) | exit_sweep_v1 plateau |
| Momentum lookback | 26 w | Gate-3 plateau (13–39 all work) |
| Vol window | 26 w | a priori |
| Redundancy ceiling | 0.85 corr / 26 w | Gate-3 plateau |
| Regime exposure | 1.0 / 0.5 / 0.0 | validated gate |
| Risk | 1% trade · 20% pos · 40% sleeve · 18% vol | validated risk layer |

## Documented negative results (tested, rejected — do not re-add casually)
ExtPct<0 exit condition · single-benchmark ranking · defensive bond
overlay · trail-only exits · per-window parameter optimization ·
VSA / CMF standalone edge · sleeve-rotation score (no forward spread,
rotation_screen_v1).

**Pending verdicts (data accruing, not yet testable):** options-flow tilt
(CP-IV spread / skew / GEX) — flow tape v2 logging via the 📡 button;
verdict when ~6-12 months of snapshots exist. FINRA daily short-volume is
the next testable external dataset (free, full history).
""")
