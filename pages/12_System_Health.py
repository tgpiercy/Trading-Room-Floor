"""
pages/12_System_Health.py
SYSTEM HEALTH — the live-vs-validated audit. The validation produced a
specific behavioral fingerprint; this page checks whether the LIVE system
(via the trade journal) is actually exhibiting it, and whether the rules'
live outcomes match what the labs predicted. Live data is the only
out-of-sample test that can't be gamed — this page is where it reports.

Sections:
  1. Fingerprint vs validation — median hold, exit mix, book size,
     exposure path, with in/out-of-band status per metric
  2. Outcome audit — forward returns AFTER each decision type
     (releases should underperform post-release; entries outperform)
  3. Equity curve — mark-to-market of the tracked book, snapshotted to
     the equity_curve worksheet on every visit (one per day)
Needs journal history to say anything — early on it will mostly report
"insufficient data", which is honest, not broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.journal import (load_journal, log_equity, load_equity,
                           SYSTEM_VERSION, storage_status)

# ── Validated fingerprint (from exit_stage2_v1 / selection_lab_v1) ───────────
EXPECT = {
    "Median hold (wk)":   {"target": 10.0, "lo": 6.0,  "hi": 16.0},
    "Trail share of exits": {"target": 0.03, "lo": 0.0, "hi": 0.12},
    "Avg book size":      {"target": 18.6, "lo": 13.0, "hi": 24.0},
}
FWD_W = 4

st.set_page_config(page_title="System Health", layout="wide")
st.title("🩺 System Health — live vs validated")
st.caption(f"Config: **{SYSTEM_VERSION}** · journal storage: "
           f"**{storage_status()}** · validated fingerprint from "
           f"exit_stage2_v1 + selection_lab_v1")

j = load_journal()
if j.empty:
    st.info("No journal history yet. Run Rebalance and press **📓 Log to "
            "trade journal** — this page comes alive as entries accrue.")
    st.stop()

j["date"] = pd.to_datetime(j["date"], errors="coerce")
j = j.dropna(subset=["date"]).sort_values("date")
run_dates = sorted(j["date"].unique())
st.caption(f"{len(j)} journal rows · {len(run_dates)} logged runs · "
           f"{j['date'].min().date()} → {j['date'].max().date()}")

BOOK = {"ENTER", "HELD", "HOLD-BAND"}
EXITS = {"RELEASE-DECAY", "EXIT-TRAIL"}

# ── 1. Fingerprint ────────────────────────────────────────────────────────────
st.subheader("1 · Behavioral fingerprint vs validation")

# realized holds: pair each ticker's entry with its subsequent exit
holds = []
for tk, g in j[j["decision"].isin(BOOK | EXITS)].groupby("ticker"):
    g = g.sort_values("date")
    entry_d = None
    for _, r in g.iterrows():
        if r["decision"] in BOOK and entry_d is None:
            entry_d = r["date"]
        elif r["decision"] in EXITS and entry_d is not None:
            holds.append({"ticker": tk, "weeks":
                          (r["date"] - entry_d).days / 7,
                          "exit": r["decision"]})
            entry_d = None

exit_rows = j[j["decision"].isin(EXITS)]
n_exits = len(exit_rows)
trail_share = (float((exit_rows["decision"] == "EXIT-TRAIL").mean())
               if n_exits else None)
med_hold = (float(np.median([h["weeks"] for h in holds]))
            if len(holds) >= 3 else None)
book_sizes = j[j["decision"].isin(BOOK)].groupby("date").size()
avg_book = float(book_sizes.mean()) if len(book_sizes) else None

live = {"Median hold (wk)": med_hold,
        "Trail share of exits": trail_share,
        "Avg book size": avg_book}
rows = []
for k, exp in EXPECT.items():
    v = live[k]
    if v is None:
        status = "⏳ insufficient data"
    elif exp["lo"] <= v <= exp["hi"]:
        status = "✅ in band"
    else:
        status = "🚨 OUT OF BAND — investigate"
    rows.append({"Metric": k, "Live": round(v, 2) if v is not None else None,
                 "Validated": exp["target"],
                 "Band": f"{exp['lo']}–{exp['hi']}", "Status": status})
fp = pd.DataFrame(rows)
st.dataframe(fp, use_container_width=True, hide_index=True)
st.caption(f"Completed round-trips so far: {len(holds)} · exits logged: "
           f"{n_exits}. Bands are deliberately wide — a young journal is "
           "noisy; OUT OF BAND on small N is a prompt to look, not panic.")

reg = j[j["decision"] == "REGIME"].dropna(subset=["exposure"])
if len(reg) >= 2:
    st.line_chart(reg.set_index("date")["exposure"].astype(float))
    st.caption("Exposure path (Layer 1) — should match what the Cockpit "
               "showed at each run.")

# ── 2. Outcome audit — were the rules right, live? ───────────────────────────
st.subheader(f"2 · Outcome audit — forward {FWD_W}w return vs SPY after "
             "each decision")
st.caption("RELEASE-DECAY names should tend to UNDERPERFORM after release "
           "(the rule sold the right things); ENTER names should tend to "
           "outperform. This is live falsification of every rule.")

aud_kinds = ["ENTER", "RELEASE-DECAY", "EXIT-TRAIL", "SKIP-REDUNDANT"]
aud = j[j["decision"].isin(aud_kinds)].copy()
mature = aud[aud["date"] <= (pd.Timestamp.now()
                             - pd.Timedelta(weeks=FWD_W))]
if mature.empty:
    st.info(f"Outcome audit needs decisions at least {FWD_W} weeks old — "
            "none yet. It will populate automatically.")
else:
    tickers = sorted(set(mature["ticker"]) | {"SPY"})
    with st.spinner(f"Fetching prices for {len(tickers)} names…"):
        from utils.data_fetcher import fetch_ohlcv_batch
        from utils.watchlist import yf_sym
        px = fetch_ohlcv_batch(tuple(yf_sym(t) for t in tickers), period="2y")
    if not px or "SPY" not in px:
        st.warning("Price fetch failed — outcome audit skipped this visit.")
    else:
        spy = px["SPY"]["Close"]

        def fwd_rel(tk, d0):
            try:
                c = px[yf_sym(tk)]["Close"]
                i0 = c.index.searchsorted(d0)
                i1 = i0 + FWD_W
                if i1 >= len(c):
                    return None
                s0 = spy.index.searchsorted(d0)
                if s0 + FWD_W >= len(spy):
                    return None
                return float((c.iloc[i1] / c.iloc[i0])
                             - (spy.iloc[s0 + FWD_W] / spy.iloc[s0])) * 100
            except Exception:
                return None

        mature["fwd_rel"] = [fwd_rel(r["ticker"], r["date"])
                             for _, r in mature.iterrows()]
        oa = mature.dropna(subset=["fwd_rel"]).groupby("decision")["fwd_rel"]\
                   .agg(**{"Mean vs SPY %": "mean",
                           "Win %": lambda s: (s > 0).mean() * 100,
                           "N": "count"}).round(2)
        st.dataframe(oa, use_container_width=True)
        verdicts = []
        if "RELEASE-DECAY" in oa.index and oa.loc["RELEASE-DECAY", "N"] >= 10:
            ok = oa.loc["RELEASE-DECAY", "Mean vs SPY %"] < 0
            verdicts.append(("✅" if ok else "⚠️") + " released names "
                            + ("underperformed" if ok else "OUTPERFORMED")
                            + " after release"
                            + ("" if ok else " — exit rule leaking alpha, "
                               "flag for review"))
        if "ENTER" in oa.index and oa.loc["ENTER", "N"] >= 10:
            ok = oa.loc["ENTER", "Mean vs SPY %"] > 0
            verdicts.append(("✅" if ok else "⚠️") + " entries "
                            + ("outperformed" if ok else "did NOT outperform")
                            + " post-entry")
        for v in verdicts:
            st.markdown("- " + v)

# ── 3. Equity curve — snapshot + history ─────────────────────────────────────
st.subheader("3 · Equity curve (tracked book, mark-to-market)")
try:
    from utils.portfolio import load_holdings
    from utils.data_fetcher import get_current_price
    from utils.watchlist import yf_sym as _ys
    hl = [h for h in load_holdings()
          if str(h.get("ticker", "")).strip() and float(h.get("shares") or 0) > 0]
    mtm, priced = 0.0, 0
    for h in hl:
        try:
            p = get_current_price(str(h["ticker"]).upper().strip())
            if p:
                mtm += float(h["shares"]) * float(p)
                priced += 1
        except Exception:
            continue
    expo = None
    if len(reg):
        expo = float(reg["exposure"].astype(float).iloc[-1])
    if priced:
        fresh = log_equity(mtm, priced, expo,
                           note=f"{priced}/{len(hl)} priced")
        st.metric("Holdings MTM", f"${mtm:,.0f}",
                  f"{priced} position(s) priced"
                  + (" · snapshot saved" if fresh else " · already "
                     "snapshotted today"))
    else:
        st.info("No priced holdings — add positions in Portfolio to start "
                "the equity curve.")
except Exception as e:
    st.warning(f"MTM snapshot unavailable: {e}")

eq = load_equity()
if not eq.empty and len(eq) >= 2:
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    eq = eq.dropna(subset=["date"]).sort_values("date")
    st.line_chart(eq.set_index("date")["holdings_mtm"].astype(float))
    st.caption("One snapshot per day, persisted to the equity_curve sheet. "
               "Visit this page (or any day the app is open) to extend it.")

# ── Export ────────────────────────────────────────────────────────────────────
import json as _json
st.subheader("📋 Results for Claude")
payload = {"stage": "system_health_v1",
           "system_version": SYSTEM_VERSION,
           "journal": {"rows": int(len(j)), "runs": int(len(run_dates)),
                       "round_trips": len(holds), "exits": int(n_exits)},
           "fingerprint": fp.to_dict("records"),
           "equity_points": int(len(eq)) if not eq.empty else 0}
try:
    payload["outcomes"] = oa.reset_index().to_dict("records")
except Exception:
    payload["outcomes"] = "insufficient history"
st.code(_json.dumps(payload, indent=1, default=str), language="json")
