"""
pages/4_Rebalance.py
Model portfolio + rebalance orders — the executable layer. Computes today's
target book from the validated RS-Extension model, diffs it against your
holdings (Portfolio tracker), and emits the BUY / ADD / TRIM / SELL orders to
align the book. Decision support — you place the orders. Not financial advice.
"""
import streamlit as st
import pandas as pd

from utils.watchlist import PORTFOLIO, yf_sym
from utils.data_fetcher import fetch_ohlcv_batch, get_current_price

try:
    from utils.rebalance import build_model_portfolio, build_orders, ACTION_COLOR
    from utils.portfolio import load_holdings
    _OK, _ERR = True, None
except Exception as e:
    _OK, _ERR = False, f"{type(e).__name__}: {e}"

try:
    from utils.risk import apply_risk_layer
    _RISK_OK = True
except Exception:
    _RISK_OK = False

st.title("⚖️ Model Portfolio & Rebalance")
st.caption("Today's target book from the validated RS-Extension model → the exact orders "
           "to align your holdings. Weekly-rebalance position algorithm on EOD data. "
           "Decision support, not financial advice.")

if not _OK:
    st.error(f"**Import failed:** `{_ERR}`")
    st.warning("Push **utils/rebalance.py** + **utils/portfolio.py** (and ensure "
               "strategy_backtest.py is deployed).")
    st.stop()

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.1])
account = c1.number_input("Account value ($)", min_value=0.0, value=100_000.0, step=1000.0)
top_n = c2.slider("Entry rank (N)", 5, 30, 10,
                  help="Validated: enter ≤10; held names persist to rank 30 (hold band).")
cost_bps = c3.slider("Cost (bps/side)", 0, 50, 10, step=5)
signal_label = c4.radio("Signal", ["RS Extension", "RS + trend filter"], index=0,
                        help="RS Extension is the validated primary. RS + trend filter adds the "
                             "chop/rollover exclusion (validate before relying on it).")
signal = "extpct_filtered" if "filter" in signal_label else "extpct"

core_pct = st.slider("Core (indices) allocation %", 0, 100, 50, step=5,
                     help="Share of exposure allocated to the CORE sleeve "
                          "(INDICES group): selected/exited by the same "
                          "validated machinery scaled to the sleeve (3/6 "
                          "band), stops apply, risk layer does NOT. The "
                          "remainder goes to the GROWTH sleeve (10/30 + "
                          "redundancy filter + risk layer).") / 100

if _RISK_OK:
    with st.expander("🛡️ Risk layer — GROWTH sleeve only (vol target · caps · per-trade risk)",
                     expanded=False):
        apply_risk = st.checkbox("Apply risk layer to the GROWTH sleeve "
                                 "(CORE indices keep stops, skip the risk layer)",
                                 value=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        target_vol = rc1.slider("Target vol %", 5, 30, 18) / 100
        max_pos = rc2.slider("Max position %", 5, 40, 20) / 100
        max_sector = rc3.slider("Max sector %", 20, 100, 40, step=5) / 100
        per_trade_risk = rc4.slider("Per-trade risk %", 0.25, 3.0, 1.0, step=0.25) / 100
else:
    apply_risk = False
    st.info("Push **utils/risk.py** to enable the risk layer (vol target + caps).")

run = st.button("⚖️ Build rebalance plan", type="primary", width="stretch")
if run or st.session_state.get("rb_run"):
    st.session_state["rb_run"] = True
    pairs = list(PORTFOLIO)
    syms = set(yf_sym(t) for p in pairs for t in (p[0], p[1])) | {"SPY", "IEF", "^VIX"}
    with st.spinner("Loading universe…"):
        ohlcv = fetch_ohlcv_batch(tuple(syms), period="2y")
    if not ohlcv:
        st.error("Data fetch failed (rate-limited). Wait and retry.")
        st.stop()

    # ── Data-QA gate: don't emit orders on a thin or stale universe ───────────
    from utils.data_qa import qa_report
    qa = qa_report(ohlcv, syms)
    _bench_missing = [b for b in ("SPY", "IEF", "^VIX") if b not in ohlcv]
    if qa["verdict"] == "block" or _bench_missing:
        st.error(qa["summary"])
        if _bench_missing:
            st.error(f"Critical benchmark(s) missing: {', '.join(_bench_missing)} "
                     "— the regime gate can't be computed.")
        st.caption("Coverage is below the safe floor (or a benchmark is "
                   "missing), so the model would build a different book than "
                   "intended. Retry the fetch; if a name is delisted (e.g. "
                   "LYTE), remove it from the universe.")
        if not st.checkbox("⚠️ Override and build anyway — I accept the book "
                           "may be incomplete", value=False):
            st.stop()
    elif qa["verdict"] == "warn":
        st.warning(qa["summary"])
        with st.expander("Data-QA detail"):
            if qa["missing"]:
                st.caption("Missing (no data this fetch): " + ", ".join(qa["missing"]))
            if qa["stale"]:
                st.caption("Stale (last bar lagging the universe): "
                           + ", ".join(qa["stale"]))
            st.caption("Proceeding — the book is built from the names that did "
                       "fetch. Persistent misses usually mean a delisting or a "
                       "ticker change worth fixing in the universe.")
    else:
        st.caption("🟢 " + qa["summary"])

    holdings = load_holdings()
    model = build_model_portfolio(ohlcv, pairs, top_n=top_n, signal=signal,
                                  current_holdings=holdings,
                                  core_pct=core_pct)
    if "error" in model:
        st.warning(model["error"])
        st.stop()
    _b = model.get("budgets", {})
    st.caption(f"🧭 Selector: **{model.get('selector','')}** · "
               f"Exit stack: **{model.get('exit_spec','')}** · sleeves: "
               f"CORE {_b.get('CORE',0)*100:.0f}% / GROWTH "
               f"{_b.get('GROWTH',0)*100:.0f}% of equity · "
               f"{model.get('n_hold_band',0)} name(s) riding the hold band.")

    # Regime / exposure banner
    exp = model["exposure"]
    if exp >= 0.99:
        st.success(f"🟢 Regime risk-ON — model fully invested ({exp*100:.0f}%), "
                   f"{model['n_held']} names.")
    elif exp <= 0.01:
        st.error("🔴 Regime risk-OFF — model is in **cash**. No target longs today.")
    else:
        st.warning(f"🟡 Regime caution — model {exp*100:.0f}% invested, "
                   f"{model['cash_weight']*100:.0f}% cash, {model['n_held']} names.")

    # ── Risk layer ──────────────────────────────────────────────────────────────
    risk_report = None
    _growth = [h for h in model["holdings"] if h.get("sleeve") == "GROWTH"]
    _core = [h for h in model["holdings"] if h.get("sleeve") != "GROWTH"]
    if _RISK_OK and apply_risk and _growth:
        _gbudget = model.get("budgets", {}).get("GROWTH",
                                                model.get("exposure", 1.0))
        adj, risk_report = apply_risk_layer(
            _growth, target_vol=target_vol, max_pos=max_pos,
            max_sector=max_sector, per_trade_risk=per_trade_risk,
            max_gross=max(_gbudget, 0.0001))
        model["holdings"] = _core + adj
        _inv = sum(h["weight"] for h in model["holdings"])
        model["cash_weight"] = round(max(0.0, 1 - _inv), 3)
        st.markdown(f"**🛡️ Risk layer applied — GROWTH sleeve "
                    f"({len(adj)} names); CORE ({len(_core)}) untouched**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. portfolio vol", f"{risk_report['vol_final_pct']}%",
                  f"{risk_report['vol_final_pct'] - risk_report['vol_raw_pct']:+.1f} vs raw",
                  delta_color="inverse")
        m2.metric("Gross invested", f"{risk_report['gross_after_pct']}%",
                  f"{risk_report['gross_after_pct'] - risk_report['gross_before_pct']:+.1f}")
        m3.metric("Max per-trade risk", f"{risk_report['max_trade_risk_pct']}%",
                  help=f"Budget {risk_report['per_trade_budget_pct']}% of equity per name")
        m4.metric("Cash", f"{risk_report['cash_pct']}%")
        bits = [f"target vol {risk_report['target_vol_pct']}%"]
        if risk_report["capped_names"]:
            bits.append("position/risk-capped: " + ", ".join(risk_report["capped_names"][:6]))
        if risk_report["capped_sectors"]:
            bits.append("sector-capped: " + ", ".join(
                f"{k} (was {v}%)" for k, v in risk_report["capped_sectors"].items()))
        st.caption(" · ".join(bits) + ". Weights below are after risk shaping.")

    # Prices: universe last close, else live fetch for held names off-universe
    price_of = {}
    want = {h["ticker"] for h in model["holdings"]} | {
        str(h.get("ticker", "")).upper().strip() for h in holdings}
    for tk in want:
        if not tk:
            continue
        yt = yf_sym(tk)
        if yt in ohlcv and not ohlcv[yt].empty:
            price_of[tk] = float(ohlcv[yt]["Close"].iloc[-1])
        else:
            try:
                p = get_current_price(tk)
                if p:
                    price_of[tk] = float(p)
            except Exception:
                pass

    # ── Target book ───────────────────────────────────────────────────────────
    st.subheader("🎯 Target portfolio")
    tdf = pd.DataFrame(model["holdings"])
    if not tdf.empty:
        tdf = tdf.assign(weight=(tdf["weight"] * 100).round(1)).rename(columns={
            "ticker": "Ticker", "weight": "Target %", "ext": "ExtPct",
            "rank": "Rank", "band": "Band",
            "price": "Price", "stop": "Stop", "sector": "Sector",
            "trade_risk_pct": "Risk %"})
        if "sleeve" in tdf.columns:
            tdf = tdf.rename(columns={"sleeve": "Sleeve"})
        cols = ["Ticker", "Sleeve", "Rank", "Band", "Target %", "ExtPct", "Price", "Stop"]
        cols = [c for c in cols if c in tdf.columns]
        if "Sector" in tdf.columns:
            cols.insert(2, "Sector")
        if "Risk %" in tdf.columns:
            cols.append("Risk %")
        st.dataframe(tdf[cols],
                     width="stretch", hide_index=True,
                     column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Stop": st.column_config.NumberColumn(format="$%.2f"),
                                    "Target %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Risk %": st.column_config.NumberColumn(
                                        format="%.2f%%", help="Equity lost if stopped out")})

    # ── Orders ──────────────────────────────────────────────────────────────────
    df, summary = build_orders(model, holdings, account, price_of)
    st.subheader("📝 Rebalance orders")
    if df.empty:
        st.info("No orders — your book already matches the model (or no prices available).")
    else:
        def _css(v):
            return f"background-color:{ACTION_COLOR.get(v, '#888')}33; font-weight:700"
        try:
            sty = df.style
            (sty.map if hasattr(sty, "map") else sty.applymap)(_css, subset=["Action"])
            st.dataframe(sty, width="stretch", hide_index=True,
                         column_config={"Δ $": st.column_config.NumberColumn(format="$%.0f"),
                                        "Cur %": st.column_config.NumberColumn(format="%.1f%%"),
                                        "Tgt %": st.column_config.NumberColumn(format="%.1f%%"),
                                        "Stop": st.column_config.NumberColumn(format="$%.2f")})
        except Exception:
            st.dataframe(df, width="stretch", hide_index=True)

        est_cost = summary["gross_turn_dollars"] * (cost_bps / 10000.0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", f"{summary['n_buy']+summary['n_sell']}",
                  f"{summary['n_buy']} buy / {summary['n_sell']} sell")
        m2.metric("Turnover", f"{summary['turnover_pct']:.0f}%")
        m3.metric("Invested / Cash", f"{summary['gross_exposure_pct']:.0f}% / {summary['cash_pct']:.0f}%")
        m4.metric("Est. cost", f"${est_cost:,.0f}", f"@ {cost_bps} bps/side")

    st.caption("Target = validated RS-Extension ranking → top-N, inverse-vol weights scaled by the "
               "regime gate, with validated chandelier (4×ATR) stops. Orders bring your tracked holdings to that target. "
               "Prices are last close (delayed); real fills differ. Verify before trading.")

# The decision matrix, flow snapshot, and journal below all require the built
# `model`/`df` from the run-guard above. On initial page load (before the plan
# is built) those names don't exist yet — gate the whole tail on the same
# condition so the page shows just the intro + button until a plan is built.
if not (run or st.session_state.get("rb_run")):
    st.stop()

st.divider()
st.subheader("🧾 Decision matrix — why each call was made")
dec_df = pd.DataFrame(model.get("decisions", []))
if not dec_df.empty:
    show_cols = [c for c in ["ticker", "decision", "gate", "sleeve", "rank", "mom_pct",
                             "extadj_pct", "weeks_breach", "blocked_by",
                             "corr", "price", "stop", "weight", "reason"]
                 if c in dec_df.columns]
    st.dataframe(dec_df[show_cols], width="stretch", hide_index=True)
    st.caption("Every name the system touched this run, the gate that "
               "decided it, and the quantitative reason. ENTER/HELD = "
               "selector · HOLD-BAND = riding 11-30 · RELEASE-DECAY / "
               "EXIT-TRAIL = exit layers · SKIP-REDUNDANT = correlation "
               "filter. mom_pct / extadj_pct are the two composite "
               "components (1.00 = strongest in universe).")
    try:
        from utils.journal import log_decisions, storage_status as _jstat
        # merge risk-layer final weights into journal rows before logging
        _final_w = {h["ticker"]: h["weight"] for h in model["holdings"]}
        _rows = []
        for r in model["decisions"]:
            rr = dict(r)
            if rr.get("ticker") in _final_w:
                rr["weight"] = _final_w[rr["ticker"]]
            _rows.append(rr)
        # orders → journal too (execution audit trail)
        _today = model["decisions"][0]["date"]
        if not df.empty:
            for _, _o in df.iterrows():
                if _o["Action"] == "HOLD":
                    continue
                _rows.append({"date": _today, "ticker": _o["Ticker"],
                              "decision": f"ORDER-{_o['Action']}",
                              "gate": "execution",
                              "weight": round(float(_o["Tgt %"]) / 100, 4),
                              "stop": _o.get("Stop"),
                              "reason": f"{_o['Δ sh']:+.1f} sh "
                                        f"(${_o['Δ $']:+,.0f}) → "
                                        f"{_o['Tgt %']:.1f}%"})
        jc1, jc2 = st.columns([1, 2])
        if jc1.button("📓 Log to trade journal", width="stretch"):
            n = log_decisions(_rows)
            jc2.success(f"Journaled {n} new decision(s) → {_jstat()}")
        else:
            jc2.caption(f"Journal storage: {_jstat()}. Log after you place "
                        "the orders so the journal reflects acted decisions.")
    except Exception as _je:
        st.warning(f"Journal unavailable: {_je}")

st.divider()
st.subheader("📡 Flow snapshot (research dataset)")
st.caption("Logs today's options-flow snapshot for the target book + your "
           "holdings to the flow_history sheet — building the dataset for "
           "future flow-tilt validation. Run after each rebalance.")
if st.session_state.get("rb_run") and st.button("📡 Log flow snapshot",
                                                width="stretch"):
    try:
        from utils.flow_logger import log_flow_snapshot
        names = [h["ticker"] for h in model.get("holdings", [])]
        names += [str(h.get("ticker", "")).upper().strip() for h in holdings]
        fprog = st.progress(0.0, text="Snapshotting flow…")
        out = log_flow_snapshot(names, lambda f, tk: fprog.progress(
            min(f, 1.0), text=f"Flow: {tk}"))
        fprog.empty()
        st.success(f"Logged {out['logged']} snapshot(s) "
                   f"({out['skipped']} skipped) → {out['storage']}")
    except Exception as e:
        st.error(f"Flow logging failed: {e}")

st.divider()
st.caption("⚖️ Executable layer over the validated model. You place the orders — "
           "systematic decision support, not personalized financial advice.")
