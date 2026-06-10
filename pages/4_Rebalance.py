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
top_n = c2.slider("Target names (N)", 5, 30, 15)
cost_bps = c3.slider("Cost (bps/side)", 0, 50, 10, step=5)
signal_label = c4.radio("Signal", ["RS Extension", "RS + trend filter"], index=0,
                        help="RS Extension is the validated primary. RS + trend filter adds the "
                             "chop/rollover exclusion (validate before relying on it).")
signal = "extpct_filtered" if "filter" in signal_label else "extpct"

if _RISK_OK:
    with st.expander("🛡️ Risk layer (vol target · position/sector caps · per-trade risk)",
                     expanded=False):
        apply_risk = st.checkbox("Apply risk layer to the target book", value=True)
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

    model = build_model_portfolio(ohlcv, pairs, top_n=top_n, signal=signal)
    if "error" in model:
        st.warning(model["error"])
        st.stop()

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
    if _RISK_OK and apply_risk and model["holdings"]:
        adj, risk_report = apply_risk_layer(
            model["holdings"], target_vol=target_vol, max_pos=max_pos,
            max_sector=max_sector, per_trade_risk=per_trade_risk)
        model["holdings"] = adj
        model["cash_weight"] = round(risk_report["cash_pct"] / 100, 3)
        st.markdown("**🛡️ Risk layer applied**")
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

    holdings = load_holdings()

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
            "price": "Price", "stop": "Stop", "sector": "Sector",
            "trade_risk_pct": "Risk %"})
        cols = ["Ticker", "Target %", "ExtPct", "Price", "Stop"]
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
               "regime gate, with GW2/ATR stops. Orders bring your tracked holdings to that target. "
               "Prices are last close (delayed); real fills differ. Verify before trading.")

st.divider()
st.caption("⚖️ Executable layer over the validated model. You place the orders — "
           "systematic decision support, not personalized financial advice.")
