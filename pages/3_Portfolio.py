"""
pages/3_Portfolio.py
Owned-position tracker: holdings → balanced decisions (hold / add / trim / raise
stop / exit) with transparent reasoning, continuous score history, durable
Google Sheets (or local JSON) persistence. Decision SUPPORT, not advice.
"""
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_fetcher import get_stock_data, get_options_cboe

try:
    from utils.portfolio import (load_holdings, save_holdings, decide_position,
                                 append_scores, load_score_history, export_json,
                                 import_json, storage_status, DECISION_COLOR, HOLD_COLS)
    _OK, _ERR = True, None
except Exception as e:
    _OK, _ERR = False, f"{type(e).__name__}: {e}"

st.title("💼 Portfolio")
st.caption("Owned-position tracking with balanced decisions and continuous scoring. "
           "Signal-based decision support — not financial advice.")

if not _OK:
    st.error(f"**Import failed:** `{_ERR}`")
    st.warning("Push **utils/portfolio.py** (and ensure flow_analysis/strategy/rs_indicators "
               "are deployed). If it names gspread, add it to requirements.txt.")
    st.stop()

status = storage_status()
icon = "🟢" if "Sheets" in status else "🟡"
st.caption(f"{icon} Persistence: **{status}**")
if "Sheets" not in status:
    with st.expander("⚙️ Enable durable Google Sheets storage (survives reboots)"):
        st.markdown(
            "Local JSON resets when the app reboots. For durable storage:\n"
            "1. In Google Cloud, create a **service account** + JSON key; enable the "
            "Sheets & Drive APIs.\n"
            "2. Create a Google Sheet named **StratFlow Portfolio** and **share it** with "
            "the service-account email (Editor).\n"
            "3. In Streamlit **Manage app → Settings → Secrets**, add the key under "
            "`[gcp_service_account]` (paste the JSON fields), and optionally "
            "`portfolio_sheet_key = \"<sheet id>\"`.\n"
            "Once set, this page auto-switches to Sheets — no code change.")

# ── Holdings editor ───────────────────────────────────────────────────────────
st.subheader("📋 Holdings")
holds = load_holdings()
df = pd.DataFrame(holds) if holds else pd.DataFrame(columns=HOLD_COLS)
for c in HOLD_COLS:
    if c not in df.columns:
        df[c] = None
edited = st.data_editor(
    df[HOLD_COLS], num_rows="dynamic", width="stretch", key="pf_editor",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "shares": st.column_config.NumberColumn("Shares", format="%d"),
        "entry_price": st.column_config.NumberColumn("Entry $", format="$%.2f"),
        "entry_date": st.column_config.TextColumn("Entry Date"),
        "stop": st.column_config.NumberColumn("Stop $", format="$%.2f"),
        "notes": st.column_config.TextColumn("Notes"),
    })

c1, c2, c3 = st.columns(3)
if c1.button("💾 Save holdings", width="stretch"):
    recs = [{k: r.get(k) for k in HOLD_COLS} for r in edited.to_dict("records")
            if str(r.get("ticker", "")).strip()]
    for r in recs:
        r["ticker"] = str(r["ticker"]).upper().strip()
    save_holdings(recs)
    st.success(f"Saved {len(recs)} position(s).")
c2.download_button("⬇️ Export backup", export_json(), file_name="stratflow_portfolio.json",
                   width="stretch")
up = c3.file_uploader("⬆️ Import backup", type="json", label_visibility="collapsed")
if up is not None:
    try:
        import_json(up.read().decode())
        st.success("Imported. Reload to see holdings.")
    except Exception as e:
        st.error(f"Import failed: {e}")

st.divider()

# ── Analyze positions ─────────────────────────────────────────────────────────
if st.button("📊 Analyze positions", type="primary", width="stretch"):
    st.session_state["pf_analyze"] = True

if st.session_state.get("pf_analyze"):
    holds = load_holdings()
    if not holds:
        st.info("Add holdings above and Save, then Analyze.")
        st.stop()
    with st.spinner("Loading benchmark…"):
        bench = get_stock_data("SPY", period="2y", interval="1d")
    rows, snaps = [], []
    today = str(dt.date.today())
    prog = st.progress(0.0, text="Scoring positions…")
    for i, h in enumerate(holds):
        tk = str(h.get("ticker", "")).upper().strip()
        prog.progress((i + 1) / len(holds), text=f"Scoring {tk} ({i+1}/{len(holds)})…")
        if not tk:
            continue
        daily = get_stock_data(tk, period="2y", interval="1d")
        if daily is None or daily.empty:
            continue
        spot = None; calls = puts = None; expiry = None
        try:
            spot, expmap = get_options_cboe(tk)
            if expmap:
                e = sorted(expmap.keys())[0]
                calls, puts = expmap[e]; expiry = e
        except Exception:
            pass
        d = decide_position(h, daily, bench, calls, puts, spot, expiry)
        rows.append(d)
        snaps.append({"date": today, "ticker": tk, "score": d["score"],
                      "decision": d["decision"], "price": d["spot"], "stop": d["stop"]})
    prog.empty()
    append_scores(snaps)   # continuous scoring

    if not rows:
        st.warning("No positions could be scored (check tickers / data availability).")
        st.stop()

    # Decisions table
    st.subheader("🎯 Decisions")
    show = pd.DataFrame([{
        "Ticker": r["ticker"], "Decision": r["decision"], "Score": r["score"],
        "RS State": r["rs_state"], "Regime": r["regime"],
        "Price": r["spot"], "P&L %": r["pnl_pct"], "R": r["r_multiple"],
        "Stop": r["stop"] or None,
        "↑ Stop": r["suggested_stop"] if r["raise_stop"] else None,
        "Why": r["reasons"],
    } for r in rows])

    def _dc(v):
        return f"background-color:{DECISION_COLOR.get(v, '#888')}33;font-weight:700"
    try:
        sty = show.style
        _cell = sty.map if hasattr(sty, "map") else sty.applymap
        _cell(_dc, subset=["Decision"])
        st.dataframe(sty, width="stretch", hide_index=True,
                     column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Stop": st.column_config.NumberColumn(format="$%.2f"),
                                    "↑ Stop": st.column_config.NumberColumn(format="$%.2f"),
                                    "P&L %": st.column_config.NumberColumn(format="%.1f%%")})
    except Exception:
        st.dataframe(show, width="stretch", hide_index=True)

    # Action summary
    acts = show["Decision"].value_counts().to_dict()
    chips = " · ".join(f"{k}: {v}" for k, v in acts.items())
    st.caption(f"Summary: {chips}. Decisions blend flow outlook + RS-trend state + GW2 "
               "stops + your P&L. Every call shows its reasoning under **Why**.")

    # ── Continuous score history ──────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Score History (continuous)")
    hist = load_score_history()
    if hist.empty or len(hist) < 2:
        st.info("Score history builds over time — each Analyze run records today's score "
                "per position. Come back over days to see trajectories.")
    else:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date"]).sort_values("date")
        tickers = sorted(hist["ticker"].unique())
        sel = st.selectbox("Position", tickers)
        h = hist[hist["ticker"] == sel]
        fig = go.Figure(go.Scatter(x=h["date"], y=pd.to_numeric(h["score"], errors="coerce"),
                        mode="lines+markers", line=dict(color="#00ff88", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(template="plotly_dark", height=300, yaxis_title="Flow Outlook Score",
                          yaxis=dict(range=[-10, 10]), margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{sel} score trajectory — rising = strengthening thesis, rolling over = "
                   "deteriorating (consider trimming / tightening stop).")

st.divider()
st.caption("⚠️ Decision support from systematic signals on 15-min-delayed/EOD data — "
           "not personalized financial advice. You own every decision. Verify before acting.")
