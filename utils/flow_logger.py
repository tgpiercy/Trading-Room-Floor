"""
utils/flow_logger.py
Daily options-flow snapshot logger. PURPOSE: build the historical flow
dataset that does not currently exist, so a flow-based selection tilt can
be validated in 6-12 months. Logging now costs nothing; not logging costs
research that can never be recovered.

Each snapshot row (one per ticker per date, deduped):
    date, ticker, spot, expiry, call_prem, put_prem, prem_pcr, call_pct,
    n_vol_gt_oi  (contracts with Volume > OI — new positioning proxy)

Persistence mirrors utils/portfolio.py: Google Sheets worksheet
'flow_history' in the StratFlow Portfolio spreadsheet when secrets are set,
local JSON fallback otherwise (fallback resets on reboot — Sheets strongly
recommended for this module, the whole point is durability).
"""
import os
import json
import datetime as _dt
import pandas as pd
import streamlit as st

# v2 schema — every options feature with literature support, captured daily
# so the predictive-model dataset accrues. (v1 tab left untouched; v2 writes
# to its own worksheet.)
FLOW_COLS = ["date", "ticker", "spot", "exp_near", "exp_month",
             "call_prem", "put_prem", "prem_pcr", "call_pct",
             "oi_call", "oi_put", "pcr_oi", "vol_oi", "n_vol_gt_oi",
             "atm_iv", "iv_skew", "cpiv_spread", "net_gex_mm"]
_WS_NAME = "flow_history_v2"
_JSON = "/tmp/.stratflow_flow_history_v2.json"


def _ws():
    """Worksheet 'flow_history' in the portfolio spreadsheet, or None."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scopes)
        gc = gspread.authorize(creds)
        key = st.secrets.get("portfolio_sheet_key")
        sh = gc.open_by_key(key) if key else gc.open("StratFlow Portfolio")
        try:
            return sh.worksheet(_WS_NAME)
        except Exception:
            w = sh.add_worksheet(title=_WS_NAME, rows=20000,
                                 cols=len(FLOW_COLS))
            w.update([FLOW_COLS])
            return w
    except Exception:
        return None


def storage_status() -> str:
    return "Google Sheets (durable)" if _ws() else "Local JSON (resets!)"


def _bs_gamma(spot, strike, iv, t_years, r=0.045):
    """Black-Scholes gamma (same for calls and puts)."""
    import math
    if not (spot and strike and iv and t_years) or iv <= 0 or t_years <= 0:
        return 0.0
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) \
             / (iv * math.sqrt(t_years))
        phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        return phi / (spot * iv * math.sqrt(t_years))
    except Exception:
        return 0.0


def _chain_features(calls, puts, spot, expiry, r):
    """Literature-backed features from one expiry's chain.
    iv_skew      : mean OTM-put IV (0.90-0.97 moneyness) − mean OTM-call IV
                   (1.03-1.10). Steep positive skew = downside fear priced.
    cpiv_spread  : OI-weighted mean(call IV − put IV) on matched near-ATM
                   strikes (Cremers-Weinbaum proxy; positive predicts
                   outperformance at ~weekly horizon in the literature).
    net_gex_mm   : Σ gamma·OI·spot²·0.01 (calls +, puts −), $mm per 1% move
                   — dealer-positioning/vol-regime context.
    """
    import datetime as _dt
    import pandas as pd
    out = {"atm_iv": None, "iv_skew": None, "cpiv_spread": None,
           "net_gex_mm": None}
    try:
        t = max(( _dt.date.fromisoformat(str(expiry)[:10])
                  - _dt.date.today()).days, 1) / 365.0
    except Exception:
        t = 21 / 365.0
    try:
        c = calls.dropna(subset=["strike"]).copy()
        p = puts.dropna(subset=["strike"]).copy()
        for df in (c, p):
            df["iv"] = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
            df["m"] = df["strike"] / spot
        c_atm = c[(c["m"] >= 0.97) & (c["m"] <= 1.03)]
        p_atm = p[(p["m"] >= 0.97) & (p["m"] <= 1.03)]
        ivs = pd.concat([c_atm["iv"], p_atm["iv"]]).dropna()
        if len(ivs):
            out["atm_iv"] = round(float(ivs.mean()), 4)
        p_otm = p[(p["m"] >= 0.90) & (p["m"] <= 0.97)]["iv"].dropna()
        c_otm = c[(c["m"] >= 1.03) & (c["m"] <= 1.10)]["iv"].dropna()
        if len(p_otm) and len(c_otm):
            out["iv_skew"] = round(float(p_otm.mean() - c_otm.mean()), 4)
        merged = c_atm.merge(p_atm, on="strike", suffixes=("_c", "_p"))
        merged = merged.dropna(subset=["iv_c", "iv_p"])
        if len(merged):
            w = (merged["openInterest_c"].fillna(0)
                 + merged["openInterest_p"].fillna(0)).clip(lower=1)
            out["cpiv_spread"] = round(float(
                ((merged["iv_c"] - merged["iv_p"]) * w).sum() / w.sum()), 4)
        gex = 0.0
        for df, sign in ((c, +1), (p, -1)):
            for _, row in df.iterrows():
                g = _bs_gamma(spot, float(row["strike"]),
                              float(row["iv"]) if row["iv"] == row["iv"] else 0,
                              t, r)
                gex += sign * g * float(row.get("openInterest") or 0) \
                       * 100 * spot * spot * 0.01
        out["net_gex_mm"] = round(gex / 1e6, 2)
    except Exception:
        pass
    return out


def snapshot_ticker(tk: str) -> dict | None:
    """Full v2 flow snapshot: premium flow + OI structure from the NEAR
    expiry, IV features (skew / CP-IV spread / GEX) from the ~MONTHLY
    expiry (more stable IVs than weeklies)."""
    from utils.data_fetcher import get_options_cboe
    from utils.order_flow import premium_flow
    try:
        from utils.fred import get_risk_free_rate
        r = get_risk_free_rate()
    except Exception:
        r = 0.045
    try:
        import datetime as _dt
        spot, expmap = get_options_cboe(tk)
        if not expmap or not spot:
            return None
        exps = sorted(expmap.keys())
        exp_near = exps[0]
        target = _dt.date.today() + _dt.timedelta(days=30)
        exp_month = min(exps, key=lambda e: abs(
            (_dt.date.fromisoformat(str(e)[:10]) - target).days))
        calls_n, puts_n = expmap[exp_near]
        pf = premium_flow(calls_n, puts_n)
        n_new = 0
        oi_c = oi_p = vol_t = 0.0
        for df, is_call in ((calls_n, True), (puts_n, False)):
            try:
                v = df["volume"].fillna(0)
                oi = df["openInterest"].fillna(0)
                n_new += int((v > oi).sum())
                vol_t += float(v.sum())
                if is_call:
                    oi_c += float(oi.sum())
                else:
                    oi_p += float(oi.sum())
            except Exception:
                pass
        calls_m, puts_m = expmap[exp_month]
        feats = _chain_features(calls_m, puts_m, float(spot), exp_month, r)
        return {"date": str(_dt.date.today()), "ticker": tk,
                "spot": round(float(spot), 2),
                "exp_near": str(exp_near), "exp_month": str(exp_month),
                "call_prem": round(pf["call_premium"], 0),
                "put_prem": round(pf["put_premium"], 0),
                "prem_pcr": pf["prem_pcr"], "call_pct": pf["call_pct"],
                "oi_call": int(oi_c), "oi_put": int(oi_p),
                "pcr_oi": round(oi_p / oi_c, 2) if oi_c else None,
                "vol_oi": round(vol_t / (oi_c + oi_p), 3)
                          if (oi_c + oi_p) else None,
                "n_vol_gt_oi": n_new, **feats}
    except Exception:
        return None


def log_flow_snapshot(tickers: list, progress_cb=None) -> dict:
    """Snapshot each ticker and append (deduped by date+ticker).
    Returns {logged, skipped, storage}."""
    snaps = []
    tickers = list(dict.fromkeys(t for t in tickers if t))
    for i, tk in enumerate(tickers):
        if progress_cb:
            progress_cb((i + 1) / len(tickers), tk)
        s = snapshot_ticker(tk)
        if s:
            snaps.append(s)
    logged = _append(snaps)
    return {"logged": logged, "skipped": len(tickers) - len(snaps),
            "storage": storage_status()}


def _append(snaps: list) -> int:
    if not snaps:
        return 0
    ws = _ws()
    if ws:
        try:
            seen = {(str(r["date"]), str(r["ticker"]))
                    for r in ws.get_all_records()}
            new = [[s.get(c, "") for c in FLOW_COLS] for s in snaps
                   if (s["date"], s["ticker"]) not in seen]
            if new:
                ws.append_rows(new)
            return len(new)
        except Exception:
            pass
    try:
        hist = json.load(open(_JSON)) if os.path.exists(_JSON) else []
    except Exception:
        hist = []
    seen = {(str(r["date"]), str(r["ticker"])) for r in hist}
    new = [s for s in snaps if (s["date"], s["ticker"]) not in seen]
    hist.extend(new)
    try:
        json.dump(hist, open(_JSON, "w"))
    except Exception:
        return 0
    return len(new)


def load_flow_history() -> pd.DataFrame:
    ws = _ws()
    rows = []
    if ws:
        try:
            rows = ws.get_all_records()
        except Exception:
            rows = []
    if not rows:
        try:
            rows = json.load(open(_JSON)) if os.path.exists(_JSON) else []
        except Exception:
            rows = []
    return pd.DataFrame(rows)
