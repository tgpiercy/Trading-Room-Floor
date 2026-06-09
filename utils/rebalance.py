"""
utils/rebalance.py
Turns the validated RS-Extension strategy into an executable algorithm:
  1. build_model_portfolio() — today's TARGET book (rank by causal ExtPct, take
     top-N, inverse-vol weights, scaled by the regime gate, with GW2/ATR stops).
     Uses the SAME precompute_series the walk-forward validated, so the live
     model is identical to what was tested.
  2. build_orders() — diff the target against current holdings (from the Portfolio
     tracker) → the exact BUY / SELL / ADD / TRIM orders, with share counts,
     dollar deltas, turnover and estimated cost.

Decision support, not advice. Weekly-rebalance position algorithm on EOD data.
"""
import pandas as pd
import numpy as np

from utils.strategy_backtest import precompute_series
from utils.strategy import calc_price_indicators, calc_stops
from utils.watchlist import yf_sym

ACTION_COLOR = {"BUY": "#00cc66", "ADD": "#26a69a", "HOLD": "#4fc3f7",
                "TRIM": "#ff9800", "SELL": "#ff4444"}


def build_model_portfolio(ohlcv, pairs, top_n: int = 15, vol_lookback: int = 13,
                          signal: str = "extpct", with_stops: bool = True) -> dict:
    """Today's target portfolio from the validated strategy."""
    pc = precompute_series(ohlcv, pairs, vol_lookback, signal=signal)
    if "error" in pc:
        return pc
    data, exposure = pc["data"], pc["exposure"]
    e = float(exposure.iloc[-1]) if len(exposure) else 0.0

    scored = []
    for tk, d in data.items():
        ext, px = d["ext"].iloc[-1], d["close"].iloc[-1]
        if pd.notna(ext) and pd.notna(px):
            scored.append((tk, float(ext), d))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    invs = []
    for tk, ext, d in top:
        v = d["vol"].iloc[-1]
        invs.append(1.0 / v if (v == v and v > 0) else 0.0)
    ssum = sum(invs) or 1.0

    holds = []
    for (tk, ext, d), iv in zip(top, invs):
        w = (iv / ssum) * e                      # inverse-vol, scaled by regime
        px = float(d["close"].iloc[-1])
        stop = None
        if with_stops:
            try:
                wk = calc_price_indicators(ohlcv[yf_sym(tk)].copy())
                sp = calc_stops(wk).get("program_stop")
                stop = round(float(sp), 2) if sp else None
            except Exception:
                stop = None
        holds.append({"ticker": tk, "weight": round(w, 4), "ext": round(ext, 2),
                      "price": round(px, 2), "stop": stop})

    invested = sum(h["weight"] for h in holds)
    return {"exposure": round(e, 2), "cash_weight": round(max(0.0, 1 - invested), 3),
            "holdings": holds, "top_n": top_n, "n_held": len(holds)}


def build_orders(model: dict, current_holdings: list, account: float,
                 price_of: dict, min_trade_frac: float = 0.005):
    """Diff target vs current → orders. price_of maps ticker→latest price.
    min_trade_frac: ignore resizes smaller than this fraction of the account."""
    target = {h["ticker"]: h for h in model.get("holdings", [])}
    cur = {}
    for h in current_holdings or []:
        tk = str(h.get("ticker", "")).upper().strip()
        if not tk:
            continue
        try:
            cur[tk] = float(h.get("shares") or 0)
        except Exception:
            cur[tk] = 0.0

    rows, gross_turn = [], 0.0
    for tk in sorted(set(target) | set(cur)):
        px = price_of.get(tk)
        if not px or px <= 0:
            continue
        tw = target.get(tk, {}).get("weight", 0.0)
        tgt_sh = (tw * account) / px
        cur_sh = cur.get(tk, 0.0)
        cur_w = (cur_sh * px) / account if account else 0.0
        dsh = tgt_sh - cur_sh
        dval = dsh * px

        if cur_sh > 0 and tw > 0 and abs(dval) < min_trade_frac * account:
            action = "HOLD"
        elif cur_sh <= 0 and tw > 0:
            action = "BUY"
        elif tw <= 0 and cur_sh > 0:
            action = "SELL"
        elif dsh > 0:
            action = "ADD"
        else:
            action = "TRIM"

        if action != "HOLD":
            gross_turn += abs(dval)
        rows.append({"Action": action, "Ticker": tk,
                     "Cur sh": round(cur_sh, 2), "Tgt sh": round(tgt_sh, 2),
                     "Δ sh": round(dsh, 2), "Δ $": round(dval, 0),
                     "Cur %": round(cur_w * 100, 1), "Tgt %": round(tw * 100, 1),
                     "Stop": target.get(tk, {}).get("stop")})

    df = pd.DataFrame(rows)
    n_buy = int(df["Action"].isin(["BUY", "ADD"]).sum()) if not df.empty else 0
    n_sell = int(df["Action"].isin(["SELL", "TRIM"]).sum()) if not df.empty else 0
    summary = {"gross_exposure_pct": round(sum(h["weight"] for h in model.get("holdings", [])) * 100, 1),
               "cash_pct": round(model.get("cash_weight", 0) * 100, 1),
               "turnover_pct": round(gross_turn / account * 100, 1) if account else 0.0,
               "est_cost": round(gross_turn / account * 100, 1),  # placeholder; ×bps in page
               "n_buy": n_buy, "n_sell": n_sell, "gross_turn_dollars": round(gross_turn, 0)}
    return df, summary

