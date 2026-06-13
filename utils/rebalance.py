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


# ── Sleeve architecture (v4) ──────────────────────────────────────────────────
# CORE  = the INDICES group: RRSP-style core, selected and exited by the same
#         validated machinery scaled to the sleeve (entry top-3 of 9, band 6,
#         2wk confirm, 4×ATR trail). NO risk layer, NO redundancy filter
#         (indices are intentionally overlapping); stops/exits fully apply.
# GROWTH= everything else: validated 10/30 selector + redundancy + exit stack,
#         risk layer applies to this sleeve only.
# NOTE: the core 3/6 scaling preserves the validated 10%/30% proportions on
# the 9-name sleeve but has NOT been independently lab-validated — flagged
# for a confirmation arm. Regime exposure (Layer 1) gates BOTH sleeves.
CORE_GROUPS = ("INDICES",)
CORE_ENTRY = 3
CORE_BAND = 6


def build_model_portfolio(ohlcv, pairs, top_n: int = 10, vol_lookback: int = 13,
                          signal: str = "extpct", with_stops: bool = True,
                          current_holdings: list | None = None,
                          selector: str = "composite_v1",
                          core_pct: float = 0.50) -> dict:
    """Two-sleeve target book: CORE (indices) at core_pct of exposure,
    GROWTH at the remainder. Both run the validated selector + exit stack
    within their own sleeve; only GROWTH gets the redundancy filter and
    (on the page) the risk layer. Decision matrix tags every row by sleeve.
    """
    from utils.exits import (decay_matrix, wilder_atr, trail_level,
                             EXIT_RANK, CONFIRM_WEEKS, TRAIL_K)
    from utils.selection import (rs_frames, composite_components,
                                 redundancy_filter_today, SELECTOR_VERSION)
    import datetime as _dt

    pc = precompute_series(ohlcv, pairs, vol_lookback, signal=signal)
    if "error" in pc:
        return pc
    data, exposure, cal = pc["data"], pc["exposure"], pc["cal"]
    e = float(exposure.iloc[-1]) if len(exposure) else 0.0
    core_pct = min(max(float(core_pct), 0.0), 1.0)
    budget = {"CORE": core_pct * e, "GROWTH": (1.0 - core_pct) * e}

    ext_df = pd.DataFrame({tk: d["ext"] for tk, d in data.items()}, index=cal)
    rs_df, close_df = rs_frames(ohlcv, pairs, cal)
    mom_pct, extadj_pct, _ = composite_components(ext_df, rs_df)
    score = mom_pct + extadj_pct

    group_of = {p[0]: p[2] for p in pairs}
    sleeve_of = {tk: ("CORE" if group_of.get(tk) in CORE_GROUPS else "GROWTH")
                 for tk in ext_df.columns}
    cols = {"CORE": [t for t in ext_df.columns if sleeve_of[t] == "CORE"],
            "GROWTH": [t for t in ext_df.columns if sleeve_of[t] == "GROWTH"]}
    params = {"CORE": {"entry": CORE_ENTRY, "band": CORE_BAND},
              "GROWTH": {"entry": top_n, "band": EXIT_RANK}}

    held_meta = {}
    for h in (current_holdings or []):
        tk = str(h.get("ticker", "")).upper().strip()
        if tk in data:
            held_meta[tk] = h

    holds, decisions, trail_exits, skips_all = [], [], [], []
    today = str(_dt.date.today())
    decisions.append({"date": today, "ticker": "—", "decision": "REGIME",
                      "gate": "layer 1", "exposure": round(e, 2),
                      "reason": f"gate → {e*100:.0f}% gross · split "
                                f"CORE {core_pct*100:.0f}% / GROWTH "
                                f"{(1-core_pct)*100:.0f}%"})

    def _comp(df, tk):
        try:
            v = float(df[tk].iloc[-1])
            return round(v, 2) if v == v else None
        except Exception:
            return None

    for sl in ("CORE", "GROWTH"):
        sc = cols[sl]
        if not sc or budget[sl] <= 0:
            continue
        rank_s = score[sc].rank(axis=1, ascending=False)
        last_rank = rank_s.iloc[-1]
        band = params[sl]["band"]; entry_n = params[sl]["entry"]
        decay_now = decay_matrix(rank_s, band, CONFIRM_WEEKS).iloc[-1]

        def _wb(tk):
            n = 0
            for v in reversed(rank_s[tk].tolist()):
                if (v != v) or v > band:
                    n += 1
                else:
                    break
            return n

        if sl == "GROWTH" and selector == "composite_v1":
            entries, skips = redundancy_filter_today(
                last_rank, close_df[sc], entry_n, return_skips=True)
            skips_all.extend(skips)
        else:
            row = last_rank.dropna().sort_values()
            entries = list(row[row <= entry_n].index)

        book = {tk: ("HELD" if tk in held_meta else "NEW") for tk in entries}
        for tk in held_meta:
            if sleeve_of.get(tk) != sl or tk in book:
                continue
            if not bool(decay_now.get(tk, True)):
                book[tk] = "HOLD-BAND"
            else:
                decisions.append({"date": today, "ticker": tk,
                                  "decision": "RELEASE-DECAY",
                                  "gate": "layer 2", "sleeve": sl,
                                  "rank": int(last_rank.get(tk))
                                          if last_rank.get(tk) == last_rank.get(tk) else None,
                                  "weeks_breach": _wb(tk),
                                  "reason": f"rank > {band} for "
                                            f"{CONFIRM_WEEKS}+ wks ({sl})"})

        sleeve_rows = []
        for tk, b in book.items():
            d = data[tk]
            px = d["close"].iloc[-1]
            if px != px:
                continue
            v = d["vol"].iloc[-1]
            stop = None
            if with_stops:
                try:
                    wk = ohlcv[yf_sym(tk)]
                    atr = wilder_atr(wk["High"].to_numpy(),
                                     wk["Low"].to_numpy(),
                                     wk["Close"].to_numpy())
                    entry_date = (held_meta.get(tk, {}) or {}).get("entry_date")
                    if tk in held_meta and entry_date:
                        try:
                            peak = float(wk["Close"].loc[str(entry_date):].max())
                        except Exception:
                            peak = float(wk["Close"].iloc[-1])
                    else:
                        peak = float(wk["Close"].iloc[-1])
                    sp = trail_level(peak, float(atr[-1]), TRAIL_K)
                    stop = round(sp, 2) if sp and sp > 0 else None
                    if (tk in held_meta and stop is not None
                            and float(wk["Close"].iloc[-1]) < stop):
                        trail_exits.append(tk)
                        decisions.append({"date": today, "ticker": tk,
                                          "decision": "EXIT-TRAIL",
                                          "gate": "layer 3", "sleeve": sl,
                                          "reason": f"close below {TRAIL_K}×ATR "
                                                    f"chandelier ({sl})"})
                        continue
                except Exception:
                    stop = None
            sleeve_rows.append({"ticker": tk, "band": b, "sleeve": sl,
                                "price": round(float(px), 2), "stop": stop,
                                "vol": float(v) if v == v and v > 0 else None,
                                "sector": group_of.get(tk, "—"),
                                "rank": int(last_rank[tk])
                                        if last_rank[tk] == last_rank[tk] else None,
                                "ext": _comp(ext_df, tk)})
        tot_iv = sum((1.0 / r["vol"]) for r in sleeve_rows if r["vol"]) or 1.0
        for r in sleeve_rows:
            iv = (1.0 / r["vol"]) if r["vol"] else 0.0
            r["weight"] = round((iv / tot_iv) * budget[sl], 4)
            dec = {"NEW": "ENTER", "HELD": "HELD",
                   "HOLD-BAND": "HOLD-BAND"}[r["band"]]
            wb = _wb(r["ticker"]) if r["band"] == "HOLD-BAND" else 0
            why = (f"rank {r['rank']} ≤ {entry_n} in {sl}" if dec != "HOLD-BAND"
                   else f"{sl} band {entry_n+1}..{band}; decay {wb}/"
                        f"{CONFIRM_WEEKS} wk")
            decisions.append({"date": today, "ticker": r["ticker"],
                              "decision": dec, "gate": "selector"
                              if dec != "HOLD-BAND" else "band",
                              "sleeve": sl, "rank": r["rank"],
                              "mom_pct": _comp(mom_pct, r["ticker"]),
                              "extadj_pct": _comp(extadj_pct, r["ticker"]),
                              "band": r["band"], "weeks_breach": wb,
                              "price": r["price"], "stop": r["stop"],
                              "weight": r["weight"], "exposure": round(e, 2),
                              "reason": why})
        holds.extend(sleeve_rows)

    for s in skips_all:
        decisions.append({"date": today, "ticker": s["ticker"],
                          "decision": "SKIP-REDUNDANT", "gate": "filter",
                          "sleeve": "GROWTH",
                          "blocked_by": s["blocked_by"], "corr": s["corr"],
                          "reason": f"{s['corr']:.2f} corr (26w) with "
                                    f"accepted {s['blocked_by']} > 0.85"})

    holds.sort(key=lambda x: (x["sleeve"], x["rank"] is None, x["rank"]))
    invested = sum(h["weight"] for h in holds)
    return {"exposure": round(e, 2), "core_pct": core_pct,
            "budgets": {k: round(v, 3) for k, v in budget.items()},
            "cash_weight": round(max(0.0, 1 - invested), 3),
            "holdings": holds, "top_n": top_n, "n_held": len(holds),
            "n_hold_band": sum(1 for h in holds if h["band"] == "HOLD-BAND"),
            "trail_exits": trail_exits,
            "decisions": decisions,
            "selector": (SELECTOR_VERSION if selector == "composite_v1"
                         else "raw ExtPct (legacy)"),
            "exit_spec": f"CORE {CORE_ENTRY}/{CORE_BAND} · GROWTH "
                         f"{top_n}/{EXIT_RANK} · {CONFIRM_WEEKS}wk · "
                         f"trail {TRAIL_K}×ATR"}


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
