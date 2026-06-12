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


def build_model_portfolio(ohlcv, pairs, top_n: int = 10, vol_lookback: int = 13,
                          signal: str = "extpct", with_stops: bool = True,
                          current_holdings: list | None = None,
                          selector: str = "composite_v1") -> dict:
    """Today's target portfolio from the validated strategy + exit stack.

    v3 — validated SELECTOR (selection_lab_v1 F + Gate-3 sweep) on top of
    the validated hold-band (exit_stage2_v1 D/F):
      RANK  : composite_v1 (utils.selection) — 26w RS momentum ⊕ vol-adj
              ExtPct; selector="extpct" is the legacy escape hatch
      ENTER : composite rank ≤ top_n, AFTER the redundancy filter (skip
              candidates >0.85 corr/26w with an accepted stronger name)
      HOLD  : already-held names stay while rank ≤ EXIT_RANK, released only
              after CONFIRM_WEEKS consecutive weeks beyond it (Layer 2)
      STOP  : chandelier TRAIL_K × ATR(14) below peak close since entry
              (Layer 3 backstop — display/order level)
      Regime exposure (Layer 1) scales the whole book, as before.

    current_holdings: the Portfolio tracker's list of dicts (ticker,
    entry_date, …). None → pure top-N book (legacy behaviour).
    Holdings outside the watchlist universe are ignored here and will diff
    to SELL in build_orders, as before.
    """
    from utils.exits import (decay_matrix, wilder_atr, trail_level,
                             EXIT_RANK, CONFIRM_WEEKS, TRAIL_K)
    from utils.selection import (rs_frames, composite_components,
                                 redundancy_filter_today, SELECTOR_VERSION)

    pc = precompute_series(ohlcv, pairs, vol_lookback, signal=signal)
    if "error" in pc:
        return pc
    data, exposure, cal = pc["data"], pc["exposure"], pc["cal"]
    e = float(exposure.iloc[-1]) if len(exposure) else 0.0

    ext_df = pd.DataFrame({tk: d["ext"] for tk, d in data.items()}, index=cal)
    mom_pct = extadj_pct = None
    if selector == "composite_v1":
        rs_df, close_df = rs_frames(ohlcv, pairs, cal)
        mom_pct, extadj_pct, rank_df = composite_components(ext_df, rs_df)
    else:                                    # legacy raw-ExtPct escape hatch
        rank_df = ext_df.rank(axis=1, ascending=False)
    last_rank = rank_df.iloc[-1]
    decay_full = decay_matrix(rank_df, EXIT_RANK, CONFIRM_WEEKS)
    decay_now = decay_full.iloc[-1]

    def _weeks_breach(tk):
        col = rank_df[tk] if tk in rank_df.columns else None
        if col is None:
            return None
        n = 0
        for v in reversed(col.tolist()):
            if (v != v) or v > EXIT_RANK:
                n += 1
            else:
                break
        return n

    def _comp(tk, key_df):
        try:
            v = float(key_df[tk].iloc[-1])
            return round(v, 2) if v == v else None
        except Exception:
            return None

    held_meta = {}
    for h in (current_holdings or []):
        tk = str(h.get("ticker", "")).upper().strip()
        if tk in data:
            held_meta[tk] = h

    book = {}   # tk -> band label
    skips = []
    if selector == "composite_v1":
        entries, skips = redundancy_filter_today(last_rank, close_df, top_n,
                                                 return_skips=True)
    else:
        row = last_rank.dropna().sort_values()
        entries = list(row[row <= top_n].index)
    for tk in entries:
        book[tk] = "HELD" if tk in held_meta else "NEW"
    for tk in held_meta:
        if tk in book:
            continue
        if not bool(decay_now.get(tk, True)):
            book[tk] = "HOLD-BAND"        # rank 11..EXIT_RANK, decay unconfirmed

    group_of = {p[0]: p[2] for p in pairs}
    rows = []
    for tk, band in book.items():
        d = data[tk]
        px = d["close"].iloc[-1]
        if px != px:
            continue
        v = d["vol"].iloc[-1]
        rows.append((tk, band, float(px),
                     float(v) if v == v and v > 0 else None))

    trail_exits = []
    invs = [(1.0 / v) if v else 0.0 for _, _, _, v in rows]
    ssum = sum(invs) or 1.0

    holds = []
    for (tk, band, px, v), iv in zip(rows, invs):
        w = (iv / ssum) * e
        stop = None
        if with_stops:
            try:
                wk = ohlcv[yf_sym(tk)]
                atr = wilder_atr(wk["High"].to_numpy(), wk["Low"].to_numpy(),
                                 wk["Close"].to_numpy())
                entry_date = (held_meta.get(tk, {}) or {}).get("entry_date")
                if tk in held_meta and entry_date:
                    try:
                        peak = float(wk["Close"].loc[str(entry_date):].max())
                    except Exception:
                        peak = float(wk["Close"].iloc[-1])
                else:
                    # NEW entry (or no entry_date): chandelier anchors at NOW
                    peak = float(wk["Close"].iloc[-1])
                sp = trail_level(peak, float(atr[-1]), TRAIL_K)
                stop = round(sp, 2) if sp and sp > 0 else None
                # Layer 3 enforcement: a held name already below its trail is
                # an EXIT, not a holding — release it from the book.
                if (tk in held_meta and stop is not None
                        and float(wk["Close"].iloc[-1]) < stop):
                    trail_exits.append(tk)
                    continue
            except Exception:
                stop = None
        holds.append({"ticker": tk, "weight": round(w, 4),
                      "ext": round(float(ext_df[tk].iloc[-1]), 2)
                             if ext_df[tk].iloc[-1] == ext_df[tk].iloc[-1] else None,
                      "rank": int(last_rank[tk]) if last_rank[tk] == last_rank[tk] else None,
                      "band": band, "price": round(px, 2), "stop": stop,
                      "vol": v, "sector": group_of.get(tk, "—")})
    holds.sort(key=lambda x: (x["rank"] is None, x["rank"]))
    # Renormalize inverse-vol weights over the surviving book × exposure
    tot_iv = sum((1.0 / h["vol"]) for h in holds if h.get("vol")) or 1.0
    for h in holds:
        iv = (1.0 / h["vol"]) if h.get("vol") else 0.0
        h["weight"] = round((iv / tot_iv) * e, 4)

    invested = sum(h["weight"] for h in holds)

    # ── Decision matrix (reason per gate) for the trade journal ──────────────
    import datetime as _dt
    today = str(_dt.date.today())
    decisions = [{"date": today, "ticker": "—", "decision": "REGIME",
                  "gate": "layer 1", "exposure": round(e, 2),
                  "reason": f"validated SPY/IEF/VIX gate → {e*100:.0f}% "
                            f"gross exposure cap"}]
    by_tk = {h["ticker"]: h for h in holds}
    for h in holds:
        tk = h["ticker"]
        dec = {"NEW": "ENTER", "HELD": "HELD",
               "HOLD-BAND": "HOLD-BAND"}[h["band"]]
        gate = "band" if h["band"] == "HOLD-BAND" else "selector"
        wb = _weeks_breach(tk) if h["band"] == "HOLD-BAND" else 0
        if dec == "ENTER":
            why = (f"composite rank {h['rank']} ≤ {top_n}; passed "
                   f"redundancy filter")
        elif dec == "HELD":
            why = f"owned and still rank {h['rank']} ≤ {top_n}"
        else:
            why = (f"owned, rank {h['rank']} in band {top_n+1}..{EXIT_RANK}; "
                   f"decay {wb}/{CONFIRM_WEEKS} wk toward release")
        decisions.append({"date": today, "ticker": tk, "decision": dec,
                          "gate": gate, "rank": h["rank"],
                          "mom_pct": _comp(tk, mom_pct) if mom_pct is not None else None,
                          "extadj_pct": _comp(tk, extadj_pct) if extadj_pct is not None else None,
                          "band": h["band"], "weeks_breach": wb,
                          "price": h["price"], "stop": h["stop"],
                          "weight": h["weight"], "exposure": round(e, 2),
                          "reason": why})
    for s in skips:
        decisions.append({"date": today, "ticker": s["ticker"],
                          "decision": "SKIP-REDUNDANT", "gate": "filter",
                          "rank": int(last_rank.get(s["ticker"]))
                                  if last_rank.get(s["ticker"]) == last_rank.get(s["ticker"]) else None,
                          "blocked_by": s["blocked_by"], "corr": s["corr"],
                          "reason": f"{s['corr']:.2f} corr (26w) with "
                                    f"accepted {s['blocked_by']} > 0.85"})
    for tk in held_meta:
        if tk in by_tk or tk in trail_exits:
            continue
        if bool(decay_now.get(tk, True)):
            decisions.append({"date": today, "ticker": tk,
                              "decision": "RELEASE-DECAY", "gate": "layer 2",
                              "rank": int(last_rank.get(tk))
                                      if last_rank.get(tk) == last_rank.get(tk) else None,
                              "weeks_breach": _weeks_breach(tk),
                              "reason": f"rank > {EXIT_RANK} for "
                                        f"{CONFIRM_WEEKS}+ consecutive weeks"})
    for tk in trail_exits:
        decisions.append({"date": today, "ticker": tk,
                          "decision": "EXIT-TRAIL", "gate": "layer 3",
                          "reason": f"close below {TRAIL_K}×ATR chandelier "
                                    f"(disaster backstop)"})

    return {"exposure": round(e, 2),
            "decisions": decisions, "cash_weight": round(max(0.0, 1 - invested), 3),
            "holdings": holds, "top_n": top_n, "n_held": len(holds),
            "n_hold_band": sum(1 for h in holds if h["band"] == "HOLD-BAND"),
            "trail_exits": trail_exits,
            "selector": (SELECTOR_VERSION if selector == "composite_v1"
                         else "raw ExtPct (legacy)"),
            "exit_spec": f"enter≤{top_n} · hold≤{EXIT_RANK} · "
                         f"release {CONFIRM_WEEKS}wk · trail {TRAIL_K}×ATR"}


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

