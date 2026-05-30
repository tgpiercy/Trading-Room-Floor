"""
utils/backtest.py
Signal edge engine — rigorous forward-return event study.

Method (signal isolation, no lookahead):
  • For each ticker, precompute weekly indicator/RS/A-D frames ONCE. Rolling
    calcs are causal (row T uses only data ≤ T), so this is point-in-time safe.
  • Walk forward week by week. At each week T (with ≥ min_hist history),
    compute each signal using data ≤ T, then record forward returns measured
    from T's close to T+n's close.
  • Aggregate cross-sectionally: bucket every (ticker, week) observation by
    signal value and compare mean forward return to the universe baseline
    ("random pick from your watchlist").

A signal has real edge if its favourable bucket beats the universe baseline
consistently across horizons, with a decent win rate and sample size.
"""
import numpy as np
import pandas as pd

from utils.rs_indicators import build_rs_df, classify_state
from utils.strategy import (calc_price_indicators, calc_ad, calc_gw2_score,
                            calc_impulse_state, calc_rs_momentum)
from utils.rotation import calc_rrg, rrg_quadrant


FWD_WEEKS = (1, 4, 12)
MIN_HIST  = 45   # need 40 for SMA40 + a little buffer


def build_signal_panel(ohlcv: dict, pairs: list, progress_cb=None) -> pd.DataFrame:
    """
    Build the long panel: one row per (ticker, week) with each signal's value
    and forward returns. ohlcv = {yf_symbol: weekly_ohlcv_df}.
    pairs = list of (display_ticker, display_bench, group) using yf-mapped lookups.
    """
    from utils.watchlist import yf_sym
    records = []
    npairs = len(pairs)

    for idx, (dt, db, grp) in enumerate(pairs):
        if progress_cb:
            progress_cb(idx / npairs, dt)
        yt, yb = yf_sym(dt), yf_sym(db)
        if yt not in ohlcv or yb not in ohlcv:
            continue
        wk_t, wk_b = ohlcv[yt], ohlcv[yb]
        if len(wk_t) < MIN_HIST + max(FWD_WEEKS) + 1:
            continue
        try:
            # Precompute causal frames ONCE
            price_df = calc_price_indicators(wk_t)
            ad_df    = calc_ad(wk_t)
            rs_df    = build_rs_df(wk_t["Close"], wk_b["Close"])
            ratio, mom = calc_rrg(wk_t["Close"], wk_b["Close"])
            closes = price_df["Close"].values
            dates  = price_df.index
            n = len(price_df)

            for i in range(MIN_HIST, n - 1):
                # Need at least the 1-week forward return
                ps = price_df.iloc[:i + 1]
                ads = ad_df.iloc[:i + 1]
                rss = rs_df.iloc[:i + 1]

                rs_state, rs_score, _, _, rs_ext = classify_state(rss)
                gw2 = calc_gw2_score(ps, ads, rss)
                impulse = calc_impulse_state(ps, ads, rss, gw2)
                rs_mom = calc_rs_momentum(rss)

                # RRG quadrant at week i (ratio/mom are causal series)
                quad = "Unknown"
                if ratio is not None and not ratio.empty:
                    d = dates[i]
                    if d in ratio.index and d in mom.index:
                        quad = rrg_quadrant(float(ratio.loc[d]), float(mom.loc[d]))

                fwd = {}
                for w in FWD_WEEKS:
                    if i + w < n:
                        fwd[f"fwd_{w}w"] = (closes[i + w] / closes[i] - 1) * 100

                records.append({
                    "date": dates[i], "ticker": dt, "group": grp,
                    "rs_state": rs_state,
                    "gw2_bucket": ("5-7" if gw2["score"] >= 5 else
                                   "3-4" if gw2["score"] >= 3 else "0-2"),
                    "gw2_score": gw2["score"],
                    "impulse": impulse,
                    "rs_mom": rs_mom,
                    "quadrant": quad,
                    **fwd,
                })
        except Exception:
            continue

    if progress_cb:
        progress_cb(1.0, "done")
    return pd.DataFrame(records)


def universe_baseline(panel: pd.DataFrame) -> dict:
    """Mean forward return across ALL observations — the random-pick benchmark."""
    out = {}
    for w in FWD_WEEKS:
        col = f"fwd_{w}w"
        if col in panel.columns:
            s = panel[col].dropna()
            out[w] = {"mean": s.mean(), "win": (s > 0).mean() * 100, "n": len(s)}
    return out


def edge_table(panel: pd.DataFrame, signal_col: str, baseline: dict,
               primary_w: int = 4) -> pd.DataFrame:
    """
    For one signal, bucket observations by value and measure forward-return edge
    vs the universe baseline. Sorted by primary-horizon mean return.
    """
    col = f"fwd_{primary_w}w"
    if signal_col not in panel.columns or col not in panel.columns:
        return pd.DataFrame()
    rows = []
    base = baseline.get(primary_w, {}).get("mean", 0.0)
    for val, grp in panel.groupby(signal_col):
        s = grp[col].dropna()
        if len(s) < 20:      # ignore tiny samples
            continue
        row = {"Bucket": str(val), "N": len(s),
               f"Mean {primary_w}w %": round(s.mean(), 2),
               "Win %": round((s > 0).mean() * 100, 1),
               f"Edge vs Univ": round(s.mean() - base, 2)}
        # Other horizons
        for w in FWD_WEEKS:
            if w == primary_w:
                continue
            c = f"fwd_{w}w"
            if c in grp.columns:
                row[f"{w}w %"] = round(grp[c].dropna().mean(), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(f"Mean {primary_w}w %", ascending=False).reset_index(drop=True)


def signal_edge_ranking(panel: pd.DataFrame, baseline: dict,
                        primary_w: int = 4) -> pd.DataFrame:
    """
    Rank signals by the spread between their best and worst buckets at the
    primary horizon — a quick read on which signals discriminate returns.
    """
    signals = {
        "RS State":  "rs_state",
        "GW2 Score": "gw2_bucket",
        "Impulse":   "impulse",
        "RS Momentum": "rs_mom",
        "RRG Quadrant": "quadrant",
    }
    col = f"fwd_{primary_w}w"
    rows = []
    for label, scol in signals.items():
        if scol not in panel.columns:
            continue
        means = []
        for val, grp in panel.groupby(scol):
            s = grp[col].dropna()
            if len(s) >= 20:
                means.append(s.mean())
        if len(means) >= 2:
            rows.append({
                "Signal": label,
                "Best Bucket %": round(max(means), 2),
                "Worst Bucket %": round(min(means), 2),
                "Spread (discrimination)": round(max(means) - min(means), 2),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Spread (discrimination)", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS  (resolve whether an edge is real or a regime/concentration mirage)
# ══════════════════════════════════════════════════════════════════════════════
def multi_horizon_table(panel: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    Every bucket of a signal × every horizon (1w/4w/12w) mean return + win%.
    Reveals whether 'strong' buckets win short-term (momentum working) even if
    they lose at 12w (mean reversion). Sorted by the SHORTEST horizon.
    """
    if signal_col not in panel.columns:
        return pd.DataFrame()
    rows = []
    for val, grp in panel.groupby(signal_col):
        if len(grp) < 20:
            continue
        row = {"Bucket": str(val), "N": len(grp)}
        for w in FWD_WEEKS:
            c = f"fwd_{w}w"
            if c in grp.columns:
                s = grp[c].dropna()
                row[f"{w}w %"]   = round(s.mean(), 2) if len(s) else None
                row[f"{w}w win"] = round((s > 0).mean() * 100, 0) if len(s) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = f"{FWD_WEEKS[0]}w %"
    if not df.empty and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def concentration_check(panel: pd.DataFrame, signal_col: str,
                        bucket_value: str, primary_w: int = 4) -> dict:
    """
    Is a bucket's return broad or driven by a few names?
      • breadth   = % of distinct names with positive mean return
      • mean_ex_top1 = bucket mean with the single biggest contributor removed
        (if the edge collapses, it was one name's story — an artifact)
      • contrib_pp = each name's additive contribution to the bucket mean,
        in percentage points (sums to the bucket mean; robust to negatives)
    """
    col = f"fwd_{primary_w}w"
    grp = panel[panel[signal_col].astype(str) == str(bucket_value)].dropna(subset=[col])
    if grp.empty:
        return {}
    n_total = len(grp)
    stats = grp.groupby("ticker")[col].agg(["mean", "count", "sum"])
    stats["contrib_pp"] = (stats["sum"] / n_total).round(3)   # adds up to bucket mean
    stats = stats.sort_values("contrib_pp", ascending=False)

    breadth   = (stats["mean"] > 0).mean() * 100
    top_name  = stats.index[0]
    ex_top    = grp[grp["ticker"] != top_name][col]
    mean_ex   = round(ex_top.mean(), 2) if len(ex_top) else None

    top_tbl = stats.head(8).reset_index()[["ticker", "mean", "count", "contrib_pp"]]
    top_tbl["mean"] = top_tbl["mean"].round(2)

    return {
        "n_obs": n_total,
        "n_names": grp["ticker"].nunique(),
        "n_months": grp["date"].dt.to_period("M").nunique(),
        "breadth": round(breadth, 0),
        "mean": round(grp[col].mean(), 2),
        "mean_ex_top1": mean_ex,
        "top_name": top_name,
        "top_names": top_tbl,
    }


def subperiod_split(panel: pd.DataFrame, signal_col: str,
                    bucket_value: str, primary_w: int = 4) -> pd.DataFrame:
    """Mean forward return per calendar year for a bucket — does edge persist?"""
    col = f"fwd_{primary_w}w"
    grp = panel[panel[signal_col].astype(str) == str(bucket_value)].dropna(subset=[col]).copy()
    if grp.empty:
        return pd.DataFrame()
    grp["Year"] = grp["date"].dt.year
    out = grp.groupby("Year")[col].agg(
        **{f"Mean {primary_w}w %": "mean", "N": "count",
           "Win %": lambda s: (s > 0).mean() * 100}).reset_index()
    out[f"Mean {primary_w}w %"] = out[f"Mean {primary_w}w %"].round(2)
    out["Win %"] = out["Win %"].round(0)
    return out
