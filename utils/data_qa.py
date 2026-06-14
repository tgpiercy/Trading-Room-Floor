"""
utils/data_qa.py
Pre-flight data-quality gate for the Rebalance order path.

A weekly fetch can silently come back thin (delisted names like LYTE, transient
"database is locked" dropouts) or stale (a halted/holiday feed), and the model
would then build a DIFFERENT book than intended without anyone noticing. This
gate inspects the fetched OHLCV against the expected universe and returns a
coverage/freshness report with a verdict:

  ok    — coverage high, data fresh → proceed
  warn  — some names missing or stale → proceed, but surface it loudly
  block — coverage below the floor → do not emit orders silently

Pure and dependency-light (pandas only) so it's unit-testable without network.
"""
from __future__ import annotations
import pandas as pd

COVERAGE_BLOCK = 0.70      # below this, refuse to emit orders silently
COVERAGE_WARN = 0.90       # below this (or any stale), warn
MAX_STALE_DAYS = 10        # last bar older than this vs the universe = stale


def qa_report(ohlcv: dict, expected, asof=None,
              max_stale_days: int = MAX_STALE_DAYS) -> dict:
    """Inspect fetched OHLCV against the expected universe.

    ohlcv    : {symbol: DataFrame with a DatetimeIndex and a 'Close' column}
    expected : iterable of symbols that SHOULD have been fetched
    asof     : reference 'today' (defaults to the universe's latest bar)
    """
    expected = list(dict.fromkeys(expected))
    n_exp = len(expected)

    fetched, last_bar = [], {}
    for s in expected:
        df = ohlcv.get(s)
        if df is None or len(df) == 0 or "Close" not in getattr(df, "columns", []):
            continue
        closes = df["Close"].dropna()
        if closes.empty:
            continue
        fetched.append(s)
        last_bar[s] = df.index[-1]

    missing = [s for s in expected if s not in fetched]
    universe_latest = max(last_bar.values()) if last_bar else None
    ref = pd.Timestamp(asof) if asof is not None else universe_latest

    stale = []
    if ref is not None:
        for s, d in last_bar.items():
            try:
                if (ref - d).days > max_stale_days:
                    stale.append(s)
            except Exception:
                continue

    coverage = (len(fetched) / n_exp) if n_exp else 0.0
    if coverage < COVERAGE_BLOCK:
        verdict = "block"
    elif coverage < COVERAGE_WARN or stale:
        verdict = "warn"
    else:
        verdict = "ok"

    return {
        "verdict": verdict,
        "coverage": round(coverage, 3),
        "n_expected": n_exp,
        "n_fetched": len(fetched),
        "missing": missing,
        "stale": stale,
        "universe_latest": (str(universe_latest.date())
                            if universe_latest is not None else None),
        "summary": _summary(verdict, coverage, len(fetched), n_exp,
                            missing, stale, universe_latest),
    }


def _summary(verdict, coverage, n_fetched, n_exp, missing, stale, latest):
    bits = [f"{n_fetched}/{n_exp} names ({coverage*100:.0f}%)"]
    if latest is not None:
        bits.append(f"latest bar {pd.Timestamp(latest).date()}")
    if missing:
        head = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
        bits.append(f"missing: {head}")
    if stale:
        head = ", ".join(stale[:6]) + (" …" if len(stale) > 6 else "")
        bits.append(f"stale: {head}")
    label = {"ok": "✅ Data OK", "warn": "🟠 Data warning",
             "block": "🛑 Data insufficient"}[verdict]
    return f"{label} — " + " · ".join(bits)
