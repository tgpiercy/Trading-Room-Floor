"""
utils/watchlist.py
Ordered portfolio watchlist — ratio pairs with group structure.
Order is preserved exactly as specified. Each entry defines its own benchmark.

SYMBOL_MAP remaps display tickers to yfinance-compatible symbols.
Canadian TSX-listed securities use the .TO suffix to ensure
currency consistency when benchmarked against Canadian ETFs (HXT, XBB).
"""

# ── yfinance symbol remapping ─────────────────────────────────────────────────
SYMBOL_MAP = {
    # Canadian ETF benchmarks
    "HXT":    "HXT.TO",
    "XBB":    "XBB.TO",
    # Canadian equities (TSX listings — CAD, consistent with HXT.TO benchmark)
    "RY":     "RY.TO",
    "BN":     "BN.TO",
    "BAM":    "BAM.TO",
    "IFC":    "IFC.TO",
    "CNQ":    "CNQ.TO",
    "ENB":    "ENB.TO",
    "FTS":    "FTS.TO",
    "CP":     "CP.TO",
    "WSP":    "WSP.TO",
    "CAE":    "CAE.TO",
    "WCN":    "WCN.TO",
    "SHOP":   "SHOP.TO",
    "ATD":    "ATD.TO",
    "CSU":    "CSU.TO",
    "CCO":    "CCO.TO",
    "AEM":    "AEM.TO",
    "NTR":    "NTR.TO",
    "TECK.B": "TECK-B.TO",
}


def yf_sym(ticker: str) -> str:
    """Resolve display ticker to yfinance symbol."""
    return SYMBOL_MAP.get(ticker, ticker)


# ── Portfolio definition — order preserved exactly ────────────────────────────
# Format: (display_ticker, display_bench, group_label)
PORTFOLIO: list[tuple[str, str, str]] = [

    # ── INDICES ───────────────────────────────────────────────────────────────
    ("SPY",  "IEF",  "INDICES"),
    ("EEM",  "SPY",  "INDICES"),
    ("ACWX", "SPY",  "INDICES"),
    ("FNDF", "SPY",  "INDICES"),
    ("DIA",  "SPY",  "INDICES"),
    ("QQQ",  "SPY",  "INDICES"),
    ("IWM",  "SPY",  "INDICES"),
    ("IWC",  "SPY",  "INDICES"),
    ("HXT",  "XBB",  "INDICES"),

    # ── SECTORS ───────────────────────────────────────────────────────────────
    ("XLB",  "SPY",  "SECTORS"),
    ("XLC",  "SPY",  "SECTORS"),
    ("XLE",  "SPY",  "SECTORS"),
    ("XLF",  "SPY",  "SECTORS"),
    ("XLI",  "SPY",  "SECTORS"),
    ("XLK",  "SPY",  "SECTORS"),
    ("XLP",  "SPY",  "SECTORS"),
    ("XLRE", "SPY",  "SECTORS"),
    ("XLU",  "SPY",  "SECTORS"),
    ("XLV",  "SPY",  "SECTORS"),
    ("XLY",  "SPY",  "SECTORS"),

    # ── THEMES ────────────────────────────────────────────────────────────────
    ("BOTZ", "SPY",  "THEMES"),
    ("XBI",  "SPY",  "THEMES"),
    ("REMX", "SPY",  "THEMES"),
    ("ICLN", "SPY",  "THEMES"),
    ("GDX",  "SPY",  "THEMES"),

    # ── PRECIOUS METALS ───────────────────────────────────────────────────────
    ("GLD",  "SPY",  "PRECIOUS METALS"),
    ("SLV",  "SPY",  "PRECIOUS METALS"),

    # ── INDUSTRIES · Robotics ─────────────────────────────────────────────────
    ("AIQ",  "BOTZ", "INDUSTRIES · Robotics"),
    ("PLTR", "BOTZ", "INDUSTRIES · Robotics"),

    # ── INDUSTRIES · Biotech ──────────────────────────────────────────────────
    ("IBB",  "XBI",  "INDUSTRIES · Biotech"),
    ("ARKG", "XBI",  "INDUSTRIES · Biotech"),

    # ── INDUSTRIES · Clean Energy ─────────────────────────────────────────────
    ("TAN",  "ICLN", "INDUSTRIES · Clean Energy"),
    ("NLR",  "ICLN", "INDUSTRIES · Clean Energy"),
    ("URA",  "ICLN", "INDUSTRIES · Clean Energy"),
    ("CEG",  "ICLN", "INDUSTRIES · Clean Energy"),

    # ── INDUSTRIES · GDX ──────────────────────────────────────────────────────
    ("GDXJ", "GDX",  "INDUSTRIES · GDX"),
    ("SIL",  "GDX",  "INDUSTRIES · GDX"),

    # ── INDUSTRIES · XLB ──────────────────────────────────────────────────────
    ("COPX", "XLB",  "INDUSTRIES · XLB"),
    ("AA",   "XLB",  "INDUSTRIES · XLB"),
    ("SLX",  "XLB",  "INDUSTRIES · XLB"),
    ("DD",   "XLB",  "INDUSTRIES · XLB"),
    ("MOO",  "XLB",  "INDUSTRIES · XLB"),

    # ── INDUSTRIES · XLC ──────────────────────────────────────────────────────
    ("META",  "XLC", "INDUSTRIES · XLC"),
    ("GOOGL", "XLC", "INDUSTRIES · XLC"),

    # ── INDUSTRIES · XLE ──────────────────────────────────────────────────────
    ("XOP",  "XLE",  "INDUSTRIES · XLE"),
    ("IEO",  "XLE",  "INDUSTRIES · XLE"),
    ("OIH",  "XLE",  "INDUSTRIES · XLE"),
    ("MLPX", "XLE",  "INDUSTRIES · XLE"),

    # ── INDUSTRIES · XLF ──────────────────────────────────────────────────────
    ("KBE",  "XLF",  "INDUSTRIES · XLF"),
    ("KRE",  "XLF",  "INDUSTRIES · XLF"),
    ("V",    "XLF",  "INDUSTRIES · XLF"),
    ("BLK",  "XLF",  "INDUSTRIES · XLF"),
    ("IAK",  "XLF",  "INDUSTRIES · XLF"),

    # ── INDUSTRIES · XLI ──────────────────────────────────────────────────────
    ("ITA",  "XLI",  "INDUSTRIES · XLI"),
    ("IYT",  "XLI",  "INDUSTRIES · XLI"),
    ("GRID", "XLI",  "INDUSTRIES · XLI"),
    ("PAVE", "XLI",  "INDUSTRIES · XLI"),

    # ── INDUSTRIES · XLK ──────────────────────────────────────────────────────
    ("SMCI", "XLK",  "INDUSTRIES · XLK"),
    ("DELL", "XLK",  "INDUSTRIES · XLK"),
    ("WDC",  "XLK",  "INDUSTRIES · XLK"),
    ("P",    "XLK",  "INDUSTRIES · XLK"),
    ("ANET", "XLK",  "INDUSTRIES · XLK"),
    ("IGV",  "XLK",  "INDUSTRIES · XLK"),
    ("MSFT", "XLK",  "INDUSTRIES · XLK"),
    ("SMH",  "XLK",  "INDUSTRIES · XLK"),
    ("NVDA", "XLK",  "INDUSTRIES · XLK"),
    ("MU",   "XLK",  "INDUSTRIES · XLK"),
    ("ARM",  "XLK",  "INDUSTRIES · XLK"),
    ("CDNS", "XLK",  "INDUSTRIES · XLK"),
    ("ASML", "XLK",  "INDUSTRIES · XLK"),
    ("VRT",  "XLK",  "INDUSTRIES · XLK"),
    ("CRWD", "XLK",  "INDUSTRIES · XLK"),
    ("CIBR", "XLK",  "INDUSTRIES · XLK"),

    # ── INDUSTRIES · XLV ──────────────────────────────────────────────────────
    ("PPH",  "XLV",  "INDUSTRIES · XLV"),
    ("IHI",  "XLV",  "INDUSTRIES · XLV"),

    # ── INDUSTRIES · XLY ──────────────────────────────────────────────────────
    ("ITB",  "XLY",  "INDUSTRIES · XLY"),
    ("TSLA", "XLY",  "INDUSTRIES · XLY"),
    ("AMZN", "XLY",  "INDUSTRIES · XLY"),

    # ── INDUSTRIES · HXT (Canadian) ───────────────────────────────────────────
    ("RY",     "HXT", "INDUSTRIES · HXT"),
    ("BN",     "HXT", "INDUSTRIES · HXT"),
    ("BAM",    "HXT", "INDUSTRIES · HXT"),
    ("IFC",    "HXT", "INDUSTRIES · HXT"),
    ("CNQ",    "HXT", "INDUSTRIES · HXT"),
    ("ENB",    "HXT", "INDUSTRIES · HXT"),
    ("FTS",    "HXT", "INDUSTRIES · HXT"),
    ("CP",     "HXT", "INDUSTRIES · HXT"),
    ("WSP",    "HXT", "INDUSTRIES · HXT"),
    ("CAE",    "HXT", "INDUSTRIES · HXT"),
    ("WCN",    "HXT", "INDUSTRIES · HXT"),
    ("SHOP",   "HXT", "INDUSTRIES · HXT"),
    ("ATD",    "HXT", "INDUSTRIES · HXT"),
    ("CSU",    "HXT", "INDUSTRIES · HXT"),
    ("CCO",    "HXT", "INDUSTRIES · HXT"),
    ("AEM",    "HXT", "INDUSTRIES · HXT"),
    ("NTR",    "HXT", "INDUSTRIES · HXT"),
    ("TECK.B", "HXT", "INDUSTRIES · HXT"),
]

# Unique yfinance symbols needed for batch download
ALL_YF_SYMBOLS: list[str] = list(dict.fromkeys(
    yf_sym(t) for pair in PORTFOLIO for t in (pair[0], pair[1])
))

# Group order for display
GROUP_ORDER: list[str] = list(dict.fromkeys(p[2] for p in PORTFOLIO))
