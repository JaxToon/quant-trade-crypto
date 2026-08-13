"""
Static pairwise correlation reference for the crypto watchlist — used by
PaperTradingLedger's portfolio-level risk cap to recognize that 7
simultaneously open positions on correlated majors (avg correlation 0.62,
peak 0.86 for BTC-ETH) is not 7 independent bets, it's a handful of real
ones wearing different symbols.

Computed once from 30 days of hourly returns (see backtest.py's data
source — same yfinance tickers). Crypto large-cap correlation structure is
fairly stable (most alts are BTC/ETH-beta-dominated), so a static snapshot
refreshed occasionally is a reasonable trade-off against the complexity of
computing this live from a rolling price buffer. Re-run the generation
snippet in DEPLOYMENT.md-adjacent notes if this goes stale (or ask to
regenerate — it's a ~10-line script).

Stock correlations are NOT covered — ASSET_TYPE=stock falls back to
treating all pairs as uncorrelated (0.0) here, which is a known gap, not a
claim that stocks are actually uncorrelated.
"""
from __future__ import annotations

# (symbol_a, symbol_b) -> Pearson correlation of hourly returns, 30d window.
# Symmetric — look up either order via get_correlation().
CRYPTO_CORRELATION_MATRIX: dict[tuple[str, str], float] = {
    ("BTCUSDT", "ETHUSDT"): 0.856,
    ("BTCUSDT", "BNBUSDT"): 0.687,
    ("BTCUSDT", "SOLUSDT"): 0.813,
    ("BTCUSDT", "ADAUSDT"): 0.477,
    ("BTCUSDT", "AVAXUSDT"): 0.584,
    ("BTCUSDT", "DOTUSDT"): 0.556,
    ("BTCUSDT", "XRPUSDT"): 0.738,
    ("ETHUSDT", "BNBUSDT"): 0.680,
    ("ETHUSDT", "SOLUSDT"): 0.842,
    ("ETHUSDT", "ADAUSDT"): 0.505,
    ("ETHUSDT", "AVAXUSDT"): 0.635,
    ("ETHUSDT", "DOTUSDT"): 0.582,
    ("ETHUSDT", "XRPUSDT"): 0.756,
    ("BNBUSDT", "SOLUSDT"): 0.669,
    ("BNBUSDT", "ADAUSDT"): 0.422,
    ("BNBUSDT", "AVAXUSDT"): 0.508,
    ("BNBUSDT", "DOTUSDT"): 0.481,
    ("BNBUSDT", "XRPUSDT"): 0.611,
    ("SOLUSDT", "ADAUSDT"): 0.547,
    ("SOLUSDT", "AVAXUSDT"): 0.667,
    ("SOLUSDT", "DOTUSDT"): 0.597,
    ("SOLUSDT", "XRPUSDT"): 0.782,
    ("ADAUSDT", "AVAXUSDT"): 0.459,
    ("ADAUSDT", "DOTUSDT"): 0.509,
    ("ADAUSDT", "XRPUSDT"): 0.569,
    ("AVAXUSDT", "DOTUSDT"): 0.560,
    ("AVAXUSDT", "XRPUSDT"): 0.604,
    ("DOTUSDT", "XRPUSDT"): 0.583,
}


def get_correlation(symbol_a: str, symbol_b: str) -> float:
    """Returns 1.0 for the same symbol, else the looked-up pairwise
    correlation, else 0.0 (unknown pair — e.g. stock symbols not covered)."""
    if symbol_a == symbol_b:
        return 1.0
    return (
        CRYPTO_CORRELATION_MATRIX.get((symbol_a, symbol_b))
        or CRYPTO_CORRELATION_MATRIX.get((symbol_b, symbol_a))
        or 0.0
    )
