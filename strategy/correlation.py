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

Stock correlations (STOCK_CORRELATION_MATRIX) computed from 730 days of
hourly returns (~5072 observations/symbol) — extended from an original 30d
snapshot (~209 observations/symbol) that was noisy enough to show
implausible results (e.g. AAPL-AMZN at -0.523, negative correlation between
two large-cap tech names that should co-move with the market). The 730d
recompute lands everywhere in a sane 0.26-0.58 positive range, consistent
with expected large-cap tech beta correlation — confirms the original
snapshot was sample-size noise, not a real relationship. Still a static
snapshot like the crypto matrix; re-run if it goes stale.
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

# (symbol_a, symbol_b) -> Pearson correlation of hourly returns, 730d window.
STOCK_CORRELATION_MATRIX: dict[tuple[str, str], float] = {
    ("AAPL", "MSFT"): 0.298,
    ("AAPL", "GOOGL"): 0.341,
    ("AAPL", "AMZN"): 0.315,
    ("AAPL", "NVDA"): 0.297,
    ("AAPL", "META"): 0.277,
    ("AAPL", "TSLA"): 0.339,
    ("AAPL", "AVGO"): 0.265,
    ("MSFT", "GOOGL"): 0.350,
    ("MSFT", "AMZN"): 0.507,
    ("MSFT", "NVDA"): 0.424,
    ("MSFT", "META"): 0.410,
    ("MSFT", "TSLA"): 0.283,
    ("MSFT", "AVGO"): 0.327,
    ("GOOGL", "AMZN"): 0.496,
    ("GOOGL", "NVDA"): 0.345,
    ("GOOGL", "META"): 0.373,
    ("GOOGL", "TSLA"): 0.331,
    ("GOOGL", "AVGO"): 0.312,
    ("AMZN", "NVDA"): 0.447,
    ("AMZN", "META"): 0.542,
    ("AMZN", "TSLA"): 0.366,
    ("AMZN", "AVGO"): 0.361,
    ("NVDA", "META"): 0.414,
    ("NVDA", "TSLA"): 0.361,
    ("NVDA", "AVGO"): 0.582,
    ("META", "TSLA"): 0.302,
    ("META", "AVGO"): 0.329,
    ("TSLA", "AVGO"): 0.352,
}


def get_correlation(symbol_a: str, symbol_b: str) -> float:
    """Returns 1.0 for the same symbol, else the looked-up pairwise
    correlation from whichever matrix contains the pair (crypto and stock
    symbol namespaces never overlap — "BTCUSDT" vs "AAPL"), else 0.0 for a
    genuinely unknown pair."""
    if symbol_a == symbol_b:
        return 1.0
    for matrix in (CRYPTO_CORRELATION_MATRIX, STOCK_CORRELATION_MATRIX):
        val = matrix.get((symbol_a, symbol_b))
        if val is None:
            val = matrix.get((symbol_b, symbol_a))
        if val is not None:
            return val
    return 0.0
