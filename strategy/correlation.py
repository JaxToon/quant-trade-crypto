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

Stock correlations (STOCK_CORRELATION_MATRIX) computed the same way, 30d
hourly returns via yfinance. Much noisier than the crypto matrix — hourly
stock data only exists during ~6.5h/day trading sessions (209 observations
over 30 days vs crypto's ~720 for the same window), and large-cap tech
correlations came out weaker and less consistent than expected (e.g.
AAPL-AMZN at -0.523, where same-sector large caps are usually positively
correlated) — plausibly idiosyncratic, earnings-driven moves during this
specific window rather than a stable structural relationship. Treat the
stock matrix as lower-confidence than the crypto one; a longer window
would help but wasn't re-run before this snapshot was taken.
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

# (symbol_a, symbol_b) -> Pearson correlation of hourly returns, 30d window.
# See module docstring — noisier / lower-confidence than the crypto matrix.
STOCK_CORRELATION_MATRIX: dict[tuple[str, str], float] = {
    ("AAPL", "MSFT"): -0.048,
    ("AAPL", "GOOGL"): -0.111,
    ("AAPL", "AMZN"): -0.523,
    ("AAPL", "NVDA"): -0.138,
    ("AAPL", "META"): -0.020,
    ("AAPL", "TSLA"): 0.006,
    ("AAPL", "AVGO"): 0.004,
    ("MSFT", "GOOGL"): 0.194,
    ("MSFT", "AMZN"): 0.434,
    ("MSFT", "NVDA"): 0.199,
    ("MSFT", "META"): -0.194,
    ("MSFT", "TSLA"): 0.264,
    ("MSFT", "AVGO"): 0.178,
    ("GOOGL", "AMZN"): 0.545,
    ("GOOGL", "NVDA"): 0.166,
    ("GOOGL", "META"): 0.413,
    ("GOOGL", "TSLA"): 0.382,
    ("GOOGL", "AVGO"): 0.008,
    ("AMZN", "NVDA"): 0.222,
    ("AMZN", "META"): 0.175,
    ("AMZN", "TSLA"): 0.239,
    ("AMZN", "AVGO"): -0.028,
    ("NVDA", "META"): 0.049,
    ("NVDA", "TSLA"): 0.365,
    ("NVDA", "AVGO"): 0.579,
    ("META", "TSLA"): 0.252,
    ("META", "AVGO"): -0.181,
    ("TSLA", "AVGO"): 0.322,
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
