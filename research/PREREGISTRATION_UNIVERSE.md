# Pre-registration: which coins should C4 trade? (written 2026-11-06, before any result)

Question from the owner: are the current 8 coins the best universe for C4,
or would other coins do better? If something is better, adopt it.

## The trap this design avoids

Picking coins because their backtest looks best uses hindsight: we already
know which coins went up. So every rule below picks coins using only data
available at the time, and the pool includes coins that later died or were
delisted (LUNA, FTT and the like), otherwise the test only sees survivors.

## Candidates (strategy rules are C4 unchanged; only the coin list differs)

| ID | Universe | Selection rule (ex-ante) |
|---|---|---|
| U8 | the current 8: BTC ETH BNB XRP ADA SOL AVAX DOT | fixed (baseline) |
| T10 | top 10 by trailing 90-day USDT quote volume | re-selected on the 1st of each month |
| T20 | top 20 by trailing 90-day USDT quote volume | re-selected on the 1st of each month |
| WF4 | best 4 of the U8 coins by trailing 365-day standalone C4 return | re-selected each 1 January |
| WF8 | best 8 of the T20 pool by trailing 365-day standalone C4 return | re-selected each 1 January |

Pool: every USDT spot pair Binance has ever listed (data.binance.vision
archive), excluding stablecoins, fiat, leveraged tokens (UP/DOWN/BULL/BEAR),
wrapped/staked duplicates (WBTC, WBETH, BNSOL, ...) and tokenised gold/stocks.
A coin must have 365 daily bars before C4 may hold it (as in the original).
Weights: C4's vol-target weight with N = number of coins in the universe.

## Data, costs, periods

- Binance daily bars, signal at close, trade at next open (research/engine.py).
- Cost 0.4322% per side (Tokocrypto fee + 0.05% slippage), as before.
- DEV 2018-08-17 to 2024-12-31; HOLDOUT 2025-01-01 to 2026-10-10.
- Contamination disclosure: the holdout has already been seen once (the C4
  verdict). It is used here as a second check, not as fresh evidence.

## Decision rule (locked)

Replace U8 only if a candidate, compared with U8:
1. has a higher Sharpe in BOTH DEV and HOLDOUT, and
2. has a max drawdown no more than 5 percentage points worse in either period.

If several pass, take the highest DEV Sharpe. Live trading then holds only
coins tradeable on Tokocrypto at run time. If none pass, keep U8.
Whatever is chosen still has to prove itself in the forward test.
