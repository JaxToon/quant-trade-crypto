# Pre-registration: strategy comparison (written 2026-10-11, before any candidate was run)

Everything below is fixed before a single candidate is evaluated on any data.
The commit that adds this file is the timestamp. Any later change to a
parameter, a period or a rule must be a new, separately dated section, and
results produced under the old rules must still be reported.

## Question

Which automated, rule-based strategy (no ML) gives the best risk-adjusted
return on this project's 8-coin universe AFTER realistic execution and real
Tokocrypto costs, and does the winner hold up on data no decision has touched?

## Universe and data

- Binance spot USDT pairs: BTC, ETH, BNB, XRP, ADA, SOL, AVAX, DOT (the
  system's watchlist). Daily klines in UTC from `data-api.binance.vision`,
  cached under `data/klines/`. SMC candidates also use 1h and 4h klines from the
  same source.
- A symbol becomes tradable once it has at least 365 daily bars.
- **Known bias, stated up front:** survivorship. All 8 are coins that are still
  large in 2026, which inflates every long-only result. The benchmarks carry the
  same bias, so relative comparisons are fairer than absolute ones.

## Periods

- Warm-up: data before 2018-01-01 is used only to compute indicators.
- **DEV:** 2018-01-01 to 2024-12-31. All ranking and selection happens here.
- **HOLDOUT:** 2025-01-01 to the last complete daily bar (2026-10-10). Run once,
  after selection, for every candidate. Only the selected candidate is gated.
- **Contamination disclosure:** (1) my training knowledge reaches into 2026, so
  I may know the broad market direction of 2025; (2) a web summary seen during
  this research said one BTC Donchian replication had a negative 2025; (3) every
  parameter below is taken from the cited literature or set here without
  tuning, which limits how much (1) and (2) can leak in.

## Execution and costs

- A signal computed on the daily close of day t is traded at the open of day
  t+1. No decision uses a price it could not have seen.
- Cost per unit of notional traded, each side: Tokocrypto fee 0.3822%
  (published USDT schedule, incl. PPh 0.21% and CFX 0.0222%) plus 0.05%
  slippage = **0.4322%**.
- Sensitivity, reported but never used for decisions: 0.10% fee + 0.05%
  slippage.
- Long-only, spot, no leverage: gross exposure at most 100% of equity. Idle cash
  earns 0%.
- Rebalancing: a position is traded when its target turns on or off, or on a
  Monday when the held weight differs from the target by more than 20% of the
  target.

## Benchmarks (not candidates)

- **BH-BTC:** buy and hold BTC.
- **BH-EW:** equal weight across eligible coins, rebalanced on the first day of
  each month.

## Candidates (parameters fixed here)

`N` is the number of eligible coins that day. `sigma90` is the annualised
standard deviation of the last 90 daily close-to-close returns (x sqrt(365)).

| ID | Name | Rule | Weight per coin | Source |
|---|---|---|---|---|
| C1 | SMA200 | hold while close > 200-day SMA | s / N | Le & Ruthbah (2024) test 20/65/150/200 |
| C2 | TSMOM4W | on Mondays, hold for the week if the 28-day return > 0 | s / N | Liu & Tsyvinski (2021): 1-4 week time-series momentum |
| C3 | DONCH-BTC | Donchian ensemble, BTC only (rule below) | s x min(1, 0.25 / sigma90) | Zarattini, Pagani & Barbon (2025) |
| C4 | DONCH-VT | Donchian ensemble, all coins | s x min(1, 0.25 / sigma90) / N | same |
| C5 | DONCH-EW | Donchian ensemble, all coins | s / N | same, without vol sizing |
| C6 | XSMOM3 | on Mondays, hold the top 3 coins by 14-day return, only those with return > 0 | 1/3 each | Liu, Tsyvinski & Wu (2022): 2-week cross-sectional momentum |
| C7 | RSI2 | enter when RSI(2) < 10 and close > 200-day SMA; exit when close > 5-day SMA | s / N | Connors-style mean reversion; included as a contrast |
| C8 | SMC-1H | current SMC/FVG engine, spot (long) only, entry at the signal bar's close | ledger sizing: risk 0.25%, notional cap 1.0x | this project |
| C9 | SMC-4H | same, 4h candles | same | this project |
| C10 | SMC-1D | same, daily candles | same | this project |

**Donchian ensemble rule (C3-C5).** Lookbacks L in {5, 10, 20, 30, 60, 90, 150,
250, 360}. For each L, using the previous L closes (today excluded):
`hi = max`, `lo = min`. A sub-model turns on when today's close > hi. While on,
its stop is `max(previous stop, (hi + lo) / 2)`, updated daily including the
entry day. From the day after entry it turns off when the close < stop. The coin's
signal `s` is the mean of the 9 sub-model states, so it is in [0, 1].

**SMC candidates (C8-C10)** reuse `SmartMoneyAnalyzer`, `FairValueGapAnalyzer`,
`build_entry_insight` and `PaperTradingLedger` unchanged, with the ledger's own
cost model at the Tokocrypto fee (0.3822%/side) and the measured 0.119% stop
slippage. Entry is booked at the signal bar's close (the MARKET arm of
`experiments/entry_fill_realism.py`), not at the FVG midpoint. Equity is
built from realised trades.

## Metrics

Net daily returns give CAGR, annualised volatility, Sharpe (rf = 0, x sqrt(365)),
Sortino, maximum drawdown, Calmar, % positive months, average gross exposure,
annual turnover and fees paid. Per coin and holding period ("trade"): count,
win rate, average win, average loss, profit factor, expectancy (% of equity per
trade), average holding days. Results are reported per calendar year as well.

## Statistical checks

- Stationary block bootstrap of daily net returns (mean block 20 days, 5,000
  resamples): 95% CI of Sharpe.
- Random-timing null for C1-C7: shift each coin's signal series circularly by an
  independent random offset in [60, len - 60] days, recompute net Sharpe, 1,000
  draws. Report the percentile of the real Sharpe.
- Deflated Sharpe of the selected candidate with N = 10 trials and the
  cross-candidate variance of DEV Sharpe.

## Selection rule (DEV only)

A candidate is eligible if, on DEV, all of these hold:

1. Lower bound of the Sharpe 95% CI > 0.
2. Sharpe above the 95th percentile of its random-timing null (not applicable
   to C8-C10).
3. Maximum drawdown < 40%.
4. Calmar above BH-EW's Calmar.

The eligible candidate with the highest DEV Sharpe is selected. Candidates within
0.05 Sharpe of each other are tie-broken toward fewer parameters. If none is
eligible, no candidate is selected; holdout numbers are still reported for
information.

## Holdout gate (selected candidate only)

PASS if, on HOLDOUT, net return > 0 **and** maximum drawdown < 40%. Sharpe,
Calmar and the comparison with BH-EW are reported alongside but do not change
the verdict.

- PASS: recommended for Phase 3 of `docs/IMPROVEMENT_PLAN.md` (paper, testnet,
  then small real money).
- FAIL: no real money. Report it and stop.

## Also reported, outside the candidate set

Funding-rate carry (long spot, short perpetual) on BTC and ETH from Binance's
public funding history, as descriptive analysis only: it needs a futures venue,
which the spot-only setup does not have.
