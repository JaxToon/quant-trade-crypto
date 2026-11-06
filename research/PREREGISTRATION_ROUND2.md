# Pre-registration, research round 2 (written 2026-11-06, before any result)

Two additions to C4, each tested against C4 as it runs today (U8, long-only spot, x1).
The fund later applies the owner's x2 risk setting to whatever is chosen, so the
comparison is made at x1, where both sides are unlevered.

## R2-A: long/short trend (C4-LS)

C4 sits in cash when markets fall (2025-26: -3.9%). Trend-following funds trade both
directions. The short side mirrors C4 exactly:

- For each lookback L in (5, 10, 20, 30, 60, 90, 150, 250, 360): a short sub-model turns
  on when the close breaks BELOW the lowest close of the previous L days; its stop starts
  at the channel midpoint and only ratchets DOWN; it turns off when the close rises above
  the stop. (Mirror of `research.strategies.donchian_ensemble`.)
- Net signal per coin = (long sub-models on - short sub-models on) / 9, in [-1, 1].
- Weight = net signal x min(1, 25% / sigma90) / N eligible, gross |w| capped at 1.
- Execution: next open, as `research/engine.py`, with signed weights. Long legs are spot
  (Tokocrypto 0.3822% + 0.05% slippage per side); short legs are perpetual futures.
- Perpetual costs: fee + slippage **0.15% per side** (Bittime's official fee is not
  verified, so this is set above typical exchange taker fees), plus real Binance USDT-M
  funding: a short receives positive funding and pays negative funding. Before a coin's
  perpetual listed, its short side is not allowed.
- Sensitivity only (not part of the decision): perp cost at 0.4322% per side.

## R2-B: gold as a ninth asset (C4+GOLD)

Gold often moves differently from crypto, which is why trend-following funds trade it.
PAXG (one token = one troy ounce, on Tokocrypto and Binance) is added to the U8 universe
with C4's rules unchanged (long-only spot, N becomes 9 once PAXG is eligible).

## Periods, data

- Binance daily bars (`research/data.py` cache for U8; PAXGUSDT from data-api), funding
  from data.binance.vision.
- DEV 2018-08-17 to 2024-12-31; HOLDOUT 2025-01-01 to 2026-10-09.
- Contamination disclosure: the holdout has been seen (C4 verdict, universe study). It is
  a second check, not fresh evidence. Anything adopted still goes through the forward test.

## Decision rule (locked)

A candidate replaces the current C4 only if, against C4 at x1:
1. Sharpe is higher in BOTH DEV and HOLDOUT, and
2. max drawdown is no more than 5 percentage points worse in either period.

If both pass, they may be combined (LS with gold), but only if the combination also passes
the same rule; otherwise the single passing candidate with the higher DEV Sharpe is taken.
If neither passes, the fund stays as it is.
