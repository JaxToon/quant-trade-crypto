# Pre-registration: intraday regression-trend (written 2026-11-06, before any result)

Owner's request: try an intra-hour tactic like the one in his TradingView screenshot
(SOPHUSDT 15m perp: a regression "angle/theta" + Pearson R regime indicator with
red/green regime shading, and a short position). Owner's target: 40%/week. That target
is not a pass criterion here; the study reports how far from it the results are.

## Strategy RT (regression trend), per coin, decided at each bar's close

- On the last N bars: Pearson correlation r between log(close) and time (r near +1 =
  straight uptrend, near -1 = straight downtrend), i.e. the screenshot's "Pearson R" with
  the sign of its "angle".
- Enter long when r > R*, short when r < -R*. Exit (go flat) when r crosses 0 (the
  regime flips). No other stops or targets.
- Execution at the next bar (signal at close t earns the return from close t to close
  t+1; one bar of delay is not added on top, which slightly flatters intraday results).
- Size: each coin 1/8 x min(1, 25% / realized vol), vol from the last 30 days of bars,
  annualized. Gross never above 100% (no leverage), as C4.
- Costs: perpetual futures. Fee + slippage per side 0.10% (base) and 0.15% (stress).
  Real Binance USDT-M funding charged on held positions (spread evenly over each day).
- Universe: the 8 C4 coins (BTC ETH BNB XRP ADA SOL AVAX DOT), Binance USDT-M
  perpetual prices from data.binance.vision. Micro-caps like SOPH are not tested: too
  little history and liquidity to trust a backtest.

## Variants (12 trials, all reported)

Timeframe {15m, 1h} x N {24, 48, 96} bars x R* {0.6, 0.8}.

## Periods

DEV 2020-01-01 to 2024-12-31 (or each coin's perp listing); HOLDOUT 2025-01-01 to
2026-10-09. The 8 coins were already used in earlier studies; the intraday data and this
rule were not.

## Decision rule (locked)

Pick the variant with the best DEV Sharpe at 0.10%/side. It becomes a paper sleeve next to
C4 only if ALL hold:
1. DEV deflated Sharpe probability > 0.95 with n_trials = 12;
2. HOLDOUT net return > 0 and HOLDOUT Sharpe > 0.5 at the 0.15%/side stress cost;
3. HOLDOUT max drawdown < 30% (owner's tolerance).
Otherwise: rejected, and no parameter is tuned further on this history.
