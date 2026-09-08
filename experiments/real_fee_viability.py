"""
Does the strategy survive Tokocrypto's REAL fees?

Every backtest in this project charges PAPER_TRADING_FEE_PCT = 0.1% per
side. Tokocrypto's published USDT-market schedule is:

    maker 0.15%  taker 0.15%  PPh 0.21%  CFX 0.0222%
    TOTAL 0.3822% PER SIDE

That is 3.82x what we model, and 0.7644% round-trip against our 0.2% — an
extra 0.56% on every trade. For scale, the slippage work established this
strategy's backtested edge dies somewhere around 0.51% of stop slippage.
An unmodelled cost of the same magnitude is not a detail.

The fee also cascades. `MIN_STOP_DISTANCE_PCT` is derived from it:

    MIN_STOP_DISTANCE_PCT = (2 * fee) / MAX_FEE_TO_RISK_RATIO

so at 0.1% the floor is 0.67%, and at 0.3822% it becomes 2.55%. Setups with
stops tighter than that get rejected outright rather than traded at a loss —
which is the guard working as intended, but it changes WHICH trades exist,
not just their PnL. Modelling only the PnL hit would understate the change.

Arms (all on the spot-viable config, risk 0.25% + notional cap 1.0x):

  0.100%  what every previous result used
  0.2622% Indodax USDT maker — the cheapest local venue, for contrast
  0.3822% Tokocrypto USDT — the venue actually planned
  0.4220% Tokocrypto IDR taker-sell — the worst local leg

PRE-REGISTERED: the spot plan survives only if, at 0.3822%, the strategy is
still profitable in >= 5 of 6 segments. Anything less and the deployment
target has to change venue or the plan is off.

Usage: python -m experiments.real_fee_viability
"""
from __future__ import annotations

import statistics

from config import settings

FEES = [0.001, 0.002622, 0.003822, 0.00422]
LABELS = {0.001: "0.100% (modelled)", 0.002622: "0.2622% Indodax",
          0.003822: "0.3822% Tokocrypto", 0.00422: "0.4220% TKO IDR sell"}
SLIP = 0.00119
RISK, CAP = 0.0025, 1.0


def main():
    from backtest import YF_MAX_DAYS, fetch_candles
    from strategy.entry_insight import build_entry_insight
    from strategy.fvg_analyzer import FairValueGapAnalyzer
    from strategy.smart_money import SmartMoneyAnalyzer

    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    start = settings.PAPER_TRADING_STARTING_BALANCE
    print(f"Fetching {days}d of {tf}...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]

    results = {}
    for fee in FEES:
        # The fee feeds MIN_STOP_DISTANCE_PCT, which is computed at import
        # time — so the module has to be reloaded for each level, otherwise
        # only the PnL changes and the stop-distance floor silently stays put.
        settings.PAPER_TRADING_FEE_PCT = fee
        import importlib
        import backtrack.paper_trading as pt
        importlib.reload(pt)
        settings.PAPER_TRADING_RISK_PCT = RISK
        settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = CAP

        seg_pnl, seg_dd, trades = [], [], []
        for i in range(n_seg):
            part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
            led = pt.PaperTradingLedger(path=f"backtrack_logs/_fee_{fee}_{i}.json",
                                        starting_balance=start, stop_slippage_pct=SLIP)
            led.balance = led.starting_balance
            led.open_positions.clear(); led.closed_trades.clear()
            led.max_closed_trades = 1_000_000

            smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in part}
            fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in part}
            m = min(len(c) for c in part.values())
            for t in range(m):
                for sym, cs in part.items():
                    c = cs[t]
                    sig = smc[sym].update(c)
                    hist = list(smc[sym].candles)
                    if len(hist) >= 3:
                        fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                            current_price=c.close, volume=c.volume)
                    fvg[sym].update_all(c.close)
                    ctx = {**sig, **fvg[sym].get_trading_signal(
                        current_price=c.close, trend=sig.get("market_structure", "unknown")),
                        "timestamp": c.timestamp.isoformat()}
                    ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
                    at = c.timestamp.isoformat()
                    led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
                    led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=at)
                    led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=at)

            tr = list(led.closed_trades)
            trades.extend(tr)
            seg_pnl.append(sum(x["pnl_dollars"] for x in tr))
            eq = peak = start
            worst = 0.0
            for x in tr:
                eq += x["pnl_dollars"]
                peak = max(peak, eq)
                if peak > 0:
                    worst = max(worst, (peak - eq) / peak * 100)
            seg_dd.append(worst)
        results[fee] = (trades, seg_pnl, seg_dd, pt.MIN_STOP_DISTANCE_PCT)
        print(f"  fee {fee*100:.4f}% done ({len(trades)} trades)", flush=True)

    base = sum(results[0.001][1])
    print(f"\n{'=' * 96}")
    print(f"REAL FEE VIABILITY — {days}d, {n_seg} segments, slippage {SLIP*100:.3f}%")
    print(f"{'=' * 96}")
    print(f"{'fee/side':<22}{'minStop':<10}{'n':<7}{'WR':>6}  {'Total PnL':>13}{'%base':>8}"
          f"  {'worstDD':>9}{'medDD':>8}  {'prof':>6}")
    print("-" * 96)
    for fee in FEES:
        tr, sp, sd, minstop = results[fee]
        if not tr:
            print(f"{LABELS[fee]:<22}{minstop*100:>8.2f}%  (no trades)"); continue
        w = sum(1 for t in tr if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in tr)
        prof = sum(1 for v in sp if v > 0)
        print(f"{LABELS[fee]:<22}{minstop*100:>8.2f}%  {len(tr):<7}{w/len(tr)*100:>5.1f}%  "
              f"${pnl:>+12,.0f}{pnl/base*100:>7.0f}%  {max(sd):>8.1f}%{statistics.median(sd):>7.1f}%"
              f"  {prof}/{n_seg}")
    print("-" * 96)

    tr, sp, sd, _ = results[0.003822]
    prof = sum(1 for v in sp if v > 0)
    print(f"\nPRE-REGISTERED GATE — Tokocrypto 0.3822%: profitable in {prof}/{n_seg} segments")
    if prof >= 5:
        print("  PASS. The spot plan survives real Tokocrypto fees.")
    else:
        print("  FAIL. The strategy does not survive its venue's real fees. The plan needs")
        print("  a cheaper venue, or it does not go live at all. Every prior PnL figure in")
        print("  this project was computed at 0.1% and is therefore optimistic.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
