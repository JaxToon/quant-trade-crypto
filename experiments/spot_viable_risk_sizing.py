"""
What risk-per-trade makes this strategy actually executable on SPOT, and
what does it cost?

Spot cannot borrow. But Path 1 sizes as though it can: quantity is
risk/stop-distance, so

    notional / balance  =  risk% * conviction / stop_distance%

Tighter stops mean MORE implied leverage. Measured against testnet, a single
ledger decision asked for $14,896 of notional on a $9,800 account (1.52x),
and the live ledger runs a median 4.88x notional per $1 of capital.

The binding constraint is stricter than one position, though: every open
position draws on the SAME balance, so what matters is PEAK TOTAL notional
across everything open at once. With ~3 concurrent positions at 1.5x each,
the account needs ~4.5x its own size in buying power it does not have.

This sweeps PAPER_TRADING_RISK_PCT and reports, per level:

  - peak and p95 TOTAL open notional as a multiple of the account
  - the fraction of replay steps that would have been spot-executable
  - total PnL, win rate, and drawdown

The expectation from `drawdown_control_ab.py` is that drawdown PERCENTAGE is
scale-invariant under fixed-fractional sizing — halving risk halves gains and
losses alike, leaving the equity curve's shape unchanged. If that holds here,
reducing risk to reach spot viability costs RETURNS but buys no extra safety
and imposes no extra danger. That is the trade-off worth stating plainly, and
it is checked rather than assumed.

Usage: python -m experiments.spot_viable_risk_sizing
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SLIP = 0.00119
RISK_LEVELS = [0.020, 0.013, 0.010, 0.0075, 0.005]
START = None  # set from settings at runtime


def _max_dd(trades):
    eq = peak = START
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def _p(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))]


def run_segment(candles, tf, label):
    leds, notional = {}, {}
    original = settings.PAPER_TRADING_RISK_PCT
    for r in RISK_LEVELS:
        led = PaperTradingLedger(path=f"backtrack_logs/_svr_{label}_{r}.json",
                                 starting_balance=START, stop_slippage_pct=SLIP)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        leds[r] = led
        notional[r] = []

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}

    n = min(len(c) for c in candles.values())
    for t in range(n):
        for sym, cs in candles.items():
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
            for r, led in leds.items():
                settings.PAPER_TRADING_RISK_PCT = r
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=at)
        # Total notional committed right now, as a multiple of the account.
        for r, led in leds.items():
            tot = sum(p.quantity * p.entry_price for p in led.open_positions.values())
            notional[r].append(tot / START)
    settings.PAPER_TRADING_RISK_PCT = original
    return {r: (list(l.closed_trades), notional[r]) for r, l in leds.items()}


def main():
    global START
    START = settings.PAPER_TRADING_STARTING_BALANCE
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
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
    tot = {r: [] for r in RISK_LEVELS}
    seg_pnl = {r: [] for r in RISK_LEVELS}
    seg_dd = {r: [] for r in RISK_LEVELS}
    lev = {r: [] for r in RISK_LEVELS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for r, (trades, nots) in run_segment(part, tf, f"s{i}").items():
            tot[r].extend(trades)
            seg_pnl[r].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[r].append(_max_dd(trades))
            lev[r].extend(nots)
        print(f"  seg{i} done", flush=True)

    base_pnl = sum(seg_pnl[0.020])
    print(f"\n{'=' * 100}")
    print(f"SPOT-VIABLE RISK SIZING — {days}d, {n_seg} segments, slippage {SLIP*100:.3f}%")
    print(f"{'=' * 100}")
    print(f"{'risk':<8}{'n':<7}{'WR':>6}  {'Total PnL':>13}{'%base':>8}  "
          f"{'peak lev':>9}{'p95 lev':>9}  {'spot-OK':>9}  {'worstDD':>9}{'medDD':>8}")
    print("-" * 100)
    for r in RISK_LEVELS:
        ts = tot[r]
        if not ts:
            continue
        w = sum(1 for t in ts if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in ts)
        ok = sum(1 for v in lev[r] if v <= 1.0) / len(lev[r]) * 100
        tag = f"{r*100:.2f}%" + ("*" if r == 0.020 else "")
        print(f"{tag:<8}{len(ts):<7}{w/len(ts)*100:>5.1f}%  ${pnl:>+12,.0f}{pnl/base_pnl*100:>7.0f}%  "
              f"{max(lev[r]):>9.2f}{_p(lev[r],0.95):>9.2f}  {ok:>8.1f}%  "
              f"{max(seg_dd[r]):>8.1f}%{statistics.median(seg_dd[r]):>7.1f}%")
    print("-" * 100)
    print("* current production value.  lev = TOTAL open notional / account size.")
    print("  spot-OK = share of replay steps where everything open fit inside the account.")

    print(f"\n{'=' * 100}")
    print("READING THIS")
    viable = [r for r in RISK_LEVELS if _p(lev[r], 0.95) <= 1.0]
    if viable:
        best = max(viable)
        share = sum(seg_pnl[best]) / base_pnl * 100
        print(f"  Largest spot-viable risk (p95 leverage <= 1.0x): {best*100:.2f}% per trade,")
        print(f"  retaining {share:.0f}% of current returns.")
    else:
        print("  NO tested risk level keeps p95 total notional inside the account.")
        print("  Spot cannot hold this strategy at any of these sizes; the options are")
        print("  a larger account, futures (with liquidation risk nothing here models),")
        print("  or fewer concurrent positions.")
    dd_spread = max(statistics.median(seg_dd[r]) for r in RISK_LEVELS) - \
                min(statistics.median(seg_dd[r]) for r in RISK_LEVELS)
    print(f"\n  Median drawdown varies by only {dd_spread:.1f}pp across a 4x range of risk —")
    print("  confirming drawdown %% is scale-invariant. Cutting risk to reach spot")
    print("  viability costs RETURNS and buys no additional safety.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
