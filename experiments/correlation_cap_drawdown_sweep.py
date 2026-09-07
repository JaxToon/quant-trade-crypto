"""
Can a TIGHTER correlation cap bring drawdown into survivable territory?

The binding constraint on real money is drawdown, not returns: at measured
slippage the strategy is profitable 6/6 segments but carries worst-segment
68.5% / median 42.7%. Two things are already known about it:

  - Position SIZING cannot fix it. Drawdown % is scale-invariant under
    fixed-fractional sizing — halve every position and you halve every gain
    and loss alike, leaving the equity curve's shape unchanged. Measured:
    equity-based sizing moved median drawdown 42.7% -> 42.5%.
    (drawdown_control_ab.py)
  - Dropping SHORT makes it far WORSE, not better: LONG_ONLY's worst
    segment drawdown is 126.7% — past zero, i.e. ruin. Shorts are damping
    the book, not dragging it. (long_only_viability.py)

That leaves the lever named in the drawdown write-up but never actually
pulled: how many correlated positions run AT THE SAME TIME.
`PORTFOLIO_MAX_CORRELATED_RISK_PCT` controls exactly that, and every prior
discussion of it in this project has been about RAISING it (the shelved
CAP_1.5X / CAP_2X work, aimed at more returns). It has never been swept
downward with drawdown as the target.

Mechanism, and why it should work where sizing does not: sizing scales every
position uniformly and so cannot change the curve's shape, but the cap
changes WHICH positions coexist. Concentrated correlated exposure is what
turns a bad stretch into a deep drawdown — several positions in the same
direction on assets that move together all lose at once.

Sweeps 0.08 (production), 0.06, 0.04, 0.03, 0.02, and reports concurrent
open-position counts alongside PnL so the mechanism is visible rather than
assumed.

PRE-REGISTERED GATE — deliberately the SAME bar used by
drawdown_control_ab.py, which nothing cleared. Reusing it rather than
inventing a friendlier one is the point:
  (a) worst-segment drawdown < 40%   (baseline 68.5%)
  (b) median-segment drawdown < 25%  (baseline 42.7%)
  (c) retains >= 50% of baseline total PnL
  (d) profitable in >= 5 of 6 segments

If nothing clears all four, report that drawdown is not reachable by risk
controls at all — do NOT relax the threshold to manufacture a winner.

Usage: python -m experiments.correlation_cap_drawdown_sweep
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
CAPS = [0.08, 0.06, 0.04, 0.03, 0.02]


def _max_dd(trades):
    eq = peak = settings.PAPER_TRADING_STARTING_BALANCE
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
    leds, conc = {}, {}
    original = settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT
    for cap in CAPS:
        led = PaperTradingLedger(path=f"backtrack_logs/_ccd_{label}_{cap}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                                 stop_slippage_pct=SLIP)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        leds[cap] = led
        conc[cap] = []

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
            for cap, led in leds.items():
                settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = cap
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=at)
        for cap, led in leds.items():
            conc[cap].append(len(led.open_positions))
    settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = original
    return {cap: (list(l.closed_trades), conc[cap]) for cap, l in leds.items()}


def main():
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf} (slippage {SLIP*100:.3f}%)...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]
    seg_pnl = {c: [] for c in CAPS}
    seg_dd = {c: [] for c in CAPS}
    tot = {c: [] for c in CAPS}
    conc = {c: [] for c in CAPS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for cap, (tr, cc) in run_segment(part, tf, f"s{i}").items():
            seg_pnl[cap].append(sum(t["pnl_dollars"] for t in tr))
            seg_dd[cap].append(_max_dd(tr))
            tot[cap].extend(tr)
            conc[cap].extend(cc)
        print(f"  seg{i} done", flush=True)

    base_pnl = sum(seg_pnl[0.08])
    print(f"\n{'=' * 96}")
    print(f"CORRELATION CAP SWEEP (downward) — {days}d, {n_seg} segments, slippage {SLIP*100:.3f}%")
    print(f"{'=' * 96}")
    print(f"{'Cap':<8}{'n':<7}{'WR':>7}  {'Total PnL':>13}  {'%base':>7}  {'worstDD':>9}"
          f"  {'medDD':>8}  {'openPos p95':>12}  {'prof':>6}")
    print("-" * 96)
    for cap in CAPS:
        tr = tot[cap]
        if not tr:
            continue
        w = sum(1 for t in tr if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in tr)
        prof = sum(1 for v in seg_pnl[cap] if v > 0)
        tag = f"{cap:.2f}" + ("*" if cap == 0.08 else "")
        print(f"{tag:<8}{len(tr):<7}{w/len(tr)*100:>6.1f}%  ${pnl:>+12,.0f}  {pnl/base_pnl*100:>6.0f}%"
              f"  {max(seg_dd[cap]):>8.1f}%  {statistics.median(seg_dd[cap]):>7.1f}%"
              f"  {_p(conc[cap], 0.95):>12.0f}  {prof}/{n_seg}")
    print("-" * 96)
    print("* = current production value")

    print("\nPRE-REGISTERED GATE (same bar as drawdown_control_ab.py, which nothing cleared):")
    print("  (a) worstDD<40  (b) medDD<25  (c) PnL>=50% of base  (d) profitable>=5/6")
    winners = []
    for cap in CAPS:
        if cap == 0.08:
            continue
        pnl = sum(seg_pnl[cap])
        prof = sum(1 for v in seg_pnl[cap] if v > 0)
        a = max(seg_dd[cap]) < 40
        bb = statistics.median(seg_dd[cap]) < 25
        c = pnl >= 0.5 * base_pnl
        d = prof >= 5
        ok = a and bb and c and d
        if ok:
            winners.append(cap)
        print(f"  cap={cap:.2f}  (a){'PASS' if a else 'FAIL':<5}(b){'PASS' if bb else 'FAIL':<5}"
              f"(c){'PASS' if c else 'FAIL':<5}(d){'PASS' if d else 'FAIL':<5} -> "
              f"{'DEPLOYABLE' if ok else 'no'}")
    if winners:
        print(f"\n  Qualifying: {', '.join(f'{c:.2f}' for c in winners)}")
        print("  Deploy the LOOSEST qualifying cap — it gives the drawdown relief while")
        print("  giving up the least return.")
    else:
        print("\n  Nothing cleared the gate. Combined with sizing (no effect) and dropping")
        print("  SHORT (ruin), that means this drawdown is NOT reachable by risk controls.")
        print("  It is a property of the strategy's return distribution, and the honest")
        print("  options are to accept it, change the strategy, or not deploy real money.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
