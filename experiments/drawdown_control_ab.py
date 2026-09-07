"""
Can this strategy's drawdown be brought into survivable territory without
destroying its returns?

Why: at the measured 0.119% stop slippage, baseline drawdown is
worst-segment 68.5%, median 42.7% (drawdown_at_real_slippage.py). It is
profitable in 6/6 segments and never actually ruins an account, so this is
not a maths problem — it is a "nobody keeps funding a system that takes
two-thirds of their money" problem. And Path 1 currently has NO loss limit
of any kind: DAILY_DRAWDOWN_LIMIT lives only in strategy/execution.py
(Path 2, disabled).

Two independent levers, tested separately and together:

  EQUITY sizing  — size risk off current equity instead of the fixed
                   starting balance. Today a $200 risk unit is 2% of a
                   $10,000 account but 6.3% of what remains after a 68.5%
                   drawdown, so the risk FRACTION grows exactly when the
                   account can least afford it. Capped at starting_balance
                   so winnings never inflate the unit (keeps the PnL curve
                   interpretable; see _sizing_base).
  DD HALT        — stop opening new positions once equity is X% below its
                   peak; resume at X/2 (hysteresis). Open positions keep
                   their stops either way.

PRE-REGISTERED SUCCESS CRITERIA (fixed before running; not to be
renegotiated afterwards). A variant is worth deploying only if ALL hold:
  (a) worst-segment drawdown < 40%   (baseline 68.5%)
  (b) median-segment drawdown < 25%  (baseline 42.7%)
  (c) retains >= 50% of baseline total PnL
  (d) profitable in >= 5 of 6 segments

If nothing clears all four, the honest report is that drawdown cannot be
controlled by sizing alone — NOT a relaxed threshold picked to let a
winner through.

Usage: python -m experiments.drawdown_control_ab
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SLIP = 0.00119   # measured mean, n=31 live stop exits

# name -> (equity_based_sizing, max_drawdown_halt_pct)
ARMS = {
    "BASELINE":        (False, 0.0),
    "EQUITY":          (True,  0.0),
    "HALT_25":         (False, 25.0),
    "HALT_35":         (False, 35.0),
    "EQUITY+HALT_25":  (True,  25.0),
    "EQUITY+HALT_35":  (True,  35.0),
}


def _max_dd(trades):
    eq = peak = settings.PAPER_TRADING_STARTING_BALANCE
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def run_segment(symbols_candles, timeframe, label):
    ledgers = {}
    for name, (eq, halt) in ARMS.items():
        led = PaperTradingLedger(path=f"backtrack_logs/_ddc_{label}_{name}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                                 stop_slippage_pct=SLIP,
                                 equity_based_sizing=eq, max_drawdown_halt_pct=halt)
        led.balance = led.starting_balance
        led.peak_balance = led.starting_balance
        led._halted = False
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

    n = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(n):
        for sym, candles in symbols_candles.items():
            c = candles[t]
            sig = smc[sym].update(c)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2, current_price=c.close, volume=c.volume)
            fvg[sym].update_all(c.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(current_price=c.close,
                   trend=sig.get("market_structure", "unknown")), "timestamp": c.timestamp.isoformat()}
            ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
            at = c.timestamp.isoformat()
            for led in ledgers.values():
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=timeframe)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=timeframe, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=timeframe, opened_at=at)

    return {n_: list(l.closed_trades) for n_, l in ledgers.items()}


def main():
    tf, days, n_seg = "1h", YF_MAX_DAYS.get("1h", 730), 6
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
    seg_pnl = {k: [] for k in ARMS}
    seg_dd = {k: [] for k in ARMS}
    seg_n = {k: 0 for k in ARMS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for name, tr in run_segment(part, tf, f"s{i}").items():
            seg_pnl[name].append(sum(t["pnl_dollars"] for t in tr))
            seg_dd[name].append(_max_dd(tr))
            seg_n[name] += len(tr)
        print(f"  seg{i} done", flush=True)

    base_pnl = sum(seg_pnl["BASELINE"])
    print(f"\n{'=' * 92}")
    print(f"DRAWDOWN CONTROL A/B — {days}d, {n_seg} segments, slippage {SLIP*100:.3f}%")
    print(f"{'=' * 92}")
    print(f"{'Arm':<17}{'Trades':<9}{'Total PnL':<16}{'%of base':<11}{'worstDD':<10}{'medDD':<10}{'Profitable':<11}")
    print("-" * 92)
    for name in ARMS:
        pnl = sum(seg_pnl[name]); prof = sum(1 for v in seg_pnl[name] if v > 0)
        print(f"{name:<17}{seg_n[name]:<9}${pnl:>+13,.0f}  {pnl/base_pnl*100:<11.0f}"
              f"{max(seg_dd[name]):<10.1f}{statistics.median(seg_dd[name]):<10.1f}{prof}/{n_seg:<11}")
    print("-" * 92)

    print("\nPRE-REGISTERED GATE  (a) worstDD<40  (b) medDD<25  (c) PnL>=50% of base  (d) profitable>=5/6")
    winners = []
    for name in ARMS:
        if name == "BASELINE":
            continue
        pnl = sum(seg_pnl[name]); prof = sum(1 for v in seg_pnl[name] if v > 0)
        a = max(seg_dd[name]) < 40
        bb = statistics.median(seg_dd[name]) < 25
        c = pnl >= 0.5 * base_pnl
        d = prof >= 5
        ok = a and bb and c and d
        if ok:
            winners.append(name)
        print(f"  {name:<17}(a){'PASS' if a else 'FAIL':<6}(b){'PASS' if bb else 'FAIL':<6}"
              f"(c){'PASS' if c else 'FAIL':<6}(d){'PASS' if d else 'FAIL':<6} -> "
              f"{'DEPLOYABLE' if ok else 'no'}")
    print(f"\n{'Winners: ' + ', '.join(winners) if winners else 'NOTHING cleared all four gates — report that, do not lower the bar.'}")
    print(f"{'=' * 92}")


if __name__ == "__main__":
    main()
