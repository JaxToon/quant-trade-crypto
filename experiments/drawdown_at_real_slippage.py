"""
What drawdown does the live strategy actually carry at its MEASURED
slippage — and would a real account have survived it?

`breakeven_under_slippage.py` reported worst-segment drawdown alongside
PnL, and the numbers were alarming: 55.0% at zero slippage, 85.3% at 0.25%,
116.9% at 0.40%. A drawdown above 100% means the account went through zero
— ruin, not a bad month.

Those were run at assumed slippage levels. The real figure is now known:
0.119% mean, reconstructed from 31 live stop exits
(`reconstruct_historical_slippage.py`). This runs baseline at exactly that
level, reports per-segment drawdown rather than only the worst, and checks
whether a simple daily-loss halt would have changed the outcome.

That last part matters because DAILY_DRAWDOWN_LIMIT (5%) is enforced ONLY
in strategy/execution.py — Path 2, currently disabled. Path 1, the path
with the actual evidence behind it and the one any real-money deployment
would use, has no loss limit of any kind.

Usage: python -m experiments.drawdown_at_real_slippage
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

MEASURED_SLIPPAGE = 0.00119   # mean, n=31 live stop exits
LEVELS = [0.0, MEASURED_SLIPPAGE]


def _equity_curve(trades):
    eq = settings.PAPER_TRADING_STARTING_BALANCE
    out = [eq]
    for t in trades:
        eq += t["pnl_dollars"]
        out.append(eq)
    return out


def _max_dd(trades):
    eq = peak = settings.PAPER_TRADING_STARTING_BALANCE
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def _ruined(trades):
    """Did equity ever touch or pass zero?"""
    return min(_equity_curve(trades)) <= 0


def run_segment(symbols_candles, timeframe, label):
    ledgers = {}
    for slip in LEVELS:
        led = PaperTradingLedger(path=f"backtrack_logs/_dd_{label}_{slip}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                                 stop_slippage_pct=slip)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[slip] = led

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
            for slip, led in ledgers.items():
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=timeframe)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=timeframe, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=timeframe, opened_at=at)

    return {slip: list(led.closed_trades) for slip, led in ledgers.items()}


def main():
    tf, days, n_seg = "1h", YF_MAX_DAYS.get("1h", 730), 6
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
    seg = {slip: [] for slip in LEVELS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        res = run_segment(part, tf, f"s{i}")
        for slip, tr in res.items():
            seg[slip].append(tr)
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 82}")
    print(f"DRAWDOWN AT MEASURED SLIPPAGE ({MEASURED_SLIPPAGE*100:.3f}%) — {days}d, {n_seg} segments")
    print(f"{'=' * 82}")
    for slip in LEVELS:
        tag = "ZERO (old assumption)" if slip == 0 else f"MEASURED {slip*100:.3f}%"
        print(f"\n{tag}")
        print(f"  {'seg':<6}{'trades':<9}{'PnL':<16}{'maxDD%':<10}{'ruined?':<9}")
        print("  " + "-" * 52)
        dds, ruins = [], 0
        for i, tr in enumerate(seg[slip]):
            dd = _max_dd(tr); r = _ruined(tr)
            dds.append(dd); ruins += 1 if r else 0
            print(f"  {i:<6}{len(tr):<9}${sum(t['pnl_dollars'] for t in tr):>+13,.0f}  {dd:<10.1f}{'YES' if r else 'no':<9}")
        print("  " + "-" * 52)
        print(f"  worst {max(dds):.1f}%   median {statistics.median(dds):.1f}%   "
              f"segments ruined: {ruins}/{n_seg}")

    print(f"\n{'=' * 82}")
    print("A drawdown of even 40-50% is not survivable in practice with real money —")
    print("not because the maths breaks, but because nobody keeps funding a system")
    print("through it. Read the per-segment column, not the average.")
    print(f"{'=' * 82}")


if __name__ == "__main__":
    main()
