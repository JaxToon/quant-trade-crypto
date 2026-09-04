"""
Final deploy gate for the breakeven-stop finding: does it quietly increase
CONCURRENT portfolio exposure?

`breakeven_lookahead_check.py` established the effect is real (it survives —
and strengthens under — honest intra-bar handling: BE_0.5R_LAGGED
+$752,072 / 28.5% drawdown vs baseline +$289,399 / 55.0%). But that
drawdown figure is computed from the SEQUENCE OF CLOSED TRADES, and BE
roughly doubles trade count while making ~55-63% of exits scratches (~$0).
A curve made of twice as many, mostly-tiny steps is smoother almost by
construction — so the apparent risk improvement may be a metric artifact
rather than genuinely less risk.

Real portfolio risk is about what's open AT THE SAME TIME. If positions
close early at breakeven and are immediately replaced, simultaneous
exposure can go UP even as trade-sequence drawdown goes DOWN. Deploying
something that looks safer while actually carrying more concurrent risk is
exactly the failure this check exists to prevent.

Measures, sampled at every replay step:
  - open position COUNT (mean / p95 / max)
  - open NOTIONAL (sum of quantity x entry price)
  - open RISK (sum of quantity x |entry - current stop|) -- the real
    "how much could this portfolio lose if everything stopped out now",
    and the one that matters. Note BE mechanically shrinks this for any
    position already moved to breakeven (its remaining risk is ~0), so a
    fair reading looks at count and notional too.

Usage: python -m experiments.breakeven_exposure_check [--segments 6]
"""
from __future__ import annotations

import argparse
import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

ARMS: dict[str, float | None] = {
    "BASELINE": None,
    "BE_0.5R_LAGGED": 0.5,
    "BE_0.3R_LAGGED": 0.3,
}


def _apply_breakeven(ledger: PaperTradingLedger, symbol: str, high: float, low: float, activation_r: float) -> None:
    for pos in ledger.open_positions.values():
        if pos.symbol != symbol:
            continue
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 0:
            continue
        if pos.direction == "long":
            if (high - pos.entry_price) / risk >= activation_r:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            if (pos.entry_price - low) / risk >= activation_r:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)


def _snapshot(ledger: PaperTradingLedger) -> tuple[int, float, float]:
    count = len(ledger.open_positions)
    notional = sum(p.quantity * p.entry_price for p in ledger.open_positions.values())
    risk = sum(p.quantity * abs(p.entry_price - p.stop_loss) for p in ledger.open_positions.values())
    return count, notional, risk


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    ledgers = {}
    samples: dict[str, dict[str, list]] = {}
    for name in ARMS:
        led = PaperTradingLedger(
            path=f"backtrack_logs/_beexp_{label}_{name}.json",
            starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
        )
        led.balance = led.starting_balance
        led.open_positions.clear()
        led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led
        samples[name] = {"count": [], "notional": [], "risk": []}

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}
    prev_bar: dict[str, tuple[float, float]] = {}

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            hist = list(smc.candles)
            if len(hist) >= 3:
                fvg.detect_fvg(candles=hist, index=len(hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            prior = prev_bar.get(sym)

            for name, be_r in ARMS.items():
                led = ledgers[name]
                if be_r is not None and prior is not None:
                    _apply_breakeven(led, sym, prior[0], prior[1], be_r)
                led.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                led.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                led.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            prev_bar[sym] = (candle.high, candle.low)

        # One portfolio-wide snapshot per replay step (after all symbols processed)
        for name, led in ledgers.items():
            c, n, r = _snapshot(led)
            samples[name]["count"].append(c)
            samples[name]["notional"].append(n)
            samples[name]["risk"].append(r)

    out = {}
    for name, led in ledgers.items():
        led.open_positions.clear()
        out[name] = {"samples": samples[name], "pnl": sum(t["pnl_dollars"] for t in led.closed_trades),
                     "trades": len(led.closed_trades)}
    return out


def _p(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    tf = args.timeframe
    days = min(args.days if args.days is not None else YF_MAX_DAYS.get(tf, 730), YF_MAX_DAYS.get(tf, 730))

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — fetching {days}d of {tf}...\n", flush=True)
    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        c = fetch_candles(symbol, days, tf)
        if c:
            all_candles[symbol] = c
    if not all_candles:
        print("No data — aborting.")
        return
    print(f"{len(all_candles)} symbols loaded.\n", flush=True)

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]

    agg = {name: {"count": [], "notional": [], "risk": [], "pnl": 0.0, "trades": 0} for name in ARMS}
    for i in range(n_seg):
        seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()}
        res = run_segment(seg, tf, f"seg{i}")
        for name, r in res.items():
            agg[name]["count"].extend(r["samples"]["count"])
            agg[name]["notional"].extend(r["samples"]["notional"])
            agg[name]["risk"].extend(r["samples"]["risk"])
            agg[name]["pnl"] += r["pnl"]
            agg[name]["trades"] += r["trades"]
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 104}")
    print("CONCURRENT EXPOSURE — does breakeven quietly carry MORE risk at the same time?")
    print(f"{'=' * 104}")
    print(f"{'Arm':<18}{'Trades':<9}{'PnL':<15}{'OpenPos mean':<14}{'p95':<7}{'max':<7}"
          f"{'Notional p95':<15}{'OpenRisk p95':<14}{'OpenRisk max':<13}")
    print("-" * 104)
    for name in ARMS:
        a = agg[name]
        print(f"{name:<18}{a['trades']:<9}${a['pnl']:>+12,.0f}  "
              f"{statistics.mean(a['count']):<14.2f}{_p(a['count'],0.95):<7.0f}{max(a['count']):<7.0f}"
              f"${_p(a['notional'],0.95):>13,.0f}  ${_p(a['risk'],0.95):>12,.0f}  ${max(a['risk']):>11,.0f}")
    print("-" * 104)
    print("OpenRisk = sum of quantity x |entry - current stop| across open positions: what the")
    print("portfolio loses if everything stops out at once. BE mechanically shrinks it for any")
    print("position already at breakeven, so read OpenPos count and Notional alongside it --")
    print("those show whether MORE capital is committed simultaneously regardless of stop placement.")
    print(f"{'=' * 104}")


if __name__ == "__main__":
    main()
