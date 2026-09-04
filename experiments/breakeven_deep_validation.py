"""
Deep validation of the BE_0.5R breakeven-stop finding before deploying it.

`trailing_breakeven_robustness.py` found BE_0.5R beats baseline 6/6 segments
(+85% PnL) and `breakeven_vs_capacity_disambiguation.py` proved it's an
independent lever (super-additive with the correlation cap, unlike
SHORT_20PCT) with the only better-returns-AND-lower-drawdown profile in
the project. This runs the checks that should come BEFORE deployment.

Four questions, all on the same 6-segment/730d bar:

1. PER-SYMBOL CONSISTENCY (robustness, NOT selection). Does BE_0.5R beat
   baseline on most/all 8 symbols individually, or is the aggregate carried
   by one or two? Consistent = real effect. Concentrated = fragile finding
   that should be rejected.
   Explicitly NOT a shortlist-the-winners exercise: picking which symbols
   get the rule based on which backtested well is textbook curve-fitting
   (this project already has a live example of why -- SOLUSDT shows 9/9
   wins and ~91% of all live crypto profit on a sample far too small to
   mean anything).

2. TIGHTER PARAMETER SWEEP. 0.5R > 0.7R > 1.0R was monotonic in the first
   sweep, so 0.5R might be an edge-of-plateau artifact rather than a peak.
   Adds 0.3R and 0.4R. If tighter keeps improving without limit, that's a
   warning sign -- it would mean the "edge" is really just "take risk off
   the table almost immediately", which trivially reduces variance and
   should be treated as a degenerate solution, not an edge.

3. PER-DIRECTION SPLIT. Does it help LONG and SHORT equally? If breakeven
   disproportionately rescues SHORT, that's directly relevant to the
   project's longest-running unsolved problem.

4. MECHANISM CHECK. Confirms the win-rate collapse (57%->33%) really is
   scratches rather than new losses, by reporting the share of exits that
   land within +/-0.15% of entry.

Usage:
  python -m experiments.breakeven_deep_validation                 # crypto
  ASSET_TYPE=stock python -m experiments.breakeven_deep_validation  # transfer test
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

ARMS: dict[str, float | None] = {
    "BASELINE": None,
    "BE_0.3R": 0.3,
    "BE_0.4R": 0.4,
    "BE_0.5R": 0.5,
    "BE_0.6R": 0.6,
}
SCRATCH_BAND_PCT = 0.15  # |exit - entry| within this % counts as a scratch


def _apply_breakeven(ledger: PaperTradingLedger, symbol: str, high: float, low: float, activation_r: float) -> None:
    for pos in ledger.open_positions.values():
        if pos.symbol != symbol:
            continue
        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if initial_risk <= 0:
            continue
        if pos.direction == "long":
            if (high - pos.entry_price) / initial_risk >= activation_r:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            if (pos.entry_price - low) / initial_risk >= activation_r:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)


def _max_dd(trades: list[dict]) -> float:
    equity = peak = settings.PAPER_TRADING_STARTING_BALANCE
    worst = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak * 100)
    return worst


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, list[dict]]:
    ledgers = {}
    for name in ARMS:
        led = PaperTradingLedger(
            path=f"backtrack_logs/_bedeep_{label}_{name}.json",
            starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
        )
        led.balance = led.starting_balance
        led.open_positions.clear()
        led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

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

            for name, be_r in ARMS.items():
                led = ledgers[name]
                if be_r is not None:
                    _apply_breakeven(led, sym, candle.high, candle.low, be_r)
                led.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                led.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                led.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    out = {}
    for name, led in ledgers.items():
        led.open_positions.clear()
        out[name] = list(led.closed_trades)
    return out


def _summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "scratch_pct": 0.0}
    wins = [t for t in trades if t["pnl_dollars"] > 0]
    scratches = [
        t for t in trades
        if t["entry_price"] and abs(t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100 <= SCRATCH_BAND_PCT
    ]
    return {
        "n": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pnl": sum(t["pnl_dollars"] for t in trades),
        "scratch_pct": len(scratches) / len(trades) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    days = min(args.days if args.days is not None else YF_MAX_DAYS.get(timeframe, 730), YF_MAX_DAYS.get(timeframe, 730))

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — fetching {days}d of {timeframe}...\n", flush=True)
    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        c = fetch_candles(symbol, days, timeframe)
        if c:
            all_candles[symbol] = c
            print(f"  {symbol}: {len(c)} candles", flush=True)
    if not all_candles:
        print("No data — aborting.")
        return

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]

    all_trades: dict[str, list[dict]] = {name: [] for name in ARMS}
    seg_pnl: dict[str, list[float]] = {name: [] for name in ARMS}
    seg_dd: dict[str, list[float]] = {name: [] for name in ARMS}

    for i in range(n_seg):
        seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()}
        res = run_segment(seg, timeframe, f"seg{i}")
        for name, trades in res.items():
            all_trades[name].extend(trades)
            seg_pnl[name].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[name].append(_max_dd(trades))
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 104}")
    print(f"[{settings.ASSET_TYPE.upper()}] BREAKEVEN DEEP VALIDATION — {n_seg} segments, {days}d")
    print(f"{'=' * 104}")

    # ---- 2. parameter sweep + overall ----
    print(f"\n[2] PARAMETER SWEEP (is 0.5R a peak, or does tighter keep 'improving' degenerately?)")
    print(f"{'Arm':<12}{'Trades':<9}{'WR':<9}{'Total PnL':<17}{'WorstSegDD':<13}{'Scratch%':<11}{'Segs beat base':<15}")
    print("-" * 104)
    base_seg = seg_pnl["BASELINE"]
    for name in ARMS:
        s = _summarize(all_trades[name])
        beats = sum(1 for a, b in zip(seg_pnl[name], base_seg) if a > b)
        beat_str = "-" if name == "BASELINE" else f"{beats}/{n_seg}"
        print(f"{name:<12}{s['n']:<9}{s['wr']:<9.1f}${s['pnl']:>+14,.0f}   {max(seg_dd[name]):<13.1f}{s['scratch_pct']:<11.1f}{beat_str:<15}")

    # ---- 1. per-symbol consistency ----
    print(f"\n[1] PER-SYMBOL CONSISTENCY of BE_0.5R (robustness check — NOT a shortlist)")
    print(f"{'Symbol':<12}{'Base PnL':<16}{'BE_0.5R PnL':<16}{'Delta':<16}{'Better?':<9}")
    print("-" * 104)
    base_by_sym: dict[str, float] = defaultdict(float)
    be_by_sym: dict[str, float] = defaultdict(float)
    for t in all_trades["BASELINE"]:
        base_by_sym[t["symbol"]] += t["pnl_dollars"]
    for t in all_trades["BE_0.5R"]:
        be_by_sym[t["symbol"]] += t["pnl_dollars"]
    better = 0
    for sym in sorted(set(base_by_sym) | set(be_by_sym)):
        b, e = base_by_sym[sym], be_by_sym[sym]
        ok = e > b
        better += 1 if ok else 0
        print(f"{sym:<12}${b:>+14,.0f}  ${e:>+14,.0f}  ${e-b:>+14,.0f}  {'YES' if ok else 'no':<9}")
    total_syms = len(set(base_by_sym) | set(be_by_sym))
    print("-" * 104)
    print(f"BE_0.5R better on {better}/{total_syms} symbols.  Consistent = real effect; concentrated = fragile.")

    # ---- 3. per-direction ----
    print(f"\n[3] PER-DIRECTION (does breakeven disproportionately rescue SHORT?)")
    print(f"{'Direction':<12}{'Base n':<9}{'Base WR':<10}{'Base PnL':<16}{'BE n':<9}{'BE WR':<10}{'BE PnL':<16}{'Delta':<14}")
    print("-" * 104)
    for direction in ("long", "short"):
        b = _summarize([t for t in all_trades["BASELINE"] if t.get("direction") == direction])
        e = _summarize([t for t in all_trades["BE_0.5R"] if t.get("direction") == direction])
        print(f"{direction:<12}{b['n']:<9}{b['wr']:<10.1f}${b['pnl']:>+14,.0f}  {e['n']:<9}{e['wr']:<10.1f}${e['pnl']:>+14,.0f}  ${e['pnl']-b['pnl']:>+12,.0f}")

    print(f"\n{'=' * 104}")
    print("Deploy only if: [1] wins on most symbols (not 1-2 carrying it), [2] 0.5R is a genuine")
    print("peak rather than 'tighter is always better' (which would be a degenerate no-risk solution),")
    print("[3] no direction is made worse, and the drawdown improvement holds.")
    print(f"{'=' * 104}")


if __name__ == "__main__":
    main()
