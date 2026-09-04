"""
MFE/MAE (Maximum Favorable / Adverse Excursion) analysis, split by trade
direction — asks whether this system's SHORT problem is an EXIT-geometry
problem rather than the entry-selection problem every prior experiment
assumed.

Motivation: ~70 entry-filter configurations have been tested across this
project (VWAP, EMA, the 11-indicator survey, all combinations,
market_structure alignment, GNN confluence) and NONE beat baseline. Exit
design has barely been touched — only `backtest_target_matrix.py`'s target
comparison. Field practice says that's backwards: exit optimization is
usually the higher-leverage half and the more neglected one, and MFE/MAE is
the standard tool for it.

The specific hypothesis this tests, which no prior experiment here has
asked: live data shows LONG and SHORT run essentially IDENTICAL stop/target
geometry (~1.2% target, ~1.0-1.17% stop realized on both) but wildly
different win rates (76.5% LONG vs 37.1% SHORT). A single stop distance
applied to both directions may simply be too tight for SHORT, whose adverse
moves in crypto (squeezes/pumps) are sharper than the adverse moves against
a LONG — meaning the stop cuts SHORT trades that would have won.

Method: replays the same 730d/1h multi-symbol window every other robustness
pass in this project uses, driving the REAL `PaperTradingLedger` so entries
and exits are exactly what production logic would do — MFE/MAE is only
*observed* alongside, never fed back into any decision. For each open
position, each candle updates:
  MFE = best the trade ever got (high for long, low for short)
  MAE = worst it ever got against you (low for long, high for short)
both as % of entry, and both recorded at close.

Reads out:
  1. MAE of WINNERS, per direction — how deep did eventual winners dip?
     If SHORT winners routinely dip near the stop while LONG winners don't,
     the shared stop distance is the problem, not SHORT entry quality.
  2. MFE of LOSERS, per direction — did losers show real profit first?
     High MFE on losers means a breakeven/trailing rule could rescue them.
  3. MFE vs realized PnL of WINNERS — profit left on the table.

Usage: python -m experiments.mfe_mae_exit_analysis [--days 730] [--timeframe 1h]
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


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0 if b else 0.0


def run(symbols_candles: dict[str, list[Candle]], timeframe: str) -> list[dict]:
    """Returns one record per closed trade with its MFE/MAE attached."""
    ledger = PaperTradingLedger(
        path="backtrack_logs/_mfe_mae_scratch.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    ledger.max_closed_trades = 1_000_000

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

    # key -> {"mfe_pct": float, "mae_pct": float} for currently-open positions.
    excursions: dict[tuple, dict] = {}
    records: list[dict] = []
    closed_seen = 0

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        step_candles: dict[str, Candle] = {}
        smc_context_by_symbol: dict[str, dict] = {}

        for sym, candles in symbols_candles.items():
            candle = candles[t]
            step_candles[sym] = candle
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            hist = list(smc.candles)
            if len(hist) >= 3:
                fvg.detect_fvg(candles=hist, index=len(hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context_by_symbol[sym] = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

        # Update excursions for every open position BEFORE the close check --
        # a position closing on this candle still experienced this candle's
        # range, so its final MFE/MAE has to include it.
        for key, pos in ledger.open_positions.items():
            candle = step_candles.get(pos.symbol)
            if candle is None:
                continue
            ex = excursions.setdefault(key, {"mfe_pct": 0.0, "mae_pct": 0.0})
            if pos.direction == "long":
                fav, adv = _pct(candle.high, pos.entry_price), _pct(candle.low, pos.entry_price)
                ex["mfe_pct"] = max(ex["mfe_pct"], fav)
                ex["mae_pct"] = min(ex["mae_pct"], adv)
            else:  # short: favorable = price falls, adverse = price rises
                fav, adv = -_pct(candle.low, pos.entry_price), -_pct(candle.high, pos.entry_price)
                ex["mfe_pct"] = max(ex["mfe_pct"], fav)
                ex["mae_pct"] = min(ex["mae_pct"], adv)

        for sym, candle in step_candles.items():
            opened_at = candle.timestamp.isoformat()
            keys_before = set(ledger.open_positions)
            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            closed_keys = keys_before - set(ledger.open_positions)

            # Pair each newly-closed position with the trade record the ledger
            # just appended, so MFE/MAE lands on the right trade.
            for key in closed_keys:
                ex = excursions.pop(key, None)
                if ex is None:
                    continue
                while closed_seen < len(ledger.closed_trades):
                    trade = ledger.closed_trades[closed_seen]
                    closed_seen += 1
                    if trade["symbol"] == key[0] and trade["market"] == key[1]:
                        records.append({**trade, "mfe_pct": ex["mfe_pct"], "mae_pct": ex["mae_pct"]})
                        break

            insight = build_entry_insight(sym, candle.close, smc_context_by_symbol[sym], gnn_signal=0.0)
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    return records


def _describe(values: list[float], label: str) -> None:
    if not values:
        print(f"    {label}: (no trades)")
        return
    vals = sorted(values)
    n = len(vals)
    def pct(p):
        return vals[min(n - 1, int(n * p))]
    print(f"    {label}: n={n}  median={statistics.median(vals):+.3f}%  "
          f"p75={pct(0.75):+.3f}%  p90={pct(0.90):+.3f}%  worst={vals[0] if vals[0] < 0 else vals[-1]:+.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols")
    print(f"Fetching {days}d of {timeframe} candles...\n", flush=True)

    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles", flush=True)

    if not all_candles:
        print("No data — aborting.")
        return

    print("\nReplaying (real PaperTradingLedger, MFE/MAE observed alongside)...", flush=True)
    records = run(all_candles, timeframe)
    print(f"{len(records)} closed trades recorded.\n", flush=True)

    print("=" * 92)
    print("MFE/MAE EXIT ANALYSIS — is the shared stop geometry wrong for SHORT?")
    print("=" * 92)

    for direction in ("long", "short"):
        d = [r for r in records if r.get("direction") == direction]
        if not d:
            continue
        wins = [r for r in d if r["pnl_dollars"] > 0]
        losses = [r for r in d if r["pnl_dollars"] <= 0]
        wr = len(wins) / len(d) * 100 if d else 0
        print(f"\n{direction.upper()}  (n={len(d)}, win rate {wr:.1f}%)")

        print("  [1] How deep did eventual WINNERS dip against us? (MAE of winners)")
        print("      -> if this approaches the stop distance, the stop is cutting would-be winners")
        _describe([r["mae_pct"] for r in wins], "winners' MAE")

        print("  [2] How much profit did LOSERS show before failing? (MFE of losers)")
        print("      -> high values mean a breakeven/trailing rule could have rescued them")
        _describe([r["mfe_pct"] for r in losses], "losers' MFE")

        print("  [3] Profit left on the table by WINNERS (MFE vs realized)")
        realized = [abs(r["pnl_pct"]) * 100 for r in wins]
        mfes = [r["mfe_pct"] for r in wins]
        if wins:
            print(f"      median MFE={statistics.median(mfes):.3f}%  vs  median realized={statistics.median(realized):.3f}%"
                  f"  (captured {statistics.median(realized)/statistics.median(mfes)*100:.0f}% of the peak)"
                  if statistics.median(mfes) else "      (no MFE)")

    print("\n" + "=" * 92)
    print("Read [1] first: if SHORT winners' MAE sits much closer to the stop than LONG winners',")
    print("the same stop distance is systematically harsher on SHORT — an EXIT-geometry problem,")
    print("not the entry-quality problem every prior experiment in this project assumed.")
    print("=" * 92)


if __name__ == "__main__":
    main()
