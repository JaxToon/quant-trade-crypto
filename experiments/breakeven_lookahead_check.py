"""
Is the BE_0.5R breakeven "edge" real, or intra-candle look-ahead bias?

The tell that prompted this: in `breakeven_deep_validation.py` the
parameter sweep improved MONOTONICALLY as the trigger got tighter, on BOTH
PnL and drawdown, with no peak --

    BE_0.6R +$463,701 / 49.1% DD   ... BE_0.3R +$694,508 / 18.3% DD

-- and the scratch share climbed to 63%. A real edge has an optimum; "the
tighter the better, forever" is the signature of a simulation artifact.

The suspected artifact, confirmed by reading `check_and_close_ohlc`: within
ONE candle it updates the stop from the candle's FAVORABLE extreme first,
then tests the ADVERSE extreme against the already-tightened stop:

    _maybe_update_trailing_stop(pos, high if long else low)   # favourable first
    hit_sl = low <= effective_stop                            # then adverse

i.e. it assumes the favourable move happened BEFORE the adverse one, inside
a bar whose true path OHLC cannot reveal. The codebase already documents
this for trailing ("could in principle overstate how far a trail tightened
within one bar — an inherent limit of OHLC-only data") and already refuses
the same optimism elsewhere ("If both SL and TP fall inside the same
candle's range, conservatively assume SL was hit first — avoids an
optimistic look-ahead bias"). A breakeven rule triggering at 0.3-0.5R is
hit by this far harder than trailing is, because a far larger share of
trades reach that small a move.

The honest test: allow the breakeven stop to move ONLY on information from
candles that have already fully closed — never from the same bar being
evaluated. No intra-bar ordering assumption at all.

  BASELINE            no breakeven
  BE_0.5R_OPTIMISTIC  same-bar trigger (what the earlier runs measured)
  BE_0.5R_LAGGED      trigger from prior completed bars only  <- the honest one
  BE_0.3R_OPTIMISTIC  the most extreme same-bar case, for scale
  BE_0.3R_LAGGED      its honest counterpart

If LAGGED keeps most of the advantage, the effect is real and deployable.
If the advantage largely disappears, the whole finding was look-ahead bias
and must NOT be deployed -- and the same doubt then extends to the
production trailing implementation, which shares the ordering.

Usage: python -m experiments.breakeven_lookahead_check [--segments 6]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# name -> (activation_R, lagged?)
ARMS: dict[str, tuple | None] = {
    "BASELINE": None,
    "BE_0.5R_OPTIMISTIC": (0.5, False),
    "BE_0.5R_LAGGED": (0.5, True),
    "BE_0.3R_OPTIMISTIC": (0.3, False),
    "BE_0.3R_LAGGED": (0.3, True),
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
            path=f"backtrack_logs/_belook_{label}_{name}.json",
            starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
        )
        led.balance = led.starting_balance
        led.open_positions.clear()
        led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}
    prev_bar: dict[str, tuple[float, float]] = {}  # symbol -> (high, low) of the last COMPLETED bar

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

            for name, spec in ARMS.items():
                led = ledgers[name]
                if spec is not None:
                    activation_r, lagged = spec
                    if lagged:
                        # Only price action from bars that already fully closed
                        # may move the stop -- no same-bar ordering assumption.
                        if prior is not None:
                            _apply_breakeven(led, sym, prior[0], prior[1], activation_r)
                    else:
                        _apply_breakeven(led, sym, candle.high, candle.low, activation_r)
                led.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                led.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                led.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            prev_bar[sym] = (candle.high, candle.low)

    out = {}
    for name, led in ledgers.items():
        led.open_positions.clear()
        out[name] = list(led.closed_trades)
    return out


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

    trades_by_arm: dict[str, list[dict]] = {k: [] for k in ARMS}
    seg_pnl: dict[str, list[float]] = {k: [] for k in ARMS}
    seg_dd: dict[str, list[float]] = {k: [] for k in ARMS}

    for i in range(n_seg):
        seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()}
        res = run_segment(seg, tf, f"seg{i}")
        for name, trades in res.items():
            trades_by_arm[name].extend(trades)
            seg_pnl[name].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[name].append(_max_dd(trades))
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 100}")
    print(f"LOOK-AHEAD CHECK — does the breakeven edge survive honest intra-bar handling? ({days}d, {n_seg} seg)")
    print(f"{'=' * 100}")
    print(f"{'Arm':<22}{'Trades':<9}{'WR':<9}{'Total PnL':<18}{'WorstSegDD':<13}{'Segs beat base':<15}")
    print("-" * 100)
    base = seg_pnl["BASELINE"]
    for name in ARMS:
        ts = trades_by_arm[name]
        wins = sum(1 for t in ts if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in ts)
        beats = sum(1 for a, b in zip(seg_pnl[name], base) if a > b)
        beat_str = "-" if name == "BASELINE" else f"{beats}/{n_seg}"
        wr = wins / len(ts) * 100 if ts else 0
        print(f"{name:<22}{len(ts):<9}{wr:<9.1f}${pnl:>+15,.0f}   {max(seg_dd[name]):<13.1f}{beat_str:<15}")
    print("-" * 100)

    opt = sum(t["pnl_dollars"] for t in trades_by_arm["BE_0.5R_OPTIMISTIC"])
    lag = sum(t["pnl_dollars"] for t in trades_by_arm["BE_0.5R_LAGGED"])
    bas = sum(t["pnl_dollars"] for t in trades_by_arm["BASELINE"])
    if opt > bas:
        retained = (lag - bas) / (opt - bas) * 100
        print(f"\nBE_0.5R: LAGGED retains {retained:.0f}% of the OPTIMISTIC edge over baseline.")
        print("Near 100% -> the effect is real. Near 0% (or negative) -> it was look-ahead bias,")
        print("do NOT deploy, and the production trailing implementation inherits the same doubt.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
