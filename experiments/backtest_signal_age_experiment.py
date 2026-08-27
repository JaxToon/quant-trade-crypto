"""
Signal age/decay experiment — Path 1 (SMC/FVG paper trading) ONLY.
Step 3 of the agreed entry-selection research roadmap (see README),
crypto-first per the standing rule.

Hypothesis: does an FVG zone's predictive value change as it ages between
formation and the moment a trade actually opens off it?

METHODOLOGY DISCIPLINE (explicit, carried over from the step 2 lesson):
the age definition and bucket boundaries below are fixed BEFORE this
script is ever run against real data, and will not be adjusted after
seeing a result. The 5 buckets are taken verbatim from the original
external research memo (0-2h, 2-8h, 8-24h, 1-3d, 3-7d) rather than
chosen to fit this project's own data -- notably, the outer edge (7d)
already matches this system's own configured FVG_MAX_AGE_HOURS=168
independently, which is a good sign the memo's buckets weren't arbitrary.
If a later run suggests different boundaries would be "cleaner," that is
a separate, explicitly-labeled refinement -- never a silent edit to this
one, mirroring the step 2 precedent of not collapsing categories after
seeing results.

Purely observational, same as step 2: every baseline trade fires exactly
as entry_insight already decides, unfiltered. This only tags each one
with the AGE of the FVG zone that produced it and asks whether outcome
varies by age bucket -- nothing added or removed from the trade set.

Age source: FairValueGapAnalyzer.get_nearest_fvg(current_price, fvg_type)
-- called here with the EXACT SAME arguments get_trading_signal() itself
uses internally (see fvg_analyzer.py's get_trading_signal, which calls
self.get_nearest_fvg(current_price, "bullish"/"bearish") right before
building the signal dict this script's own build_entry_insight() call
consumes) -- so this deterministically recovers the identical FVGZone
object that produced the trade, including its created_at timestamp,
without modifying fvg_analyzer.py at all.

age = trade's opened_at (candle.timestamp) - zone.created_at

Reuses the same causally-correct scaffolding pattern as
backtest_htf_bias_experiment.py / backtest_concurrence_experiment.py,
kept as its own self-contained script (same anti-premature-abstraction
discipline already agreed -- shared infrastructure gets extracted only
once a third experiment reveals genuine common structure, not before).

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py,
strategy/smart_money.py, strategy/entry_insight.py, or any live code.
Read/replay only. Fully isolated from any live main.py process.

Usage: python backtest_signal_age_experiment.py [--days N] [--segments N] [--timeframe 5m|15m|1h]
(single timeframe per run, unlike the concurrence experiment -- age is a
property of the zone itself, not a cross-timeframe question, and running
one timeframe at a time keeps candle volume manageable)
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

# Fixed before running -- see module docstring. (label, max_hours) pairs,
# lower bound is the previous bucket's max_hours (0 for the first).
AGE_BUCKETS = [
    ("0-2h", 2),
    ("2-8h", 8),
    ("8-24h", 24),
    ("1-3d", 72),
    ("3-7d", 168),
]


def _bucket_for_age(age_hours: float) -> str | None:
    prev_max = 0
    for label, max_hours in AGE_BUCKETS:
        if prev_max <= age_hours < max_hours:
            return label
        prev_max = max_hours
    return None  # older than 7d -- FVG_MAX_AGE_HOURS=168 means this shouldn't occur for active zones


def _split_into_segments(candles: list, n_segments: int) -> list[list]:
    if not candles:
        return [[] for _ in range(n_segments)]
    start, end = candles[0].timestamp, candles[-1].timestamp
    span = (end - start) / n_segments
    bounds = [start + span * i for i in range(n_segments + 1)]
    segments: list[list] = []
    for i in range(n_segments):
        lo, hi = bounds[i], bounds[i + 1]
        seg = [c for c in candles if lo <= c.timestamp < hi] if i < n_segments - 1 else [c for c in candles if lo <= c.timestamp <= hi]
        segments.append(seg)
    return segments


def replay(symbols_candles: dict[str, list], timeframe: str) -> tuple[PaperTradingLedger, dict[str, str]]:
    """Cold-started per call. symbols_candles: {symbol: [candles]} for ONE
    timeframe. Returns the ledger (baseline trades, completely unfiltered)
    plus a trade_marker -> age_bucket map."""
    events: list[tuple] = []
    for symbol, candles in symbols_candles.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(path="backtrack_logs/_signal_age_scratch.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    age_bucket_by_trade: dict[str, str] = {}

    for timestamp, symbol, candle in events:
        smc, fvg = smc_by_symbol[symbol], fvg_by_symbol[symbol]
        smc_signals = smc.update(candle)
        candle_hist = list(smc.candles)
        if len(candle_hist) >= 3:
            fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
        fvg.update_all(candle.close)
        ms_trend = smc_signals.get("market_structure", "unknown")
        fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
        smc_context = {**smc_signals, **fvg_signal, "timestamp": timestamp.isoformat()}

        ledger.check_and_close_ohlc(symbol, candle.low, candle.high, closed_at=timestamp.isoformat(), timeframe=timeframe)

        insight = build_entry_insight(symbol, candle.close, smc_context, gnn_signal=0.0)
        opened_at = timestamp.isoformat()
        direction = insight.get("direction", "neutral")

        before_open = set(ledger.open_positions.keys())
        ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
        ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)
        newly_opened = set(ledger.open_positions.keys()) - before_open

        if newly_opened and direction in ("long", "short"):
            fvg_type = "bullish" if direction == "long" else "bearish"
            # Same call, same arguments, get_trading_signal() already made
            # internally -- deterministically recovers the identical zone.
            zone = fvg.get_nearest_fvg(candle.close, fvg_type)
            if zone is not None:
                age_hours = (timestamp - zone.created_at).total_seconds() / 3600.0
                bucket = _bucket_for_age(age_hours)
                if bucket is not None:
                    for pos_key in newly_opened:
                        trade_marker = f"{pos_key[0]}|{pos_key[1]}|{pos_key[2]}|{opened_at}"
                        age_bucket_by_trade[trade_marker] = bucket

    ledger.open_positions.clear()  # unresolved at segment end -- not counted
    return ledger, age_bucket_by_trade


def _score(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    equity = 10_000.0
    peak = equity
    max_dd_pct = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100)
    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "expectancy_dollars": total_pnl / n,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal age/decay: does FVG zone age at entry condition baseline trade outcome? Purely observational. (Path 1 only)")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--timeframe", choices=list(YF_MAX_DAYS.keys()), default="15m")
    args = parser.parse_args()
    timeframe = args.timeframe
    days = min(args.days if args.days is not None else YF_MAX_DAYS[timeframe], YF_MAX_DAYS[timeframe])
    n_segments = max(2, args.segments)

    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    symbols = list(tickers.keys())
    bucket_labels = [b[0] for b in AGE_BUCKETS]
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — signal age/decay over {days}d of {timeframe}, {n_segments} non-overlapping segments, symbols: {symbols}")
    print(f"Age buckets (fixed before running, from the original research memo): {bucket_labels}\n")

    all_candles: dict[str, list] = {}
    for symbol in symbols:
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles")
    print()

    per_symbol_segments = {sym: _split_into_segments(candles, n_segments) for sym, candles in all_candles.items()}

    agg_by_bucket: dict[str, list[dict]] = {b: [] for b in bucket_labels}
    for seg_idx in range(n_segments):
        seg_candles = {sym: per_symbol_segments[sym][seg_idx] for sym in all_candles if per_symbol_segments[sym][seg_idx]}
        n_candles = sum(len(c) for c in seg_candles.values())
        print(f"segment {seg_idx + 1}/{n_segments}: {n_candles} candles")
        ledger, age_bucket_by_trade = replay(seg_candles, timeframe)

        by_bucket: dict[str, list[dict]] = {b: [] for b in bucket_labels}
        unclassified = 0
        for t in ledger.closed_trades:
            marker = f"{t['symbol']}|{'spot' if t['market'] == 'spot' else 'derivative'}|{t['timeframe']}|{t['opened_at']}"
            bucket = age_bucket_by_trade.get(marker)
            if bucket is None:
                unclassified += 1
                continue
            by_bucket[bucket].append(t)
            agg_by_bucket[bucket].append(t)

        header = f"  {'Bucket':<10}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
        print(header)
        for b in bucket_labels:
            s = _score(by_bucket[b])
            if s["trades"] == 0:
                print(f"  {b:<10}{'0':<8}(no trades)")
                continue
            print(f"  {b:<10}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")
        if unclassified:
            print(f"  ({unclassified} closed trades unclassified -- no FVG zone found at open, or neutral direction; expected, not an error)")
        print()

    print("=" * 78)
    print(f"AGGREGATE ACROSS ALL {n_segments} SEGMENTS (pooled, for a larger per-bucket sample)")
    print("=" * 78)
    header = f"{'Bucket':<10}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(header)
    for b in bucket_labels:
        s = _score(agg_by_bucket[b])
        if s["trades"] == 0:
            print(f"{b:<10}{'0':<8}(no trades)")
            continue
        print(f"{b:<10}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or any live strategy code regardless of which bucket wins.")
    print("No trades were added or removed from the baseline set; this only conditions the SAME trades by zone age.")


if __name__ == "__main__":
    main()
