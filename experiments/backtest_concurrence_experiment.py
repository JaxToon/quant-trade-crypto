"""
Timeframe concurrence experiment — Path 1 (SMC/FVG paper trading) ONLY.
Step 2 of the agreed entry-selection research roadmap (see README), run
crypto-first per the standing rule: crypto establishes or disproves the
hypothesis, stock only tests transferability.

Deliberately NOT a repeat of the HTF bias experiment's design. That one
imposed a hard veto (block a 5m/15m entry if 1h structure disagrees) --
a decision intervention. This one changes nothing about which trades are
taken. Every baseline trade fires exactly as it already does, unfiltered;
this only TAGS each one with the OTHER two timeframes' current structure
at that moment and asks whether outcome varies by that tag. Purely
observational, per the explicit framing agreed before building this:

    "When independently generated 5m/15m/1h SMC states agree, conflict,
     or partially agree, how does that condition the outcome of the
     baseline trade?"

not

    "should 1h control 15m?"

Reuses backtest_htf_bias_experiment.py's causally-correct scaffolding
pattern (merged multi-timeframe event queue, incrementally-updated
per-(symbol,timeframe) analyzer state, non-overlapping walk-forward
segments with cold-started state) rather than importing it as a library
-- kept as its own self-contained script, per the explicit pushback
already agreed on generic-harness-before-real-experiments (this
project's Section 9 critique: build the harness once two real
experiments reveal a genuine common structure, not speculatively before
either exists).

Concurrence state source: each layer's SmartMoneyAnalyzer.market_structure
.trend ("bullish"/"bearish"/"unknown") for the same symbol -- the same
field backtest_htf_bias_experiment.py already validated as causally
correct to track across timeframes (only updated when that (symbol, tf)'s
own event fires, so a lower-timeframe trade only ever sees a higher- or
same-timeframe structure computed from data that had actually closed by
that moment).

For every baseline trade (on any of the 3 timeframes, taken exactly as
entry_insight already decides -- nothing added, nothing removed), records
the OTHER two timeframes' current structure and classifies:
  full_agreement    - both other layers' structure matches the trade's
                       own direction (bullish~long, bearish~short)
  partial_agreement - exactly one other layer matches, the other is
                       "unknown" (no signal yet, not a disagreement)
  no_context         - both other layers are "unknown"
  partial_conflict   - one other layer actively opposes, the other
                       matches or is unknown
  full_conflict      - both other layers actively oppose

Then groups the SAME baseline trade set by concurrence class and compares
win rate / expectancy / drawdown / trade count per class -- a pure
segmentation of already-taken trades, not a new filter.

Walk-forward: same non-overlapping-segment discipline as the HTF bias
experiment (that one's first version compared 30d-vs-60d nested windows
and wrongly called them independent -- not repeating that here).

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py,
strategy/smart_money.py, strategy/entry_insight.py, or any live code.
Read/replay only. Fully isolated from any live main.py process.

Usage: python backtest_concurrence_experiment.py [--days N] [--segments N]
(always all 3 timeframes at once, crypto by default per ASSET_TYPE)
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

TIMEFRAMES = settings.BINANCE_SMC_TIMEFRAMES  # e.g. ["5m", "15m", "1h"]


def _fetch_all(symbols: list[str], days: int) -> dict[tuple[str, str], list]:
    candles_by_symbol_tf: dict[tuple[str, str], list] = {}
    for symbol in symbols:
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, days, tf)
            if candles:
                candles_by_symbol_tf[(symbol, tf)] = candles
    return candles_by_symbol_tf


def _split_into_segments(candles_by_symbol_tf: dict, n_segments: int) -> list[dict]:
    """Splits by CALENDAR TIME, not candle index -- 5m/15m/1h have very
    different candle counts for the same days, so index-slicing would
    misalign the date range each timeframe's segment i actually covers."""
    all_ts = [c.timestamp for candles in candles_by_symbol_tf.values() for c in candles]
    start, end = min(all_ts), max(all_ts)
    span = (end - start) / n_segments
    bounds = [start + span * i for i in range(n_segments + 1)]

    segments: list[dict] = [dict() for _ in range(n_segments)]
    for key, candles in candles_by_symbol_tf.items():
        for i in range(n_segments):
            lo, hi = bounds[i], bounds[i + 1]
            seg_candles = [c for c in candles if lo <= c.timestamp < hi] if i < n_segments - 1 else [c for c in candles if lo <= c.timestamp <= hi]
            if seg_candles:
                segments[i][key] = seg_candles
    return segments


def _classify_concurrence(direction: str, other_structures: list[str]) -> str:
    """direction: the trade's own "long"/"short". other_structures: the
    OTHER two timeframes' current market_structure ("bullish"/"bearish"/
    "unknown"), NOT including the trade's own timeframe."""
    own = "bullish" if direction == "long" else "bearish"
    opposite = "bearish" if direction == "long" else "bullish"
    matches = sum(1 for s in other_structures if s == own)
    opposes = sum(1 for s in other_structures if s == opposite)
    unknown = sum(1 for s in other_structures if s == "unknown")

    if matches == 2:
        return "full_agreement"
    if opposes == 2:
        return "full_conflict"
    if unknown == 2:
        return "no_context"
    if opposes >= 1:
        return "partial_conflict"
    return "partial_agreement"  # matches==1, other is unknown


def replay(candles_by_symbol_tf: dict) -> tuple[PaperTradingLedger, dict[str, str]]:
    """Cold-started per call. Returns the ledger (baseline trades,
    completely unfiltered -- identical to what live/backtest already
    produces) plus a trade_id -> concurrence_class map built alongside it,
    without ever feeding that classification back into which trades open."""
    events: list[tuple] = []
    for (symbol, tf), candles in candles_by_symbol_tf.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, tf, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(path="backtrack_logs/_concurrence_scratch.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

    smc_by_key = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in candles_by_symbol_tf}
    fvg_by_key = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in candles_by_symbol_tf}
    latest_structure: dict[tuple[str, str], str] = {}
    # trade_key (symbol|market|timeframe|opened_at) -> concurrence class at
    # the moment it opened. Keyed loosely since PaperTradingLedger doesn't
    # expose a stable trade id -- opened_at is unique enough per (symbol, tf).
    concurrence_by_trade: dict[str, str] = {}

    for timestamp, symbol, tf, candle in events:
        key = (symbol, tf)
        ledger.check_and_close_ohlc(symbol, candle.low, candle.high, closed_at=timestamp.isoformat(), timeframe=tf)

        smc, fvg = smc_by_key[key], fvg_by_key[key]
        smc_signals = smc.update(candle)
        candle_hist = list(smc.candles)
        if len(candle_hist) >= 3:
            fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
        fvg.update_all(candle.close)
        ms_trend = smc_signals.get("market_structure", "unknown")
        latest_structure[key] = ms_trend
        fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
        smc_context = {**smc_signals, **fvg_signal, "timestamp": timestamp.isoformat()}

        insight = build_entry_insight(symbol, candle.close, smc_context, gnn_signal=0.0)
        opened_at = timestamp.isoformat()
        direction = insight.get("direction", "neutral")

        before_open = set(ledger.open_positions.keys())
        ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=tf, opened_at=opened_at)
        ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=tf, opened_at=opened_at)
        newly_opened = set(ledger.open_positions.keys()) - before_open

        if newly_opened and direction in ("long", "short"):
            other_structures = [latest_structure.get((symbol, other_tf), "unknown") for other_tf in TIMEFRAMES if other_tf != tf]
            cls = _classify_concurrence(direction, other_structures)
            for pos_key in newly_opened:
                trade_marker = f"{pos_key[0]}|{pos_key[1]}|{pos_key[2]}|{opened_at}"
                concurrence_by_trade[trade_marker] = cls

    ledger.open_positions.clear()  # unresolved at segment end -- not counted
    return ledger, concurrence_by_trade


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


CLASSES = ["full_agreement", "partial_agreement", "no_context", "partial_conflict", "full_conflict"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeframe concurrence: does 5m/15m/1h agreement condition baseline trade outcome? Purely observational -- no trades added or removed. (Path 1 only)")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--segments", type=int, default=3)
    args = parser.parse_args()
    days = min(args.days if args.days is not None else YF_MAX_DAYS["5m"], YF_MAX_DAYS["5m"])
    n_segments = max(2, args.segments)

    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    symbols = list(tickers.keys())
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — timeframe concurrence over {days}d in {n_segments} non-overlapping segments, symbols: {symbols}")
    print(f"Timeframes: {TIMEFRAMES} (observational only -- baseline trade set is identical to unfiltered SMC/FVG, nothing added or removed)\n")

    full = _fetch_all(symbols, days)
    segments = _split_into_segments(full, n_segments)

    agg_by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for seg_idx, seg in enumerate(segments):
        n_candles = sum(len(c) for c in seg.values())
        print(f"segment {seg_idx + 1}/{n_segments}: {n_candles} candles")
        ledger, concurrence_by_trade = replay(seg)

        by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
        unclassified = 0
        for t in ledger.closed_trades:
            marker = f"{t['symbol']}|{'spot' if t['market'] == 'spot' else 'derivative'}|{t['timeframe']}|{t['opened_at']}"
            cls = concurrence_by_trade.get(marker)
            if cls is None:
                unclassified += 1
                continue
            by_class[cls].append(t)
            agg_by_class[cls].append(t)

        header = f"  {'Class':<18}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
        print(header)
        for cls in CLASSES:
            s = _score(by_class[cls])
            if s["trades"] == 0:
                print(f"  {cls:<18}{'0':<8}(no trades)")
                continue
            print(f"  {cls:<18}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")
        if unclassified:
            print(f"  ({unclassified} closed trades unclassified -- opened before this segment, from a prior segment's warmup, or neutral-direction; expected, not an error)")
        print()

    print("=" * 78)
    print(f"AGGREGATE ACROSS ALL {n_segments} SEGMENTS (pooled, for a larger per-class sample)")
    print("=" * 78)
    header = f"{'Class':<18}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(header)
    for cls in CLASSES:
        s = _score(agg_by_class[cls])
        if s["trades"] == 0:
            print(f"{cls:<18}{'0':<8}(no trades)")
            continue
        print(f"{cls:<18}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or any live strategy code regardless of which class wins.")
    print("No trades were added or removed from the baseline set; this only conditions the SAME trades by context.")


if __name__ == "__main__":
    main()
