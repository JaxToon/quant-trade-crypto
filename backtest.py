"""
Historical backtest: replays real closed candles through the same SMC/FVG
entry-insight pipeline main.py uses live, and scores every recommendation
against what actually happened next — giving an immediate win-rate estimate
instead of waiting days for main.py to accumulate live paper trades.

Data source is yfinance, not Binance klines: this sandbox's network can't
reach production api.binance.com, and Binance's testnet only retains a few
days of kline history — nowhere near enough for a real backtest. Yahoo
Finance serves up to 60 days of 5-minute candles, which is what we use.

Splits the fetched window in half and replays each half through its own
ledger (in-sample = first half, out-of-sample = second half) — a single
combined win rate can't tell you whether the edge holds up on a period the
strategy wasn't eyeballed against, or whether it was just one lucky window.

GNN confluence is NOT applied here: entry_insight's GNN-agreement boost/veto
needs a live GNN signal, and there's no point-in-time GNN snapshot to replay
against historical candles (the graph doesn't have historical state). This
backtest measures pure SMC/FVG signal quality; the GNN confluence effect can
only be evaluated going forward, via main.py's live paper trading.

Uses a separate ledger file (backtrack_logs/backtest_ledger.json) so it
never touches the live paper-trading ledger main.py maintains.

Usage: python backtest.py [--days N]  (N is capped at 60 — Yahoo's 5m limit)
"""
from __future__ import annotations

import argparse
import logging

import yfinance as yf

from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

logging.basicConfig(level=logging.WARNING)  # keep noise down; we print our own summary
logger = logging.getLogger("backtest")

YF_MAX_DAYS_5M = 60  # Yahoo Finance's hard limit for 5-minute-interval history

# Binance symbol -> yfinance ticker (kept in sync with train_gnn.py's map;
# not imported from there directly to avoid pulling in torch/GNN at import time).
SYMBOL_TICKERS = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    "SOLUSDT": ["SOL-USD", "SOL1-USD"],
    "BNBUSDT": ["BNB-USD", "BNB-USDT"],
    "XRPUSDT": ["XRP-USD"],
    "ADAUSDT": ["ADA-USD"],
    "AVAXUSDT": ["AVAX-USD"],
    "DOTUSDT": ["DOT-USD"],
}


def fetch_candles(symbol: str, days: int) -> list[Candle]:
    tickers = SYMBOL_TICKERS.get(symbol, [])
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{days}d", interval="5m", progress=False, auto_adjust=True)
            if df.empty:
                continue

            opens = df["Open"].values.flatten().tolist()
            highs = df["High"].values.flatten().tolist()
            lows = df["Low"].values.flatten().tolist()
            closes = df["Close"].values.flatten().tolist()
            volumes = df["Volume"].values.flatten().tolist() if "Volume" in df.columns else [0.0] * len(closes)
            timestamps = [ts.to_pydatetime().replace(tzinfo=None) for ts in df.index]

            return [
                Candle(timestamp=timestamps[i], open=opens[i], high=highs[i], low=lows[i],
                       close=closes[i], volume=volumes[i])
                for i in range(len(closes))
            ]
        except Exception as exc:
            logger.warning("%s (%s): fetch failed — %s", symbol, ticker, exc)
            continue
    return []


def replay_symbol(symbol: str, candles: list[Candle], ledger: PaperTradingLedger) -> int:
    """
    Walks candles in order. At each closed candle: first check open paper
    positions against that candle's [low, high] range (so a position opened
    on candle t is only ever evaluated starting candle t+1 — no same-bar
    look-ahead), THEN compute a fresh entry_insight and maybe open a new one.
    """
    smc = SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES)
    fvg = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)

    n_candles = 0
    for candle in candles:
        # 1. Resolve any open positions against this candle's range FIRST.
        ledger.check_and_close_ohlc(symbol, candle.low, candle.high, closed_at=candle.timestamp.isoformat())

        # 2. Update SMC/FVG state with this now-closed candle.
        smc_signals = smc.update(candle)
        candle_hist = list(smc.candles)
        if len(candle_hist) >= 3:
            fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
        fvg.update_all(candle.close)

        ms_trend = smc_signals.get("market_structure", "unknown")
        fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
        smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

        # 3. Compute the same recommendation the live dashboard would show,
        #    and let the ledger open a position off it (next candle onward).
        #    gnn_signal=0.0: no point-in-time GNN snapshot exists for
        #    historical candles — see module docstring.
        insight = build_entry_insight(symbol, candle.close, smc_context, gnn_signal=0.0)
        ledger.maybe_open(symbol, "spot", insight["spot"])
        ledger.maybe_open(symbol, "derivative", insight["derivative"])

        n_candles += 1
    return n_candles


def run_replay(symbols_candles: dict[str, list[Candle]], path: str) -> tuple[PaperTradingLedger, int, int]:
    ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    # The live ledger's 1000-trade cap would silently drop early trades from
    # a full historical replay — a backtest wants the true, untrimmed count.
    ledger.max_closed_trades = 1_000_000

    total_candles = 0
    for symbol, candles in symbols_candles.items():
        total_candles += replay_symbol(symbol, candles, ledger)

    # Anything still open at the end of the window never resolved — drop it
    # from the stats rather than counting it as a phantom win or loss.
    open_count = len(ledger.open_positions)
    ledger.open_positions.clear()
    ledger.save()
    return ledger, total_candles, open_count


def print_report(title: str, ledger: PaperTradingLedger, total_candles: int, open_count: int) -> None:
    stats = ledger.stats()
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Candles replayed : {total_candles}")
    print(f"Starting balance : ${stats['starting_balance']:,.2f}")
    print(f"Ending balance   : ${stats['balance']:,.2f}  ({stats['total_return_pct']:+.2f}%)")
    print(f"Unresolved at end: {open_count} (excluded from win rate)")
    print()
    for label, key in [("OVERALL", "overall"), ("SPOT", "spot"), ("DERIVATIVE", "derivative")]:
        b = stats[key]
        wr = f"{b['win_rate_pct']:.1f}%" if b["win_rate_pct"] is not None else "—"
        print(f"{label:<11} trades={b['trades']:<5} wins={b['wins']:<5} losses={b['losses']:<5} "
              f"win_rate={wr:<7} pnl=${b['total_pnl_dollars']:+.2f}")
    if stats["overall"]["trades"] < 20:
        print("Sample size is small — treat this win rate as directional, not conclusive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest entry_insight recommendations on historical candles.")
    parser.add_argument("--days", type=int, default=min(settings.BACKTEST_DAYS, YF_MAX_DAYS_5M))
    args = parser.parse_args()
    days = min(args.days, YF_MAX_DAYS_5M)
    if args.days > YF_MAX_DAYS_5M:
        print(f"Note: capping to {YF_MAX_DAYS_5M}d — Yahoo Finance doesn't serve 5m candles further back than that.\n")

    print(f"Fetching {days}d of 5m candles across {len(settings.WATCHLIST)} symbols (via yfinance)...\n")

    first_half: dict[str, list] = {}
    second_half: dict[str, list] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days)
        if not candles:
            print(f"  {symbol}: no historical data, skipping")
            continue
        mid = len(candles) // 2
        first_half[symbol] = candles[:mid]
        second_half[symbol] = candles[mid:]
        print(f"  {symbol}: {len(candles)} candles ({candles[0].timestamp} -> {candles[-1].timestamp})"
              f" — split at {candles[mid].timestamp}")

    ledger_is, candles_is, open_is = run_replay(first_half, "backtrack_logs/backtest_ledger_insample.json")
    ledger_oos, candles_oos, open_oos = run_replay(second_half, "backtrack_logs/backtest_ledger_oos.json")

    print_report(f"IN-SAMPLE (first half, ~{days // 2}d)", ledger_is, candles_is, open_is)
    print_report(f"OUT-OF-SAMPLE (second half, ~{days - days // 2}d)", ledger_oos, candles_oos, open_oos)

    wr_is = ledger_is.stats()["overall"]["win_rate_pct"]
    wr_oos = ledger_oos.stats()["overall"]["win_rate_pct"]
    print(f"\n{'=' * 60}")
    if wr_is is not None and wr_oos is not None:
        drift = wr_oos - wr_is
        print(f"Win rate: in-sample {wr_is:.1f}% -> out-of-sample {wr_oos:.1f}%  ({drift:+.1f}pp)")
        if abs(drift) > 15:
            print("Large drift between halves — treat the edge as unstable, not confirmed.")
        else:
            print("Win rate held up reasonably well across both halves.")
    print(f"{'=' * 60}")
    print("\nNote: pnl_dollars assumes each trade's $ risk is independent of every other")
    print("open position — it is NOT a realistic capital-constrained return projection.")
    print("Win rate is the number to trust here; the dollar figures are directional.")
    print("\nFull trade logs: backtrack_logs/backtest_ledger_insample.json, backtest_ledger_oos.json")


if __name__ == "__main__":
    main()
