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

By default gnn_signal is 0.0 for every symbol (pure SMC/FVG signal quality,
no Neo4j/torch touched) — pass --with-gnn to layer in a real point-in-time
GNN signal instead, recomputed at every candle from data available up to
that point and run through the same trained model used live (see
replay_lockstep and GNNTrainer.predict_from_features).

All symbols are replayed in true lockstep (aligned by relative candle
index), not one-symbol-fully-then-the-next — the portfolio correlated-risk
cap needs genuine cross-symbol concurrency to evaluate correctly, and an
earlier sequential-per-symbol design silently couldn't provide that: on an
identical 30d/1h window it found 25 trades (689 blocked by the correlation
cap, out of 2087 attempts) vs lockstep's 53 (551 blocked) — same signals,
same candles, ~2x different trade count from execution order alone. See
replay_lockstep's docstring for the full story.

Uses separate, asset-type-tagged ledger files (backtrack_logs/backtest_ledger_*_{crypto,stock}.json)
so it never touches the live paper-trading ledger main.py maintains.

Usage: python backtest.py [--days N] [--timeframe 5m|15m|1h] [--segments N] [--with-gnn]
Days are capped per timeframe by Yahoo's own limits: 60d for 5m/15m,
730d for 1h — see YF_MAX_DAYS below. --segments splits the fetched window
into N equal chunks and reports each one separately (default 2, i.e. the
original in-sample/out-of-sample split) — useful for checking whether an
edge measured over a longer 1h window actually holds up across more than
one market regime, instead of a single lucky stretch.
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

# Yahoo Finance's hard per-interval history limits (sub-daily intervals are
# far more restricted than 1h+ — confirmed empirically, not just documented).
YF_MAX_DAYS = {"5m": 60, "15m": 60, "1h": 730}
YF_MAX_DAYS_5M = YF_MAX_DAYS["5m"]  # kept for backwards compat within this file

# Binance symbol -> yfinance ticker (kept in sync with train_gnn.py's map;
# not imported from there directly to avoid pulling in torch/GNN at import time).
CRYPTO_TICKERS = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    "SOLUSDT": ["SOL-USD", "SOL1-USD"],
    "BNBUSDT": ["BNB-USD", "BNB-USDT"],
    "XRPUSDT": ["XRP-USD"],
    "ADAUSDT": ["ADA-USD"],
    "AVAXUSDT": ["AVAX-USD"],
    "DOTUSDT": ["DOT-USD"],
}
STOCK_TICKERS = {sym: [sym] for sym in settings.STOCK_WATCHLIST}
SYMBOL_TICKERS = STOCK_TICKERS if settings.ASSET_TYPE == "stock" else CRYPTO_TICKERS


def fetch_candles(symbol: str, days: int, timeframe: str) -> list[Candle]:
    tickers = SYMBOL_TICKERS.get(symbol, [])
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{days}d", interval=timeframe, progress=False, auto_adjust=True)
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


def replay_lockstep(
    symbols_candles: dict[str, list[Candle]], ledger: PaperTradingLedger, timeframe: str, trainer=None,
) -> int:
    """
    Steps ALL symbols in true lockstep (aligned by relative candle index, not
    calendar time — the same simplification train_gnn.py already accepts),
    processing every symbol's close-check and open-attempt for a given step
    before moving to the next step.

    This matters beyond just enabling a real GNN signal (see trainer below):
    replay_symbol's original per-symbol-sequential design (replay ALL of
    BTCUSDT fully, THEN all of ETHUSDT, etc.) meant the portfolio correlated-
    risk cap in paper_trading.py could only ever see ONE symbol's own
    positions at a time — cross-symbol correlated exposure was structurally
    invisible to it, even though main.py's live event loop processes all
    symbols' real-time ticks genuinely concurrently. Measured on identical
    30d/1h data: the old per-symbol-sequential design found 25 trades (689
    rejected by the correlation cap, out of 2087 actionable attempts); this
    lockstep replay found 53 (551 rejected) on the exact same candles, exact
    same signals — a ~2x difference from execution order alone, not signal
    quality. run_replay uses this engine exclusively now.

    trainer: optional GNNTrainer. If given, RSI/MACD/Bollinger + SMC state
    are recomputed from data available up to that candle only (no look-
    ahead) and run through the SAME trained model used live, feeding a real
    point-in-time GNN signal into build_entry_insight. If None (the default,
    and what every non-GNN backtest run uses), gnn_signal is 0.0 for every
    symbol and no Neo4j/torch is touched at all — same behavior as the old
    hardcoded-0.0 path, just correctly lockstepped. Graph topology (edges,
    Protocol/Macro_Event features) still comes from the live Neo4j graph and
    is held fixed across the whole replay when a trainer IS given — there's
    no historical snapshot of those to replay against either.
    """
    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    closes_by_symbol: dict[str, list[float]] = {sym: [] for sym in symbols_candles}

    if trainer is not None:
        from train_gnn import _rsi, _macd, _bollinger, _encode_market_structure  # local: keeps the no-GNN path torch-free

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    n_candles = 0
    for t in range(min_len):
        step_candles: dict[str, Candle] = {}
        smc_context_by_symbol: dict[str, dict] = {}

        for sym, candles in symbols_candles.items():
            candle = candles[t]
            step_candles[sym] = candle
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            candle_hist = list(smc.candles)
            if len(candle_hist) >= 3:
                fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context_by_symbol[sym] = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}
            if trainer is not None:
                closes_by_symbol[sym].append(candle.close)

        if trainer is not None:
            # Build the same 18-dim feature vector (TOKEN_FEATURES order) live
            # inference feeds the model, for every symbol at this exact step —
            # a joint forward pass is required since the GNN message-passes
            # across all Token nodes at once.
            features: dict[str, list[float]] = {}
            for sym, candle in step_candles.items():
                closes = closes_by_symbol[sym]
                rsi = _rsi(closes)
                macd, macd_signal, macd_hist = _macd(closes)
                bb_upper, bb_mid, bb_lower = _bollinger(closes)
                smc_signals = smc_context_by_symbol[sym]
                tech = [candle.close, candle.volume, rsi, macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower]
                smc_feats = [
                    smc_signals.get("smc_signal", 0.0),
                    _encode_market_structure(smc_signals.get("market_structure", "unknown")),
                    float(smc_signals.get("active_order_blocks", 0)),
                    float(smc_signals.get("active_fvgs", 0)),
                    float(smc_signals.get("liquidity_zones", 0)),
                    1.0 if smc_signals.get("in_bullish_ob") else 0.0,
                    1.0 if smc_signals.get("in_bearish_ob") else 0.0,
                    1.0 if smc_signals.get("in_bullish_fvg") else 0.0,
                    1.0 if smc_signals.get("in_bearish_fvg") else 0.0,
                ]
                features[sym] = tech + smc_feats
            gnn_signals = trainer.predict_from_features(features)
        else:
            gnn_signals = {sym: 0.0 for sym in step_candles}

        for sym, candle in step_candles.items():
            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context_by_symbol[sym], gnn_signal=gnn_signals.get(sym, 0.0))
            opened_at = candle.timestamp.isoformat()
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

        n_candles += len(step_candles)
    return n_candles


def run_replay(
    symbols_candles: dict[str, list[Candle]], path: str, timeframe: str,
    slippage_pct: float = 0.0, persist: bool = True, trainer=None,
) -> tuple[PaperTradingLedger, int, int]:
    """
    The default replay path — lockstep across all symbols (see
    replay_lockstep for why this matters: the portfolio correlation cap
    needs genuine cross-symbol concurrency to be evaluated correctly, and
    the old per-symbol-sequential design couldn't provide that). trainer is
    None by default (gnn_signal=0.0 for everyone, no Neo4j/torch touched) —
    pass a loaded GNNTrainer to layer in a real historical GNN signal.
    """
    ledger = PaperTradingLedger(
        path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE, slippage_pct=slippage_pct,
    )
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    # The live ledger's 1000-trade cap would silently drop early trades from
    # a full historical replay — a backtest wants the true, untrimmed count.
    ledger.max_closed_trades = 1_000_000

    total_candles = replay_lockstep(symbols_candles, ledger, timeframe, trainer=trainer)

    # Anything still open at the end of the window never resolved — drop it
    # from the stats rather than counting it as a phantom win or loss.
    open_count = len(ledger.open_positions)
    ledger.open_positions.clear()
    if persist:
        ledger.save()
    return ledger, total_candles, open_count


def run_gnn_ablation(symbols_candles: dict[str, list[Candle]], timeframe: str) -> None:
    """
    Runs the SAME full window through both paths — SMC-only (gnn_signal
    0.0) and SMC+GNN (a real point-in-time GNN signal recomputed at every
    candle from data available up to that point) — and diffs the results,
    to answer whether GNN confluence adds real edge over SMC/FVG alone or
    is just adding noise on top of it.

    Both paths use the SAME replay_lockstep engine, differing ONLY in
    whether a trainer is passed — this used to compare the sequential
    SMC-only path against the lockstep SMC+GNN path, which was confounding
    "does GNN help" with "does lockstep vs sequential execution order
    change the correlation cap's behavior" (it does, substantially — see
    replay_lockstep's docstring). That confound is why an earlier run of
    this comparison showed a 60.0% vs 62.3% win-rate gap that vanished
    entirely (62.3% vs 62.3%, identical trade count) once execution order
    was held constant and only the GNN signal was varied.
    """
    from gnn.trainer import GNNTrainer

    trainer = GNNTrainer()
    if not trainer.load():
        print(f"\nNo trained GNN checkpoint found for ASSET_TYPE={settings.ASSET_TYPE} — "
              "run train_gnn.py first. Skipping ablation.")
        return

    smc_path = f"backtrack_logs/backtest_ledger_ablation_smc_{settings.ASSET_TYPE}_{timeframe}.json"
    gnn_path = f"backtrack_logs/backtest_ledger_ablation_gnn_{settings.ASSET_TYPE}_{timeframe}.json"

    ledger_smc, candles_smc, open_smc = run_replay(symbols_candles, smc_path, timeframe, trainer=None)
    ledger_gnn, candles_gnn, open_gnn = run_replay(symbols_candles, gnn_path, timeframe, trainer=trainer)

    print_report(f"SMC-ONLY (GNN disabled) [{timeframe}]", ledger_smc, candles_smc, open_smc)
    print_report(f"SMC + GNN confluence (real historical signal) [{timeframe}]", ledger_gnn, candles_gnn, open_gnn)

    s_smc, s_gnn = ledger_smc.stats()["overall"], ledger_gnn.stats()["overall"]
    print(f"\n{'=' * 60}")
    print("GNN ABLATION SUMMARY")
    print(f"{'=' * 60}")
    wr_smc = s_smc["win_rate_pct"]
    wr_gnn = s_gnn["win_rate_pct"]
    if wr_smc is not None and wr_gnn is not None:
        print(f"Win rate:  SMC-only {wr_smc:.1f}%  vs  SMC+GNN {wr_gnn:.1f}%  ({wr_gnn - wr_smc:+.1f}pp)")
    print(f"Trades:    SMC-only {s_smc['trades']}  vs  SMC+GNN {s_gnn['trades']} "
          f"(GNN confluence can veto or boost signals, so counts can differ)")
    print(f"Total PnL: SMC-only ${s_smc['total_pnl_dollars']:+.2f}  vs  SMC+GNN ${s_gnn['total_pnl_dollars']:+.2f}")
    print(
        "\nCaveat: graph topology (edges, Protocol/Macro_Event features) is the "
        "LIVE graph held fixed across the whole replay, not a historical "
        "snapshot — only each symbol's own technical+SMC features are truly "
        "point-in-time. Treat this as a real but imperfect ablation."
    )
    print(f"{'=' * 60}")


# Adverse slippage applied to both legs of every fill — real friction the
# rest of this backtest doesn't otherwise model (no order-book depth).
SLIPPAGE_SWEEP_LEVELS = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003]


def run_slippage_sweep(symbols_candles: dict[str, list[Candle]], timeframe: str) -> None:
    """
    Runs the full (unsplit) window at increasing slippage assumptions to
    find the level at which the backtested edge breaks even — answers
    "how much real-world friction can this strategy actually tolerate?"
    rather than assuming fills are always exact.
    """
    print(f"\n{'=' * 70}")
    print(f"SLIPPAGE SENSITIVITY SWEEP [{timeframe}]")
    print(f"{'=' * 70}")
    print(f"{'Slippage/fill':<15}{'Trades':<9}{'Win Rate':<11}{'Return %':<12}{'Total PnL $':<14}")
    print("-" * 70)

    breakeven_flagged = False
    prev_slip = 0.0
    for slip in SLIPPAGE_SWEEP_LEVELS:
        ledger, _, _ = run_replay(
            symbols_candles, path="backtrack_logs/_slippage_sweep_scratch.json",
            timeframe=timeframe, slippage_pct=slip, persist=False,
        )
        s = ledger.stats()
        overall = s["overall"]
        wr = f"{overall['win_rate_pct']:.1f}%" if overall["win_rate_pct"] is not None else "—"
        print(
            f"{slip * 100:>6.2f}%        {overall['trades']:<9}{wr:<11}"
            f"{s['total_return_pct']:>+8.2f}%   ${overall['total_pnl_dollars']:>+11.2f}"
        )
        if not breakeven_flagged and overall["total_pnl_dollars"] < 0:
            print(f"  ^ edge turns net-negative somewhere between {prev_slip*100:.2f}% and {slip*100:.2f}% slippage per fill")
            breakeven_flagged = True
        prev_slip = slip

    if not breakeven_flagged:
        print(f"\nEdge stayed net-positive across the whole {SLIPPAGE_SWEEP_LEVELS[-1]*100:.1f}% sweep range.")
    print(
        "\nFor reference: PAPER_TRADING_FEE_PCT is already 0.1%/side (0.2% round trip) on top of this —"
        "\nthis sweep is testing ADDITIONAL slippage beyond that fee, i.e. real order-book/execution friction."
    )


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
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument(
        "--timeframe", choices=settings.BINANCE_SMC_TIMEFRAMES, default=settings.BINANCE_KLINE_INTERVAL,
        help="Which SMC/FVG layer to backtest — run each one separately to compare layers.",
    )
    parser.add_argument(
        "--slippage-sweep", action="store_true",
        help="Instead of the in-sample/out-of-sample report, run the full window at increasing "
             "slippage assumptions to find where the edge breaks even.",
    )
    parser.add_argument(
        "--with-gnn", action="store_true",
        help="Instead of the in-sample/out-of-sample report, run the full window once with "
             "gnn_signal hardcoded to 0.0 (SMC-only, the default everywhere else in this file) "
             "and once with a real point-in-time GNN signal recomputed at every candle, then "
             "diff the two — isolates whether GNN confluence adds real edge over SMC/FVG alone.",
    )
    parser.add_argument(
        "--segments", type=int, default=2,
        help="Split the fetched window into N equal chunks and report each separately "
             "(default 2 = in-sample/out-of-sample). Use a longer --days with a 1h "
             "timeframe and more segments to check whether the edge holds across "
             "multiple market regimes rather than one stretch.",
    )
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, YF_MAX_DAYS_5M)
    requested_days = args.days if args.days is not None else min(settings.BACKTEST_DAYS, max_days)
    days = min(requested_days, max_days)
    if requested_days > max_days:
        print(f"Note: capping to {max_days}d — Yahoo Finance doesn't serve {timeframe} candles further back than that.\n")

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
    print(f"Fetching {days}d of {timeframe} candles (via yfinance)...\n")

    all_candles: dict[str, list] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days, timeframe)
        if not candles:
            print(f"  {symbol}: no historical data, skipping")
            continue
        all_candles[symbol] = candles
        print(f"  {symbol}: {len(candles)} candles ({candles[0].timestamp} -> {candles[-1].timestamp})")

    if args.slippage_sweep:
        run_slippage_sweep(all_candles, timeframe)
        return

    if args.with_gnn:
        run_gnn_ablation(all_candles, timeframe)
        return

    n_segments = max(2, args.segments)
    segments: list[dict[str, list]] = [dict() for _ in range(n_segments)]
    for symbol, candles in all_candles.items():
        n = len(candles)
        bounds = [round(n * i / n_segments) for i in range(n_segments + 1)]
        for i in range(n_segments):
            segments[i][symbol] = candles[bounds[i]:bounds[i + 1]]

    if n_segments == 2:
        # Preserve original in-sample/out-of-sample naming and framing.
        is_path = f"backtrack_logs/backtest_ledger_insample_{settings.ASSET_TYPE}_{timeframe}.json"
        oos_path = f"backtrack_logs/backtest_ledger_oos_{settings.ASSET_TYPE}_{timeframe}.json"
        ledger_is, candles_is, open_is = run_replay(segments[0], is_path, timeframe)
        ledger_oos, candles_oos, open_oos = run_replay(segments[1], oos_path, timeframe)

        print_report(f"IN-SAMPLE [{timeframe}] (first half, ~{days // 2}d)", ledger_is, candles_is, open_is)
        print_report(f"OUT-OF-SAMPLE [{timeframe}] (second half, ~{days - days // 2}d)", ledger_oos, candles_oos, open_oos)

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
        print(f"\nFull trade logs: {is_path}, {oos_path}")
    else:
        # Multi-regime mode: N equal chronological chunks, reported side by side.
        days_per_segment = days / n_segments
        win_rates: list[float | None] = []
        print(f"\n{'=' * 70}")
        print(f"MULTI-REGIME [{timeframe}] — {n_segments} segments x ~{days_per_segment:.1f}d each")
        print(f"{'=' * 70}")
        print(f"{'Segment':<10}{'Period':<25}{'Trades':<9}{'Win Rate':<11}{'Return %':<10}")
        print("-" * 70)
        for i, seg in enumerate(segments):
            path = f"backtrack_logs/backtest_ledger_seg{i}_{settings.ASSET_TYPE}_{timeframe}.json"
            ledger, n_candles, _open = run_replay(seg, path, timeframe)
            stats = ledger.stats()
            overall = stats["overall"]
            wr = overall["win_rate_pct"]
            win_rates.append(wr)
            start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
            end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
            period = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "—"
            wr_str = f"{wr:.1f}%" if wr is not None else "—"
            print(f"seg{i:<7}{period:<25}{overall['trades']:<9}{wr_str:<11}{stats['total_return_pct']:>+7.2f}%")

        valid = [w for w in win_rates if w is not None]
        print("-" * 70)
        if len(valid) >= 2:
            spread = max(valid) - min(valid)
            print(f"Win rate range across segments: {min(valid):.1f}% - {max(valid):.1f}%  (spread {spread:.1f}pp)")
            if spread > 25:
                print("Wide spread across regimes — the edge looks regime-dependent, not consistent.")
            else:
                print("Win rate stayed in a reasonably tight band across all segments.")
        print(f"{'=' * 70}")

    print("\nNote: pnl_dollars assumes each trade's $ risk is independent of every other")
    print("open position — it is NOT a realistic capital-constrained return projection.")
    print("Win rate is the number to trust here; the dollar figures are directional.")
    print(f"\nCompare layers: python backtest.py --timeframe 15m   /   --timeframe 1h")


if __name__ == "__main__":
    main()
