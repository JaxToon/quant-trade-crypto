"""
Main orchestrator.

Startup sequence:
  1. Initialize Neo4j schema
  2. Fit HMM regime model
  3. Load or build GNN
  4. Start data pipelines (Binance WebSocket + RSS stream)
  5. Main decision loop: GNN inference → agent debate → execute
  6. Schedule nightly post-mortem at 00:00 UTC

Run: python main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from datetime import datetime

import schedule
from binance.client import Client

from config import settings
from backtrack.post_mortem import run_nightly_postmortem
from gnn.trainer import GNNTrainer
from graph.neo4j_client import Neo4jClient
from graph.schema import initialize_schema
from ingestion.kg_enrichment import run_enrichment
from ingestion.integrated_stream import IntegratedMarketStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # trader.log accumulates across restarts by design (README) for
        # "what happened since the system started" -- but a plain
        # FileHandler here meant genuinely unbounded growth for a system
        # meant to run for weeks/months (found at 30MB after ~1 week of
        # intermittent runtime). Rotates at 50MB, keeps 5 backups (~250MB
        # ceiling) -- same "found unbounded growth, added a bound" fix
        # already applied to the Macro_Event graph earlier this session.
        logging.handlers.RotatingFileHandler(
            "trader.log", maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("main")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Optional NLP modules (may fail if spaCy DLL blocked by App Control)
try:
    from ingestion.unstructured_stream import UnstructuredStream
    from knowledge.graph_populator import GraphPopulator
    NLP_AVAILABLE = True
except ImportError as e:
    logger.warning("NLP modules unavailable: %s - News stream disabled", e)
    NLP_AVAILABLE = False
    UnstructuredStream = None
    GraphPopulator = None
from backtrack.paper_trading import PaperTradingLedger
from risk.circuit_breaker import CircuitBreaker
from risk.hmm_regime import RegimeDetector
from strategy.consensus import evaluate_trade
from strategy.entry_insight import build_entry_insight
from strategy.execution import ExecutionEngine

# ── Shared state ──────────────────────────────────────────────────────────
# Keyed by (symbol, timeframe) — SMC/FVG runs independently per layer
# (settings.BINANCE_SMC_TIMEFRAMES), so ticks from 5m/15m/1h streams for the
# same symbol must not overwrite each other under one key.
_latest_indicators: dict[tuple[str, str], dict] = {}
_shutdown = threading.Event()
_paper_ledger = PaperTradingLedger() if settings.PAPER_TRADING_ENABLED else None

# Rolling window for the dashboard's session charts (equity curve, price
# movement). One point per decision-loop cycle (~60s) — 5000 points is
# ~3.5 days. Persisted to disk and reloaded at startup so restarting
# main.py doesn't reset the chart back to a blank slate — see DEPLOYMENT.md.
# This is NOT the model's learning (that's the GNN checkpoint + paper
# ledger, both already persisted); it's purely the dashboard's memory of
# what happened, across however many runs you've done.
# Asset-type-suffixed so a crypto process and a stock process can run
# simultaneously without silently overwriting each other's dashboard state
# (found while checking feasibility of running both at once — these three
# files, plus dashboard/api.py's STATUS_FILE, were the only genuinely
# shared/colliding paths; GNN checkpoints and paper ledgers were already
# namespaced by asset type).
HISTORY_FILE = os.path.join("dashboard", f"history_{settings.ASSET_TYPE}.json")
MAX_HISTORY_POINTS = 5000


def _load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        logger.info("No session history at %s — dashboard charts start fresh.", HISTORY_FILE)
        return []
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        logger.info("Session history restored from %s (%d points).", HISTORY_FILE, len(data))
        return data
    except Exception as exc:
        logger.warning("Session history load failed, starting fresh: %s", exc)
        return []


_history: list[dict] = _load_history()

# Separate from _history above (that's paper-trading/price data). This is
# the LLM-consensus path's own decision history — added after auditing that
# path and finding it had been completely blocked since the system started
# (an HMM regime bug pinned bear_veto_active true 100% of the time; see
# risk/hmm_regime.py's fix). Once unblocked, every P(Trade) evaluation came
# back well under TRADE_THRESHOLD — the formula's own mathematical ceiling
# (W1_BULL - 0, at Bull=1/Bear=0/C_Risk=0) is only 0.6, so a 0.55 threshold
# sits in roughly the top 8% of its whole possible range. Whether that's
# genuinely well-calibrated (this path has never closed a single trade, so
# there's no outcome data either way) isn't something to guess at from one
# afternoon of samples — this history exists so it CAN be evidence-based
# once enough real decisions have accumulated, instead of requiring a manual
# grep through trader.log every time the question comes up.
CONSENSUS_HISTORY_FILE = os.path.join("dashboard", f"consensus_history_{settings.ASSET_TYPE}.json")
MAX_CONSENSUS_HISTORY_POINTS = 2000


def _load_consensus_history() -> list[dict]:
    if not os.path.exists(CONSENSUS_HISTORY_FILE):
        return []
    try:
        with open(CONSENSUS_HISTORY_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Consensus decision history load failed, starting fresh: %s", exc)
        return []


_consensus_history: list[dict] = _load_consensus_history()


def _record_consensus_decision(decision) -> None:
    _consensus_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": decision.symbol,
        "p_trade": decision.p_trade,
        "bull_score": decision.debate.bull_score,
        "bear_score": decision.debate.bear_score,
        "c_risk": decision.debate.c_risk,
        "execute": decision.execute,
        "direction": decision.direction,
    })
    if len(_consensus_history) > MAX_CONSENSUS_HISTORY_POINTS:
        del _consensus_history[: len(_consensus_history) - MAX_CONSENSUS_HISTORY_POINTS]
    try:
        with open(CONSENSUS_HISTORY_FILE, "w") as f:
            json.dump(_consensus_history, f)
    except Exception as exc:
        logger.debug("Consensus decision history save failed: %s", exc)


def _consensus_stats() -> dict:
    if not _consensus_history:
        return {"count": 0}
    p_trades = [d["p_trade"] for d in _consensus_history]
    executed = sum(1 for d in _consensus_history if d["execute"])
    return {
        "count": len(_consensus_history),
        "executed": executed,
        "max_p_trade": round(max(p_trades), 4),
        "mean_p_trade": round(sum(p_trades) / len(p_trades), 4),
        "trade_threshold": settings.TRADE_THRESHOLD,
        "formula_ceiling": round(settings.W1_BULL, 4),  # max possible: W1_BULL*1 - W2_BEAR*0 - 0
    }


def _validate_config() -> None:
    missing = []
    if not settings.NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    if not settings.BINANCE_API_KEY:
        missing.append("BINANCE_API_KEY")
    if not settings.BINANCE_API_SECRET:
        missing.append("BINANCE_API_SECRET")
    if not settings.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Copy .env.example → .env and fill in your credentials.")
        sys.exit(1)


# ── Decision loop (runs every 60 seconds) ────────────────────────────────

def _build_entry_insights(gnn_signals: dict) -> dict:
    """Returns {"SYMBOL_timeframe": insight} — one entry per (symbol, layer)."""
    insights = {}
    for (symbol, tf), indicators in _latest_indicators.items():
        try:
            price = indicators.get("price", 0.0)
            insight = build_entry_insight(
                symbol, price, indicators.get("smc"), gnn_signals.get(symbol, 0.0)
            )
            insight["timeframe"] = tf
            insights[f"{symbol}_{tf}"] = insight
        except Exception as exc:
            logger.debug("Entry insight failed for %s [%s]: %s", symbol, tf, exc)
    return insights


def _run_paper_trading(insights: dict, prices: dict) -> dict:
    """
    Scores entry_insight recommendations against live price with a virtual
    account — opens simulated positions on actionable insights, closes them
    on SL/TP cross. No real or testnet orders are placed here.

    insights: keyed "SYMBOL_timeframe" (see _build_entry_insights) — each
    layer gets its own independent paper position, so e.g. a 5m and a 1h
    derivative signal on the same symbol can both be open at once.

    Returns {"SYMBOL_timeframe": {"spot": reason, "derivative": reason}} for
    any insight that said "actionable" but the ledger rejected anyway — so a
    rejection like "stop too tight for fees" is visible on the dashboard
    instead of silently invisible (it used to only log at debug level, which
    the default INFO config never shows).
    """
    rejections: dict[str, dict] = {}
    if _paper_ledger is None:
        return rejections
    try:
        for insight_key, ins in insights.items():
            symbol = ins.get("symbol", insight_key)
            tf = ins.get("timeframe", settings.BINANCE_KLINE_INTERVAL)
            # confidence lives on the parent insight dict, not the spot/
            # derivative sub-blocks — carry it along so the ledger records it.
            spot = {**ins.get("spot", {}), "confidence": ins.get("confidence", 0.0)}
            derivative = {**ins.get("derivative", {}), "confidence": ins.get("confidence", 0.0)}
            # Use the SAME cross-timeframe price check_and_close will use
            # below, not ins["price"] — that's the signal's OWN timeframe's
            # last candle price (e.g. a 1h layer's price only updates once
            # an hour), which can already be stale relative to the live
            # price check_and_close checks moments later. Comparing the
            # stale price against itself always passes, defeating the guard
            # entirely (confirmed live: NVDA still re-opened/instant-closed
            # after the current_price guard was first deployed).
            current_price = prices.get(symbol)
            spot_reason = _paper_ledger.maybe_open(symbol, "spot", spot, timeframe=tf, current_price=current_price)
            deriv_reason = _paper_ledger.maybe_open(symbol, "derivative", derivative, timeframe=tf, current_price=current_price)
            if spot_reason or deriv_reason:
                rejections[insight_key] = {"spot": spot_reason, "derivative": deriv_reason}

        _paper_ledger.check_and_close(prices)
        _paper_ledger.save()
    except Exception as exc:
        logger.warning("Paper trading update failed: %s", exc)
    return rejections


def _record_history(prices: dict) -> None:
    _history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "balance": round(_paper_ledger.balance, 2) if _paper_ledger else None,
        "prices": prices,
    })
    if len(_history) > MAX_HISTORY_POINTS:
        del _history[: len(_history) - MAX_HISTORY_POINTS]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(_history, f)
    except Exception as exc:
        logger.debug("Session history save failed: %s", exc)


def _current_prices() -> dict:
    """One representative price per symbol, preferring the primary timeframe
    when available (any layer's price is close enough for display/closing
    purposes — they're all the same live market, just different candle
    granularity)."""
    prices: dict[str, float] = {}
    for (sym, tf), ind in _latest_indicators.items():
        price = ind.get("price")
        if not price:
            continue
        if sym not in prices or tf == settings.BINANCE_KLINE_INTERVAL:
            prices[sym] = price
    return prices


def _write_status(circuit: "CircuitBreaker", signals: dict) -> None:
    try:
        insights = _build_entry_insights(signals)
        prices = _current_prices()
        paper_trade_rejections = _run_paper_trading(insights, prices)
        _record_history(prices)
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "asset_type": settings.ASSET_TYPE,
            "regime": circuit.regime_label,
            "bear_veto": circuit.bear_veto_active,
            "history": _history,
            "gnn_signals": {k: round(float(v), 4) for k, v in signals.items()},
            "entry_insights": insights,
            "paper_trade_rejections": paper_trade_rejections,
            "paper_trading": _paper_ledger.stats() if _paper_ledger else None,
            # Most-recent-first slice of actual closed trades — separate
            # from paper_trading.stats() above, which is aggregates only
            # (win rate, counts, PnL sums), never the individual trade list.
            # The dashboard's "Trade History" card previously rendered from
            # graph.recent_trades (Neo4j Trade nodes, Path 2's testnet
            # execution path) — always empty, since that path has never
            # closed a single trade. This is the ledger that actually has
            # real closed trades right now.
            "paper_trading_recent_trades": (
                list(reversed(_paper_ledger.closed_trades[-20:])) if _paper_ledger else []
            ),
            "consensus_decisions": {
                "recent": _consensus_history[-50:],
                "stats": _consensus_stats(),
            },
        }
        with open(f"dashboard/status_{settings.ASSET_TYPE}.json", "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def decision_loop(
    gnn: GNNTrainer,
    execution: ExecutionEngine,
    circuit: CircuitBreaker,
    binance_client: Client,
) -> None:
    """Called every 60s: run GNN → debate → execute."""
    # Update regime
    regime = circuit.update(binance_client)

    # Always get signals so dashboard stays current
    signals = gnn.predict_all()
    _write_status(circuit, signals)

    # Skip all LLM debate when bear veto is fully active — saves OpenAI credits
    if circuit.bear_veto_active:
        logger.info("Bear veto active (%s) — skipping debate cycle.", circuit.regime_label)
        execution.monitor_positions()
        return

    for symbol in settings.WATCHLIST:
        # LLM-consensus execution stays on the primary timeframe only — the
        # multi-timeframe layers (15m/1h) feed entry_insight/paper trading,
        # not this real-testnet-order path, which shouldn't change behavior.
        indicators = _latest_indicators.get((symbol, settings.BINANCE_KLINE_INTERVAL), {})
        if not indicators:
            logger.debug("%s: no indicators yet, skipping.", symbol)
            continue

        gnn_signal = signals.get(symbol, 0.0)  # signals populated above

        # Skip near-zero signals (model uncertain)
        if abs(gnn_signal) < settings.GNN_CONVICTION_THRESHOLD:
            continue

        decision = evaluate_trade(
            symbol=symbol,
            gnn_signal=gnn_signal,
            indicators=indicators,
            bear_veto=circuit.bear_veto_active,
            smc_context=indicators.get('smc'),  # Extract SMC data from enriched indicators
        )
        _record_consensus_decision(decision)

        if decision.execute:
            execution.execute(decision)

    # Monitor open positions for SL/TP breach
    execution.monitor_positions()

    # Apply temporal decay to edge weights
    Neo4jClient.get().apply_temporal_decay(settings.TEMPORAL_DECAY_GAMMA)


# ── Binance tick callback ─────────────────────────────────────────────────

def on_tick(symbol: str, indicators: dict) -> None:
    tf = indicators.get("interval", settings.BINANCE_KLINE_INTERVAL)
    _latest_indicators[(symbol, tf)] = indicators


def _start_dashboard() -> None:
    """Serve the monitoring dashboard in-process so `python main.py` alone is enough."""
    if not settings.DASHBOARD_ENABLED:
        return
    try:
        import uvicorn
    except ImportError:
        logger.warning(
            "Dashboard disabled: fastapi/uvicorn not installed. "
            "pip install fastapi uvicorn to enable, or run run_dashboard.py separately."
        )
        return

    def _serve():
        uvicorn.run("dashboard.api:app", host="0.0.0.0", port=settings.DASHBOARD_PORT, log_level="warning")

    threading.Thread(target=_serve, daemon=True, name="Dashboard").start()
    logger.info("Dashboard available at http://localhost:%d", settings.DASHBOARD_PORT)

# ── Async entry point ─────────────────────────────────────────────────────

async def async_main() -> None:
    _validate_config()
    logger.info("=== TraderStuff Starting ===")

    # 1. Neo4j schema
    logger.info("Initializing Neo4j schema...")
    initialize_schema()

    _start_dashboard()

    # 2. Sync Binance client (for HMM fitting + order execution)
    binance_client = Client(
        settings.BINANCE_API_KEY,
        settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )

    # 3. HMM regime detector
    logger.info("Setting up regime detector...")
    regime_detector = RegimeDetector()
    circuit = CircuitBreaker(regime_detector)
    if not hasattr(regime_detector, "_model") or regime_detector._model is None:
        regime_detector.fit(binance_client, days=60)

    # 4. GNN
    logger.info("Initializing GNN...")
    gnn = GNNTrainer()
    gnn.build_model()
    if not gnn.load():
        logger.info("No checkpoint found — model needs training data. Running with zero signal until trained.")

    # 5. Execution engine
    execution = ExecutionEngine()

    # 6. Unstructured stream (RSS + EDGAR) - Optional if NLP available
    if NLP_AVAILABLE:
        logger.info("Starting news stream (RSS + EDGAR)...")
        news_stream = UnstructuredStream()
        news_stream.start()

        populator = GraphPopulator(news_stream)
        populator.start()

        # 7. External KG enrichment at startup + every 6 hours
        logger.info("Running initial KG enrichment (CoinGecko + DeFiLlama)...")
        threading.Thread(target=run_enrichment, daemon=True, name="KGEnrich-Init").start()
        schedule.every(6).hours.do(run_enrichment)
    else:
        logger.warning("News stream disabled (NLP unavailable) - running with market data + SMC only")
        news_stream = None
        populator = None

    # Schedule nightly post-mortem
    schedule.every().day.at("00:00").do(run_nightly_postmortem, gnn_trainer=gnn)
    # Re-fit HMM weekly
    schedule.every().monday.at("01:00").do(regime_detector.fit, client=binance_client, days=90)

    # Graph pruning — Macro_Event nodes have no cap/expiry otherwise (one
    # new node per qualifying article, forever) and apply_temporal_decay()
    # only ever weakens edges, never removes them. Daily is plenty; this is
    # maintenance, not something that needs sub-day freshness.
    def run_graph_pruning():
        client = Neo4jClient.get()
        try:
            deleted_events = client.prune_old_macro_events(settings.GRAPH_MACRO_EVENT_MAX_AGE_DAYS)
            deleted_edges = client.prune_decayed_edges(settings.GRAPH_MIN_EDGE_CONFIDENCE)
            logger.info("Graph pruning: removed %d stale Macro_Event nodes, %d decayed edges", deleted_events, deleted_edges)
        except Exception as exc:
            logger.warning("Graph pruning failed: %s", exc)

    schedule.every().day.at("00:30").do(run_graph_pruning)

    # Continual learning from realized paper-trade outcomes — fires every
    # LEARNING_INTERVAL_HOURS, no-ops if fewer than LEARNING_MIN_NEW_TRADES
    # have closed since the last update. See continual_learning.py docstring
    # for why this isn't formal reinforcement learning.
    if _paper_ledger is not None:
        def run_paper_trade_learning():
            from backtrack.continual_learning import learn_from_paper_trades
            try:
                learn_from_paper_trades(gnn, _paper_ledger)
            except Exception as exc:
                logger.warning("Paper-trade learning update failed: %s", exc)

        schedule.every(settings.LEARNING_INTERVAL_HOURS).hours.do(run_paper_trade_learning)

    def run_scheduler():
        while not _shutdown.is_set():
            schedule.run_pending()
            time.sleep(30)

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="Scheduler")
    scheduler_thread.start()

    # 8. Decision loop (every 60 seconds in a background thread)
    def run_decision_loop():
        while not _shutdown.is_set():
            try:
                decision_loop(gnn, execution, circuit, binance_client)
            except Exception as exc:
                logger.error("Decision loop error: %s", exc, exc_info=True)
            _shutdown.wait(60)

    decision_thread = threading.Thread(target=run_decision_loop, daemon=True, name="DecisionLoop")
    decision_thread.start()

    # 9. Integrated Market Stream with SMC analysis (async — blocks here until shutdown)
    stream = IntegratedMarketStream(source="auto")
    stream.add_callback(on_tick)

    logger.info("=== All systems live. Watching: %s ===", settings.WATCHLIST)

    def shutdown_handler(*_):
        logger.info("Shutdown signal received.")
        _shutdown.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        await stream.start()
    except asyncio.CancelledError:
        pass
    finally:
        await stream.stop()
        if news_stream:
            news_stream.stop()
        if populator:
            populator.stop()
        Neo4jClient.get().close()
        logger.info("=== TraderStuff Stopped ===")


if __name__ == "__main__":
    asyncio.run(async_main())
