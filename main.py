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
        logging.FileHandler("trader.log", encoding="utf-8"),
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
HISTORY_FILE = os.path.join("dashboard", "history.json")
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
            spot_reason = _paper_ledger.maybe_open(symbol, "spot", spot, timeframe=tf)
            deriv_reason = _paper_ledger.maybe_open(symbol, "derivative", derivative, timeframe=tf)
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
        }
        with open("dashboard/status.json", "w") as f:
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
