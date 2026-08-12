"""
Computes P(Trade) from agent debate scores.
P(Trade) = W1 * S_Bull - W2 * S_Bear - C_Risk
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings
from graph.neo4j_client import Neo4jClient
from strategy.agents import AgentDebate

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    symbol: str
    p_trade: float
    execute: bool
    direction: str          # "LONG" | "SHORT" | "FLAT"
    debate: AgentDebate
    gnn_signal: float


def compute_p_trade(debate: AgentDebate) -> float:
    """
    Implements the PDF formula:
    P(Trade) = w1 * S_Bull - w2 * S_Bear - C_Risk
    """
    p = settings.W1_BULL * debate.bull_score - settings.W2_BEAR * debate.bear_score - debate.c_risk
    return round(p, 4)


def _build_graph_context(symbol: str, for_bear: bool = False) -> str:
    """Pull relevant neighbour relationships from Neo4j for agent prompts."""
    try:
        rows = Neo4jClient.get().get_neighbours(symbol, hops=2)
        lines = []
        for row in rows[:10]:
            node = dict(row.get("n", {}))
            rels = row.get("rels", [])
            rel_types = [r.type for r in rels if hasattr(r, "type")]
            lines.append(f"  - {node.get('name', node.get('symbol', '?'))} via {rel_types}")
        return "\n".join(lines) or "No graph neighbours found."
    except Exception as exc:
        logger.warning("Graph context query failed: %s", exc)
        return "Graph context unavailable."


def _get_backtrack_summary() -> str:
    try:
        losses = Neo4jClient.get().get_recent_losses(limit=5)
        if not losses:
            return ""
        lines = []
        for row in losses:
            t = dict(row.get("t", {}))
            lines.append(
                f"  LOSS: {t.get('symbol')} {t.get('direction')} "
                f"pnl={t.get('pnl_pct', 0)*100:.1f}%"
            )
        return "\n".join(lines)
    except Exception:
        return ""


def evaluate_trade(
    symbol: str,
    gnn_signal: float,
    indicators: dict,
    bear_veto: bool = False,
    smc_context: dict = None,
) -> TradeDecision:
    """
    Full pipeline: build context → run debate → compute P(Trade) → decide.
    Now includes Smart Money Concept (SMC) and Fair Value Gap (FVG) analysis.

    bear_veto: set True by HMM circuit breaker in turbulent regime.
    smc_context: SMC indicators (order blocks, FVG, market structure)
    """
    bull_ctx = _build_graph_context(symbol, for_bear=False)
    bear_ctx = _build_graph_context(symbol, for_bear=True)
    backtrack_summary = _get_backtrack_summary()

    # Re-use the losses already fetched in _get_backtrack_summary via get_recent_losses
    try:
        loss_rows = Neo4jClient.get().get_recent_losses(limit=3)
        loss_history = [
            f"{dict(r.get('t', {})).get('symbol')} crashed — {dict(r.get('t', {})).get('direction', '')}"
            for r in loss_rows
        ]
    except Exception:
        loss_history = []

    from strategy.agents import run_debate
    debate = run_debate(
        symbol=symbol,
        gnn_signal=gnn_signal,
        indicators=indicators,
        bull_graph_context=bull_ctx,
        bear_graph_context=bear_ctx,
        backtrack_summary=backtrack_summary,
        loss_history=loss_history,
        smc_context=smc_context,
    )

    p_trade = compute_p_trade(debate)

    # Apply SMC adjustments to P(Trade)
    if smc_context:
        smc_signal = smc_context.get('smc_signal', 0.0)

        # Boost P(Trade) if SMC aligns with GNN signal
        if (gnn_signal > 0 and smc_signal > 0) or (gnn_signal < 0 and smc_signal < 0):
            # Confluence: both signals agree
            smc_boost = abs(smc_signal) * 0.15  # Max 15% boost
            p_trade += smc_boost
            logger.debug("%s: SMC confluence boost +%.3f", symbol, smc_boost)

        # Penalize if SMC contradicts GNN
        elif abs(smc_signal) > 0.3 and (gnn_signal * smc_signal < 0):
            # Divergence: signals disagree
            smc_penalty = abs(smc_signal) * 0.20  # Max 20% penalty
            p_trade -= smc_penalty
            logger.debug("%s: SMC divergence penalty -%.3f", symbol, smc_penalty)

        # Extra boost for being inside high-quality zones
        if smc_context.get('in_bullish_ob') or smc_context.get('in_bullish_fvg'):
            if gnn_signal > 0:  # Aligned with bullish setup
                p_trade += 0.10
                logger.debug("%s: Inside bullish institutional zone +0.10", symbol)

        if smc_context.get('in_bearish_ob') or smc_context.get('in_bearish_fvg'):
            if gnn_signal < 0:  # Aligned with bearish setup
                p_trade += 0.10
                logger.debug("%s: Inside bearish institutional zone +0.10", symbol)

    # Clamp P(Trade) to [0, 1]
    p_trade = max(0.0, min(1.0, p_trade))

    # Bear Agent absolute veto in turbulent regime
    if bear_veto and debate.bear_score > 0.5:
        logger.warning("%s: Bear veto active (turbulent regime). Blocking trade.", symbol)
        return TradeDecision(symbol, p_trade, False, "FLAT", debate, gnn_signal)

    execute = p_trade >= settings.TRADE_THRESHOLD
    direction = "LONG" if gnn_signal > 0 else "SHORT"

    smc_info = ""
    if smc_context:
        ms = smc_context.get('market_structure', 'unknown')
        smc_sig = smc_context.get('smc_signal', 0.0)
        smc_info = f" | SMC={smc_sig:.2f} ({ms})"

    logger.info(
        "%s | P(Trade)=%.3f | Bull=%.2f | Bear=%.2f | C_Risk=%.2f%s | %s",
        symbol, p_trade, debate.bull_score, debate.bear_score, debate.c_risk,
        smc_info,
        "EXECUTE" if execute else "SKIP",
    )

    return TradeDecision(symbol, p_trade, execute, direction if execute else "FLAT", debate, gnn_signal)
