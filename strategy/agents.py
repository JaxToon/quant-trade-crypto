"""
Multi-agent LLM consensus network.
Bull Agent → S_Bull ∈ [0,1]
Bear Agent → S_Bear ∈ [0,1]
Critic      → C_Risk ∈ [0,0.3]

P(Trade) = W1 * S_Bull - W2 * S_Bear - C_Risk
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_debate_cache: dict[str, tuple[float, "AgentDebate"]] = {}  # symbol → (timestamp, result)
_DEBATE_CACHE_TTL = 300  # 5 minutes — don't re-run debate if market context unchanged


def _format_smc_context(smc_context: dict, perspective: str = "bullish") -> str:
    """Format Smart Money Concept context for LLM agents."""
    if not smc_context:
        return "No Smart Money data available."

    lines = []
    lines.append(f"Market Structure: {smc_context.get('market_structure', 'unknown').upper()}")
    lines.append(f"SMC Signal: {smc_context.get('smc_signal', 0.0):.3f} (range -1 to 1)")

    # Order Blocks
    in_bull_ob = smc_context.get('in_bullish_ob', False)
    in_bear_ob = smc_context.get('in_bearish_ob', False)
    active_obs = smc_context.get('active_order_blocks', 0)

    if perspective == "bullish":
        if in_bull_ob:
            lines.append("✓ Price IN BULLISH Order Block (institutional demand zone)")
        else:
            lines.append(f"  {active_obs} active Order Blocks nearby")
    else:
        if in_bear_ob:
            lines.append("✓ Price IN BEARISH Order Block (institutional supply zone)")
        else:
            lines.append(f"  {active_obs} active Order Blocks nearby")

    # Fair Value Gaps
    in_bull_fvg = smc_context.get('in_bullish_fvg', False)
    in_bear_fvg = smc_context.get('in_bearish_fvg', False)
    active_fvgs = smc_context.get('active_fvgs', 0)

    if perspective == "bullish":
        if in_bull_fvg:
            lines.append("✓ Price IN BULLISH FVG (imbalance support)")
        else:
            lines.append(f"  {active_fvgs} active FVGs in the chart")
    else:
        if in_bear_fvg:
            lines.append("✓ Price IN BEARISH FVG (imbalance resistance)")
        else:
            lines.append(f"  {active_fvgs} active FVGs in the chart")

    # Liquidity Zones
    liquidity_zones = smc_context.get('liquidity_zones', 0)
    if liquidity_zones > 0:
        lines.append(f"  {liquidity_zones} liquidity zones identified (potential stop hunts)")

    # FVG detailed info if available
    if 'fvg_context' in smc_context and smc_context['fvg_context']:
        lines.append(f"  FVG Detail: {smc_context['fvg_context']}")

    return "\n".join(lines)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _call_llm(model: str, system: str, user: str) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _parse_json_score(text: str, key: str, default: float) -> tuple[float, str]:
    """Extract {key: float, reasoning: str} from LLM output robustly."""
    try:
        # Try direct JSON parse
        obj = json.loads(text)
        return float(obj.get(key, default)), obj.get("reasoning", "")
    except json.JSONDecodeError:
        # Try extracting from markdown code block
        import re
        m = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
                return float(obj.get(key, default)), obj.get("reasoning", "")
            except json.JSONDecodeError:
                pass
        return default, text[:200]


@dataclass
class AgentDebate:
    symbol: str
    bull_score: float
    bull_reasoning: str
    bear_score: float
    bear_reasoning: str
    c_risk: float
    critic_reasoning: str


def run_bull_agent(symbol: str, gnn_signal: float, indicators: dict, graph_context: str, smc_context: dict = None) -> tuple[float, str]:
    system = (
        "You are a senior institutional order-flow trader with 15+ years reading Smart Money "
        "footprints for a proprietary trading desk — the same rigor a Citadel or hedge-fund quant "
        "desk applies before greenlighting a position. You are building the bull case, but you are "
        "not a cheerleader: an unconvincing setup should score low, not a hedged 0.5. "
        "Output ONLY valid JSON: {\"score\": 0.X, \"reasoning\": \"...\"}. "
        "score is your bullish conviction from 0.0 (no conviction) to 1.0 (maximum bullish). "
        "reasoning must be one tight sentence naming the single strongest piece of confluence (or "
        "the lack of one) — not a restatement of the input data."
    )

    smc_context = smc_context or {}
    smc_summary = _format_smc_context(smc_context, "bullish")

    user = f"""Symbol: {symbol}
GNN Signal: {gnn_signal:.3f} (positive=bullish, negative=bearish, range -1 to 1)

Technical Indicators:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} | Signal: {indicators.get('macd_signal', 'N/A')}
- Price: {indicators.get('price', 0):.2f} | BB_upper: {indicators.get('bb_upper', 0):.2f} | BB_lower: {indicators.get('bb_lower', 0):.2f}

Smart Money Concepts:
{smc_summary}

Graph Context (fundamental signals):
{graph_context}

Build the bull case like a pre-trade memo. Work through:
- Confluence: does the GNN signal agree with the SMC picture, or are you leaning on just one of them?
- Are bullish Order Blocks / FVGs actually being respected, or merely present?
- Is market structure confirming (higher highs/higher lows), or is this a countertrend bet?
- Is there a live liquidity-sweep-then-reversal pattern, or just proximity to a zone?

Rate your BULLISH conviction 0.0-1.0 — let it be low if the setup is weak. JSON only."""

    try:
        text = _call_llm(settings.BULL_BEAR_MODEL, system, user)
        score, reasoning = _parse_json_score(text, "score", 0.5)
        return max(0.0, min(1.0, score)), reasoning
    except Exception as exc:
        logger.warning("Bull agent error: %s", exc)
        # Fallback: derive from GNN signal and SMC
        smc_signal = smc_context.get('smc_signal', 0.0) if smc_context else 0.0
        fallback = ((gnn_signal + 1) / 2) * 0.7 + (smc_signal + 1) / 2 * 0.3
        return max(0.0, min(1.0, fallback)), "LLM unavailable; GNN+SMC-derived score"


def run_bear_agent(symbol: str, gnn_signal: float, indicators: dict, graph_context: str, loss_history: list[str], smc_context: dict = None) -> tuple[float, str]:
    system = (
        "You are a senior risk officer conducting adversarial due diligence on a trade before it's "
        "approved — your job is to find every real reason this position fails, the way a hedge "
        "fund's risk desk stress-tests a book before the PM is allowed to size up. You are not here "
        "to be balanced; you are here to break the bull case if it can be broken. "
        "Output ONLY valid JSON: {\"score\": 0.X, \"reasoning\": \"...\"}. "
        "score is your BEARISH/RISK conviction from 0.0 (no risk) to 1.0 (maximum risk). "
        "reasoning must name the single most concrete danger (or state plainly that none stands out) "
        "— not a generic hedge."
    )

    smc_context = smc_context or {}
    smc_summary = _format_smc_context(smc_context, "bearish")
    loss_summary = "\n".join(loss_history[-3:]) if loss_history else "No recent losses."

    user = f"""Symbol: {symbol}
GNN Signal: {gnn_signal:.3f}

Technical Indicators:
- RSI: {indicators.get('rsi', 'N/A')} (>70 overbought, <30 oversold)
- MACD Histogram: {indicators.get('macd_hist', 'N/A')}

Smart Money Concepts:
{smc_summary}

Graph Risk Context:
{graph_context}

Recent System Losses (learn from these — is this setup rhyming with one of them?):
{loss_summary}

Stress-test this trade like a risk-committee memo. Work through:
- Distribution: are bearish Order Blocks / FVGs forming against this move?
- Structure break: any sign of lower lows undermining the bullish read?
- Liquidity trap: does this look like buy-side liquidity being swept before a reversal down?
- Precedent: does this setup resemble a documented past loss, or is it genuinely new territory?

Rate the BEARISH RISK level 0.0-1.0 — don't inflate it just to look cautious. JSON only."""

    try:
        text = _call_llm(settings.BULL_BEAR_MODEL, system, user)
        score, reasoning = _parse_json_score(text, "score", 0.5)
        return max(0.0, min(1.0, score)), reasoning
    except Exception as exc:
        logger.warning("Bear agent error: %s", exc)
        smc_signal = smc_context.get('smc_signal', 0.0) if smc_context else 0.0
        fallback = ((-gnn_signal + 1) / 2) * 0.7 + (-smc_signal + 1) / 2 * 0.3
        return max(0.0, min(1.0, fallback)), "LLM unavailable; GNN+SMC-derived score"


def run_critic(
    symbol: str,
    bull_score: float,
    bull_reasoning: str,
    bear_score: float,
    bear_reasoning: str,
    backtrack_summary: str,
) -> tuple[float, str]:
    system = (
        "You are the risk-committee reviewer for a systematic trading desk, applying a radical-"
        "transparency standard: your job is to hold the bull case to documented evidence, not to "
        "split the difference between the two agents. "
        "Output ONLY valid JSON: {\"c_risk\": 0.X, \"reasoning\": \"...\"}. "
        "c_risk is a penalty from 0.0 (bull argument is novel and sound) to 0.3 (argument repeats a "
        "known failure or leans on a single signal with no confluence). "
        "reasoning must state the specific match to a past loss, or the specific lack of confluence — "
        "or state plainly that the argument holds up."
    )
    user = f"""Symbol: {symbol}

Bull Agent said (score={bull_score:.2f}): {bull_reasoning}
Bear Agent said (score={bear_score:.2f}): {bear_reasoning}

Historical Backtrack Losses (past mistakes to avoid):
{backtrack_summary if backtrack_summary else 'No historical losses recorded yet.'}

Two checks:
1. Is the bull argument repeating a documented past failure pattern above?
2. Is the bull case leaning on a single signal (e.g. GNN alone, or SMC alone) with no real
   confluence between them — the kind of thin setup that looks fine until it isn't?

Output c_risk penalty 0.0-0.3. JSON only."""

    try:
        text = _call_llm(settings.CRITIC_MODEL, system, user)
        c_risk, reasoning = _parse_json_score(text, "c_risk", 0.0)
        return max(0.0, min(0.3, c_risk)), reasoning
    except Exception as exc:
        logger.warning("Critic agent error: %s", exc)
        return 0.0, "Critic unavailable"


def run_debate(
    symbol: str,
    gnn_signal: float,
    indicators: dict,
    bull_graph_context: str,
    bear_graph_context: str,
    backtrack_summary: str,
    loss_history: list[str],
    smc_context: dict = None,
) -> AgentDebate:
    """Runs the full Bull→Bear→Critic debate and returns scored result.
    Results are cached per symbol for 5 minutes to avoid redundant API calls
    when market context hasn't meaningfully changed."""
    now = time.time()
    cached_ts, cached = _debate_cache.get(symbol, (0, None))

    # Cache hit: same symbol, signal direction unchanged, within TTL
    if (cached is not None
            and now - cached_ts < _DEBATE_CACHE_TTL
            and abs(gnn_signal - (1 if cached.bull_score > 0.5 else -1)) < 0.3):
        logger.debug("%s: returning cached debate (age=%.0fs)", symbol, now - cached_ts)
        return cached

    logger.info("Running agent debate for %s (GNN=%.3f, SMC=%.3f)", symbol, gnn_signal, smc_context.get('smc_signal', 0.0) if smc_context else 0.0)

    bull_score, bull_reason = run_bull_agent(symbol, gnn_signal, indicators, bull_graph_context, smc_context)
    bear_score, bear_reason = run_bear_agent(symbol, gnn_signal, indicators, bear_graph_context, loss_history, smc_context)
    c_risk, critic_reason = run_critic(symbol, bull_score, bull_reason, bear_score, bear_reason, backtrack_summary)

    result = AgentDebate(
        symbol=symbol,
        bull_score=bull_score,
        bull_reasoning=bull_reason,
        bear_score=bear_score,
        bear_reasoning=bear_reason,
        c_risk=c_risk,
        critic_reasoning=critic_reason,
    )
    _debate_cache[symbol] = (now, result)
    return result
