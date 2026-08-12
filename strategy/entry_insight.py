"""
Read-only entry-point insight for the dashboard.

Turns the SMC/FVG signals already computed by IntegratedMarketStream into a
human-readable recommendation for spot (long-only) and derivative/futures
(long or short, with a rough liquidation estimate) markets. This module
never places orders — it only summarizes what the analyzers already found.
"""
from __future__ import annotations

from config import settings

_NO_SETUP = {
    "actionable": False,
    "direction": "neutral",
    "entry_zone": None,
    "stop_loss": None,
    "target": None,
    "note": "Tidak ada setup SMC/FVG yang jelas saat ini.",
}


def build_entry_insight(symbol: str, price: float, smc: dict | None, gnn_signal: float = 0.0) -> dict:
    """
    Args:
        symbol: e.g. "BTCUSDT"
        price: last known price
        smc: the merged smc+fvg dict produced by
             IntegratedMarketStream._enrich_with_smc (indicators['smc'])
        gnn_signal: the GNN's directional signal for this symbol, in [-1, +1].
             Used purely as a confluence check — reusing the same agree/boost,
             disagree/penalize logic strategy/consensus.py already applies to
             the LLM-execution path, so both decision paths agree on what
             "GNN and SMC disagree" means. 0.0 (no GNN signal available, e.g.
             during backtest replay) is treated as neutral — no adjustment.

    Returns a dict with `spot` and `derivative` recommendation blocks.
    """
    smc = smc or {}
    direction = smc.get("direction", "neutral")  # from FVG signal: long/short/neutral
    confidence = round(smc.get("confidence", 0.0), 1)
    entry_zone = smc.get("entry_zone")
    stop_loss = smc.get("stop_loss")
    target = smc.get("target")
    context = smc.get("fvg_context") or _structure_context(smc)

    confluence, confidence = _apply_gnn_confluence(direction, confidence, gnn_signal)
    vetoed = confluence == "disagree" and confidence < settings.ENTRY_INSIGHT_MIN_CONFIDENCE

    base = {
        "symbol": symbol,
        "price": price,
        "market_structure": smc.get("market_structure", "unknown"),
        "smc_signal": smc.get("smc_signal", 0.0),
        "gnn_signal": round(gnn_signal, 4),
        "confluence": confluence,
        "direction": direction,
        "confidence": confidence,
        "context": context,
    }

    if direction == "neutral" or entry_zone is None or vetoed:
        base["spot"] = dict(_NO_SETUP)
        base["derivative"] = dict(_NO_SETUP)
        if vetoed:
            veto_note = "Diveto: sinyal GNN berlawanan arah dengan SMC/FVG — confidence turun di bawah ambang batas."
            base["spot"]["note"] = base["derivative"]["note"] = veto_note
        return base

    base["spot"] = _spot_recommendation(direction, entry_zone, stop_loss, target)
    base["derivative"] = _derivative_recommendation(direction, entry_zone, stop_loss, target, price)
    return base


def _apply_gnn_confluence(direction: str, confidence: float, gnn_signal: float) -> tuple[str, float]:
    """
    Mirrors strategy/consensus.py's confluence-boost / divergence-penalty
    formula so both decision paths treat "GNN agrees with SMC" the same way.
    """
    if direction == "neutral" or abs(gnn_signal) < settings.GNN_CONVICTION_THRESHOLD:
        return "neutral", confidence

    gnn_direction = "long" if gnn_signal > 0 else "short"
    if gnn_direction == direction:
        boosted = min(100.0, confidence + abs(gnn_signal) * 30)
        return "agree", round(boosted, 1)

    penalized = max(0.0, confidence - abs(gnn_signal) * 50)
    return "disagree", round(penalized, 1)


def _structure_context(smc: dict) -> str:
    if smc.get("in_bullish_ob"):
        return "Price di dalam Bullish Order Block"
    if smc.get("in_bearish_ob"):
        return "Price di dalam Bearish Order Block"
    return f"Market structure: {smc.get('market_structure', 'unknown')}"


def _spot_recommendation(direction: str, entry_zone, stop_loss, target) -> dict:
    if direction == "long":
        return {
            "actionable": True,
            "direction": "long",
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "target": target,
            "note": "Bullish setup — bisa entry beli langsung di spot.",
        }
    # Spot can't short — bearish signal is informational only.
    return {
        "actionable": False,
        "direction": direction,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "target": target,
        "note": "Sinyal bearish — spot tidak bisa short. Pertimbangkan hindari entry atau tunggu.",
    }


def _derivative_recommendation(direction: str, entry_zone, stop_loss, target, price: float) -> dict:
    leverage = settings.INSIGHT_DERIVATIVE_LEVERAGE
    entry_ref = price
    if entry_zone:
        entry_ref = sum(entry_zone) / 2

    liquidation = None
    if entry_ref:
        if direction == "long":
            liquidation = round(entry_ref * (1 - 1 / leverage), 4)
        elif direction == "short":
            liquidation = round(entry_ref * (1 + 1 / leverage), 4)

    return {
        "actionable": direction in ("long", "short"),
        "direction": direction,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "target": target,
        "suggested_leverage": leverage,
        "est_liquidation": liquidation,
        "note": f"Estimasi kasar @{leverage}x isolated, tidak menghitung funding rate/maintenance margin.",
    }
