"""
Regression tests for strategy/entry_insight.py — specifically targeting the
confidence-propagation bug found in this project's history: _spot_recommendation()
and _derivative_recommendation() didn't include a "confidence" key in their
returned dicts, so paper_trading.py's maybe_open() (which reads rec["confidence"])
silently defaulted to 0.0 on every single trade, live and backtest, making
conviction-scaled sizing completely inert without ever raising an error.
"""
from __future__ import annotations

from strategy.entry_insight import build_entry_insight


def _bullish_smc(confidence: float = 40.0) -> dict:
    return {
        "direction": "long",
        "confidence": confidence,
        "entry_zone": [100.0, 101.0],
        "stop_loss": 95.0,
        "target": 110.0,
        "smc_signal": 0.5,
        "market_structure": "bullish",
    }


def _bearish_smc(confidence: float = 40.0) -> dict:
    return {
        "direction": "short",
        "confidence": confidence,
        "entry_zone": [100.0, 101.0],
        "stop_loss": 105.0,
        "target": 90.0,
        "smc_signal": -0.5,
        "market_structure": "bearish",
    }


def test_confidence_key_present_in_spot_and_derivative():
    """The actual regression: these dicts used to have NO "confidence" key
    at all, so rec.get("confidence", 0.0) in paper_trading.py always hit
    the default. This must never silently regress again."""
    insight = build_entry_insight("BTCUSDT", 100.5, _bullish_smc(confidence=40.0), gnn_signal=0.0)
    assert "confidence" in insight["spot"]
    assert "confidence" in insight["derivative"]


def test_confidence_matches_top_level():
    """Sub-dict confidence must equal the top-level confidence used to
    compute it — not a stale default, not a different value."""
    insight = build_entry_insight("BTCUSDT", 100.5, _bullish_smc(confidence=40.0), gnn_signal=0.0)
    assert insight["spot"]["confidence"] == insight["confidence"]
    assert insight["derivative"]["confidence"] == insight["confidence"]
    assert insight["spot"]["confidence"] == 40.0


def test_gnn_agreement_boosts_confidence_and_it_still_propagates():
    """Confluence boost changes the top-level confidence value — confirm
    the BOOSTED value (not the original) reaches the sub-dicts too."""
    base = _bullish_smc(confidence=40.0)
    insight = build_entry_insight("BTCUSDT", 100.5, base, gnn_signal=0.5)  # agrees with "long"
    assert insight["confluence"] == "agree"
    assert insight["confidence"] > 40.0
    assert insight["spot"]["confidence"] == insight["confidence"]
    assert insight["derivative"]["confidence"] == insight["confidence"]


def test_gnn_disagreement_penalizes_confidence():
    base = _bullish_smc(confidence=40.0)
    insight = build_entry_insight("BTCUSDT", 100.5, base, gnn_signal=-0.5)  # disagrees with "long"
    assert insight["confluence"] == "disagree"
    assert insight["confidence"] < 40.0


def test_weak_gnn_signal_below_threshold_is_neutral():
    """Signals below GNN_CONVICTION_THRESHOLD should be treated as no
    opinion — confidence unchanged, not boosted or penalized."""
    base = _bullish_smc(confidence=40.0)
    insight = build_entry_insight("BTCUSDT", 100.5, base, gnn_signal=0.001)
    assert insight["confluence"] == "neutral"
    assert insight["confidence"] == 40.0


def test_spot_long_is_actionable():
    insight = build_entry_insight("BTCUSDT", 100.5, _bullish_smc(), gnn_signal=0.0)
    assert insight["spot"]["actionable"] is True
    assert insight["spot"]["direction"] == "long"


def test_spot_short_is_not_actionable():
    """Spot can't short — bearish signals must show up as informational
    only, never actionable."""
    insight = build_entry_insight("BTCUSDT", 100.5, _bearish_smc(), gnn_signal=0.0)
    assert insight["spot"]["actionable"] is False
    assert insight["spot"]["direction"] == "short"


def test_derivative_can_go_either_direction():
    long_insight = build_entry_insight("BTCUSDT", 100.5, _bullish_smc(), gnn_signal=0.0)
    short_insight = build_entry_insight("BTCUSDT", 100.5, _bearish_smc(), gnn_signal=0.0)
    assert long_insight["derivative"]["actionable"] is True
    assert short_insight["derivative"]["actionable"] is True


def test_no_setup_returns_inactionable_for_both_markets():
    insight = build_entry_insight("BTCUSDT", 100.5, None, gnn_signal=0.0)
    assert insight["spot"]["actionable"] is False
    assert insight["derivative"]["actionable"] is False
