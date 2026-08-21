"""
Tests for build_candidate_dataset.py -- the offline join between the
candidate-trade log (features at decision time) and the paper ledger
(realized outcomes). Written and validated against synthetic data before
any real candidate log exists (see RESEARCH_NOTES.md), since the join
logic's correctness doesn't depend on having real data, only its
usefulness does.
"""
from __future__ import annotations

from build_candidate_dataset import build_dataset


def _candidate(symbol="BTCUSDT", market="spot", timeframe="5m", decision="accepted",
                opened_at="2026-08-22T10:00:00", reason=None):
    return {
        "timestamp": opened_at,
        "symbol": symbol,
        "timeframe": timeframe,
        "market": market,
        "direction": "long",
        "entry_zone": [100.0, 100.0],
        "stop_loss": 98.0,
        "target": 104.0,
        "confidence": 70.0,
        "atr": 1.5,
        "decision": decision,
        "reason": reason,
        "opened_at": opened_at if decision == "accepted" else None,
        "features": {"market_structure": "bullish"},
    }


def _closed_trade(symbol="BTCUSDT", market="spot", timeframe="5m",
                   opened_at="2026-08-22T10:00:00", outcome="win", pnl=250.0):
    return {"symbol": symbol, "market": market, "timeframe": timeframe,
            "opened_at": opened_at, "closed_at": "2026-08-22T10:30:00",
            "outcome": outcome, "pnl_dollars": pnl}


def _open_position(symbol="BTCUSDT", market="spot", timeframe="5m", opened_at="2026-08-22T10:00:00"):
    return {"symbol": symbol, "market": market, "timeframe": timeframe, "opened_at": opened_at}


class TestRejectedCandidates:
    def test_rejected_gets_rejected_label_no_join_attempted(self):
        cand = _candidate(decision="rejected", reason="Stop terlalu rapat")
        rows = build_dataset([cand], closed_trades=[_closed_trade()], open_positions=[])
        assert rows[0]["label"] == "rejected"
        assert rows[0]["pnl_dollars"] is None


class TestAcceptedResolvedCandidates:
    def test_exact_timestamp_match_wins(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        closed = _closed_trade(opened_at="2026-08-22T10:00:00", outcome="win", pnl=250.0)
        rows = build_dataset([cand], closed_trades=[closed], open_positions=[])
        assert rows[0]["label"] == "win"
        assert rows[0]["pnl_dollars"] == 250.0

    def test_near_but_not_exact_timestamp_within_tolerance_matches(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        closed = _closed_trade(opened_at="2026-08-22T10:00:04", outcome="loss", pnl=-100.0)  # 4s later
        rows = build_dataset([cand], closed_trades=[closed], open_positions=[])
        assert rows[0]["label"] == "loss"
        assert rows[0]["pnl_dollars"] == -100.0

    def test_beyond_tolerance_does_not_match(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        closed = _closed_trade(opened_at="2026-08-22T10:05:00", outcome="win", pnl=250.0)  # 5min later
        rows = build_dataset([cand], closed_trades=[closed], open_positions=[])
        assert rows[0]["label"] == "unmatched"

    def test_picks_nearest_when_multiple_candidates_same_key(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:05")
        far = _closed_trade(opened_at="2026-08-22T09:59:50", outcome="loss", pnl=-50.0)   # 15s away
        near = _closed_trade(opened_at="2026-08-22T10:00:03", outcome="win", pnl=300.0)   # 2s away
        rows = build_dataset([cand], closed_trades=[far, near], open_positions=[])
        assert rows[0]["label"] == "win"
        assert rows[0]["pnl_dollars"] == 300.0

    def test_different_market_or_timeframe_does_not_cross_match(self):
        cand = _candidate(decision="accepted", market="spot", timeframe="5m")
        wrong_market = _closed_trade(market="derivative", outcome="win", pnl=999.0)
        rows = build_dataset([cand], closed_trades=[wrong_market], open_positions=[])
        assert rows[0]["label"] == "unmatched"


class TestAcceptedPendingCandidates:
    def test_still_open_gets_pending_label(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        still_open = _open_position(opened_at="2026-08-22T10:00:01")
        rows = build_dataset([cand], closed_trades=[], open_positions=[still_open])
        assert rows[0]["label"] == "pending"
        assert rows[0]["pnl_dollars"] is None

    def test_closed_trades_checked_before_open_positions(self):
        """If a position somehow appears in both (stale open_positions
        snapshot), the resolved outcome must win, not "pending"."""
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        closed = _closed_trade(opened_at="2026-08-22T10:00:00", outcome="win", pnl=250.0)
        stale_open = _open_position(opened_at="2026-08-22T10:00:00")
        rows = build_dataset([cand], closed_trades=[closed], open_positions=[stale_open])
        assert rows[0]["label"] == "win"


class TestUnmatchedCandidates:
    def test_no_match_anywhere_is_unmatched_not_dropped(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        rows = build_dataset([cand], closed_trades=[], open_positions=[])
        assert len(rows) == 1  # not silently dropped
        assert rows[0]["label"] == "unmatched"

    def test_missing_opened_at_on_accepted_candidate_is_unmatched(self):
        cand = _candidate(decision="accepted")
        cand["opened_at"] = None  # malformed/legacy row
        rows = build_dataset([cand], closed_trades=[_closed_trade()], open_positions=[])
        assert rows[0]["label"] == "unmatched"


class TestPreservesOriginalFields:
    def test_all_original_candidate_fields_survive_in_output_row(self):
        cand = _candidate(decision="accepted", opened_at="2026-08-22T10:00:00")
        rows = build_dataset([cand], closed_trades=[_closed_trade()], open_positions=[])
        row = rows[0]
        assert row["symbol"] == "BTCUSDT"
        assert row["confidence"] == 70.0
        assert row["features"]["market_structure"] == "bullish"
        assert row["entry_zone"] == [100.0, 100.0]
