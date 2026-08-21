"""
Tests for main.py's rejection-history persistence (paper_trade_rejections
was already computed every cycle but only ever written into the live
status payload, overwritten on the next tick -- no data existed to
analyze "rejection distribution over time" from).

Two things this must prove, per the explicit guardrails agreed before
building this (2026-08-21):
  1. The history actually records correctly and stays bounded/isolated
     (tests in TestRecordRejection / TestRejectionHistoryBounds /
     TestFailureIsolation).
  2. _run_paper_trading's DECISION PATH is provably unchanged --
     maybe_open/check_and_close/save are called with the exact same
     arguments as before this was added, and rejection recording never
     runs when there's nothing to record (TestDecisionPathUnchanged).
     This is the more important half: the whole point of "observability,
     not decision infrastructure" is worthless if not actually verified.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import main


def _insight(symbol="AAPL", timeframe="15m", price=150.0, confidence=62.0):
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": price,
        "confidence": confidence,
        "spot": {"actionable": True, "direction": "long", "entry_zone": [price, price], "stop_loss": price * 0.98, "target": price * 1.02},
        "derivative": {"actionable": True, "direction": "long", "entry_zone": [price, price], "stop_loss": price * 0.98, "target": price * 1.02},
    }


class TestRecordRejection:
    def test_appends_expected_fields_and_persists(self, tmp_path, monkeypatch):
        path = str(tmp_path / "rejection_history_test.json")
        monkeypatch.setattr(main, "REJECTION_HISTORY_FILE", path)
        monkeypatch.setattr(main, "_rejection_history", [])

        main._record_rejection("BTCUSDT", "5m", "spot", "Stop terlalu rapat", 65000.0, 78.0)

        assert len(main._rejection_history) == 1
        entry = main._rejection_history[0]
        assert entry["symbol"] == "BTCUSDT"
        assert entry["timeframe"] == "5m"
        assert entry["market"] == "spot"
        assert entry["reason"] == "Stop terlalu rapat"
        assert entry["price"] == 65000.0
        assert entry["confidence"] == 78.0
        assert "timestamp" in entry

        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk == main._rejection_history

    def test_load_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "REJECTION_HISTORY_FILE", str(tmp_path / "does_not_exist.json"))
        assert main._load_rejection_history() == []


class TestRejectionHistoryBounds:
    def test_truncates_to_max_points(self, tmp_path, monkeypatch):
        path = str(tmp_path / "rejection_history_test.json")
        monkeypatch.setattr(main, "REJECTION_HISTORY_FILE", path)
        monkeypatch.setattr(main, "_rejection_history", [])
        monkeypatch.setattr(main, "MAX_REJECTION_HISTORY_POINTS", 5)

        for i in range(8):
            main._record_rejection("ETHUSDT", "1h", "spot", f"reason_{i}", 3000.0 + i, 50.0)

        assert len(main._rejection_history) == 5
        # oldest entries dropped -- the 5 most recent survive, in order
        assert [e["reason"] for e in main._rejection_history] == [f"reason_{i}" for i in range(3, 8)]


class TestFailureIsolation:
    def test_write_failure_does_not_raise(self, monkeypatch):
        """Points the history file at a path that cannot possibly be
        written (a directory that doesn't exist, no parent creation) --
        _record_rejection must swallow this, not propagate it into the
        caller (_run_paper_trading), matching the guardrail that a
        logging bug can never affect the actual trading loop."""
        monkeypatch.setattr(main, "REJECTION_HISTORY_FILE", "Z:/definitely/does/not/exist/rejection_history.json")
        monkeypatch.setattr(main, "_rejection_history", [])

        # Must not raise.
        main._record_rejection("SOLUSDT", "15m", "derivative", "correlation cap", 150.0, 90.0)

        # The in-memory list still grows even if the disk write failed --
        # only the persistence step is best-effort, not the observation itself.
        assert len(main._rejection_history) == 1


class TestDecisionPathUnchanged:
    """The important half: proves rejection recording is purely additive
    and never alters what the ledger actually decides or does."""

    def test_maybe_open_check_and_close_save_called_exactly_as_before(self, monkeypatch):
        mock_ledger = MagicMock()
        mock_ledger.maybe_open.side_effect = ["Stop terlalu rapat", None]  # spot rejected, derivative allowed
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        recorded = []
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: recorded.append((a, kw)))
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: None)  # not under test here

        insights = {"AAPL_15m": _insight()}
        prices = {"AAPL": 150.0}

        rejections = main._run_paper_trading(insights, prices)

        # Decision calls happened exactly as they always did -- same count,
        # same symbol/market/timeframe/current_price arguments.
        assert mock_ledger.maybe_open.call_count == 2
        spot_call, deriv_call = mock_ledger.maybe_open.call_args_list
        assert spot_call.args[:2] == ("AAPL", "spot")
        assert spot_call.kwargs["timeframe"] == "15m"
        assert spot_call.kwargs["current_price"] == 150.0
        assert deriv_call.args[:2] == ("AAPL", "derivative")
        mock_ledger.check_and_close.assert_called_once_with(prices)
        mock_ledger.save.assert_called_once()

        # Rejection surfaced to the dashboard payload exactly as before.
        assert rejections == {"AAPL_15m": {"spot": "Stop terlalu rapat", "derivative": None}}

        # Recorded exactly the one real rejection (spot), not the allowed derivative leg.
        assert len(recorded) == 1
        args, kwargs = recorded[0]
        assert args[:4] == ("AAPL", "15m", "spot", "Stop terlalu rapat")

    def test_no_rejection_no_recording(self, monkeypatch):
        mock_ledger = MagicMock()
        mock_ledger.maybe_open.return_value = None  # nothing rejected on either leg
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        recorded = []
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: recorded.append((a, kw)))
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: None)  # not under test here

        rejections = main._run_paper_trading({"AAPL_15m": _insight()}, {"AAPL": 150.0})

        assert rejections == {}
        assert recorded == []
        mock_ledger.check_and_close.assert_called_once()
        mock_ledger.save.assert_called_once()

    def test_paper_trading_disabled_never_touches_rejection_recording(self, monkeypatch):
        monkeypatch.setattr(main, "_paper_ledger", None)
        recorded = []
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: recorded.append((a, kw)))
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: None)  # not under test here

        rejections = main._run_paper_trading({"AAPL_15m": _insight()}, {"AAPL": 150.0})

        assert rejections == {}
        assert recorded == []
