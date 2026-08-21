"""
Tests for main.py's candidate-trade logging (Stage 1 of the entry-selection
research line, external playbook review 2026-08-22 -- see RESEARCH_NOTES.md
for the full evaluation). Persists every setup entry_insight marks
actionable, accepted or rejected, with its SMC feature snapshot -- for
future offline meta-label training. No model reads this yet.

Same two things to prove as tests/test_rejection_history.py (the pattern
this file mirrors):
  1. Recording is correct, bounded, and isolated from write failures
     (TestRecordCandidate / TestCandidateLogBounds / TestFailureIsolation).
  2. _run_paper_trading's DECISION PATH is provably unchanged -- maybe_open/
     check_and_close/save are called with the exact same arguments as
     before this was added, and candidate recording never fires for
     non-actionable ("no setup") insights (TestDecisionPathUnchanged /
     TestOnlyActionableLogged).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import main


def _insight(symbol="AAPL", timeframe="15m", price=150.0, confidence=62.0, actionable=True):
    leg = {
        "actionable": actionable,
        "direction": "long" if actionable else "neutral",
        "entry_zone": [price, price] if actionable else None,
        "stop_loss": price * 0.98 if actionable else None,
        "target": price * 1.02 if actionable else None,
        "atr": 1.5,
    }
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": price,
        "confidence": confidence,
        "spot": dict(leg),
        "derivative": dict(leg),
    }


class TestRecordCandidate:
    def test_appends_expected_fields_and_persists(self, tmp_path, monkeypatch):
        path = str(tmp_path / "candidate_trades_test.json")
        monkeypatch.setattr(main, "CANDIDATE_LOG_FILE", path)
        monkeypatch.setattr(main, "_candidate_log", [])

        rec = {"direction": "long", "entry_zone": [65000.0, 65000.0], "stop_loss": 64500.0,
               "target": 66000.0, "confidence": 78.0, "atr": 120.0}
        smc_raw = {"market_structure": "bullish", "smc_signal": 0.3, "active_order_blocks": 2,
                   "active_fvgs": 1, "liquidity_zones": 3, "in_bullish_ob": True,
                   "in_bearish_ob": False, "in_bullish_fvg": True, "in_bearish_fvg": False}

        main._record_candidate("BTCUSDT", "5m", "spot", rec, smc_raw, "accepted", None, "2026-08-22T00:00:00")

        assert len(main._candidate_log) == 1
        entry = main._candidate_log[0]
        assert entry["symbol"] == "BTCUSDT"
        assert entry["timeframe"] == "5m"
        assert entry["market"] == "spot"
        assert entry["direction"] == "long"
        assert entry["entry_zone"] == [65000.0, 65000.0]
        assert entry["stop_loss"] == 64500.0
        assert entry["target"] == 66000.0
        assert entry["confidence"] == 78.0
        assert entry["decision"] == "accepted"
        assert entry["reason"] is None
        assert entry["opened_at"] == "2026-08-22T00:00:00"
        assert entry["features"]["market_structure"] == "bullish"
        assert entry["features"]["active_order_blocks"] == 2
        assert "timestamp" in entry

        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk == main._candidate_log

    def test_rejected_candidate_records_reason_and_no_opened_at(self, tmp_path, monkeypatch):
        path = str(tmp_path / "candidate_trades_test.json")
        monkeypatch.setattr(main, "CANDIDATE_LOG_FILE", path)
        monkeypatch.setattr(main, "_candidate_log", [])

        rec = {"direction": "short", "entry_zone": [3000.0, 3000.0], "stop_loss": 3010.0,
               "target": 2970.0, "confidence": 45.0, "atr": 5.0}
        main._record_candidate("ETHUSDT", "15m", "derivative", rec, {}, "rejected",
                                "Stop terlalu rapat untuk fee", None)

        entry = main._candidate_log[0]
        assert entry["decision"] == "rejected"
        assert entry["reason"] == "Stop terlalu rapat untuk fee"
        assert entry["opened_at"] is None

    def test_load_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "CANDIDATE_LOG_FILE", str(tmp_path / "does_not_exist.json"))
        assert main._load_candidate_log() == []


class TestCandidateLogBounds:
    def test_truncates_to_max_points(self, tmp_path, monkeypatch):
        path = str(tmp_path / "candidate_trades_test.json")
        monkeypatch.setattr(main, "CANDIDATE_LOG_FILE", path)
        monkeypatch.setattr(main, "_candidate_log", [])
        monkeypatch.setattr(main, "MAX_CANDIDATE_LOG_POINTS", 5)

        for i in range(8):
            main._record_candidate("SOLUSDT", "1h", "spot", {"confidence": float(i)}, {},
                                    "accepted", None, f"2026-08-22T0{i}:00:00")

        assert len(main._candidate_log) == 5
        assert [e["confidence"] for e in main._candidate_log] == [3.0, 4.0, 5.0, 6.0, 7.0]


class TestFailureIsolation:
    def test_write_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(main, "CANDIDATE_LOG_FILE", "Z:/definitely/does/not/exist/candidate_trades.json")
        monkeypatch.setattr(main, "_candidate_log", [])

        main._record_candidate("SOLUSDT", "15m", "derivative", {"confidence": 90.0}, {},
                                "accepted", None, "2026-08-22T00:00:00")

        # In-memory list still grows even if the disk write failed.
        assert len(main._candidate_log) == 1


class TestDecisionPathUnchanged:
    """The important half: proves candidate recording is purely additive
    and never alters what the ledger actually decides or does."""

    def test_maybe_open_check_and_close_save_called_exactly_as_before(self, monkeypatch):
        mock_ledger = MagicMock()
        mock_ledger.open_positions = {}  # nothing already open -- both legs are genuinely new
        mock_ledger.maybe_open.side_effect = ["Stop terlalu rapat", None]  # spot rejected, derivative allowed
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: None)
        recorded = []
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: recorded.append((a, kw)))

        insights = {"AAPL_15m": _insight()}
        prices = {"AAPL": 150.0}

        rejections = main._run_paper_trading(insights, prices)

        assert mock_ledger.maybe_open.call_count == 2
        spot_call, deriv_call = mock_ledger.maybe_open.call_args_list
        assert spot_call.args[:2] == ("AAPL", "spot")
        assert spot_call.kwargs["timeframe"] == "15m"
        assert spot_call.kwargs["current_price"] == 150.0
        assert deriv_call.args[:2] == ("AAPL", "derivative")
        mock_ledger.check_and_close.assert_called_once_with(prices)
        mock_ledger.save.assert_called_once()

        assert rejections == {"AAPL_15m": {"spot": "Stop terlalu rapat", "derivative": None}}

        # Both legs were actionable -- both get logged, one rejected one accepted.
        assert len(recorded) == 2
        (spot_args, _), (deriv_args, _) = recorded
        assert spot_args[0:3] == ("AAPL", "15m", "spot")
        assert spot_args[5] == "rejected"
        assert spot_args[6] == "Stop terlalu rapat"
        assert deriv_args[0:3] == ("AAPL", "15m", "derivative")
        assert deriv_args[5] == "accepted"
        assert deriv_args[6] is None

    def test_paper_trading_disabled_never_touches_candidate_recording(self, monkeypatch):
        monkeypatch.setattr(main, "_paper_ledger", None)
        recorded = []
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: recorded.append((a, kw)))

        rejections = main._run_paper_trading({"AAPL_15m": _insight()}, {"AAPL": 150.0})

        assert rejections == {}
        assert recorded == []


class TestOnlyActionableLogged:
    def test_non_actionable_insight_is_not_recorded(self, monkeypatch):
        mock_ledger = MagicMock()
        mock_ledger.open_positions = {}
        mock_ledger.maybe_open.return_value = "not actionable"
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: None)
        recorded = []
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: recorded.append((a, kw)))

        main._run_paper_trading({"AAPL_15m": _insight(actionable=False)}, {"AAPL": 150.0})

        # No setup at all this tick -- nothing "became eligible to act",
        # so nothing should be logged as a candidate.
        assert recorded == []

    def test_already_open_position_is_not_relogged_as_a_new_candidate(self, monkeypatch):
        """maybe_open() returns None for BOTH "opened a new position" and
        "already open here, no-op" (see its own docstring) -- this is the
        real bug found while sanity-running the historical backtest
        version of this same logic: without checking open_positions
        BEFORE the call, every 60s cycle a still-valid zone stays
        actionable re-logs it as if it were a brand new candidate."""
        mock_ledger = MagicMock()
        mock_ledger.open_positions = {("AAPL", "spot", "15m"): object(), ("AAPL", "derivative", "15m"): object()}
        mock_ledger.maybe_open.return_value = None  # both legs: "already open" no-op
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: None)
        recorded = []
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: recorded.append((a, kw)))

        main._run_paper_trading({"AAPL_15m": _insight(actionable=True)}, {"AAPL": 150.0})

        # Both legs were already open -- maybe_open() still gets called
        # (decision path unchanged), but nothing gets logged as a "new"
        # candidate since nothing newly became eligible to act.
        assert mock_ledger.maybe_open.call_count == 2
        assert recorded == []

    def test_newly_opened_position_is_logged_but_not_the_following_already_open_cycle(self, monkeypatch):
        """End-to-end sanity check across two consecutive cycles on the
        same mutable open_positions dict, mirroring how the real ledger
        behaves: cycle 1 opens (logged), cycle 2 finds it already open
        (not re-logged)."""
        mock_ledger = MagicMock()
        mock_ledger.open_positions = {}

        def fake_maybe_open(symbol, market, rec, timeframe, current_price):
            key = (symbol, market, timeframe)
            if key in mock_ledger.open_positions:
                return None
            mock_ledger.open_positions[key] = object()  # simulate a real open
            return None

        mock_ledger.maybe_open.side_effect = fake_maybe_open
        monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
        monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: None)
        recorded = []
        monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: recorded.append((a, kw)))

        insights = {"AAPL_15m": _insight(actionable=True)}
        main._run_paper_trading(insights, {"AAPL": 150.0})
        assert len(recorded) == 2  # spot + derivative, both newly opened

        recorded.clear()
        main._run_paper_trading(insights, {"AAPL": 150.0})
        assert recorded == []  # same cycle again -- both already open now
