"""
Tests for main.py's _is_hmm_checkpoint_stale() -- the startup catch-up
check added 2026-08-26 after finding the weekly HMM refit
(schedule.every().monday.at("01:00")) has no catch-up mechanism: a process
restart/crash spanning a Monday 01:00 window silently skips that week's
refit, and the checkpoint can age indefinitely with nothing to notice.
Confirmed live: an 8+ day stale checkpoint's volatility clusters no longer
matched real market conditions, causing HIGH_STRESS_TURBULENT to fire on
~95% of regime checks (vs ~34-45% historically) -- silently blocking
Path 2's entire debate cycle via circuit.bear_veto_active.
"""
from __future__ import annotations

import os
import time

import main


class TestIsHmmCheckpointStale:
    def test_missing_checkpoint_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "HMM_CHECKPOINT", str(tmp_path / "does_not_exist.pkl"))
        assert main._is_hmm_checkpoint_stale() is True

    def test_freshly_written_checkpoint_is_not_stale(self, tmp_path, monkeypatch):
        path = tmp_path / "hmm_regime_test.pkl"
        path.write_bytes(b"fake checkpoint bytes")
        monkeypatch.setattr(main, "HMM_CHECKPOINT", str(path))
        assert main._is_hmm_checkpoint_stale() is False

    def test_checkpoint_older_than_max_age_is_stale(self, tmp_path, monkeypatch):
        path = tmp_path / "hmm_regime_test.pkl"
        path.write_bytes(b"fake checkpoint bytes")
        # Back-date mtime to 8 days ago -- past the default 7-day cadence.
        eight_days_ago = time.time() - 8 * 24 * 3600
        os.utime(path, (eight_days_ago, eight_days_ago))
        monkeypatch.setattr(main, "HMM_CHECKPOINT", str(path))
        assert main._is_hmm_checkpoint_stale() is True

    def test_checkpoint_just_under_max_age_is_not_stale(self, tmp_path, monkeypatch):
        path = tmp_path / "hmm_regime_test.pkl"
        path.write_bytes(b"fake checkpoint bytes")
        six_days_ago = time.time() - 6 * 24 * 3600
        os.utime(path, (six_days_ago, six_days_ago))
        monkeypatch.setattr(main, "HMM_CHECKPOINT", str(path))
        assert main._is_hmm_checkpoint_stale() is False

    def test_custom_max_age_days_respected(self, tmp_path, monkeypatch):
        path = tmp_path / "hmm_regime_test.pkl"
        path.write_bytes(b"fake checkpoint bytes")
        two_days_ago = time.time() - 2 * 24 * 3600
        os.utime(path, (two_days_ago, two_days_ago))
        monkeypatch.setattr(main, "HMM_CHECKPOINT", str(path))
        assert main._is_hmm_checkpoint_stale(max_age_days=7) is False
        assert main._is_hmm_checkpoint_stale(max_age_days=1) is True
