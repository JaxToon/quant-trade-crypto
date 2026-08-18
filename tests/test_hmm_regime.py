"""
Tests for risk/hmm_regime.py — written to avoid the two side effects a
naive test here would have: hitting real Binance/yfinance/alternative.me
APIs, and overwriting the REAL model_checkpoints/hmm_regime_{asset_type}.pkl
files main.py's live process actually uses.

Achieved by:
  - monkeypatching HMM_CHECKPOINT to a tmp_path for every test (never the
    real checkpoint directory)
  - monkeypatching the _fetch_benchmark_closes / _fetch_benchmark_volatility
    / _fetch_fear_greed methods directly with synthetic data, instead of
    mocking requests/yfinance at the wire level — these methods are the
    real boundary between "network I/O" and "everything this module
    actually decides", so patching there tests the real decision logic
    without needing a fake HTTP/yfinance response format.

Covers the three real bugs found and fixed in this file this session:
  1. benchmark hardcoded to BTC regardless of ASSET_TYPE
  2. checkpoint path collision between crypto and stock
  3. fit() accepting corrupted data (the mean_abs_change=1.01 incident)
Deliberately does NOT try to unit-test the training/inference feature-
definition-mismatch fix (single-candle vs 24h-cumulative change) — that
would require refactoring fit()'s internals just to make them independently
assertable, which is more invasive than warranted for a test suite this
size. The regime re-ordering logic (raw HMM state -> ordered low/normal/
high) is tested precisely instead, via a hand-constructed model with known
cluster centers, which sidesteps HMM fitting's own randomness entirely.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from hmmlearn import hmm

import risk.hmm_regime as hmm_regime
from config import settings
from risk.hmm_regime import RegimeDetector


@pytest.fixture(autouse=True)
def isolated_checkpoint(tmp_path, monkeypatch):
    """Every test in this file gets its own throwaway checkpoint path —
    never model_checkpoints/hmm_regime_{asset_type}.pkl."""
    monkeypatch.setattr(hmm_regime, "HMM_CHECKPOINT", str(tmp_path / "test_hmm.pkl"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Belt-and-suspenders: fail loudly if a test somehow reaches real
    requests.get instead of relying only on the method-level mocks below.
    Uses pytest.fail() rather than a plain assert/exception — hmm_regime.py
    methods like _fetch_fear_greed_history() wrap their body in a broad
    except Exception, which would silently swallow a regular AssertionError
    (it IS an Exception subclass) and mask the block ever having fired.
    pytest.fail() raises _pytest.outcomes.Failed, which deliberately
    inherits from BaseException instead of Exception for exactly this
    reason — application code's `except Exception` can't catch it."""
    def _blocked(*args, **kwargs):
        pytest.fail("Test attempted a real network call via requests.get")
    monkeypatch.setattr("requests.get", _blocked)


class _FakeKlinesResponse:
    def __init__(self, closes: list[float]) -> None:
        self._closes = closes

    def raise_for_status(self) -> None:
        pass

    def json(self):
        # Binance kline format: index 4 is the close price, rest unused here.
        return [[0, "0", "0", "0", str(c), "0"] for c in self._closes]


class TestFetchRecentWindowFeatures:
    """Direct coverage of the new windowed-fetch preprocessing — must match
    fit()'s feature formula exactly (requirement: identical preprocessing
    between training and live inference)."""

    def test_returns_correctly_shaped_window(self, monkeypatch):
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        closes = _synthetic_closes(n=100)
        monkeypatch.setattr("requests.get", lambda *a, **k: _FakeKlinesResponse(closes))

        result = d._fetch_recent_window_features(window_hours=50)

        assert result is not None
        X, timestamps = result
        assert X.shape == (50, 2)
        assert len(timestamps) == 50

    def test_feature_values_match_fit_formula(self, monkeypatch):
        """vol = rolling 24h std of log returns, abs_change = rolling 24h
        cumulative % move — exactly fit()'s np.column_stack([vol, abs_change])."""
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        closes = _synthetic_closes(n=60)
        monkeypatch.setattr("requests.get", lambda *a, **k: _FakeKlinesResponse(closes))

        X, _ = d._fetch_recent_window_features(window_hours=30)

        log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        expected_vol = [np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))]
        expected_change = [
            abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
            for i in range(len(log_returns))
        ]
        expected = np.column_stack([expected_vol, expected_change])[-30:]
        assert np.allclose(X, expected)

    def test_insufficient_data_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        monkeypatch.setattr("requests.get", lambda *a, **k: _FakeKlinesResponse([100.0, 101.0]))
        assert d._fetch_recent_window_features(window_hours=50) is None


def _synthetic_closes(n: int = 200, seed: int = 42) -> list[float]:
    """A plausible-looking random-walk price series — small per-step
    moves, no absurd single-hour jumps, safely under the fit() sanity
    guard's 1.0 log-return threshold."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=0.003, size=n)  # ~0.3% hourly moves
    closes = 100.0 * np.exp(np.cumsum(steps))
    return closes.tolist()


class TestBenchmarkSelection:
    """The actual bug: regime detection was hardcoded to BTC volatility
    regardless of ASSET_TYPE, so a stock process's bear-veto would react
    to crypto market stress, not equity market stress."""

    def test_crypto_uses_btc(self, monkeypatch):
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        assert d._benchmark == "BTCUSDT"
        assert d._benchmark_yf == "BTC-USD"

    def test_stock_uses_spy(self, monkeypatch):
        monkeypatch.setattr(settings, "ASSET_TYPE", "stock")
        d = RegimeDetector()
        assert d._benchmark == "SPY"
        assert d._benchmark_yf == "SPY"


class TestCheckpointIsolation:
    def test_save_then_load_roundtrips_from_isolated_path(self, tmp_path, monkeypatch):
        """Confirms _save()/_load() actually use the (monkeypatched)
        HMM_CHECKPOINT path, and never touch the real one."""
        real_checkpoint_dir = "model_checkpoints"
        before = set(__import__("os").listdir(real_checkpoint_dir)) if __import__("os").path.isdir(real_checkpoint_dir) else set()

        d = RegimeDetector()
        fake_model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        fake_model.means_ = np.array([[0.01, 0.01], [0.5, 0.5], [0.9, 0.9]])
        d._model = fake_model
        d._save()

        after = set(__import__("os").listdir(real_checkpoint_dir)) if __import__("os").path.isdir(real_checkpoint_dir) else set()
        assert before == after, "fit()/_save() must never write into the real model_checkpoints/ directory during tests"

        d2 = RegimeDetector()  # fresh instance, should _load() the isolated file
        assert d2._model is not None
        assert np.array_equal(d2._model.means_, fake_model.means_)


class TestFitSanityGuard:
    """The mean_abs_change=1.01 incident: a garbled API response that still
    parsed shouldn't get silently persisted as a valid model."""

    def test_rejects_corrupted_data_with_absurd_jump(self, monkeypatch):
        closes = _synthetic_closes()
        closes[100] *= 5.0  # a single-hour 5x jump — not a real market event
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: closes)
        d.fit(client=None, days=10)
        assert d._model is None  # aborted before ever calling _save()

    def test_rejects_too_little_data(self, monkeypatch):
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: [100.0, 101.0, 99.0])
        d.fit(client=None, days=10)
        assert d._model is None

    def test_accepts_clean_data_and_saves(self, monkeypatch):
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: _synthetic_closes())
        monkeypatch.setattr(d, "_fetch_fear_greed_history", lambda days: {})  # explicit, not relying on no_network's swallowed exception
        d.fit(client=None, days=10)
        assert d._model is not None
        assert d._model.n_components == settings.HMM_N_STATES


class TestFearGreedFeature:
    """Fear & Greed used to be fetched every detect_regime() call and then
    thrown away entirely — now a real 3rd feature dimension, crypto only
    (alternative.me is specifically a crypto sentiment index; reusing it
    for stock would repeat the exact benchmark-mismatch bug this file's
    other fix corrected)."""

    def test_crypto_fit_includes_fear_greed_as_third_dimension(self, monkeypatch):
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: _synthetic_closes())
        # One F&G value per day, well within the training window
        fg_history = {(datetime.utcnow() - timedelta(days=i)).date(): 0.5 for i in range(15)}
        monkeypatch.setattr(d, "_fetch_fear_greed_history", lambda days: fg_history)
        d.fit(client=None, days=10)
        assert d._model is not None
        assert d._model.means_.shape[1] == 3

    def test_crypto_fit_falls_back_to_two_features_if_history_unavailable(self, monkeypatch):
        """If the F&G API is down, fit() shouldn't fail or fabricate data —
        it should train the same 2-feature model as before this change."""
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: _synthetic_closes())
        monkeypatch.setattr(d, "_fetch_fear_greed_history", lambda days: {})
        d.fit(client=None, days=10)
        assert d._model is not None
        assert d._model.means_.shape[1] == 2

    def test_stock_fit_never_uses_fear_greed(self, monkeypatch):
        """Stock must stay 2-feature even if _fetch_fear_greed_history
        somehow returned data — alternative.me's index isn't meaningful
        for equities, so stock shouldn't touch it at all."""
        monkeypatch.setattr(settings, "ASSET_TYPE", "stock")
        d = RegimeDetector()
        monkeypatch.setattr(d, "_fetch_benchmark_closes", lambda days: _synthetic_closes())

        def _should_not_be_called(days):
            pytest.fail("Stock fit() must never call _fetch_fear_greed_history")
        monkeypatch.setattr(d, "_fetch_fear_greed_history", _should_not_be_called)

        d.fit(client=None, days=10)
        assert d._model is not None
        assert d._model.means_.shape[1] == 2

    def test_detect_regime_matches_loaded_models_feature_count(self, monkeypatch):
        """obs must match whatever the LOADED model expects, not just
        settings.ASSET_TYPE — an old 2-feature checkpoint should still work
        without a shape-mismatch crash even under ASSET_TYPE=crypto."""
        monkeypatch.setattr(settings, "ASSET_TYPE", "crypto")
        d = RegimeDetector()
        old_style_model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        old_style_model.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
        old_style_model.transmat_ = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        old_style_model.means_ = np.array([[0.01, 0.01], [0.5, 0.5], [0.9, 0.9]])  # only 2 features
        old_style_model.covars_ = np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]])
        d._model = old_style_model
        window = np.tile([[0.01, 0.01]], (5, 1))
        timestamps = [datetime.utcnow() for _ in range(5)]
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: (window, timestamps))

        def _should_not_be_called(days):
            pytest.fail("Should not fetch Fear & Greed history for a 2-feature (old-style) model")
        monkeypatch.setattr(d, "_fetch_fear_greed_history", _should_not_be_called)

        assert d.detect_regime(client=None) == 0  # must not raise on shape mismatch


class TestRegimeOrdering:
    """detect_regime() re-maps the HMM's raw (arbitrary) state indices to
    ordered low/normal/high-volatility by sorting on fitted mean volatility.
    Hand-constructing the model sidesteps HMM fitting's own randomness,
    testing only the re-ordering logic itself, precisely."""

    def _model_with_known_clusters(self) -> hmm.GaussianHMM:
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        model.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
        model.transmat_ = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        # Raw state 0 = mid vol, raw state 1 = LOW vol, raw state 2 = HIGH vol
        # — deliberately NOT in low/mid/high order, so a test passing here
        # actually proves the re-ordering runs, not that raw order happened
        # to already be correct.
        model.means_ = np.array([[0.5, 0.5], [0.01, 0.01], [0.9, 0.9]])
        model.covars_ = np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]])
        return model

    def _window_of(self, vol: float, change: float, n: int = 5) -> tuple[np.ndarray, list]:
        window = np.tile([[vol, change]], (n, 1))
        timestamps = [datetime.utcnow() for _ in range(n)]
        return window, timestamps

    def test_calm_observation_classified_as_low_volatility(self, monkeypatch):
        d = RegimeDetector()
        d._model = self._model_with_known_clusters()
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: self._window_of(0.01, 0.01))
        assert d.detect_regime(client=None) == 0

    def test_mid_observation_classified_as_normal(self, monkeypatch):
        d = RegimeDetector()
        d._model = self._model_with_known_clusters()
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: self._window_of(0.5, 0.5))
        assert d.detect_regime(client=None) == 1

    def test_extreme_observation_classified_as_turbulent(self, monkeypatch):
        d = RegimeDetector()
        d._model = self._model_with_known_clusters()
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: self._window_of(0.9, 0.9))
        assert d.detect_regime(client=None) == 2

    def test_no_model_defaults_to_normal(self):
        d = RegimeDetector()
        assert d._model is None
        assert d.detect_regime(client=None) == 1

    def test_window_fetch_failure_falls_back_to_last_known_state(self, monkeypatch):
        d = RegimeDetector()
        d._model = self._model_with_known_clusters()
        d._last_state = 2
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: None)
        assert d.detect_regime(client=None) == 2


class TestWindowedDecodeFixesStartprobBug:
    """
    The actual bug this session found: detect_regime() used to call
    model.predict(obs) with a SINGLE observation. For a length-1 sequence,
    Viterbi reduces to argmax(log(startprob) + log(emission)) — and a
    Baum-Welch-fitted model's startprob_ can concentrate almost all its
    mass on one state (confirmed on the live checkpoint: ~1e-134 to ~1.0
    spread), so single-frame predict() returned that SAME state for
    virtually any input, regardless of what the observation actually was.

    This test constructs exactly that pathological startprob_ and proves
    the windowed decode correctly tracks RECENT observations instead of
    being stuck on the dominant prior state — the single-observation
    version of this same setup would incorrectly return the startprob-
    dominant state (0) even though the window's later observations clearly
    belong to the turbulent cluster (2).
    """

    def _pathological_model(self) -> hmm.GaussianHMM:
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        # Mirrors the real deployed checkpoint's near-degenerate startprob_.
        model.startprob_ = np.array([1.0 - 2e-10, 1e-10, 1e-10])
        model.transmat_ = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
        model.means_ = np.array([[0.01, 0.01], [0.5, 0.5], [0.9, 0.9]])
        model.covars_ = np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]])
        return model

    def test_recent_turbulent_observations_override_the_dominant_prior(self, monkeypatch):
        d = RegimeDetector()
        d._model = self._pathological_model()
        # First observation matches the startprob-dominant "calm" cluster;
        # the rest clearly belong to the turbulent cluster — a real window
        # would look like this right after a regime shift.
        window = np.array([[0.01, 0.01]] + [[0.9, 0.9]] * 10)
        timestamps = [datetime.utcnow() for _ in range(len(window))]
        monkeypatch.setattr(d, "_fetch_recent_window_features", lambda hours: (window, timestamps))

        assert d.detect_regime(client=None) == 2, (
            "windowed decode must classify based on recent observations, "
            "not collapse to the startprob-dominant state"
        )
