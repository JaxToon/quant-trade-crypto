"""
Tests for gnn/feature_norm.py — the z-score normalization stats used to fix
the stock GNN collapse (confirmed via a held-out directional-accuracy test:
55.0% -> 76.9% after normalizing; see gnn/feature_norm.py docstring).

Isolated from real checkpoints via monkeypatching MODEL_CHECKPOINT_DIR to
tmp_path, same pattern test_hmm_regime.py uses for HMM_CHECKPOINT.
"""
from __future__ import annotations

import pytest

from config import settings
from gnn import feature_norm


@pytest.fixture(autouse=True)
def isolated_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_DIR", str(tmp_path))


def test_load_returns_none_when_no_file_exists():
    assert feature_norm.load_feature_norm("stock") is None


def test_save_then_load_roundtrip():
    means = [1.0, 2.0, 3.0]
    stds = [0.5, 1.5, 2.5]
    path = feature_norm.save_feature_norm(means, stds, asset_type="stock")
    assert path.endswith("gnn_feature_norm_stock.json")

    loaded_means, loaded_stds = feature_norm.load_feature_norm("stock")
    assert loaded_means == means
    assert loaded_stds == stds


def test_stats_are_isolated_per_asset_type():
    feature_norm.save_feature_norm([1.0], [1.0], asset_type="stock")
    assert feature_norm.load_feature_norm("crypto") is None
    assert feature_norm.load_feature_norm("stock") is not None


def test_compute_feature_stats_pools_across_symbols_and_samples():
    samples = [
        {"features": {"AAPL": [0.0, 10.0], "MSFT": [2.0, 20.0]}},
        {"features": {"AAPL": [4.0, 30.0], "MSFT": [2.0, 20.0]}},
    ]
    means, stds = feature_norm.compute_feature_stats(samples, n_features=2)
    assert means[0] == pytest.approx(2.0)   # (0+2+4+2)/4
    assert means[1] == pytest.approx(20.0)  # (10+20+30+20)/4
    assert stds[0] > 0
    assert stds[1] > 0


def test_compute_feature_stats_std_floor_avoids_division_by_zero():
    samples = [{"features": {"AAPL": [5.0]}}, {"features": {"AAPL": [5.0]}}]
    means, stds = feature_norm.compute_feature_stats(samples, n_features=1)
    assert means[0] == 5.0
    assert stds[0] > 0  # floored, not exactly 0


def test_normalize_feature_rows_zscores_each_column_independently():
    rows = [[0.0, 100.0], [10.0, 200.0]]
    means = [5.0, 150.0]
    stds = [5.0, 50.0]
    normalized = feature_norm.normalize_feature_rows(rows, means, stds)
    assert normalized == [[-1.0, -1.0], [1.0, 1.0]]


def test_should_normalize_only_stock():
    """
    Only validated for stock (55.0% -> 76.9% held-out directional accuracy).
    Regression guard for a real incident: normalizing crypto unconditionally
    once fed z-scored features into the already-running crypto process's
    still-raw-trained in-memory model — train_gnn.py must stay gated.
    """
    assert feature_norm.should_normalize("stock") is True
    assert feature_norm.should_normalize("crypto") is False
