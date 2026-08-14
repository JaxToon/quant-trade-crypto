"""
Tests for GNNTrainer._ready (gnn/trainer.py).

Guards a real bug found live: main.py calls gnn.build_model() BEFORE
gnn.load() (to lazy-init the model's Linear(-1,...) layers even in the
no-checkpoint-yet case). That left self.model non-None even when load()
then failed, so predict()/predict_all()'s old `if self.model is None`
guard never re-triggered — a failed load (missing checkpoint, or a stale
checkpoint incompatible with the current graph's edge types, both
reproduced here) silently ran live inference on random, untrained weights
while logging that it was returning zero signal. These tests never let a
DummyModel.forward() run when the trainer isn't genuinely ready — if it
did, the test itself would fail via the AssertionError in forward().
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import settings
from gnn.trainer import GNNTrainer


class DummyModel(nn.Module):
    """Stands in for HGTSignalModel — has real params (for load_state_dict
    shape-checking) but must never actually run inference in these tests."""

    def __init__(self, size: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(size, 1)

    def forward(self, x_dict, edge_index_dict):
        raise AssertionError("forward() must not run on a trainer that isn't _ready")


def _isolated_trainer(monkeypatch, tmp_path) -> GNNTrainer:
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_DIR", str(tmp_path))
    return GNNTrainer()


def test_fresh_trainer_is_not_ready(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    assert trainer._ready is False


def test_predict_all_returns_zero_signal_when_checkpoint_missing(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    # Simulate main.py's build_model() pre-build: self.model is non-None
    # even though nothing has been trained or loaded yet.
    trainer.model = DummyModel()

    result = trainer.predict_all()

    assert result == {sym: 0.0 for sym in settings.WATCHLIST}
    assert trainer._ready is False


def test_predict_returns_zero_signal_when_checkpoint_incompatible(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    # Write a checkpoint with a DIFFERENT param shape — reproduces the real
    # live failure: an older/newer graph topology changes the model's
    # architecture, and load_state_dict raises RuntimeError on shape mismatch.
    ckpt_path = tmp_path / f"hgt_latest_{settings.ASSET_TYPE}.pt"
    torch.save(DummyModel(size=8).state_dict(), ckpt_path)
    trainer.model = DummyModel(size=4)  # pre-built, incompatible shape

    result = trainer.predict("BTCUSDT")

    assert result == 0.0
    assert trainer._ready is False


def test_predict_from_features_returns_zero_signal_when_not_ready(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    trainer.model = DummyModel()

    result = trainer.predict_from_features({"BTCUSDT": [0.0] * 18})

    assert result == {"BTCUSDT": 0.0}


def test_get_attention_weights_returns_empty_when_not_ready(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    trainer.model = DummyModel()

    assert trainer.get_attention_weights("BTCUSDT") == {}


def test_successful_load_sets_ready_true(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    ckpt_path = tmp_path / f"hgt_latest_{settings.ASSET_TYPE}.pt"
    torch.save(DummyModel().state_dict(), ckpt_path)
    trainer.model = DummyModel()  # same shape -> load succeeds

    assert trainer.load() is True
    assert trainer._ready is True


def test_load_with_no_checkpoint_file_stays_not_ready(monkeypatch, tmp_path):
    trainer = _isolated_trainer(monkeypatch, tmp_path)
    assert trainer.load() is False
    assert trainer._ready is False
