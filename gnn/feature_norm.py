"""
Per-feature z-score normalization stats for GNN Token features, persisted
per asset_type so a norm file for one asset class never affects the other.

Applying this is opt-in at inference time (build_hetero_data only
normalizes if a stats file exists for the current asset_type) — this is
what keeps the crypto path byte-for-byte unchanged unless crypto is
deliberately retrained with normalization too. Confirmed via a held-out
directional-accuracy test that normalization fixes a real collapse in the
stock GNN (was returning a near-constant output regardless of input;
55.0% -> 76.9% holdout directional accuracy after normalizing).
"""
from __future__ import annotations

import json
import os

from config import settings


# Normalization is only validated for stock (55.0% -> 76.9% held-out
# directional accuracy after fixing a real output collapse). Crypto's own
# GNN ablation test showed confluence has negligible effect on backtested
# performance either way, so there's no established case to normalize it —
# and doing so once already caused a live incident: retraining crypto with
# normalization silently fed z-scored features into the ALREADY-RUNNING
# process's still-raw-trained in-memory model for as long as the stats
# file existed. train_gnn.py checks this instead of inlining the asset
# check, so the gate has one definition and is unit-testable without
# mocking the yfinance-dependent training pipeline.
NORMALIZE_ASSET_TYPES = {"stock"}


def should_normalize(asset_type: str) -> bool:
    return asset_type in NORMALIZE_ASSET_TYPES


def _path(asset_type: str | None = None) -> str:
    asset_type = asset_type or settings.ASSET_TYPE
    return os.path.join(settings.MODEL_CHECKPOINT_DIR, f"gnn_feature_norm_{asset_type}.json")


def save_feature_norm(means: list[float], stds: list[float], asset_type: str | None = None) -> str:
    path = _path(asset_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"means": means, "stds": stds}, f)
    return path


def load_feature_norm(asset_type: str | None = None) -> tuple[list[float], list[float]] | None:
    path = _path(asset_type)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["means"], data["stds"]


def compute_feature_stats(samples: list[dict], n_features: int) -> tuple[list[float], list[float]]:
    """
    samples: list of {"features": {symbol: [n_features floats]}, ...} as
    produced by train_gnn.py. Stats are pooled across every symbol and every
    sample — matches how build_hetero_data() applies one shared per-feature
    mean/std across all Token nodes at inference time.
    """
    sums = [0.0] * n_features
    sq_sums = [0.0] * n_features
    n = 0
    for s in samples:
        for feats in s["features"].values():
            for j, v in enumerate(feats):
                sums[j] += v
                sq_sums[j] += v * v
            n += 1
    means = [s / n for s in sums]
    stds = [max((sq_sums[j] / n - means[j] ** 2) ** 0.5, 1e-6) for j in range(n_features)]
    return means, stds


def normalize_feature_rows(rows: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(v - means[j]) / stds[j] for j, v in enumerate(row)] for row in rows]
