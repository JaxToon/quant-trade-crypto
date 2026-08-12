"""
GNN training loop and live inference.

Training target: predict next-period price direction (sign of return).
Labels are generated from historical Binance kline data.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from config import settings
from gnn.dataset import build_hetero_data, TOKEN_FEATURES
from gnn.model import HGTSignalModel

logger = logging.getLogger(__name__)


class GNNTrainer:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("GNN using device: %s", self.device)
        self.model: Optional[HGTSignalModel] = None
        self._index_maps: dict | None = None

    # ── Model lifecycle ────────────────────────────────────────────────────

    def build_model(self) -> None:
        data, self._index_maps = build_hetero_data()
        self.model = HGTSignalModel(
            metadata=data.metadata(),
            hidden_channels=settings.GNN_HIDDEN_CHANNELS,
            num_heads=settings.GNN_NUM_HEADS,
            num_layers=settings.GNN_NUM_LAYERS,
        ).to(self.device)

        # Store the edge types this model was built with — new NLP-discovered
        # edge types added later to Neo4j must be filtered out at inference time
        # to avoid HGTConv KeyError. Rebuild model to incorporate new types.
        self._known_edge_types: set[tuple] = set(
            et for et in data.edge_types if hasattr(data[et], "edge_index")
        )

        # Lazy Linear(-1, ...) layers need one forward pass to initialize weights
        self.model.eval()
        with torch.no_grad():
            x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
            edge_index_dict = {
                et: data[et].edge_index.to(self.device)
                for et in self._known_edge_types
            }
            self.model(x_dict, edge_index_dict)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("HGT model built: %d params | edge types: %d", n_params, len(self._known_edge_types))

    def _filter_edges(self, data) -> dict:
        """Return only edge types the model knows about, moved to device."""
        return {
            et: data[et].edge_index.to(self.device)
            for et in data.edge_types
            if hasattr(data[et], "edge_index") and et in self._known_edge_types
        }

    def save(self, path: str | None = None) -> str:
        os.makedirs(settings.MODEL_CHECKPOINT_DIR, exist_ok=True)
        path = path or os.path.join(settings.MODEL_CHECKPOINT_DIR, "hgt_latest.pt")
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)
        return path

    def load(self, path: str | None = None) -> bool:
        path = path or os.path.join(settings.MODEL_CHECKPOINT_DIR, "hgt_latest.pt")
        if not os.path.exists(path):
            return False
        if self.model is None:
            self.build_model()
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except RuntimeError as exc:
            logger.warning(
                "Checkpoint %s is incompatible with the current model architecture "
                "(%s) — ignoring stale checkpoint and running with zero signal until retrained.",
                path, exc,
            )
            return False
        logger.info("Model loaded from %s", path)
        return True

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, labels: dict[str, float], epochs: int | None = None) -> float:
        """
        labels: {symbol → target_signal} where target_signal ∈ {-1, +1}
        Built from next-period returns: +1 if price went up, -1 if down.
        Returns final training loss.
        """
        if self.model is None:
            self.build_model()

        epochs = epochs or settings.GNN_EPOCHS
        optimizer = optim.Adam(self.model.parameters(), lr=settings.GNN_LR, weight_decay=1e-4)
        # Cosine annealing: LR decays smoothly to 0 — prevents loss plateau
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        data, index_maps = build_hetero_data()
        x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
        edge_index_dict = self._filter_edges(data)

        # Build label tensor aligned with token node order
        token_symbols = data["Token"].symbol if hasattr(data["Token"], "symbol") else []
        label_vals = [labels.get(sym, 0.0) for sym in token_symbols]
        y = torch.tensor(label_vals, dtype=torch.float, device=self.device).unsqueeze(1)

        self.model.train()
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(x_dict, edge_index_dict)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            final_loss = loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — loss: %.4f | lr: %.2e | signal_range: [%.3f, %.3f]",
                            epoch + 1, epochs, final_loss,
                            scheduler.get_last_lr()[0],
                            float(out.detach().min()), float(out.detach().max()))

        return final_loss

    def train_on_history(self, samples: list[dict], epochs: int | None = None) -> float:
        """
        Train across many historical (feature, label) snapshots instead of one
        static point — `train()` above fits 150 epochs of gradient descent to a
        single fixed target per symbol, which is memorization, not learning.

        samples: list of {"features": {symbol: [18 floats, TOKEN_FEATURES order]},
                           "label": {symbol: float}}
        Graph topology (edges, Protocol/Macro_Event node features) is read once
        from the live graph and held fixed across all samples — only each
        symbol's Token feature vector and label vary per sample.
        """
        if self.model is None:
            self.build_model()
        if not samples:
            raise ValueError("train_on_history requires at least one sample")

        epochs = epochs or settings.GNN_EPOCHS
        optimizer = optim.Adam(self.model.parameters(), lr=settings.GNN_LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        base_data, index_maps = build_hetero_data()
        edge_index_dict = self._filter_edges(base_data)
        token_symbols = base_data["Token"].symbol if hasattr(base_data["Token"], "symbol") else []
        static_x = {nt: base_data[nt].x.to(self.device) for nt in base_data.node_types if nt != "Token"}

        prepared = []
        for sample in samples:
            feats, lbls = sample["features"], sample["label"]
            token_x = torch.tensor(
                [feats.get(sym, [0.0] * len(TOKEN_FEATURES)) for sym in token_symbols],
                dtype=torch.float, device=self.device,
            )
            y = torch.tensor(
                [lbls.get(sym, 0.0) for sym in token_symbols],
                dtype=torch.float, device=self.device,
            ).unsqueeze(1)
            prepared.append((token_x, y))

        self.model.train()
        final_loss = 0.0
        for epoch in range(epochs):
            random.shuffle(prepared)
            epoch_loss = 0.0
            for token_x, y in prepared:
                optimizer.zero_grad()
                x_dict = {**static_x, "Token": token_x}
                out = self.model(x_dict, edge_index_dict)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            final_loss = epoch_loss / len(prepared)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d — avg loss: %.4f | lr: %.2e | samples: %d",
                    epoch + 1, epochs, final_loss, scheduler.get_last_lr()[0], len(prepared),
                )

        return final_loss

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, symbol: str) -> float:
        """
        Returns the GNN directional signal for a single symbol ∈ [-1, +1].
        Positive → bullish, Negative → bearish.
        """
        if self.model is None:
            if not self.load():
                logger.warning("No trained model found; returning 0.0 signal")
                return 0.0

        data, index_maps = build_hetero_data()
        self.model.eval()
        with torch.no_grad():
            x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
            edge_index_dict = self._filter_edges(data)
            signals = self.model(x_dict, edge_index_dict)  # [num_tokens, 1]

        token_idx = index_maps["Token"].get(symbol, -1)
        if token_idx < 0:
            return 0.0
        return float(signals[token_idx, 0].cpu())

    def predict_all(self) -> dict[str, float]:
        """Returns {symbol → signal} for all watched tokens."""
        if self.model is None:
            if not self.load():
                return {sym: 0.0 for sym in settings.WATCHLIST}

        data, index_maps = build_hetero_data()
        self.model.eval()
        with torch.no_grad():
            x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
            edge_index_dict = self._filter_edges(data)
            signals = self.model(x_dict, edge_index_dict)

        result = {}
        for sym, idx in index_maps["Token"].items():
            result[sym] = float(signals[idx, 0].cpu())
        return result

    def get_attention_weights(
        self, symbol: str
    ) -> dict:
        """Returns raw attention weights from the last conv layer (for backtrack)."""
        if self.model is None:
            return {}
        data, _ = build_hetero_data()
        self.model.eval()
        with torch.no_grad():
            x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
            edge_index_dict = {
                et: data[et].edge_index.to(self.device)
                for et in data.edge_types
                if hasattr(data[et], "edge_index")
            }
            return self.model.attention_weights(x_dict, edge_index_dict)
