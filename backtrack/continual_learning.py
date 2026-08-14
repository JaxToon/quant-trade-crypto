"""
Contrastive loss update for the GNN after a failed trade.

Implements the PDF formula:
  L_backtrack = L_prediction + λ * Σ(1 - α_ij)²

Where α_ij is the attention weight the model gave to a critical risk node
that it should have weighted highly but didn't.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.optim as optim

from config import settings

if TYPE_CHECKING:
    from gnn.trainer import GNNTrainer
    from backtrack.paper_trading import PaperTradingLedger

logger = logging.getLogger(__name__)


def compute_backtrack_loss(
    prediction_loss: torch.Tensor,
    attention_weights: dict,
    critical_edge_indices: list[tuple[int, int]],
    lambda_weight: float = settings.BACKTRACK_LAMBDA,
) -> torch.Tensor:
    """
    L_backtrack = L_prediction + λ * Σ(1 - α_ij)²

    critical_edge_indices: list of (src_idx, dst_idx) for edges the model
    should have attended to but didn't (discovered by LLM causal discovery).
    """
    if not critical_edge_indices or not attention_weights:
        return prediction_loss

    penalty = torch.tensor(0.0, dtype=torch.float, requires_grad=True)

    for edge_type, (edge_index, alpha) in attention_weights.items():
        # alpha shape: [num_edges, num_heads]
        if alpha is None:
            continue
        mean_alpha = alpha.mean(dim=-1)  # [num_edges]

        for src_idx, dst_idx in critical_edge_indices:
            # Find if this critical edge exists in the edge_index
            match = (
                (edge_index[0] == src_idx) & (edge_index[1] == dst_idx)
            ).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                edge_alpha = mean_alpha[match[0]]
                # Penalize: model should have given α=1 to this critical edge
                penalty = penalty + (1 - edge_alpha) ** 2

    total = prediction_loss + lambda_weight * penalty
    return total


def mini_batch_update(
    trainer: "GNNTrainer",
    labels: dict[str, float],
    critical_edge_indices: list[tuple[int, int]],
) -> float:
    """
    Run one mini-batch backtrack update on the GNN.
    Returns the backtrack loss value.
    """
    from gnn.dataset import build_hetero_data

    if trainer.model is None:
        trainer.build_model()

    data, _ = build_hetero_data()
    device = trainer.device

    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {
        et: data[et].edge_index.to(device)
        for et in data.edge_types
        if hasattr(data[et], "edge_index")
    }

    token_symbols = data["Token"].symbol if hasattr(data["Token"], "symbol") else []
    label_vals = [labels.get(sym, 0.0) for sym in token_symbols]
    y = torch.tensor(label_vals, dtype=torch.float, device=device).unsqueeze(1)

    optimizer = optim.Adam(trainer.model.parameters(), lr=settings.GNN_LR * 0.1)
    criterion = torch.nn.MSELoss()

    trainer.model.train()
    optimizer.zero_grad()

    out = trainer.model(x_dict, edge_index_dict)
    pred_loss = criterion(out, y)

    attn = trainer.get_attention_weights("BTCUSDT")  # representative symbol
    total_loss = compute_backtrack_loss(pred_loss, attn, critical_edge_indices)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
    optimizer.step()

    trainer._ready = True
    loss_val = float(total_loss.item())
    logger.info("Backtrack update — L_backtrack=%.4f (L_pred=%.4f)", loss_val, float(pred_loss.item()))
    return loss_val


def learn_from_paper_trades(trainer: "GNNTrainer", ledger: "PaperTradingLedger") -> float | None:
    """
    Fine-tune the GNN on realized paper-trade outcomes — both wins and
    losses, not just losses, since a win confirms the directional call was
    right just as much as a loss confirms it was wrong.

    This is NOT reinforcement learning (no reward/policy/value function) —
    it's the same contrastive-style supervised update as mini_batch_update
    above, just fed from paper-trading closes instead of real Neo4j Trade
    nodes (which accumulate too slowly to be a useful learning signal).

    Label per symbol = realized market return, clamped to [-1, +1] at a
    ±5% move (matches the "5m-candle-driven SL/TP" scale of these trades),
    averaged across all newly closed trades for that symbol. This is the
    same momentum-label convention used in train_gnn.py, so the target
    stays consistent between bootstrap training and this ongoing update.
    """
    trades = ledger.get_unlearned_trades()
    if len(trades) < settings.LEARNING_MIN_NEW_TRADES:
        return None

    per_symbol: dict[str, list[float]] = {}
    for t in trades:
        entry, exit_ = t["entry_price"], t["exit_price"]
        if not entry:
            continue
        # Realized market direction — independent of whether we went long or
        # short, since that's what the GNN's directional signal represents.
        raw_return = (exit_ - entry) / entry
        label = max(-1.0, min(1.0, raw_return / 0.05))
        per_symbol.setdefault(t["symbol"], []).append(label)

    if not per_symbol:
        ledger.mark_learned()
        return None

    labels = {sym: sum(vals) / len(vals) for sym, vals in per_symbol.items()}
    loss_val = mini_batch_update(trainer, labels, critical_edge_indices=[])
    trainer.save()
    ledger.mark_learned()
    ledger.save()

    logger.info(
        "Learned from %d new paper trades across %d symbols — loss=%.4f",
        len(trades), len(per_symbol), loss_val,
    )
    return loss_val
