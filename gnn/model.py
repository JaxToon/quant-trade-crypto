"""
Heterogeneous Graph Transformer (HGT) model.
Outputs a single directional signal per Token node in [-1, +1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class HGTSignalModel(nn.Module):
    """
    Multi-layer HGT that aggregates information from the full supply/competitor
    graph and emits a directional trading signal per token node.
    """

    def __init__(
        self,
        metadata: tuple,                # from HeteroData.metadata()
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.hidden = hidden_channels

        # Project each node type to a common embedding dimension
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # Stack of HGT convolution layers with LayerNorm for sparse-graph stability
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.2)

        # Final head: map token embedding → scalar signal
        self.out_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            self.dropout,
            Linear(hidden_channels // 2, out_channels),
            nn.Tanh(),  # output in [-1, +1]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        # Initial linear projection per node type
        h_dict = {
            nt: self.lin_dict[nt](x).relu()
            for nt, x in x_dict.items()
        }

        # Message passing with residual skip connections + LayerNorm
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h_dict, edge_index_dict)
            h_dict = {nt: norm(h_new[nt] + h_dict[nt]).relu() for nt in h_new}

        # Predict only for Token nodes
        return self.out_head(h_dict["Token"])   # shape: [num_tokens, 1]

    def attention_weights(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict:
        """
        Returns the attention weights α_ij from the last conv layer.
        Used by the backtrack continual-learning loss.
        """
        h_dict = {nt: self.lin_dict[nt](x).relu() for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                h_dict = conv(h_dict, edge_index_dict)
                h_dict = {nt: h.relu() for nt, h in h_dict.items()}
            else:
                # Last layer: capture attention
                out, attn = conv(h_dict, edge_index_dict, return_attention_weights=True)
                return attn
        return {}
