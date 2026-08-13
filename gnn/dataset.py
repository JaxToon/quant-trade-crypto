"""
Pulls the live knowledge graph from Neo4j and converts it into
a PyTorch Geometric HeteroData object for GNN inference/training.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from config import settings
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Node feature order for Token nodes
TOKEN_FEATURES = [
    # Technical indicators (existing)
    "price", "volume", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",

    # Smart Money Concepts (new - institutional order flow)
    "smc_signal", "smc_market_structure_encoded", "smc_active_obs",
    "smc_active_fvgs", "smc_liquidity_zones",
    "smc_in_bullish_ob", "smc_in_bearish_ob",
    "smc_in_bullish_fvg", "smc_in_bearish_fvg",
]
# Now 18 features per Token node (9 technical + 9 SMC)

# All graph node types
NODE_TYPES = ["Token", "Protocol", "Macro_Event"]

# All relationship types (src_type, rel, dst_type)
EDGE_TYPES = [
    ("Token", "BUILT_ON", "Protocol"),
    ("Token", "COMPETES_WITH", "Token"),
    ("Token", "DEPENDS_ON", "Protocol"),
    ("Token", "IMPACTS", "Token"),
    ("Protocol", "DEPENDS_ON", "Token"),
    ("Macro_Event", "IMPACTS", "Token"),
    ("Token", "SUPPLIES", "Token"),
    ("Token", "REGULATES", "Token"),
]


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _encode_market_structure(ms: str) -> float:
    """Encode market structure as numeric value for GNN input."""
    if ms == "bullish":
        return 1.0
    elif ms == "bearish":
        return -1.0
    else:  # ranging or unknown
        return 0.0


def build_hetero_data(asset_type: str | None = None) -> tuple[HeteroData, dict[str, dict]]:
    """
    Queries Neo4j and builds a HeteroData graph.
    Returns (data, index_maps) where index_maps maps node_type → {id → int_index}.

    asset_type: defaults to settings.ASSET_TYPE — the GNN is trained and run
    separately per asset class (crypto vs stock), so by default only that
    class's Token nodes are pulled in. Edges/Protocol/Macro_Event nodes
    connected only to the other asset class are naturally excluded too, since
    any edge referencing a filtered-out token gets skipped below (its index
    lookup misses and the edge is dropped).
    """
    asset_type = asset_type or settings.ASSET_TYPE
    client = Neo4jClient.get()
    data = HeteroData()
    index_maps: dict[str, dict] = {nt: {} for nt in NODE_TYPES}

    # ── Token nodes ────────────────────────────────────────────────────────
    tokens = client.get_all_tokens(asset_type=asset_type)
    token_feats = []
    for i, row in enumerate(tokens):
        node = dict(row["t"])
        sym = node.get("symbol", f"unk_{i}")
        index_maps["Token"][sym] = i

        # Extract features with SMC support
        feats = []
        for f in TOKEN_FEATURES:
            if f == "smc_market_structure_encoded":
                # Special handling for categorical encoding
                ms = node.get("smc_market_structure", "unknown")
                feats.append(_encode_market_structure(ms))
            elif f.startswith("smc_in_"):
                # Boolean features: convert True/False to 1.0/0.0
                feats.append(1.0 if node.get(f, False) else 0.0)
            else:
                # Numeric features: use safe_float
                feats.append(_safe_float(node.get(f)))

        token_feats.append(feats)

    if token_feats:
        data["Token"].x = torch.tensor(token_feats, dtype=torch.float)
        data["Token"].symbol = [dict(row["t"]).get("symbol", "") for row in tokens]
    else:
        data["Token"].x = torch.zeros((1, len(TOKEN_FEATURES)), dtype=torch.float)

    # ── Protocol nodes (simple 1-hot presence feature) ────────────────────
    protocols = client.run_read("MATCH (p:Protocol) RETURN p")
    for i, row in enumerate(protocols):
        pid = dict(row["p"]).get("pid", f"proto_{i}")
        index_maps["Protocol"][pid] = i
    n_proto = max(len(protocols), 1)
    data["Protocol"].x = torch.eye(n_proto, dtype=torch.float)

    # ── Macro_Event nodes ──────────────────────────────────────────────────
    events = client.run_read("MATCH (e:Macro_Event) RETURN e")
    for i, row in enumerate(events):
        eid = dict(row["e"]).get("eid", f"ev_{i}")
        index_maps["Macro_Event"][eid] = i
        # sentiment_score as single feature
    n_ev = max(len(events), 1)
    ev_feats = []
    for row in events:
        score = _safe_float(dict(row["e"]).get("sentiment_score", 0.5))
        ev_feats.append([score])
    data["Macro_Event"].x = torch.tensor(ev_feats if ev_feats else [[0.0]], dtype=torch.float)

    # ── Edges ──────────────────────────────────────────────────────────────
    edge_list = client.get_edge_list()
    edge_buckets: dict[tuple, tuple[list, list, list]] = {}

    for row in edge_list:
        src_type = row["src_type"]
        dst_type = row["dst_type"]
        rel = row["rel_type"]
        key = (src_type, rel, dst_type)

        src_sym = row.get("src_symbol") or str(row.get("src_id"))
        dst_sym = row.get("dst_symbol") or str(row.get("dst_id"))

        src_idx = index_maps.get(src_type, {}).get(src_sym, -1)
        dst_idx = index_maps.get(dst_type, {}).get(dst_sym, -1)
        conf = float(row.get("confidence", 1.0))

        if src_idx < 0 or dst_idx < 0:
            continue
        if key not in edge_buckets:
            edge_buckets[key] = ([], [], [])
        edge_buckets[key][0].append(src_idx)
        edge_buckets[key][1].append(dst_idx)
        edge_buckets[key][2].append(conf)

    for (src_type, rel, dst_type), (srcs, dsts, confs) in edge_buckets.items():
        if not srcs:
            continue
        data[src_type, rel, dst_type].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        data[src_type, rel, dst_type].edge_attr = torch.tensor(confs, dtype=torch.float).unsqueeze(1)

    return data, index_maps
