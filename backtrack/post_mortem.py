"""
Nightly post-mortem audit.
Pipeline: [Closed at Loss] → [Extract News] → [LLM Causal Discovery] → [Heal Graph]

Runs as a scheduled job every night at 00:00 UTC.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from openai import OpenAI
import feedparser

from backtrack.clustering import cluster_news, NewsCluster
from backtrack.continual_learning import mini_batch_update
from config import settings
from graph.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from gnn.trainer import GNNTrainer

logger = logging.getLogger(__name__)


def _fetch_news_for_window(hours_back: int = 24) -> list[str]:
    """Pull article titles/summaries from RSS feeds for the last N hours."""
    texts = []
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    for feed_url in settings.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                published_str = entry.get("published", "")
                text = f"{entry.get('title', '')}. {entry.get('summary', '')}"
                texts.append(text)
        except Exception as exc:
            logger.warning("RSS fetch failed: %s", exc)
    return texts


def _llm_causal_discovery(symbol: str, cluster: NewsCluster, client: OpenAI) -> str:
    """Ask the LLM to deduce the missing economic relationship from the anomaly cluster."""
    sample_texts = "\n".join(cluster.texts[:5])
    prompt = (
        f"Token {symbol} hit its stop-loss today. "
        f"Based on this news cluster from that period:\n\n{sample_texts}\n\n"
        "Deduce the missing economic relationship or market dependency that caused this loss. "
        "Format your answer as: SOURCE_ENTITY → RELATIONSHIP_TYPE → TARGET_ENTITY. "
        "Example: 'Federal Reserve → REGULATES → BTCUSDT' or 'ETHUSDT → DEPENDS_ON → ethereum-chain'. "
        "Be concise. One relationship per line, up to 3 lines."
    )
    try:
        response = client.chat.completions.create(
            model=settings.CRITIC_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM causal discovery failed: %s", exc)
        return ""


def _parse_causal_relationships(text: str) -> list[tuple[str, str, str]]:
    """Parse 'SRC → REL → DST' lines from LLM output."""
    results = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split("→")]
        if len(parts) == 3:
            results.append((parts[0], parts[1], parts[2]))
    return results


def _heal_graph(relationships: list[tuple[str, str, str]]) -> None:
    """Add newly discovered edges to the knowledge graph."""
    client = Neo4jClient.get()
    for src, rel, dst in relationships:
        # Determine node types (simple heuristic: USDT suffix = Token)
        src_type = "Token" if src.endswith("USDT") else "Protocol"
        dst_type = "Token" if dst.endswith("USDT") else "Protocol"
        src_key = "symbol" if src_type == "Token" else "pid"
        dst_key = "symbol" if dst_type == "Token" else "pid"
        rel_clean = rel.upper().replace(" ", "_")
        try:
            client.upsert_edge(
                src_type, src_key, src,
                rel_clean,
                dst_type, dst_key, dst,
                confidence=0.70,  # lower confidence for LLM-discovered edges
                source="backtrack_llm",
            )
            logger.info("Healed graph: (%s)-[%s]->(%s)", src, rel_clean, dst)
        except Exception as exc:
            logger.warning("Graph heal failed for %s→%s: %s", src, dst, exc)


def run_nightly_postmortem(gnn_trainer: "GNNTrainer") -> None:
    """
    Full nightly backtrack pipeline. Call this at midnight.
    """
    logger.info("=== Nightly Post-Mortem Starting ===")
    os.makedirs(settings.BACKTRACK_LOG_DIR, exist_ok=True)

    neo_client = Neo4jClient.get()
    losses = neo_client.get_recent_losses(limit=10)

    if not losses:
        logger.info("No losses to post-mortem.")
        return

    anthropic_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Fetch global news from the failure window
    news_texts = _fetch_news_for_window(hours_back=24)
    if not news_texts:
        logger.warning("No news fetched for post-mortem window.")
        return

    # Cluster to find anomaly themes
    clusters = cluster_news(news_texts, min_cluster_size=3)
    if not clusters:
        logger.info("No significant news clusters found.")
        return

    top_cluster = clusters[0]
    logger.info("Top anomaly cluster: %d articles | %s", top_cluster.size, top_cluster.centroid_text[:80])

    # Run causal discovery for each loss symbol
    all_new_relationships = []
    for row in losses:
        trade = dict(row.get("t", {}))
        symbol = trade.get("symbol", "")
        if not symbol:
            continue

        causal_text = _llm_causal_discovery(symbol, top_cluster, anthropic_client)
        logger.info("Causal discovery for %s:\n%s", symbol, causal_text)

        relationships = _parse_causal_relationships(causal_text)
        all_new_relationships.extend(relationships)

        # Log to file
        log_path = os.path.join(settings.BACKTRACK_LOG_DIR, f"{datetime.utcnow().strftime('%Y%m%d')}_{symbol}.txt")
        with open(log_path, "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] {symbol} loss\n")
            f.write(f"Cluster: {top_cluster.centroid_text}\n")
            f.write(f"Causal:\n{causal_text}\n\n")

    # Heal the graph with discovered relationships
    if all_new_relationships:
        _heal_graph(all_new_relationships)

    # Contrastive GNN update
    # Labels: symbols that lost get -1 (wrong prediction), others 0
    loss_symbols = {dict(row.get("t", {})).get("symbol", ""): -1.0 for row in losses}
    mini_batch_update(gnn_trainer, loss_symbols, critical_edge_indices=[])

    # Save updated model
    gnn_trainer.save()
    logger.info("=== Nightly Post-Mortem Complete ===")
