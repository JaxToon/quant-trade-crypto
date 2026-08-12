"""
HDBSCAN clustering over sentence embeddings of news articles.
Used to identify macro anomaly clusters that caused a failed trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_embedder = None
_hdbscan_cls = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence embedder loaded")
    return _embedder


def _get_hdbscan():
    global _hdbscan_cls
    if _hdbscan_cls is None:
        import hdbscan
        _hdbscan_cls = hdbscan.HDBSCAN
    return _hdbscan_cls


@dataclass
class NewsCluster:
    cluster_id: int
    texts: list[str]
    centroid_text: str      # most representative text (closest to centroid)
    size: int


def cluster_news(texts: list[str], min_cluster_size: int = 3) -> list[NewsCluster]:
    """
    Embed texts and cluster with HDBSCAN.
    Returns list of NewsCluster objects (excludes noise cluster -1).
    """
    if len(texts) < min_cluster_size:
        logger.debug("Too few texts (%d) for clustering.", len(texts))
        return []

    embedder = _get_embedder()
    HDBSCAN = _get_hdbscan()

    embeddings = embedder.encode(texts, show_progress_bar=False)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True)
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels) - {-1}
    clusters = []
    for label in unique_labels:
        mask = labels == label
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        cluster_embeds = embeddings[mask]
        centroid = cluster_embeds.mean(axis=0)
        dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
        centroid_text = cluster_texts[int(np.argmin(dists))]
        clusters.append(NewsCluster(
            cluster_id=int(label),
            texts=cluster_texts,
            centroid_text=centroid_text,
            size=len(cluster_texts),
        ))

    logger.info("Clustered %d texts → %d clusters", len(texts), len(clusters))
    return sorted(clusters, key=lambda c: c.size, reverse=True)
