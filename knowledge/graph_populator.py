"""
Converts NLP ExtractionResults into Neo4j nodes and edges.
Runs as a consumer of the UnstructuredStream article queue.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
from typing import Optional

from graph.neo4j_client import Neo4jClient
from ingestion.unstructured_stream import Article, UnstructuredStream
from knowledge.entity_resolver import resolve_entity
from knowledge.nlp_extractor import ExtractionResult, NLPExtractor

logger = logging.getLogger(__name__)


class GraphPopulator:
    """
    Continuously drains UnstructuredStream.article_queue,
    runs NLP extraction, and writes results into Neo4j.
    """

    def __init__(self, stream: UnstructuredStream) -> None:
        self._stream = stream
        self._extractor = NLPExtractor()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="GraphPopulator")
        self._thread.start()
        logger.info("GraphPopulator started")

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                article: Article = self._stream.article_queue.get(timeout=2.0)
                self._process(article)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("GraphPopulator error: %s", exc, exc_info=True)

    def _process(self, article: Article) -> None:
        text = f"{article.title}. {article.body}"
        result = self._extractor.extract(article.article_id, text)
        client = Neo4jClient.get()

        # ── Resolve all mentioned entities ─────────────────────────────────
        entity_map: dict[str, tuple[str, str]] = {}  # mention_text → (node_type, id)
        for ent in result.entities:
            resolved = resolve_entity(ent.text, ent.label)
            if resolved:
                entity_map[ent.text] = resolved

        # Deduplicate resolved nodes
        resolved_nodes: list[tuple[str, str]] = list(dict.fromkeys(entity_map.values()))

        # ── Store Macro_Event and connect it to every mentioned token ──────
        if result.sentiment in ("negative", "positive") and result.sentiment_score > 0.65:
            eid = f"event_{article.article_id}"
            client.upsert_macro_event(
                eid,
                article.title[:200],
                source=article.source,
                sentiment=result.sentiment,
                sentiment_score=result.sentiment_score,
            )
            # Connect Macro_Event → IMPACTS → each resolved Token/Protocol
            for node_type, node_id in resolved_nodes:
                node_key = "symbol" if node_type == "Token" else "pid"
                try:
                    client.upsert_edge(
                        "Macro_Event", "eid", eid,
                        "IMPACTS",
                        node_type, node_key, node_id,
                        confidence=result.sentiment_score,
                        source_article=article.article_id,
                    )
                    logger.debug("Macro_Event %s IMPACTS %s:%s", eid, node_type, node_id)
                except Exception as exc:
                    logger.warning("Macro_Event edge failed: %s", exc)

        # ── Write Token↔Token / Token↔Protocol edges from NLP relations ───
        for rel in result.relations:
            src = entity_map.get(rel.subject.text)
            dst = entity_map.get(rel.obj.text)
            if not src or not dst or src == dst:
                continue
            src_type, src_id = src
            dst_type, dst_id = dst
            src_key = "symbol" if src_type == "Token" else "pid"
            dst_key = "symbol" if dst_type == "Token" else "pid"

            try:
                client.upsert_edge(
                    src_type, src_key, src_id,
                    rel.relation,
                    dst_type, dst_key, dst_id,
                    confidence=rel.confidence,
                    source_article=article.article_id,
                )
                logger.debug(
                    "Edge: (%s:%s)-[%s]->(%s:%s) conf=%.2f",
                    src_type, src_id, rel.relation, dst_type, dst_id, rel.confidence,
                )
            except Exception as exc:
                logger.warning("Edge write failed: %s", exc)

        # ── Cross-mention edges: if article mentions 2+ tokens, link them ──
        token_nodes = [(t, i) for t, i in resolved_nodes if t == "Token"]
        if len(token_nodes) >= 2:
            for i in range(len(token_nodes)):
                for j in range(i + 1, len(token_nodes)):
                    _, id_a = token_nodes[i]
                    _, id_b = token_nodes[j]
                    try:
                        client.upsert_edge(
                            "Token", "symbol", id_a,
                            "COMPETES_WITH",
                            "Token", "symbol", id_b,
                            confidence=0.5,
                            source_article=article.article_id,
                        )
                    except Exception:
                        pass

    def process_once(self, article: Article) -> ExtractionResult:
        """Synchronous single-article processing (useful for testing/backtrack)."""
        self._process(article)
        text = f"{article.title}. {article.body}"
        return self._extractor.extract(article.article_id, text)
