"""
Unstructured data pipeline: RSS crypto feeds + SEC EDGAR full-text search.
Articles are queued for the NLP knowledge extractor to process.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import feedparser
import requests
from config import settings

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL = "https://efts.sec.gov/LATEST/search-index?q=%22bitcoin%22+%22cryptocurrency%22&dateRange=custom&startdt={start}&forms=8-K,10-K"


@dataclass
class Article:
    source: str
    title: str
    body: str
    url: str
    published: datetime
    article_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.article_id = hashlib.md5(self.url.encode()).hexdigest()[:12]


class UnstructuredStream:
    """
    Polls RSS feeds and SEC EDGAR on a background thread.
    Deduplicates by URL hash and puts new articles into `article_queue`.
    """

    def __init__(self) -> None:
        self.article_queue: queue.Queue[Article] = queue.Queue(maxsize=500)
        self._seen: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="UnstructuredStream")
        self._thread.start()
        logger.info("Unstructured stream started (RSS + EDGAR)")

    def stop(self) -> None:
        self._stop_event.set()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._fetch_rss()
            self._fetch_edgar()
            self._stop_event.wait(settings.NEWS_POLL_INTERVAL_SECONDS)

    # ── RSS ────────────────────────────────────────────────────────────────

    def _fetch_rss(self) -> None:
        for feed_url in settings.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    url = entry.get("link", "")
                    if not url or url in self._seen:
                        continue
                    self._seen.add(url)
                    published = self._parse_date(entry.get("published", ""))
                    body = entry.get("summary", "") or entry.get("description", "")
                    article = Article(
                        source=feed.feed.get("title", feed_url),
                        title=entry.get("title", ""),
                        body=body,
                        url=url,
                        published=published,
                    )
                    self._enqueue(article)
            except Exception as exc:
                logger.warning("RSS fetch failed for %s: %s", feed_url, exc)

    # ── SEC EDGAR ──────────────────────────────────────────────────────────

    def _fetch_edgar(self) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        url = (
            "https://efts.sec.gov/LATEST/search-index"
            "?q=%22cryptocurrency%22+%22digital+asset%22"
            f"&dateRange=custom&startdt={today}&forms=8-K"
        )
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "TraderBot/1.0 firgymatannatikkas@gmail.com"})
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            for hit in hits[:10]:
                src = hit.get("_source", {})
                filing_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + src.get("entity_id", "")
                if filing_url in self._seen:
                    continue
                self._seen.add(filing_url)
                # Build a meaningful body that FinBERT can process
                entity = src.get("entity_name", "Unknown company")
                form = src.get("form_type", "8-K")
                period = src.get("period_of_report", "")
                description = src.get("description", "") or src.get("file_date", "")
                body = (
                    f"{entity} filed a {form} report with the SEC. "
                    f"Reporting period: {period}. "
                    f"This filing relates to cryptocurrency and digital assets. "
                    f"{description}"
                )
                article = Article(
                    source="SEC EDGAR",
                    title=f"{entity} - {form} (crypto/digital asset disclosure)",
                    body=body,
                    url=filing_url,
                    published=datetime.utcnow(),
                )
                self._enqueue(article)
        except Exception as exc:
            logger.warning("EDGAR fetch failed: %s", exc)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _enqueue(self, article: Article) -> None:
        # Evict oldest seen URLs to prevent unbounded memory growth
        if len(self._seen) > 5000:
            excess = len(self._seen) - 4000
            to_remove = list(self._seen)[:excess]
            self._seen.difference_update(to_remove)
        try:
            self.article_queue.put_nowait(article)
            logger.debug("Queued article: %s", article.title[:60])
        except queue.Full:
            logger.warning("Article queue full; dropping: %s", article.title[:40])

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
        return datetime.utcnow()
