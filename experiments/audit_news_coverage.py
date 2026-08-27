"""
News/event coverage audit -- diagnoses the stock-vs-crypto Macro_Event
imbalance found live (2026-08-21): 255 crypto Token<-Macro_Event edges
vs. 4 stock ones, despite equal token counts (8 crypto, 8 stock).

Two separate, honestly-labeled measurements, NOT one blended number:

  PART A -- GRAPH STATE (exact, full history): queries Neo4j directly for
  how many Macro_Event nodes are actually linked to each watchlist symbol,
  right now, covering the graph's entire lifetime. This is precise.

  PART B -- LIVE FUNNEL SNAPSHOT (today's articles only): raw fetched
  articles are never persisted anywhere (UnstructuredStream's dedup set
  is in-memory, process-lifetime only), so a true 30/60-day historical
  fetch->parse->resolve funnel CANNOT be reconstructed -- there is no
  data to reconstruct it from. Rather than fabricate that number, this
  polls every configured source ONCE right now and runs the REAL pipeline
  (NLPExtractor, resolve_entity -- not reimplemented) against whatever
  comes back, giving an honest today-only cross-section of: articles
  fetched -> entities resolved per symbol -> would the sentiment gate
  (graph_populator.py's score > 0.65 non-neutral requirement -- a real,
  separate funnel stage independent of whether the company is even
  mentioned) actually let a Macro_Event through.

Motivation: distinguishing "no macro events" (system malfunction) from
"legitimately quiet symbol" (a stock can validly have no company-specific
news right now) requires knowing which funnel stage is actually empty --
no articles even mention it, entities aren't resolving, or events are
being created but the sentiment gate is filtering them out. Guessing
which stage is broken from the graph-edge count alone isn't enough.

Does not modify any live ingestion code, RSS_FEEDS, or the graph itself
-- read-only against Neo4j, and the live poll never enqueues into the
real UnstructuredStream/GraphPopulator (this doesn't write any Macro_Event
nodes, it only reports what WOULD happen).

Usage: python audit_news_coverage.py
"""
from __future__ import annotations

import feedparser
import requests

from config import settings
from graph.neo4j_client import Neo4jClient
from knowledge.entity_resolver import resolve_entity
from knowledge.nlp_extractor import NLPExtractor

SENTIMENT_GATE_THRESHOLD = 0.65  # matches graph_populator.py's own gate exactly


def audit_graph_state() -> dict[str, int]:
    """Part A: exact, full-history edge counts per watchlist symbol."""
    client = Neo4jClient.get()
    rows = client.run_read("""
        MATCH (e:Macro_Event)-[r]->(t:Token)
        RETURN t.symbol AS symbol, count(r) AS n
    """)
    return {r["symbol"]: r["n"] for r in rows}


def fetch_live_articles() -> list[dict]:
    """Part B step 1: poll every RSS feed once, right now. Returns raw
    {title, body, source} dicts -- doesn't touch the real article queue
    or _seen dedup set, so this has zero interaction with the live
    ingestion pipeline."""
    articles = []
    for feed_url in settings.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                body = entry.get("summary", "") or entry.get("description", "")
                if title:
                    articles.append({"title": title, "body": body, "source": feed.feed.get("title", feed_url)})
        except Exception as exc:
            print(f"  RSS fetch failed for {feed_url}: {exc}")
    return articles


def fetch_live_edgar() -> list[dict]:
    """Part B, EDGAR leg -- included for completeness, but this query is
    itself crypto-keyword-scoped (see ingestion/unstructured_stream.py),
    so it's expected to contribute ~nothing stock-relevant regardless of
    entity resolution quality -- that's part of the finding, not a bug
    in this audit."""
    import datetime
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        '?q=%22cryptocurrency%22+%22digital+asset%22'
        f"&dateRange=custom&startdt={today}&forms=8-K"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "TraderBot/1.0 audit"})
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            {"title": h.get("_source", {}).get("display_names", ["?"])[0] if h.get("_source", {}).get("display_names") else "?",
             "body": "", "source": "SEC EDGAR"}
            for h in hits[:10]
        ]
    except Exception as exc:
        print(f"  EDGAR fetch failed: {exc}")
        return []


def audit_live_funnel(articles: list[dict], watchlist: set[str]) -> dict[str, dict]:
    """Part B steps 2-4: run the REAL NLPExtractor + resolve_entity against
    each fetched article, tally per symbol: mentions (resolved entity
    present), and would_create_event (mentions AND passes the sentiment
    gate graph_populator.py itself uses)."""
    extractor = NLPExtractor()
    per_symbol = {sym: {"mentions": 0, "would_create_event": 0} for sym in watchlist}
    total_articles = len(articles)

    for i, art in enumerate(articles):
        text = f"{art['title']}. {art['body']}"
        try:
            result = extractor.extract(f"audit_{i}", text)
        except Exception as exc:
            print(f"  extraction failed for article {i}: {exc}")
            continue

        resolved_symbols = set()
        for ent in result.entities:
            resolved = resolve_entity(ent.text, ent.label)
            if resolved and resolved[0] == "Token" and resolved[1] in watchlist:
                resolved_symbols.add(resolved[1])

        passes_gate = result.sentiment in ("negative", "positive") and result.sentiment_score > SENTIMENT_GATE_THRESHOLD
        for sym in resolved_symbols:
            per_symbol[sym]["mentions"] += 1
            if passes_gate:
                per_symbol[sym]["would_create_event"] += 1

        if (i + 1) % 20 == 0:
            print(f"    processed {i + 1}/{total_articles} articles...")

    return per_symbol


def main() -> None:
    watchlist = set(settings.CRYPTO_WATCHLIST) | set(settings.STOCK_WATCHLIST)
    stock_set = set(settings.STOCK_WATCHLIST)

    print("=== Part A: graph state (exact, full history) ===")
    graph_edges = audit_graph_state()
    print(f"  {len(graph_edges)} symbols have at least one Macro_Event edge\n")

    print("=== Part B: live funnel snapshot (today's articles only -- see module docstring for why this can't be historical) ===")
    print(f"  Polling {len(settings.RSS_FEEDS)} RSS feeds...")
    articles = fetch_live_articles()
    print(f"  {len(articles)} articles fetched from RSS")

    print("  Polling SEC EDGAR (crypto-keyword-scoped query, see docstring)...")
    edgar_articles = fetch_live_edgar()
    print(f"  {len(edgar_articles)} hits from EDGAR")
    all_articles = articles + edgar_articles

    print(f"\n  Running NLP extraction on {len(all_articles)} articles (FinBERT + spaCy, may take a few minutes)...")
    funnel = audit_live_funnel(all_articles, watchlist)

    print(f"\n{'Symbol':<10}{'Class':<8}{'Live mentions':<15}{'Would create event':<20}{'Graph edges (all-time)':<25}")
    print("-" * 78)
    for sym in sorted(watchlist, key=lambda s: (s in stock_set, s)):
        asset_class = "stock" if sym in stock_set else "crypto"
        f = funnel.get(sym, {"mentions": 0, "would_create_event": 0})
        edges = graph_edges.get(sym, 0)
        print(f"{sym:<10}{asset_class:<8}{f['mentions']:<15}{f['would_create_event']:<20}{edges:<25}")

    crypto_total_edges = sum(v for k, v in graph_edges.items() if k not in stock_set)
    stock_total_edges = sum(v for k, v in graph_edges.items() if k in stock_set)
    crypto_total_mentions = sum(f["mentions"] for sym, f in funnel.items() if sym not in stock_set)
    stock_total_mentions = sum(f["mentions"] for sym, f in funnel.items() if sym in stock_set)

    print(f"\nTotals: crypto {crypto_total_mentions} live mentions / {crypto_total_edges} graph edges | "
          f"stock {stock_total_mentions} live mentions / {stock_total_edges} graph edges")
    print("\nIf stock's live-mention count is near-zero even in today's single snapshot, the gap is upstream --")
    print("no source is fetching stock-relevant articles at all, regardless of entity resolution or sentiment gating.")
    print("If mentions are nonzero but would_create_event is much lower, the sentiment gate is the bottleneck instead.")


if __name__ == "__main__":
    main()
