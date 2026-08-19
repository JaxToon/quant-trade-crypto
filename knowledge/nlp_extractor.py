"""
NLP pipeline: FinBERT sentiment + spaCy NER → structured entity/relationship output.
Runs on articles from the UnstructuredStream queue.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)

# Lazy-loaded singletons to avoid loading models at import time
_finbert = None
_nlp = None


def _get_finbert():
    global _finbert
    if _finbert is None:
        logger.info("Loading FinBERT model (first run may take a minute)...")
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0,          # GPU
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _finbert


def _get_spacy():
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy en_core_web_sm...")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Output dataclasses ─────────────────────────────────────────────────────

@dataclass
class EntityMention:
    text: str
    label: str      # ORG, PRODUCT, GPE, or custom CRYPTO
    start: int
    end: int


@dataclass
class RelationCandidate:
    """A potential edge extracted from co-occurrence in the same sentence."""
    subject: EntityMention
    relation: str   # IMPACTS, COMPETES_WITH, DEPENDS_ON, REGULATES, SUPPLIES
    obj: EntityMention
    confidence: float
    sentence: str


@dataclass
class ExtractionResult:
    article_id: str
    sentiment: str          # positive / negative / neutral
    sentiment_score: float
    entities: list[EntityMention] = field(default_factory=list)
    relations: list[RelationCandidate] = field(default_factory=list)


# ── Regex patterns for crypto-specific entities ───────────────────────────

CRYPTO_TICKER_RE = re.compile(r"\$([A-Z]{2,6})\b")
# Was missing cardano/ada, avalanche/avax entirely (and polkadot, though
# "dot" is deliberately NOT added bare here -- too common an English word/
# punctuation mark, would flood entity detection with false positives;
# "polkadot" the full name is enough to catch real mentions, same
# precision tradeoff the existing short tickers like "sol"/"eth" already
# accept). Found while reconstructing Macro_Event edges after a Neo4j
# data-loss incident: entity_resolver.py's alias map alone wasn't enough
# fixed there too — a mention has to be DETECTED here first before
# resolve_entity() ever gets a chance to map it to a symbol.
CRYPTO_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "bnb",
    "xrp", "ripple", "defi", "nft", "stablecoin", "usdt", "usdc",
    "cardano", "ada", "avalanche", "avax", "polkadot",
}

# Simple heuristic: if two entities appear in the same sentence, check verb cues
IMPACT_VERBS = {"crash", "surge", "affect", "impact", "drive", "hurt", "boost", "fall", "rise"}
COMPETE_VERBS = {"compete", "rival", "challenge", "vs", "versus", "alternative"}
DEPENDS_VERBS = {"depend", "rely", "require", "use", "built on", "powered by"}
SUPPLY_VERBS = {"supply", "provide", "source", "mine", "produce"}
REGULATE_VERBS = {"regulate", "ban", "approve", "restrict", "sanction", "fine"}


def _classify_relation(verb_tokens: list[str]) -> str:
    vl = [v.lower() for v in verb_tokens]
    for v in vl:
        if any(k in v for k in IMPACT_VERBS):
            return "IMPACTS"
        if any(k in v for k in COMPETE_VERBS):
            return "COMPETES_WITH"
        if any(k in v for k in DEPENDS_VERBS):
            return "DEPENDS_ON"
        if any(k in v for k in SUPPLY_VERBS):
            return "SUPPLIES"
        if any(k in v for k in REGULATE_VERBS):
            return "REGULATES"
    return "IMPACTS"  # default


class NLPExtractor:
    """Processes a single article text → ExtractionResult."""

    def extract(self, article_id: str, text: str) -> ExtractionResult:
        finbert = _get_finbert()
        nlp = _get_spacy()

        # ── Sentiment (FinBERT) ────────────────────────────────────────────
        scores = finbert(text[:512])[0]
        best = max(scores, key=lambda x: x["score"])
        sentiment = best["label"].lower()
        sentiment_score = best["score"]

        # ── Named Entity Recognition (spaCy) ──────────────────────────────
        doc = nlp(text)
        entities: list[EntityMention] = []

        # Standard spaCy NER
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "PERSON", "LAW", "MONEY"):
                entities.append(EntityMention(ent.text, ent.label_, ent.start_char, ent.end_char))

        # Crypto ticker mentions via regex
        for m in CRYPTO_TICKER_RE.finditer(text):
            entities.append(EntityMention(m.group(0), "CRYPTO", m.start(), m.end()))

        # Keyword-based crypto detection
        for token in doc:
            if token.text.lower() in CRYPTO_KEYWORDS:
                entities.append(EntityMention(token.text, "CRYPTO", token.idx, token.idx + len(token.text)))

        # ── Relation extraction (sentence co-occurrence) ───────────────────
        relations: list[RelationCandidate] = []
        for sent in doc.sents:
            sent_ents = [e for e in entities if sent.start_char <= e.start < sent.end_char]
            if len(sent_ents) < 2:
                continue
            verbs = [t.text for t in sent if t.pos_ == "VERB"]
            rel_type = _classify_relation(verbs)
            base_conf = sentiment_score * 0.8 if rel_type == "IMPACTS" else 0.6
            # Connect ALL unique pairs in sentence (not just first two)
            seen_pairs: set[tuple[int, int]] = set()
            for i, subj in enumerate(sent_ents):
                for obj in sent_ents[i + 1:]:
                    pair = (subj.start, obj.start)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    # Decay confidence slightly for non-adjacent pairs
                    conf = base_conf * (1.0 - 0.05 * (sent_ents.index(obj) - i - 1))
                    relations.append(RelationCandidate(subj, rel_type, obj, max(0.3, conf), sent.text))

        return ExtractionResult(article_id, sentiment, sentiment_score, entities, relations)
