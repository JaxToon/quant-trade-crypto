"""
Run once to create Neo4j constraints and indexes.
Call: from graph.schema import initialize_schema; initialize_schema()
"""
from graph.neo4j_client import Neo4jClient

CONSTRAINTS = [
    "CREATE CONSTRAINT token_symbol IF NOT EXISTS FOR (t:Token) REQUIRE t.symbol IS UNIQUE",
    "CREATE CONSTRAINT protocol_pid IF NOT EXISTS FOR (p:Protocol) REQUIRE p.pid IS UNIQUE",
    "CREATE CONSTRAINT event_eid IF NOT EXISTS FOR (e:Macro_Event) REQUIRE e.eid IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX token_updated IF NOT EXISTS FOR (t:Token) ON (t.updated_at)",
    "CREATE INDEX trade_outcome IF NOT EXISTS FOR (t:Trade) ON (t.outcome)",
    "CREATE INDEX trade_symbol IF NOT EXISTS FOR (t:Trade) ON (t.symbol)",
]

SEED_TOKENS = [
    ("BTCUSDT", "Bitcoin"),
    ("ETHUSDT", "Ethereum"),
    ("SOLUSDT", "Solana"),
    ("BNBUSDT", "Binance Coin"),
    ("XRPUSDT", "XRP"),
    ("ADAUSDT", "Cardano"),
    ("AVAXUSDT", "Avalanche"),
    ("DOTUSDT", "Polkadot"),
]

SEED_PROTOCOLS = [
    ("ethereum-chain", "Ethereum Mainnet"),
    ("solana-chain", "Solana Mainnet"),
    ("bnb-chain", "BNB Chain"),
    ("xrp-ledger", "XRP Ledger"),
    ("cardano-chain", "Cardano Mainnet"),
    ("avalanche-chain", "Avalanche C-Chain"),
    ("polkadot-chain", "Polkadot Relay Chain"),
    ("uniswap", "Uniswap"),
    ("aave", "Aave"),
]

SEED_EDGES = [
    # Token BUILT_ON Chain
    ("Token", "symbol", "ETHUSDT", "BUILT_ON", "Protocol", "pid", "ethereum-chain", 0.99),
    ("Token", "symbol", "SOLUSDT", "BUILT_ON", "Protocol", "pid", "solana-chain", 0.99),
    ("Token", "symbol", "BNBUSDT", "BUILT_ON", "Protocol", "pid", "bnb-chain", 0.99),
    ("Token", "symbol", "XRPUSDT", "BUILT_ON", "Protocol", "pid", "xrp-ledger", 0.99),
    ("Token", "symbol", "ADAUSDT", "BUILT_ON", "Protocol", "pid", "cardano-chain", 0.99),
    ("Token", "symbol", "AVAXUSDT", "BUILT_ON", "Protocol", "pid", "avalanche-chain", 0.99),
    ("Token", "symbol", "DOTUSDT", "BUILT_ON", "Protocol", "pid", "polkadot-chain", 0.99),
    # Protocol DEPENDS_ON Token (native gas token)
    ("Protocol", "pid", "uniswap", "DEPENDS_ON", "Token", "symbol", "ETHUSDT", 0.95),
    ("Protocol", "pid", "aave", "DEPENDS_ON", "Token", "symbol", "ETHUSDT", 0.90),
    # Tokens COMPETES_WITH each other (L1s compete for the same smart-contract-chain demand)
    ("Token", "symbol", "ETHUSDT", "COMPETES_WITH", "Token", "symbol", "SOLUSDT", 0.80),
    ("Token", "symbol", "ETHUSDT", "COMPETES_WITH", "Token", "symbol", "BNBUSDT", 0.70),
    ("Token", "symbol", "ETHUSDT", "COMPETES_WITH", "Token", "symbol", "ADAUSDT", 0.65),
    ("Token", "symbol", "ETHUSDT", "COMPETES_WITH", "Token", "symbol", "AVAXUSDT", 0.65),
    ("Token", "symbol", "ETHUSDT", "COMPETES_WITH", "Token", "symbol", "DOTUSDT", 0.65),
    ("Token", "symbol", "SOLUSDT", "COMPETES_WITH", "Token", "symbol", "AVAXUSDT", 0.60),
]


def initialize_schema() -> None:
    client = Neo4jClient.get()

    print("Creating constraints...")
    for stmt in CONSTRAINTS:
        client.run_write(stmt)

    print("Creating indexes...")
    for stmt in INDEXES:
        client.run_write(stmt)

    print("Seeding base token nodes...")
    for symbol, name in SEED_TOKENS:
        client.upsert_token(symbol, name)

    print("Seeding protocol nodes...")
    for pid, name in SEED_PROTOCOLS:
        client.upsert_protocol(pid, name)

    print("Seeding base edges...")
    for src_lbl, src_k, src_v, rel, dst_lbl, dst_k, dst_v, conf in SEED_EDGES:
        client.upsert_edge(src_lbl, src_k, src_v, rel, dst_lbl, dst_k, dst_v, conf)

    print("Schema initialized.")
