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
    "CREATE INDEX token_asset_type IF NOT EXISTS FOR (t:Token) ON (t.asset_type)",
    "CREATE INDEX trade_outcome IF NOT EXISTS FOR (t:Trade) ON (t.outcome)",
    "CREATE INDEX trade_symbol IF NOT EXISTS FOR (t:Trade) ON (t.symbol)",
]

# asset_type tags every Token so crypto and stock learning stay separable —
# see graph/neo4j_client.get_all_tokens() and gnn/dataset.build_hetero_data(),
# both of which filter by settings.ASSET_TYPE. Symbols don't collide between
# the two lists, so both can be seeded unconditionally and safely coexist in
# the same graph.
SEED_TOKENS = [
    ("BTCUSDT", "Bitcoin", "crypto"),
    ("ETHUSDT", "Ethereum", "crypto"),
    ("SOLUSDT", "Solana", "crypto"),
    ("BNBUSDT", "Binance Coin", "crypto"),
    ("XRPUSDT", "XRP", "crypto"),
    ("ADAUSDT", "Cardano", "crypto"),
    ("AVAXUSDT", "Avalanche", "crypto"),
    ("DOTUSDT", "Polkadot", "crypto"),
    ("AAPL", "Apple Inc.", "stock"),
    ("MSFT", "Microsoft Corp.", "stock"),
    ("GOOGL", "Alphabet Inc.", "stock"),
    ("AMZN", "Amazon.com Inc.", "stock"),
    ("NVDA", "NVIDIA Corp.", "stock"),
    ("META", "Meta Platforms Inc.", "stock"),
    ("TSLA", "Tesla Inc.", "stock"),
    ("AVGO", "Broadcom Inc.", "stock"),
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
    # Stocks COMPETES_WITH each other (big tech / semis rivalries)
    ("Token", "symbol", "AAPL", "COMPETES_WITH", "Token", "symbol", "MSFT", 0.70),
    ("Token", "symbol", "GOOGL", "COMPETES_WITH", "Token", "symbol", "META", 0.75),
    ("Token", "symbol", "GOOGL", "COMPETES_WITH", "Token", "symbol", "AMZN", 0.60),
    ("Token", "symbol", "AMZN", "COMPETES_WITH", "Token", "symbol", "MSFT", 0.55),
    ("Token", "symbol", "NVDA", "COMPETES_WITH", "Token", "symbol", "AVGO", 0.70),
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
    for symbol, name, asset_type in SEED_TOKENS:
        client.upsert_token(symbol, name, asset_type=asset_type)

    print("Seeding protocol nodes...")
    for pid, name in SEED_PROTOCOLS:
        client.upsert_protocol(pid, name)

    print("Seeding base edges...")
    for src_lbl, src_k, src_v, rel, dst_lbl, dst_k, dst_v, conf in SEED_EDGES:
        client.upsert_edge(src_lbl, src_k, src_v, rel, dst_lbl, dst_k, dst_v, conf)

    print("Schema initialized.")
