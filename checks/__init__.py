"""Runnable verification scripts against a live exchange.

These are not unit tests — they talk to a real venue (Binance testnet by
default) and some of them place and cancel real orders with fake money.
`tests/` holds the offline suite; this holds the checks that can only be
answered by an exchange.

Run from the project root as modules so `config` and `strategy` resolve:

    python -m checks.testnet_order_path_check
    python -m checks.testnet_venue_check

Each script's own docstring says what it proves and what it costs.
"""
