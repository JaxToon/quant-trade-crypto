"""Where orders go: a paper book, or a Binance-shaped spot exchange.

The exchange broker drives Tokocrypto (or Binance testnet) through
python-binance with a base-URL swap, exactly like strategy/live_mirror.py,
and it refuses to arm unless three switches agree.
"""
from __future__ import annotations

import logging
import math
import time

from config import settings
from fund import config
from fund.core import Order

log = logging.getLogger("fund")


class RefuseToArm(RuntimeError):
    pass


class ExchangeBroker:
    """Market orders on a spot exchange. The exchange balance is the book."""

    def __init__(self, client=None, venue: str = config.LIVE_VENUE,
                 enabled: bool = config.LIVE_ENABLED,
                 confirm_real_money: bool = config.LIVE_CONFIRM_REAL_MONEY):
        if not enabled:
            raise RefuseToArm("FUND_LIVE_ENABLED is not true")
        base = settings.VENUE_REST_URLS.get(venue)
        if base is None:
            raise RefuseToArm(f"unknown venue {venue!r}")
        if client is None:
            from binance.client import Client
            if venue == "tokocrypto":
                key, secret = settings.TOKOCRYPTO_API_KEY, settings.TOKOCRYPTO_API_SECRET
            else:
                key, secret = settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET
            if not key or not secret:
                raise RefuseToArm(f"no API credentials for venue {venue!r}")
            client = Client(key, secret)
            client.API_URL = base
        self.client = client
        self.venue = venue
        self.is_testnet = "testnet" in getattr(client, "API_URL", base)
        if not self.is_testnet and not confirm_real_money:
            raise RefuseToArm(f"{venue} is real money and FUND_CONFIRM_REAL_MONEY is not true")
        self._filters: dict[str, tuple[float, float]] = {}
        self._sync_clock()
        log.warning("ExchangeBroker ARMED venue=%s%s", venue, "" if self.is_testnet else "  *** REAL MONEY ***")

    def _sync_clock(self) -> None:
        """Signed requests fail (-1021) when the PC clock runs ahead of the
        exchange; this machine was measured 1s ahead on 2026-11-06."""
        if not hasattr(self.client, "get_server_time"):
            return
        local = time.time() * 1000
        server = self.client.get_server_time()["serverTime"]
        self.client.timestamp_offset = server - (local + time.time() * 1000) / 2 - 500
        log.info("exchange clock offset %+.0f ms", server - local)

    # -- reads -------------------------------------------------------------
    def prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: float(self.client.get_symbol_ticker(symbol=s)["price"]) for s in symbols}

    def positions(self, symbols: list[str]) -> tuple[dict[str, float], float]:
        bal = {b["asset"]: float(b["free"]) + float(b["locked"])
               for b in self.client.get_account()["balances"]}
        qty = {s: bal.get(s.removesuffix("USDT"), 0.0) for s in symbols}
        return qty, bal.get("USDT", 0.0)

    def _filter(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._filters:
            info = self.client.get_symbol_info(symbol)
            if not info:
                raise ValueError(f"{symbol} is not listed on {self.venue}")
            f = {x["filterType"]: x for x in info["filters"]}
            self._filters[symbol] = (float(f["LOT_SIZE"]["stepSize"]),
                                     float(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", 0.0)))
        return self._filters[symbol]

    @staticmethod
    def _fmt(qty: float, step: float) -> str:
        n = round(math.floor(qty / step + 1e-9) * step, 12)
        prec = max(0, len(f"{step:.10f}".rstrip("0").split(".")[1])) if step < 1 else 0
        return f"{n:.{prec}f}"

    # -- writes ------------------------------------------------------------
    def execute(self, orders: list[Order], cash_available: float) -> list[dict]:
        """Sells first (they fund the buys), then buys trimmed to the fund's cash.

        cash_available is the fund's own USDT, not the account's: other money
        in the account is never spent.
        """
        fills = []
        budget = cash_available
        for o in sorted(orders, key=lambda o: o.side != "SELL"):
            step, min_notional = self._filter(o.symbol)
            qty = o.qty
            if o.side == "BUY":
                free = self.positions([])[1]
                qty = min(qty, min(free, budget) * 0.995 / o.price)
            q = self._fmt(qty, step)
            if float(q) <= 0 or float(q) * o.price < max(min_notional, config.MIN_ORDER_USDT):
                log.info("skip %s %s %s: below minimum", o.side, q, o.symbol)
                continue
            r = self.client.create_order(symbol=o.symbol, side=o.side, type="MARKET", quantity=q)
            filled = float(r.get("executedQty", 0.0))
            quote = float(r.get("cummulativeQuoteQty", 0.0))
            fee = sum(float(f.get("commission", 0.0)) for f in r.get("fills", []))
            budget += quote if o.side == "SELL" else -quote
            fills.append({"symbol": o.symbol, "side": o.side, "qty": filled,
                          "price": quote / filled if filled else None, "ref_price": o.price,
                          "commission": fee, "order_id": r.get("orderId"), "status": r.get("status")})
        return fills
