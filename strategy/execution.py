"""
Binance order execution engine.
Places market orders with server-side SL/TP via OCO or stop-limit orders.
Enforces hard position limits and daily drawdown circuit breaker.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import settings
from graph.neo4j_client import Neo4jClient
from strategy.consensus import TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str          # LONG | SHORT
    entry_price: float
    quantity: float
    sl_price: float
    tp_price: float
    order_id: str
    opened_at: datetime = field(default_factory=datetime.utcnow)


class ExecutionEngine:
    """
    Manages live Binance orders with strict risk enforcement.
    Always operates in USDT-margined futures (or spot with testnet).
    """

    def __init__(self) -> None:
        self._client = Client(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET,
        )
        self._open_positions: dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        self._trading_day: date = date.today()
        self._halted: bool = False
        logger.info("ExecutionEngine ready. Testnet=%s", settings.BINANCE_TESTNET)

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._trading_day:
            self._daily_pnl = 0.0
            self._trading_day = today
            self._halted = False
            logger.info("New trading day; daily PnL reset.")

    def _get_account_balance(self) -> float:
        try:
            info = self._client.get_account()
            for asset in info.get("balances", []):
                if asset["asset"] == "USDT":
                    return float(asset["free"])
        except BinanceAPIException as exc:
            logger.error("Balance query failed: %s", exc)
        return 0.0

    def _get_price(self, symbol: str) -> float:
        ticker = self._client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def _compute_quantity(self, symbol: str, balance: float, price: float) -> float:
        raw_qty = (balance * settings.MAX_POSITION_PCT * settings.LEVERAGE) / price
        # Round to exchange lot size (simplified: 5 decimal places)
        return round(raw_qty, 5)

    def execute(self, decision: TradeDecision) -> Optional[Position]:
        """
        Attempt to open a position based on a TradeDecision.
        Returns the Position if order was placed, None otherwise.
        """
        self._reset_daily_if_needed()

        if self._halted:
            logger.warning("Trading halted (daily drawdown limit hit). Skipping %s.", decision.symbol)
            return None

        if not decision.execute:
            return None

        if decision.symbol in self._open_positions:
            logger.info("%s already has an open position.", decision.symbol)
            return None

        balance = self._get_account_balance()
        if balance < 10:
            logger.error("Insufficient USDT balance: %.2f", balance)
            return None

        price = self._get_price(decision.symbol)
        quantity = self._compute_quantity(decision.symbol, balance, price)
        if quantity <= 0:
            return None

        # Compute SL/TP prices
        if decision.direction == "LONG":
            sl_price = round(price * (1 - settings.STOP_LOSS_PCT), 4)
            tp_price = round(price * (1 + settings.TAKE_PROFIT_PCT), 4)
            side = "BUY"
            sl_side = "SELL"
        else:
            sl_price = round(price * (1 + settings.STOP_LOSS_PCT), 4)
            tp_price = round(price * (1 - settings.TAKE_PROFIT_PCT), 4)
            side = "SELL"
            sl_side = "BUY"

        try:
            # Entry market order
            order = self._client.create_order(
                symbol=decision.symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            order_id = str(order.get("orderId", ""))
            fill_price = float(order.get("fills", [{}])[0].get("price", price))

            # Server-side stop-loss (STOP_LOSS_LIMIT)
            self._client.create_order(
                symbol=decision.symbol,
                side=sl_side,
                type="STOP_LOSS_LIMIT",
                quantity=quantity,
                stopPrice=str(sl_price),
                price=str(round(sl_price * (0.999 if side == "BUY" else 1.001), 4)),
                timeInForce="GTC",
            )

            # Take-profit (TAKE_PROFIT_LIMIT)
            tp_side = "SELL" if side == "BUY" else "BUY"
            self._client.create_order(
                symbol=decision.symbol,
                side=tp_side,
                type="TAKE_PROFIT_LIMIT",
                quantity=quantity,
                stopPrice=str(tp_price),
                price=str(round(tp_price * (0.999 if tp_side == "SELL" else 1.001), 4)),
                timeInForce="GTC",
            )

            pos = Position(
                symbol=decision.symbol,
                direction=decision.direction,
                entry_price=fill_price,
                quantity=quantity,
                sl_price=sl_price,
                tp_price=tp_price,
                order_id=order_id,
            )
            self._open_positions[decision.symbol] = pos
            logger.info(
                "OPENED %s %s qty=%.5f @ %.4f | SL=%.4f TP=%.4f",
                decision.direction, decision.symbol, quantity, fill_price, sl_price, tp_price,
            )
            return pos

        except BinanceAPIException as exc:
            logger.error("Order failed for %s: %s", decision.symbol, exc)
            return None

    def close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            return

        if pos.direction == "LONG":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        outcome = "win" if pnl_pct > 0 else "loss"
        self._daily_pnl += pnl_pct

        logger.info(
            "CLOSED %s %s @ %.4f | PnL: %.2f%% | Reason: %s",
            pos.direction, symbol, exit_price, pnl_pct * 100, reason,
        )

        # Log to Neo4j for backtrack
        Neo4jClient.get().log_trade({
            "symbol": symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        })

        # Daily drawdown check
        if self._daily_pnl < -settings.DAILY_DRAWDOWN_LIMIT:
            logger.critical(
                "Daily drawdown limit hit (%.2f%%). Halting trading for today.",
                self._daily_pnl * 100,
            )
            self._halted = True

    def monitor_positions(self) -> None:
        """Poll open positions and auto-close if SL/TP thresholds are breached."""
        for symbol, pos in list(self._open_positions.items()):
            try:
                price = self._get_price(symbol)
                if pos.direction == "LONG":
                    if price <= pos.sl_price:
                        self.close_position(symbol, price, "STOP_LOSS")
                    elif price >= pos.tp_price:
                        self.close_position(symbol, price, "TAKE_PROFIT")
                else:
                    if price >= pos.sl_price:
                        self.close_position(symbol, price, "STOP_LOSS")
                    elif price <= pos.tp_price:
                        self.close_position(symbol, price, "TAKE_PROFIT")
            except Exception as exc:
                logger.warning("Monitor error for %s: %s", symbol, exc)
