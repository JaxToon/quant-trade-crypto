"""
Binance order execution engine.
Places market orders with server-side SL/TP via OCO or stop-limit orders.
Enforces hard position limits and daily drawdown circuit breaker.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import settings
from graph.neo4j_client import Neo4jClient
from strategy.consensus import TradeDecision

logger = logging.getLogger(__name__)

POSITIONS_STATE_PATH = os.path.join("backtrack_logs", "execution_open_positions.json")


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
        # python-binance's `testnet=True` does NOT redirect spot REST calls.
        # Its own docstring says the flag is "only available for vanilla
        # options at the moment", and BaseClient.__init__ overwrites
        # self.API_URL with the PRODUCTION endpoint unconditionally; the flag
        # only switches the websocket and futures/options URLs.
        #
        # So BINANCE_TESTNET was a safety switch that did not switch anything
        # for spot orders. Harmless while the configured keys are testnet keys
        # (production rejects them with -2015), but if a PRODUCTION key ever
        # landed in .env while someone trusted BINANCE_TESTNET=true to hold
        # the system back, it would have placed real orders with real money.
        #
        # Found 2026-09-07 while checking the order path end-to-end. Pin the
        # URL explicitly so the flag means what every caller assumes it means.
        if settings.BINANCE_TESTNET:
            self._client.API_URL = Client.API_TESTNET_URL
            logger.warning("TESTNET mode: REST endpoint pinned to %s", self._client.API_URL)
        else:
            logger.warning("LIVE mode: REST endpoint is %s — REAL ORDERS, REAL MONEY",
                           self._client.API_URL)
        self._filter_cache: dict[str, tuple[float, float, float]] = {}
        self._open_positions: dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        self._trading_day: date = date.today()
        self._halted: bool = False
        self._load_positions()
        # Local state is only a file; the exchange is the truth. Check them
        # against each other before trading anything, so a position that lost
        # its protection while the process was down is found now rather than
        # after it has moved against us.
        try:
            self.reconcile_with_exchange()
        except Exception as exc:
            logger.error("Startup reconciliation failed (continuing): %s", exc)
        logger.info(
            "ExecutionEngine ready. Testnet=%s | %d open position(s) restored",
            settings.BINANCE_TESTNET, len(self._open_positions),
        )

    # ── Persistence ────────────────────────────────────────────────────────
    # _open_positions used to be purely in-memory — any real order this
    # engine ever placed would silently lose tracking on restart (still open
    # on the exchange, but never checked for SL/TP again and never logged to
    # Neo4j on close). Never actually bit us because the HMM regime bug kept
    # bear_veto_active pinned true for this system's entire history, so this
    # path never placed a single order — but now that it's unblocked, a
    # restart-without-persistence here would orphan a real position.

    def _load_positions(self) -> None:
        if not os.path.exists(POSITIONS_STATE_PATH):
            return
        try:
            with open(POSITIONS_STATE_PATH) as f:
                data = json.load(f)
            for row in data:
                row = dict(row)
                row["opened_at"] = datetime.fromisoformat(row["opened_at"])
                pos = Position(**row)
                self._open_positions[pos.symbol] = pos
        except Exception as exc:
            logger.warning("Execution position state load failed, starting fresh: %s", exc)

    def _save_positions(self) -> None:
        try:
            os.makedirs(os.path.dirname(POSITIONS_STATE_PATH), exist_ok=True)
            rows = []
            for pos in self._open_positions.values():
                row = asdict(pos)
                row["opened_at"] = pos.opened_at.isoformat()
                rows.append(row)
            with open(POSITIONS_STATE_PATH, "w") as f:
                json.dump(rows, f, indent=2)
        except Exception as exc:
            logger.warning("Execution position state save failed: %s", exc)

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

    def _symbol_filters(self, symbol: str) -> tuple[float, float, float]:
        """(stepSize, tickSize, minNotional) from the exchange, cached.

        Previously this code hardcoded "round to 5 decimal places" with the
        comment "simplified". Real symbols have wildly different LOT_SIZE
        steps, so 5dp is wrong in both directions: too coarse for some pairs
        (rejected as a lot-size violation) and needlessly fine for others.
        """
        cached = self._filter_cache.get(symbol)
        if cached:
            return cached
        info = self._client.get_symbol_info(symbol)
        f = {x["filterType"]: x for x in info["filters"]}
        out = (
            float(f["LOT_SIZE"]["stepSize"]),
            float(f["PRICE_FILTER"]["tickSize"]),
            float(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", 0.0)),
        )
        self._filter_cache[symbol] = out
        return out

    @staticmethod
    def _fmt(value: float, increment: float) -> str:
        """Snap to the exchange increment and render as a FIXED-POINT string.

        Both halves matter. Python renders small floats in scientific
        notation — 0.00008 becomes "8e-05" — and Binance rejects that with
        -1100 "Illegal characters found in parameter 'quantity'". Passing a
        raw float lets json serialise it that way, so every order on a small
        account fails. Found on testnet 2026-09-08.
        """
        prec = max(0, len(f"{increment:.10f}".rstrip("0").split(".")[1])) if increment < 1 else 0
        return f"{round(value / increment) * increment:.{prec}f}"

    def _compute_quantity(self, symbol: str, balance: float, price: float) -> str:
        """Position quantity as an exchange-ready string, or "" if the trade
        cannot be placed safely.

        Sizes against the WORST price any leg will use, not the entry price.
        The stop-loss leg sits below the entry for a long, so a quantity that
        clears MIN_NOTIONAL at entry can still fail it on the protective leg —
        which would open the position and then silently fail to protect it,
        leaving it naked. Refusing the trade is the only safe response.
        """
        step, _tick, min_notional = self._symbol_filters(symbol)
        raw_qty = (balance * settings.MAX_POSITION_PCT * settings.LEVERAGE) / price
        qty = round(raw_qty / step) * step
        if qty <= 0:
            return ""
        worst_leg_price = price * (1 - settings.STOP_LOSS_PCT)
        if qty * worst_leg_price < min_notional:
            logger.error(
                "%s: quantity %.8f clears MIN_NOTIONAL at entry (%.2f) but NOT on the "
                "protective leg (%.2f < %.2f). Refusing — opening this would leave the "
                "position unprotected.",
                symbol, qty, qty * price, qty * worst_leg_price, min_notional,
            )
            return ""
        return self._fmt(qty, step)

    def reconcile_with_exchange(self, auto_heal: bool = True) -> dict:
        """Check local position state against what the exchange actually holds.

        `_load_positions` restores from a local JSON file and nothing has ever
        compared that to reality. The two drift for mundane reasons — this
        deployment restarts often (39 times in one 25-day window), and a crash
        between an order filling and the state being saved is enough. Two
        specific divergences matter:

          PROTECTION MISSING — the file says a position is guarded, but its
            STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT are no longer on the exchange
            (cancelled by hand, expired, or never placed because the process
            died mid-sequence). The position is naked and nothing notices.
            Healed by re-placing the protective orders.

          ORPHANED ORDERS — protective orders resting for a symbol we no
            longer track, usually left by a partial failure. They can fire
            later and sell an asset the system believes it does not hold.
            Healed by cancelling them.

        Returns a report dict; logs loudly for anything it cannot fix, since
        silence is the failure mode that made the naked-position bug so bad.
        """
        report = {"checked": 0, "unprotected": [], "reprotected": [], "orphans": [], "errors": []}
        try:
            tracked = set(self._open_positions)
            symbols = tracked | {s for s in self._known_order_symbols()}
            for symbol in symbols:
                report["checked"] += 1
                try:
                    live = self._client.get_open_orders(symbol=symbol)
                except Exception as exc:
                    report["errors"].append(f"{symbol}: {exc}")
                    continue
                protective = [o for o in live
                              if o.get("type") in ("STOP_LOSS_LIMIT", "STOP_LOSS",
                                                   "TAKE_PROFIT_LIMIT", "TAKE_PROFIT")]
                pos = self._open_positions.get(symbol)

                if pos is None:
                    if protective:
                        report["orphans"].append(symbol)
                        logger.warning(
                            "RECONCILE %s: %d protective order(s) resting with no tracked "
                            "position. Cancelling — these could fire later against an asset "
                            "the system does not know it holds.", symbol, len(protective))
                        if auto_heal:
                            for o in protective:
                                try:
                                    self._client.cancel_order(symbol=symbol, orderId=o["orderId"])
                                except Exception as exc:
                                    report["errors"].append(f"{symbol} cancel: {exc}")
                    continue

                if not protective:
                    report["unprotected"].append(symbol)
                    logger.critical(
                        "RECONCILE %s: tracked position (entry %.6f, qty %.8f) has NO "
                        "protective orders on the exchange. It is running naked.",
                        symbol, pos.entry_price, pos.quantity)
                    if auto_heal and self._replace_protection(pos):
                        report["reprotected"].append(symbol)
        except Exception as exc:
            report["errors"].append(str(exc))
            logger.error("Reconciliation failed: %s", exc)

        logger.info("RECONCILE done: %d symbol(s) checked, %d unprotected (%d re-protected), "
                    "%d orphan set(s), %d error(s)",
                    report["checked"], len(report["unprotected"]), len(report["reprotected"]),
                    len(report["orphans"]), len(report["errors"]))
        return report

    def _known_order_symbols(self) -> list[str]:
        """Symbols with resting orders on the exchange, so orphans belonging to
        no tracked position are still found."""
        try:
            return sorted({o["symbol"] for o in self._client.get_open_orders()})
        except Exception:
            return []

    def _replace_protection(self, pos: "Position") -> bool:
        """Re-place the SL/TP pair for a position found running unprotected."""
        try:
            step, tick, _min_notional = self._symbol_filters(pos.symbol)
            qty = self._fmt(pos.quantity, step)
            sl_side = "SELL" if pos.direction == "LONG" else "BUY"
            tp_side = sl_side
            self._client.create_order(
                symbol=pos.symbol, side=sl_side, type="STOP_LOSS_LIMIT", quantity=qty,
                stopPrice=self._fmt(pos.sl_price, tick),
                price=self._fmt(pos.sl_price * (0.999 if sl_side == "SELL" else 1.001), tick),
                timeInForce="GTC")
            self._client.create_order(
                symbol=pos.symbol, side=tp_side, type="TAKE_PROFIT_LIMIT", quantity=qty,
                stopPrice=self._fmt(pos.tp_price, tick),
                price=self._fmt(pos.tp_price * (0.999 if tp_side == "SELL" else 1.001), tick),
                timeInForce="GTC")
            logger.warning("RECONCILE %s: protection re-placed (SL %.6f / TP %.6f).",
                           pos.symbol, pos.sl_price, pos.tp_price)
            return True
        except Exception as exc:
            logger.critical(
                "RECONCILE %s: could not re-place protection (%s). Position is STILL naked "
                "and needs manual attention.", pos.symbol, exc)
            return False

    def _emergency_flatten(self, symbol: str, entry_side: str, quantity: str, order_id: str) -> None:
        """Undo a filled entry whose protective orders could not be placed.

        Cancels anything that did land, then closes the position with a market
        order in the opposite direction. This deliberately accepts a small
        realised loss (spread + fees) because the alternative — holding an
        unprotected position the system has no record of — has no bounded
        downside at all.

        If the flatten itself fails there is nothing further this process can
        do automatically, so it logs at CRITICAL: that state needs a human.
        """
        try:
            for o in self._client.get_open_orders(symbol=symbol):
                try:
                    self._client.cancel_order(symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
            close_side = "SELL" if entry_side == "BUY" else "BUY"
            self._client.create_order(symbol=symbol, side=close_side,
                                      type="MARKET", quantity=quantity)
            logger.warning(
                "ROLLBACK %s: entry %s filled but protection failed — position flattened "
                "with a %s market order (entry order_id=%s).",
                symbol, entry_side, close_side, order_id,
            )
        except Exception as exc:
            logger.critical(
                "ROLLBACK FAILED for %s: entry filled, protection failed, AND the "
                "flatten failed (%s). There is now an UNPROTECTED position on the "
                "exchange that this process cannot close. HUMAN INTERVENTION REQUIRED.",
                symbol, exc,
            )

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
        if not quantity:
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

        # Every price sent to the exchange must be a multiple of the symbol's
        # tickSize. The old code used round(x, 4), which happens to work for
        # the stop TRIGGER but not for the limit price derived from it:
        # round(76950.1 * 0.999, 4) = 76873.1499, not a multiple of 0.01, and
        # the exchange rejects it with -1013 PRICE_FILTER. Snap everything.
        _, tick, _ = self._symbol_filters(decision.symbol)
        sl_price = float(self._fmt(sl_price, tick))
        tp_price = float(self._fmt(tp_price, tick))
        sl_limit = self._fmt(sl_price * (0.999 if side == "BUY" else 1.001), tick)
        tp_side = "SELL" if side == "BUY" else "BUY"
        tp_limit = self._fmt(tp_price * (0.999 if tp_side == "SELL" else 1.001), tick)

        order_id = ""
        entry_filled = False
        try:
            # Entry market order
            order = self._client.create_order(
                symbol=decision.symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            order_id = str(order.get("orderId", ""))
            entry_filled = True

            # Average across ALL fills. Reading fills[0] only — as this did —
            # records the first price level of a market order that walked the
            # book, understating the real entry and corrupting every SL/TP
            # distance and PnL figure computed from it.
            fills = order.get("fills") or []
            filled_qty = sum(float(f["qty"]) for f in fills)
            if filled_qty > 0:
                fill_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / filled_qty
                if len(fills) > 1:
                    logger.info("%s entry filled across %d price levels, VWAP %.6f",
                                decision.symbol, len(fills), fill_price)
            else:
                fill_price = price

            # Server-side stop-loss (STOP_LOSS_LIMIT)
            self._client.create_order(
                symbol=decision.symbol,
                side=sl_side,
                type="STOP_LOSS_LIMIT",
                quantity=quantity,
                stopPrice=self._fmt(sl_price, tick),
                price=sl_limit,
                timeInForce="GTC",
            )

            # Take-profit (TAKE_PROFIT_LIMIT)
            self._client.create_order(
                symbol=decision.symbol,
                side=tp_side,
                type="TAKE_PROFIT_LIMIT",
                quantity=quantity,
                stopPrice=self._fmt(tp_price, tick),
                price=tp_limit,
                timeInForce="GTC",
            )

            pos = Position(
                symbol=decision.symbol,
                direction=decision.direction,
                entry_price=fill_price,
                # quantity is an exchange-ready STRING for the API calls above
                # (see _fmt); Position stores it numerically for PnL maths.
                quantity=float(quantity),
                sl_price=sl_price,
                tp_price=tp_price,
                order_id=order_id,
            )
            self._open_positions[decision.symbol] = pos
            self._save_positions()
            logger.info(
                "OPENED %s %s qty=%s @ %.4f | SL=%.4f TP=%.4f",
                decision.direction, decision.symbol, quantity, fill_price, sl_price, tp_price,
            )
            return pos

        except BinanceAPIException as exc:
            logger.error("Order failed for %s: %s", decision.symbol, exc)
            # THE important half. Observed on testnet 2026-09-08: the entry
            # market order FILLED, the protective order was then rejected
            # (-1013 PRICE_FILTER), the exception landed here, execute()
            # returned None — and the account was left holding the asset with
            # NO stop, NO take-profit, and NO Position record. A naked,
            # unprotected position the system did not know it owned. With real
            # money that is unbounded downside from a bug the caller never sees.
            #
            # So if the entry filled but protection did not, flatten it
            # immediately rather than returning as though nothing happened.
            if entry_filled:
                self._emergency_flatten(decision.symbol, side, quantity, order_id)
            return None

    def close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            return
        self._save_positions()

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
