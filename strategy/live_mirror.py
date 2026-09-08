"""
Mirror Path 1's paper decisions onto a real exchange, byte-for-byte.

WHY THIS EXISTS RATHER THAN REUSING ExecutionEngine
---------------------------------------------------
`strategy/execution.py` serves Path 2 (the LLM debate). It computes its own
stop, target and size from fixed settings:

    sl = price * (1 - STOP_LOSS_PCT)        # flat 3%
    tp = price * (1 + TAKE_PROFIT_PCT)      # flat 6%
    qty = balance * MAX_POSITION_PCT / price

Path 1 — the path with the actual evidence behind it — decides completely
differently. Its stops come from SMC/FVG structure, its targets from the
measured gap, and its size from risk/stop-distance with conviction scaling,
a portfolio correlation cap, a fee-aware minimum stop distance and a
notional ceiling.

Routing Path 1 signals through ExecutionEngine would therefore place orders
with different stops, different targets and different sizes than everything
that was ever validated. The 6/6-segment profitability and the
statistically-proven live LONG edge (n=82, +$166.85/trade, 95% CI
[+$91.49, +$275.83]) would simply not describe what the account was doing.

So this module does not decide anything. `PaperTradingLedger` remains the
single brain: it applies every guard it already has and produces a
`PaperPosition`. This mirrors that exact position onto the exchange — same
quantity, same stop, same target. Live behaviour is identical to the
simulation by construction, not by careful re-implementation that drifts.

SAFETY PROPERTIES
-----------------
- Disabled by default (`LIVE_TRADING_ENABLED`). Nothing here runs unless
  explicitly switched on.
- Refuses to arm at all unless the endpoint is testnet OR
  `LIVE_TRADING_CONFIRM_REAL_MONEY` is set, so real money can never be
  reached by forgetting one flag.
- Every entry is followed by exchange-held OCO protection. If protection
  cannot be placed, the entry is flattened immediately — the naked-position
  failure observed on testnet 2026-09-08.
- A drawdown kill switch, because Path 1 has never had one:
  DAILY_DRAWDOWN_LIMIT lives only in execution.py (Path 2, disabled).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class MirrorResult:
    ok: bool
    reason: str = ""
    entry_order_id: str = ""
    protection_order_ids: list[str] = field(default_factory=list)
    fill_price: float | None = None


class LiveOrderMirror:
    """Places real orders that mirror PaperTradingLedger decisions.

    Deliberately has no opinion about WHETHER to trade — that judgement lives
    in the ledger, which has been validated. This only faithfully reproduces
    a decision the ledger already made.
    """

    def __init__(self, client=None, enabled: bool | None = None) -> None:
        self.enabled = settings.LIVE_TRADING_ENABLED if enabled is None else enabled
        self._client = client
        self._filters: dict[str, tuple[float, float, float]] = {}
        self._day = date.today()
        self._day_start_balance: float | None = None
        self._halted = False
        self._halt_reason = ""

        if not self.enabled:
            logger.info("LiveOrderMirror: DISABLED (LIVE_TRADING_ENABLED=false). Paper only.")
            return

        if self._client is None:
            from binance.client import Client
            self._client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET,
                                  testnet=settings.BINANCE_TESTNET)
            if settings.BINANCE_TESTNET:
                self._client.API_URL = Client.API_TESTNET_URL

        url = getattr(self._client, "API_URL", "")
        is_testnet = "testnet" in url
        if not is_testnet and not settings.LIVE_TRADING_CONFIRM_REAL_MONEY:
            # Two independent flags must agree before real money is reachable.
            # One forgotten env var should never be the only thing between a
            # test run and a live account.
            self.enabled = False
            logger.critical(
                "LiveOrderMirror: endpoint %s is NOT testnet and "
                "LIVE_TRADING_CONFIRM_REAL_MONEY is not set. REFUSING to arm.", url)
            return

        logger.warning("LiveOrderMirror ARMED against %s%s", url,
                       "" if is_testnet else "  *** REAL MONEY ***")

    # ── exchange helpers ──────────────────────────────────────────────────

    def _symbol_filters(self, symbol: str) -> tuple[float, float, float]:
        """(stepSize, tickSize, minNotional), cached per symbol.

        Raises ValueError for a symbol the exchange does not list. Binance
        returns None rather than erroring, and indexing that produced a bare
        TypeError ("'NoneType' object is not subscriptable") that told the
        caller nothing about what was actually wrong. Found by the failure-path
        check, 2026-09-08.
        """
        if symbol in self._filters:
            return self._filters[symbol]
        info = self._client.get_symbol_info(symbol)
        if not info or "filters" not in info:
            raise ValueError(f"{symbol} is not listed on this exchange")
        f = {x["filterType"]: x for x in info["filters"]}
        out = (float(f["LOT_SIZE"]["stepSize"]),
               float(f["PRICE_FILTER"]["tickSize"]),
               float(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", 0.0)))
        self._filters[symbol] = out
        return out

    @staticmethod
    def _fmt(value: float, increment: float) -> str:
        """Snap to the exchange increment, rendered fixed-point.

        Small floats otherwise serialise as "8e-05" and Binance rejects them
        with -1100 illegal characters — which would break every order on a
        small account. Observed on testnet 2026-09-08.
        """
        prec = max(0, len(f"{increment:.10f}".rstrip('0').split('.')[1])) if increment < 1 else 0
        return f"{round(value / increment) * increment:.{prec}f}"

    def _balance(self, asset: str = "USDT") -> float:
        for b in self._client.get_account()["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    # ── kill switch ───────────────────────────────────────────────────────

    def _check_drawdown(self) -> bool:
        """True if trading should be halted. Path 1 has never had a loss
        limit — DAILY_DRAWDOWN_LIMIT is enforced only in execution.py, which
        is Path 2 and disabled. A losing streak has run unchecked by design."""
        limit = settings.LIVE_TRADING_DAILY_LOSS_LIMIT_PCT
        if limit <= 0:
            return False
        today = date.today()
        if today != self._day:
            self._day, self._day_start_balance, self._halted = today, None, False
        bal = self._balance()
        if self._day_start_balance is None:
            self._day_start_balance = bal
            return False
        if self._day_start_balance <= 0:
            return False
        drop = (self._day_start_balance - bal) / self._day_start_balance * 100
        if drop >= limit and not self._halted:
            self._halted = True
            self._halt_reason = f"daily loss {drop:.2f}% >= {limit:.2f}%"
            logger.critical("LIVE TRADING HALTED: %s. Open positions keep their "
                            "exchange-held stops; no new entries.", self._halt_reason)
        return self._halted

    # ── the mirror ────────────────────────────────────────────────────────

    def mirror_open(self, pos) -> MirrorResult:
        """Reproduce one PaperPosition on the exchange.

        `pos` is a PaperPosition the ledger has ALREADY decided to open, with
        all its guards applied. Nothing is second-guessed here except whether
        the exchange can physically accept it.
        """
        if not self.enabled:
            return MirrorResult(False, "mirror disabled")
        if self._check_drawdown():
            return MirrorResult(False, f"halted: {self._halt_reason}")

        symbol, direction = pos.symbol, pos.direction
        try:
            step, tick, min_notional = self._symbol_filters(symbol)
        except Exception as exc:
            logger.error("MIRROR %s: cannot read symbol filters — %s", symbol, exc)
            return MirrorResult(False, f"symbol unusable: {exc}")
        qty = self._fmt(pos.quantity, step)
        if float(qty) <= 0:
            return MirrorResult(False, "quantity rounds to zero at this symbol's step size")

        # The protective leg sits beyond the entry, so a size that clears
        # MIN_NOTIONAL at entry can still fail on the stop. Opening in that
        # case would leave the position unprotectable — refuse instead.
        if float(qty) * pos.stop_loss < min_notional:
            return MirrorResult(
                False,
                f"stop leg notional {float(qty) * pos.stop_loss:.2f} < minNotional "
                f"{min_notional:.2f}; opening this could not be protected")

        if direction != "long":
            # Spot cannot short, and a derivative short needs the futures API
            # (different endpoints, margin and liquidation semantics) which
            # this mirror does not implement. Refusing loudly beats silently
            # placing a spot SELL of an asset the account may not hold.
            return MirrorResult(False, "short positions need futures support; not implemented")

        side, close_side = "BUY", "SELL"

        # Spot has no leverage, but the paper ledger sizes as though it does:
        # MAX_NOTIONAL_MULTIPLE allows a position up to 5x the account, and in
        # practice the live ledger runs a median 4.88x notional per $1 of
        # capital. Mirroring such a decision onto spot simply cannot fill.
        #
        # Reject with the number the caller needs rather than letting the
        # exchange return a bare -2010: the useful information is HOW MUCH
        # capital this decision required, since that is the real constraint on
        # what a live account must hold.
        notional = float(qty) * pos.entry_price
        free = self._balance()
        if notional > free:
            msg = (f"needs ${notional:,.2f} of quote balance but only ${free:,.2f} is free "
                   f"({notional / free:.2f}x). Spot cannot borrow — either fund the account "
                   f"to at least ${notional:,.2f} or run the paper ledger with a "
                   f"starting_balance matching the real one.")
            logger.error("MIRROR %s: %s", symbol, msg)
            return MirrorResult(False, msg)

        # Snapshot what is already resting so a rollback cannot cancel another
        # position's protection along with its own.
        try:
            pre_existing = {str(o["orderId"]) for o in self._client.get_open_orders(symbol=symbol)}
        except Exception:
            pre_existing = set()

        try:
            order = self._client.create_order(symbol=symbol, side=side,
                                              type="MARKET", quantity=qty)
        except Exception as exc:
            logger.error("MIRROR %s: entry rejected — %s", symbol, exc)
            return MirrorResult(False, f"entry rejected: {exc}")

        entry_id = str(order.get("orderId", ""))
        fills = order.get("fills") or []
        filled = sum(float(f["qty"]) for f in fills)
        fill_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills) / filled
                      if filled else pos.entry_price)
        if len(fills) > 1:
            logger.info("MIRROR %s: filled across %d levels, VWAP %.6f", symbol, len(fills), fill_price)

        # Exchange-held protection. This is the whole point: it survives the
        # process dying, which a stop living in Python does not. This
        # deployment has a measured 2.5-day outage during which stops that
        # existed only in memory protected nothing.
        try:
            # Closing a long: take-profit sits ABOVE, stop-loss BELOW.
            # Binance replaced the old symmetric OCO parameters with
            # above*/below*; the legacy form returns -1102 'aboveType was not
            # sent'. Verified against testnet 2026-09-08.
            oco = self._client.create_oco_order(
                symbol=symbol, side="SELL", quantity=qty,
                aboveType="LIMIT_MAKER",
                abovePrice=self._fmt(pos.target, tick),
                belowType="STOP_LOSS_LIMIT",
                belowStopPrice=self._fmt(pos.stop_loss, tick),
                belowPrice=self._fmt(pos.stop_loss * 0.999, tick),
                belowTimeInForce="GTC",
            )
            ids = [str(r["orderId"]) for r in oco.get("orderReports", [])]
            logger.info("MIRROR %s: OCO protection placed %s", symbol, ids)
            return MirrorResult(True, "", entry_id, ids, fill_price)
        except Exception as exc:
            logger.error("MIRROR %s: protection FAILED after entry filled — %s", symbol, exc)
            flattened = self._flatten(symbol, close_side, qty, entry_id, pre_existing)
            # Report what actually happened. The first version always said
            # "entry flattened" even when the flatten itself had failed, which
            # is precisely the case a caller most needs to know about.
            note = ("protection failed, entry flattened" if flattened
                    else "protection failed AND rollback failed — POSITION IS NAKED")
            return MirrorResult(False, f"{note}: {exc}")

    def _flatten(self, symbol: str, close_side: str, qty: str, entry_id: str,
                 spare_order_ids: set | None = None) -> bool:
        """Undo a filled entry that could not be protected.

        Accepts a small spread+fee loss deliberately. The alternative — an
        unprotected position the system has no record of — has no bounded
        downside, which is exactly the failure seen on testnet 2026-09-08.

        `spare_order_ids` are orders that existed BEFORE this entry and belong
        to other positions on the same symbol. The first version cancelled
        every open order for the symbol, so rolling one entry back stripped the
        protection off unrelated positions and left THEM naked — trading one
        naked position for another. Found by the failure-path check.
        """
        spare = spare_order_ids or set()
        try:
            for o in self._client.get_open_orders(symbol=symbol):
                if str(o["orderId"]) in spare:
                    continue        # someone else's protection — leave it alone
                try:
                    self._client.cancel_order(symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
            self._client.create_order(symbol=symbol, side=close_side,
                                      type="MARKET", quantity=qty)
            logger.warning("MIRROR ROLLBACK %s: unprotectable entry (%s) flattened.",
                           symbol, entry_id)
            return True
        except Exception as exc:
            logger.critical(
                "MIRROR ROLLBACK FAILED %s: entry %s filled, protection failed, flatten "
                "failed (%s). An UNPROTECTED position exists on the exchange that this "
                "process cannot close. HUMAN INTERVENTION REQUIRED.", symbol, entry_id, exc)
            return False

    def mirror_close(self, symbol: str) -> MirrorResult:
        """Close a mirrored position early, because the ledger closed it for a
        reason the exchange's own OCO does not know about — a time stop, a
        regime veto, or a manual stop of the system."""
        if not self.enabled:
            return MirrorResult(False, "mirror disabled")
        try:
            open_orders = self._client.get_open_orders(symbol=symbol)
            for o in open_orders:
                try:
                    self._client.cancel_order(symbol=symbol, orderId=o["orderId"])
                except Exception:
                    pass
            base = symbol.replace("USDT", "")
            held = self._balance(base)
            step, _tick, min_notional = self._symbol_filters(symbol)
            qty = self._fmt(held, step)
            if float(qty) <= 0:
                return MirrorResult(True, "nothing held to close")
            self._client.create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qty)
            logger.info("MIRROR %s: position closed on request (qty %s)", symbol, qty)
            return MirrorResult(True)
        except Exception as exc:
            logger.error("MIRROR %s: close failed — %s", symbol, exc)
            return MirrorResult(False, str(exc))
