"""
Paper-trading ledger: scores the entry-insight recommendations
(strategy/entry_insight.py) against real price action WITHOUT placing any
real or testnet orders — a virtual $10k account, fixed-risk position sizing,
closed purely off SL/TP crosses on live price.

This exists to answer "does the SMC/FVG insight actually have an edge?"
independently of the bear-veto gate and the LLM-consensus execution path
(strategy/execution.py), which already places real testnet spot orders on
its own separate decision path.

Persisted to backtrack_logs/paper_ledger.json so the win-rate survives
restarts — let main.py run for a while and the sample size grows.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

# Fixed-risk sizing (quantity = risk_amount / stop_distance) blows up when
# stop_distance is near-zero — which happens often, since FVG-derived stops
# can land a fraction of a cent apart on low-priced assets. MAX_NOTIONAL_MULTIPLE
# is a hard safety net regardless of how tight the stop actually is.
MAX_NOTIONAL_MULTIPLE = 5.0      # cap position notional at 5x starting balance

# Fee-aware stop floor: fees scale with NOTIONAL (quantity * price), but risk
# is deliberately held fixed — so a stop tighter than this makes quantity so
# large that round-trip fees alone can exceed the entire risk budget on the
# trade, guaranteeing a loss regardless of win rate. A backtest confirmed
# this exactly: 65% win rate, +$163k gross PnL, but -$82k net after fees,
# because the average fee ($98.74) was ~half the $200 risk unit per trade.
# Solving fee_roundtrip / stop_pct <= MAX_FEE_TO_RISK_RATIO for stop_pct:
MAX_FEE_TO_RISK_RATIO = 0.30  # fees shouldn't eat more than 30% of the risk budget
MIN_STOP_DISTANCE_PCT = (2 * settings.PAPER_TRADING_FEE_PCT) / MAX_FEE_TO_RISK_RATIO


@dataclass
class PaperPosition:
    symbol: str
    market: str        # "spot" | "derivative"
    direction: str      # "long" | "short"
    entry_price: float
    stop_loss: float
    target: float
    quantity: float
    leverage: float
    confidence: float
    opened_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PaperTradingLedger:
    """
    Virtual account that opens/closes simulated positions based on
    entry_insight recommendations and tracks win rate over time.
    """

    def __init__(self, path: str | None = None, starting_balance: float | None = None) -> None:
        self.path = path or os.path.join(settings.BACKTRACK_LOG_DIR, "paper_ledger.json")
        self.starting_balance = starting_balance if starting_balance is not None else settings.PAPER_TRADING_STARTING_BALANCE
        self.balance = self.starting_balance
        self.open_positions: dict[tuple[str, str], PaperPosition] = {}
        self.closed_trades: list[dict] = []
        self.last_learned_count = 0  # index into closed_trades already used for GNN updates
        # Live use (main.py, running for weeks/months) needs a cap so the
        # ledger file doesn't grow forever. A bulk backtest replaying a fixed
        # historical window wants the true, untrimmed count — set this much
        # higher (backtest.py does) before running a replay.
        self.max_closed_trades = 1000
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.balance = data.get("balance", self.starting_balance)
            self.closed_trades = data.get("closed_trades", [])
            self.last_learned_count = data.get("last_learned_count", 0)
            self.open_positions = {
                (p["symbol"], p["market"]): PaperPosition(**{k: v for k, v in p.items()})
                for p in data.get("open_positions", [])
            }
        except Exception as exc:
            logger.warning("Paper ledger load failed, starting fresh: %s", exc)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            payload = {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "open_positions": [asdict(p) for p in self.open_positions.values()],
                "closed_trades": self.closed_trades,  # already capped in _record_close
                "last_learned_count": self.last_learned_count,
            }
            with open(self.path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning("Paper ledger save failed: %s", exc)

    # ── Trading ────────────────────────────────────────────────────────────

    def maybe_open(self, symbol: str, market: str, rec: dict) -> None:
        """rec: a `spot` or `derivative` block from entry_insight.build_entry_insight."""
        key = (symbol, market)
        if key in self.open_positions:
            return
        if not rec.get("actionable"):
            return
        entry_zone = rec.get("entry_zone")
        stop_loss = rec.get("stop_loss")
        target = rec.get("target")
        if not entry_zone or stop_loss is None or target is None:
            return

        entry_price = sum(entry_zone) / 2
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < entry_price * MIN_STOP_DISTANCE_PCT:
            logger.debug(
                "%s %s: stop too tight (%.8f < %.8f), skipping to avoid a sizing blowup",
                symbol, market, stop_distance, entry_price * MIN_STOP_DISTANCE_PCT,
            )
            return

        # Fixed-risk sizing off STARTING balance, not current balance:
        # sizing off current balance compounds every single trade, and over
        # hundreds of sequential trades (a full backtest run) that compounds
        # geometrically into meaningless numbers. Sizing off a fixed risk
        # unit keeps the PnL curve roughly linear and interpretable, which
        # is what "what's the win rate" actually needs.
        risk_amount = self.starting_balance * settings.PAPER_TRADING_RISK_PCT
        quantity = risk_amount / stop_distance

        # Hard cap regardless — a tight-but-not-rejected stop combined with
        # a much wider target can still produce an unrealistic win size.
        max_notional = self.starting_balance * MAX_NOTIONAL_MULTIPLE
        if quantity * entry_price > max_notional:
            quantity = max_notional / entry_price

        pos = PaperPosition(
            symbol=symbol,
            market=market,
            direction=rec.get("direction", "long"),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            leverage=rec.get("suggested_leverage", 1),
            confidence=rec.get("confidence", 0.0),
        )
        self.open_positions[key] = pos
        logger.info(
            "PAPER OPEN %s %s %s @ %.6f | SL=%.6f TP=%.6f qty=%.4f",
            market, pos.direction.upper(), symbol, entry_price, stop_loss, target, quantity,
        )

    def check_and_close(self, current_prices: dict[str, float]) -> None:
        """Live mode: check open positions against a single latest price per symbol."""
        for key, pos in list(self.open_positions.items()):
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            if pos.direction == "long":
                hit_sl, hit_tp = price <= pos.stop_loss, price >= pos.target
            else:  # short
                hit_sl, hit_tp = price >= pos.stop_loss, price <= pos.target

            if hit_sl or hit_tp:
                exit_price = pos.stop_loss if hit_sl else pos.target
                self._close(key, pos, exit_price, "loss" if hit_sl else "win", closed_at=None)

    def check_and_close_ohlc(self, symbol: str, low: float, high: float, closed_at: str) -> None:
        """
        Backtest mode: check a position against a full candle's [low, high]
        range instead of a single price, since intrabar price action isn't
        known from OHLC alone. If both SL and TP fall inside the same
        candle's range, conservatively assume SL was hit first — avoids an
        optimistic look-ahead bias in the backtested win rate.
        """
        for market in ("spot", "derivative"):
            key = (symbol, market)
            pos = self.open_positions.get(key)
            if pos is None:
                continue

            if pos.direction == "long":
                hit_sl, hit_tp = low <= pos.stop_loss, high >= pos.target
            else:  # short
                hit_sl, hit_tp = high >= pos.stop_loss, low <= pos.target

            if hit_sl or hit_tp:
                exit_price = pos.stop_loss if hit_sl else pos.target
                self._close(key, pos, exit_price, "loss" if hit_sl else "win", closed_at=closed_at)

    def _close(self, key: tuple[str, str], pos: PaperPosition, exit_price: float, outcome: str, closed_at: str | None) -> None:
        sign = 1 if pos.direction == "long" else -1
        gross_pnl_dollars = pos.quantity * (exit_price - pos.entry_price) * sign
        # Round-trip fee on both entry and exit notional — "win" / "loss"
        # still reflects whether the SL or TP was hit (signal correctness);
        # pnl_dollars reflects the realistic net economics after costs.
        fee_dollars = pos.quantity * (pos.entry_price + exit_price) * settings.PAPER_TRADING_FEE_PCT
        pnl_dollars = gross_pnl_dollars - fee_dollars
        pnl_pct = (pnl_dollars / (pos.quantity * pos.entry_price)) if pos.entry_price else 0.0

        self.balance += pnl_dollars
        self.closed_trades.append({
            "symbol": pos.symbol, "market": pos.market, "direction": pos.direction,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "pnl_dollars": round(pnl_dollars, 4), "pnl_pct": round(pnl_pct, 4),
            "fee_dollars": round(fee_dollars, 4),
            "outcome": outcome, "opened_at": pos.opened_at,
            "closed_at": closed_at or datetime.utcnow().isoformat(),
        })
        # Cap in-memory history; keep last_learned_count valid against the trim.
        overflow = len(self.closed_trades) - self.max_closed_trades
        if overflow > 0:
            self.closed_trades = self.closed_trades[overflow:]
            self.last_learned_count = max(0, self.last_learned_count - overflow)

        del self.open_positions[key]
        logger.info(
            "PAPER CLOSE %s %s %s @ %.6f | %s | PnL=$%.2f (%.2f%%) | balance=$%.2f",
            pos.market, pos.direction.upper(), pos.symbol, exit_price,
            outcome.upper(), pnl_dollars, pnl_pct * 100, self.balance,
        )

    # ── Continual learning hook ────────────────────────────────────────────

    def get_unlearned_trades(self) -> list[dict]:
        """Closed trades since the last GNN learning update."""
        return self.closed_trades[self.last_learned_count:]

    def mark_learned(self) -> None:
        self.last_learned_count = len(self.closed_trades)

    # ── Reporting ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        def _breakdown(trades: list[dict]) -> dict:
            n = len(trades)
            wins = sum(1 for t in trades if t["outcome"] == "win")
            return {
                "trades": n,
                "wins": wins,
                "losses": n - wins,
                "win_rate_pct": round(100 * wins / n, 1) if n else None,
                "total_pnl_dollars": round(sum(t["pnl_dollars"] for t in trades), 2),
            }

        spot_trades = [t for t in self.closed_trades if t["market"] == "spot"]
        deriv_trades = [t for t in self.closed_trades if t["market"] == "derivative"]

        return {
            "starting_balance": self.starting_balance,
            "balance": round(self.balance, 2),
            "total_return_pct": round(100 * (self.balance - self.starting_balance) / self.starting_balance, 2),
            "open_positions": len(self.open_positions),
            "overall": _breakdown(self.closed_trades),
            "spot": _breakdown(spot_trades),
            "derivative": _breakdown(deriv_trades),
        }
