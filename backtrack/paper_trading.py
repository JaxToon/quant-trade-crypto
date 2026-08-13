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
from strategy.correlation import get_correlation

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
    timeframe: str = "5m"  # which SMC/FVG layer this signal came from
    liquidation_price: float | None = None  # derivative only — None for spot
    opened_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PaperTradingLedger:
    """
    Virtual account that opens/closes simulated positions based on
    entry_insight recommendations and tracks win rate over time.
    """

    def __init__(
        self, path: str | None = None, starting_balance: float | None = None, slippage_pct: float = 0.0,
    ) -> None:
        # Separate file per asset class — crypto and stock win rates are
        # different markets and shouldn't be averaged into one stat.
        self.path = path or os.path.join(settings.BACKTRACK_LOG_DIR, f"paper_ledger_{settings.ASSET_TYPE}.json")
        self.starting_balance = starting_balance if starting_balance is not None else settings.PAPER_TRADING_STARTING_BALANCE
        self.balance = self.starting_balance
        # Adverse slippage applied to BOTH legs of every fill (entry and
        # exit) — 0.0 in live/default use (see backtrack_logs vs backtest.py
        # docstrings). backtest.py sweeps this to find the level at which
        # the backtested edge stops being profitable, since neither the
        # live ledger nor the backtest otherwise model real order-book depth.
        self.slippage_pct = slippage_pct
        self.open_positions: dict[tuple[str, str, str], PaperPosition] = {}
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
            logger.warning(
                "No paper ledger found at %s — starting fresh from $%.2f. "
                "If you expected prior history to carry over, check that this "
                "path is on a persistent volume (see DEPLOYMENT.md).",
                self.path, self.starting_balance,
            )
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.balance = data.get("balance", self.starting_balance)
            self.closed_trades = data.get("closed_trades", [])
            self.last_learned_count = data.get("last_learned_count", 0)
            self.open_positions = {
                (p["symbol"], p["market"], p.get("timeframe", "5m")): PaperPosition(**{k: v for k, v in p.items()})
                for p in data.get("open_positions", [])
            }
            logger.info(
                "Paper ledger restored from %s — balance=$%.2f, %d closed trades, %d open.",
                self.path, self.balance, len(self.closed_trades), len(self.open_positions),
            )
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

    def maybe_open(
        self, symbol: str, market: str, rec: dict, timeframe: str = "5m", opened_at: str | None = None,
    ) -> str | None:
        """
        rec: a `spot` or `derivative` block from entry_insight.build_entry_insight.
        timeframe: which SMC/FVG layer this signal came from (e.g. "5m",
        "15m", "1h") — kept as a separate position key so the same symbol
        can have independent open positions per layer simultaneously.
        opened_at: simulated candle time for backtest replay — omit for live
        use, where "now" (wall-clock) is correct. backtest.py MUST pass this
        (the candle's own timestamp), otherwise every backtested position
        records today's real time as when it opened, silently breaking any
        analysis that needs real trade duration or time-overlap between
        positions (found via exactly that: an overlap analysis that should
        have found concurrent positions found none, because every "opened_at"
        was hours in the future relative to its own "closed_at").

        Returns None if a position was opened, or if there was nothing to do
        (no signal, already open). Returns a short human-readable reason if
        entry_insight says "actionable" but this ledger rejected it anyway —
        surfaced to the dashboard so a rejection is never silently invisible.
        """
        key = (symbol, market, timeframe)
        if key in self.open_positions:
            return None
        if not rec.get("actionable"):
            return None
        entry_zone = rec.get("entry_zone")
        stop_loss = rec.get("stop_loss")
        target = rec.get("target")
        if not entry_zone or stop_loss is None or target is None:
            return None

        entry_price = sum(entry_zone) / 2
        direction = rec.get("direction", "long")
        if self.slippage_pct:
            # Adverse fill: pay more to get long, receive less to get short.
            entry_price *= (1 + self.slippage_pct) if direction == "long" else (1 - self.slippage_pct)

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < entry_price * MIN_STOP_DISTANCE_PCT:
            stop_pct = stop_distance / entry_price * 100
            reason = (
                f"Stop terlalu rapat untuk profitable setelah fee "
                f"({stop_pct:.3f}% < {MIN_STOP_DISTANCE_PCT * 100:.2f}% minimum)"
            )
            logger.debug("%s %s: %s", symbol, market, reason)
            return reason

        # Fixed-risk sizing off STARTING balance, not current balance:
        # sizing off current balance compounds every single trade, and over
        # hundreds of sequential trades (a full backtest run) that compounds
        # geometrically into meaningless numbers. Sizing off a fixed risk
        # unit keeps the PnL curve roughly linear and interpretable, which
        # is what "what's the win rate" actually needs.
        #
        # Conviction scaling: a confidence=0 signal risks 1x the base unit;
        # confidence=100 risks up to PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER.
        # This is the "reward bolder calls" lever — bounded on purpose (see
        # settings.py) rather than a literal leverage multiplier.
        confidence = max(0.0, min(100.0, rec.get("confidence", 0.0)))
        max_mult = settings.PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER
        conviction_multiplier = 1.0 + (confidence / 100.0) * (max_mult - 1.0)

        risk_amount = self.starting_balance * settings.PAPER_TRADING_RISK_PCT * conviction_multiplier

        # Derivative gets a modest, damped bump on top of conviction scaling
        # — real leverage doesn't make a risk-sized trade bigger (that's the
        # whole point of sizing by fixed $ risk), it makes the SAME risk
        # cheaper to hold in margin. We're not modeling margin/capital here,
        # so this is a deliberate, bounded stand-in: sqrt(leverage), not
        # leverage itself, so a "3x" market doesn't just recreate a 3x-bigger
        # version of the same trade — MAX_NOTIONAL_MULTIPLE below is still
        # the hard backstop regardless of how this multiplier is computed.
        leverage = max(1.0, float(rec.get("suggested_leverage", 1)))
        liquidation_price = None
        if market == "derivative":
            risk_amount *= leverage ** 0.5
            liquidation_price = (
                entry_price * (1 - 1 / leverage) if direction == "long"
                else entry_price * (1 + 1 / leverage)
            )

        # Portfolio-level correlated-risk cap: an individually-fine 2% risk
        # unit can still stack into a much bigger correlated bet if several
        # majors are already open at once (backtest found peak 7 simultaneous
        # positions, avg pairwise correlation 0.62 among them). Weight each
        # open position's own risk ($ at its own stop) by how correlated it
        # is to the SYMBOL being opened now, sum that up, and refuse to add
        # more once the correlation-weighted total would cross the cap —
        # even though this new trade would pass every other filter on its own.
        correlated_exposure = sum(
            pos.quantity * abs(pos.entry_price - pos.stop_loss) * get_correlation(symbol, pos.symbol)
            for pos in self.open_positions.values()
            if get_correlation(symbol, pos.symbol) >= settings.PORTFOLIO_CORRELATION_THRESHOLD
        )
        portfolio_cap = self.starting_balance * settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT
        if correlated_exposure + risk_amount > portfolio_cap:
            reason = (
                f"Portfolio correlated-risk cap tercapai "
                f"(${correlated_exposure:.0f} exposure berkorelasi + ${risk_amount:.0f} trade baru "
                f"> ${portfolio_cap:.0f} cap)"
            )
            logger.debug("%s %s: %s", symbol, market, reason)
            return reason

        quantity = risk_amount / stop_distance

        # Hard cap regardless — a tight-but-not-rejected stop combined with
        # a much wider target (or the derivative bump above) can still
        # produce an unrealistic position size.
        max_notional = self.starting_balance * MAX_NOTIONAL_MULTIPLE
        if quantity * entry_price > max_notional:
            quantity = max_notional / entry_price

        pos = PaperPosition(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            leverage=leverage,
            confidence=rec.get("confidence", 0.0),
            timeframe=timeframe,
            liquidation_price=liquidation_price,
            **({"opened_at": opened_at} if opened_at else {}),
        )
        self.open_positions[key] = pos
        liq_note = f" | liq={liquidation_price:.6f}" if liquidation_price else ""
        logger.info(
            "PAPER OPEN %s %s %s [%s] @ %.6f | SL=%.6f TP=%.6f qty=%.4f | conviction=%.2fx (confidence=%.0f%%)%s",
            market, pos.direction.upper(), symbol, timeframe, entry_price, stop_loss, target, quantity,
            conviction_multiplier, confidence, liq_note,
        )
        return None

    def check_and_close(self, current_prices: dict[str, float]) -> None:
        """Live mode: check open positions against a single latest price per symbol."""
        for key, pos in list(self.open_positions.items()):
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            # Liquidation (derivative only) takes priority — it's the more
            # severe, exchange-forced outcome if it's ever actually reached.
            if pos.liquidation_price is not None:
                hit_liq = price <= pos.liquidation_price if pos.direction == "long" else price >= pos.liquidation_price
                if hit_liq:
                    self._close(key, pos, pos.liquidation_price, "loss", closed_at=None, liquidated=True)
                    continue

            if pos.direction == "long":
                hit_sl, hit_tp = price <= pos.stop_loss, price >= pos.target
            else:  # short
                hit_sl, hit_tp = price >= pos.stop_loss, price <= pos.target

            if hit_sl or hit_tp:
                exit_price = pos.stop_loss if hit_sl else pos.target
                self._close(key, pos, exit_price, "loss" if hit_sl else "win", closed_at=None)

    def check_and_close_ohlc(self, symbol: str, low: float, high: float, closed_at: str, timeframe: str | None = None) -> None:
        """
        Backtest mode: check a position against a full candle's [low, high]
        range instead of a single price, since intrabar price action isn't
        known from OHLC alone. If both SL and TP fall inside the same
        candle's range, conservatively assume SL was hit first — avoids an
        optimistic look-ahead bias in the backtested win rate.

        timeframe: restrict the check to one layer's open positions (a 15m
        candle shouldn't resolve a 1h position). None checks every layer —
        fine for single-timeframe backtests where only one layer is ever open.
        """
        for market in ("spot", "derivative"):
            for key, pos in list(self.open_positions.items()):
                if pos.symbol != symbol or pos.market != market:
                    continue
                if timeframe is not None and pos.timeframe != timeframe:
                    continue

                if pos.liquidation_price is not None:
                    hit_liq = low <= pos.liquidation_price if pos.direction == "long" else high >= pos.liquidation_price
                    if hit_liq:
                        self._close(key, pos, pos.liquidation_price, "loss", closed_at=closed_at, liquidated=True)
                        continue

                if pos.direction == "long":
                    hit_sl, hit_tp = low <= pos.stop_loss, high >= pos.target
                else:  # short
                    hit_sl, hit_tp = high >= pos.stop_loss, low <= pos.target

                if hit_sl or hit_tp:
                    exit_price = pos.stop_loss if hit_sl else pos.target
                    self._close(key, pos, exit_price, "loss" if hit_sl else "win", closed_at=closed_at)

    def _close(
        self, key: tuple[str, str, str], pos: PaperPosition, exit_price: float, outcome: str,
        closed_at: str | None, liquidated: bool = False,
    ) -> None:
        if self.slippage_pct:
            # Adverse fill on the way out too: a long sells for less, a
            # short buys back for more than the theoretical SL/TP/liq price.
            exit_price *= (1 - self.slippage_pct) if pos.direction == "long" else (1 + self.slippage_pct)

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
            "timeframe": pos.timeframe,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "pnl_dollars": round(pnl_dollars, 4), "pnl_pct": round(pnl_pct, 4),
            "fee_dollars": round(fee_dollars, 4),
            "outcome": outcome, "liquidated": liquidated, "opened_at": pos.opened_at,
            "closed_at": closed_at or datetime.utcnow().isoformat(),
        })
        # Cap in-memory history; keep last_learned_count valid against the trim.
        overflow = len(self.closed_trades) - self.max_closed_trades
        if overflow > 0:
            self.closed_trades = self.closed_trades[overflow:]
            self.last_learned_count = max(0, self.last_learned_count - overflow)

        del self.open_positions[key]
        tag = "LIQUIDATED" if liquidated else outcome.upper()
        logger.info(
            "PAPER CLOSE %s %s %s @ %.6f | %s | PnL=$%.2f (%.2f%%) | balance=$%.2f",
            pos.market, pos.direction.upper(), pos.symbol, exit_price,
            tag, pnl_dollars, pnl_pct * 100, self.balance,
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
            liquidations = sum(1 for t in trades if t.get("liquidated"))
            return {
                "trades": n,
                "wins": wins,
                "losses": n - wins,
                "liquidations": liquidations,
                "win_rate_pct": round(100 * wins / n, 1) if n else None,
                "total_pnl_dollars": round(sum(t["pnl_dollars"] for t in trades), 2),
            }

        spot_trades = [t for t in self.closed_trades if t["market"] == "spot"]
        deriv_trades = [t for t in self.closed_trades if t["market"] == "derivative"]

        # Per-timeframe breakdown — the whole point of layering 5m/15m/1h is
        # to see which layer actually produces a fee-viable, positive edge.
        timeframes = sorted({t.get("timeframe", "5m") for t in self.closed_trades})
        by_timeframe = {
            tf: _breakdown([t for t in self.closed_trades if t.get("timeframe", "5m") == tf])
            for tf in timeframes
        }

        return {
            "starting_balance": self.starting_balance,
            "balance": round(self.balance, 2),
            "total_return_pct": round(100 * (self.balance - self.starting_balance) / self.starting_balance, 2),
            "open_positions": len(self.open_positions),
            "overall": _breakdown(self.closed_trades),
            "spot": _breakdown(spot_trades),
            "derivative": _breakdown(deriv_trades),
            "by_timeframe": by_timeframe,
        }
