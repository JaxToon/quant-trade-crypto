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
import threading
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
    # ATR (14-period, from SmartMoneyAnalyzer) at entry — sizes the trailing
    # stop's distance if PAPER_TRADING_TRAILING_STOP_ENABLED. 0.0 default so
    # positions opened/persisted before this field existed simply never
    # activate trailing (see _maybe_update_trailing_stop's guard) rather
    # than crashing or getting a fabricated ATR on load.
    atr_at_entry: float = 0.0
    trailing_stop: float | None = None


class PaperTradingLedger:
    """
    Virtual account that opens/closes simulated positions based on
    entry_insight recommendations and tracks win rate over time.
    """

    def __init__(
        self, path: str | None = None, starting_balance: float | None = None, slippage_pct: float = 0.0,
        stop_slippage_pct: float = 0.0, equity_based_sizing: bool | None = None,
        max_drawdown_halt_pct: float | None = None,
    ) -> None:
        # Separate file per asset class — crypto and stock win rates are
        # different markets and shouldn't be averaged into one stat.
        self.path = path or os.path.join(settings.BACKTRACK_LOG_DIR, f"paper_ledger_{settings.ASSET_TYPE}.json")
        self.starting_balance = starting_balance if starting_balance is not None else settings.PAPER_TRADING_STARTING_BALANCE
        # Guards every mutation of balance/open_positions/closed_trades.
        #
        # Needed since 2026-09-07, when SL/TP monitoring moved out of the
        # 60s decision_loop into its own fast thread: the ledger is now
        # touched concurrently by the monitor thread (check_and_close +
        # save), the decision loop (maybe_open + check_and_close + save)
        # and the scheduler thread (learning hooks). Without this, two
        # threads can close the same position twice, or save() can
        # serialise a half-mutated dict to disk.
        #
        # RLock, not Lock: these methods legitimately call one another
        # (check_and_close -> _close), and a plain Lock would self-deadlock.
        self._lock = threading.RLock()

        self.balance = self.starting_balance
        # Adverse slippage applied to BOTH legs of every fill (entry and
        # exit) — 0.0 in live/default use (see backtrack_logs vs backtest.py
        # docstrings). backtest.py sweeps this to find the level at which
        # the backtested edge stops being profitable, since neither the
        # live ledger nor the backtest otherwise model real order-book depth.
        self.slippage_pct = slippage_pct
        # ADDITIONAL adverse slippage applied ONLY to stop-type exits (stop
        # loss, trailing stop, liquidation) — never to target exits.
        #
        # `slippage_pct` above is symmetric across every fill, which quietly
        # misprices any rule that changes the MIX of exit types, because the
        # two exit types do not slip alike in reality:
        #   - a stop becomes a MARKET order the moment it is breached, so it
        #     fills past the trigger. Measured on 4 real live SHORT stop-outs
        #     (2026-09-03): 0.25-0.60% beyond the recorded stop.
        #   - a target is a LIMIT order, which fills at its price or better
        #     and never slips adversely.
        # Modelling both at one flat rate therefore under-penalises exactly
        # the strategies that convert target exits into stop exits (breakeven
        # and trailing rules), which is the comparison this exists to keep
        # honest. 0.0 by default: live and every existing backtest behave
        # exactly as before unless a caller opts in.
        #
        # Still NOT modelled, in the opposite direction: target exits are
        # assumed to fill whenever the candle merely touches them, so a
        # limit order grazed for one tick is counted as filled. That optimism
        # favours whichever arm takes more target exits (usually the
        # baseline), so a rule that wins despite stop slippage is winning
        # against an already-flattered comparison.
        self.stop_slippage_pct = stop_slippage_pct
        # Size off current equity rather than the never-changing starting
        # balance. See settings.PAPER_TRADING_EQUITY_BASED_SIZING for why
        # the old behaviour amplifies drawdowns instead of damping them.
        self.equity_based_sizing = (settings.PAPER_TRADING_EQUITY_BASED_SIZING
                                    if equity_based_sizing is None else equity_based_sizing)
        # Block NEW entries once equity is this far below its peak (percent);
        # 0.0 disables. Open positions keep their own stops regardless.
        self.max_drawdown_halt_pct = (settings.PAPER_TRADING_MAX_DRAWDOWN_HALT_PCT
                                      if max_drawdown_halt_pct is None else max_drawdown_halt_pct)
        self.peak_balance = self.starting_balance
        self._halted = False   # sticky until drawdown recovers past the hysteresis band
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

    def _save_unlocked(self) -> None:
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

    def _sizing_base(self) -> float:
        """Capital that risk, the correlation cap and the notional cap are
        all sized against. Current equity when equity-based sizing is on,
        else the original starting balance (the historical behaviour)."""
        if not self.equity_based_sizing:
            return self.starting_balance
        # Never size off a non-positive balance, and never scale UP past the
        # starting balance here -- growing the risk unit on winnings is a
        # separate decision this change is deliberately not making.
        return max(0.0, min(self.balance, self.starting_balance))

    def _drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return max(0.0, (self.peak_balance - self.balance) / self.peak_balance * 100)

    def _entries_halted(self) -> bool:
        """Sticky drawdown halt with hysteresis: trips at the threshold,
        clears only once drawdown recovers to half of it, so the system
        doesn't flap on and off while hovering at the boundary."""
        if self.max_drawdown_halt_pct <= 0:
            return False
        dd = self._drawdown_pct()
        if self._halted:
            if dd <= self.max_drawdown_halt_pct / 2:
                self._halted = False
                logger.info("PAPER: drawdown recovered to %.1f%% — entries resumed", dd)
        elif dd >= self.max_drawdown_halt_pct:
            self._halted = True
            logger.warning("PAPER: drawdown %.1f%% >= %.1f%% — NEW ENTRIES HALTED "
                           "(open positions keep their stops)", dd, self.max_drawdown_halt_pct)
        return self._halted

    def _maybe_open_unlocked(
        self, symbol: str, market: str, rec: dict, timeframe: str = "5m", opened_at: str | None = None,
        current_price: float | None = None,
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

        current_price: live market price at call time, if known — passed by
        main.py, omitted by backtests (which replay entry_zone/stop_loss as
        the historical fill and reconcile against price via the separate
        check_and_close_ohlc path). Used only to reject a setup that has
        already been invalidated by price before it could ever actually
        fill: entry_insight recommends off the SMC/FVG order-block zone
        without checking whether current price has already moved through
        its own stop or target, so a stale-but-still-"actionable" signal
        (an order block that price already broke through) would otherwise
        open and get closed on the very next check_and_close call a moment
        later — always as a loss, since the fill price was never actually
        reachable. Found live: NVDA re-opened and instantly stopped out on
        the same entry/SL/TP three times in ~25 minutes on 2026-08-19.

        Returns None if a position was opened, or if there was nothing to do
        (no signal, already open). Returns a short human-readable reason if
        entry_insight says "actionable" but this ledger rejected it anyway —
        surfaced to the dashboard so a rejection is never silently invisible.
        """
        key = (symbol, market, timeframe)
        if key in self.open_positions:
            return None
        # Drawdown circuit breaker — checked before anything else so a
        # halted account genuinely stops adding risk. Open positions are
        # untouched and keep their own stops; only NEW entries are blocked.
        if self._entries_halted():
            return f"Entri dihentikan: drawdown {self._drawdown_pct():.1f}% >= {self.max_drawdown_halt_pct:.1f}%"
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

        if current_price is not None:
            already_stopped = current_price <= stop_loss if direction == "long" else current_price >= stop_loss
            already_hit_target = current_price >= target if direction == "long" else current_price <= target
            if already_stopped or already_hit_target:
                reason = (
                    f"Setup sudah invalid — harga saat ini ({current_price:.6f}) sudah melewati "
                    f"{'stop-loss' if already_stopped else 'target'} sebelum posisi sempat dibuka"
                )
                logger.debug("%s %s: %s", symbol, market, reason)
                return reason

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < entry_price * MIN_STOP_DISTANCE_PCT:
            stop_pct = stop_distance / entry_price * 100
            reason = (
                f"Stop terlalu rapat untuk profitable setelah fee "
                f"({stop_pct:.3f}% < {MIN_STOP_DISTANCE_PCT * 100:.2f}% minimum)"
            )
            logger.debug("%s %s: %s", symbol, market, reason)
            return reason

        # Sizing base — see _sizing_base(). Historically this was always
        # STARTING balance, for a good measurement reason: sizing off current
        # balance compounds every trade, and over hundreds of sequential
        # trades that compounds geometrically into meaningless numbers, while
        # a fixed risk unit keeps the PnL curve roughly linear and
        # interpretable, which is what "what's the win rate" actually needs.
        #
        # That reasoning is about MEASUREMENT, and it does not survive
        # contact with real money: a fixed $200 risk unit on a $10,000
        # account is 6.3% of what's left after a 68.5% drawdown, so the risk
        # FRACTION grows exactly when the account can least afford it.
        #
        # _sizing_base() resolves the conflict asymmetrically when
        # equity-based sizing is enabled: it scales DOWN with losses (the
        # survival property) but is capped at starting_balance so winnings
        # never inflate the risk unit (the interpretability property).
        # Default stays False, so historical backtests are unaffected.
        #
        # Conviction scaling: a confidence=0 signal risks 1x the base unit;
        # confidence=100 risks up to PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER.
        # This is the "reward bolder calls" lever — bounded on purpose (see
        # settings.py) rather than a literal leverage multiplier.
        confidence = max(0.0, min(100.0, rec.get("confidence", 0.0)))
        max_mult = settings.PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER
        conviction_multiplier = 1.0 + (confidence / 100.0) * (max_mult - 1.0)

        risk_amount = self._sizing_base() * settings.PAPER_TRADING_RISK_PCT * conviction_multiplier

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
        portfolio_cap = self._sizing_base() * settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT
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
        max_notional = self._sizing_base() * MAX_NOTIONAL_MULTIPLE
        if quantity * entry_price > max_notional:
            quantity = max_notional / entry_price

        # Total-notional cap. The correlated-risk cap below governs RISK; this
        # governs CAPITAL COMMITTED, and they come apart badly — halving risk
        # per trade just admits twice as many positions, leaving total notional
        # unchanged (measured: 17.2x -> 16.5x across a 4x risk cut). Spot
        # cannot borrow, so without this the strategy is unexecutable there.
        # 0 disables, preserving the historical behaviour exactly.
        if settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE > 0:
            open_notional = sum(p.quantity * p.entry_price for p in self.open_positions.values())
            notional_cap = self._sizing_base() * settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE
            if open_notional + quantity * entry_price > notional_cap:
                return (f"Notional portfolio penuh (${open_notional:,.0f} terbuka + "
                        f"${quantity * entry_price:,.0f} baru > ${notional_cap:,.0f} batas)")


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
            atr_at_entry=float(rec.get("atr", 0.0) or 0.0),
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

    def _maybe_update_trailing_stop(self, pos: PaperPosition, favorable_price: float) -> None:
        """
        favorable_price: the best price reached since the last check —
        current price in live mode, or the candle's high (longs) / low
        (shorts) in OHLC backtest mode, representing the best price reached
        during that candle. No-ops entirely unless
        PAPER_TRADING_TRAILING_STOP_ENABLED and the position has a real
        ATR recorded at entry (0.0 for positions opened before this field
        existed, or if ATR genuinely couldn't be computed yet).

        Once profit reaches PAPER_TRADING_TRAILING_ACTIVATION_R multiples
        of the position's own initial risk, the stop starts trailing
        PAPER_TRADING_TRAILING_ATR_MULTIPLE x ATR behind the best price
        seen — and only ever tightens (max() for longs, min() for shorts),
        never loosens back out even if price pulls back before reversing
        further favorably.
        """
        if not settings.PAPER_TRADING_TRAILING_STOP_ENABLED or not pos.atr_at_entry:
            return
        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if initial_risk <= 0:
            return

        if pos.direction == "long":
            profit_r = (favorable_price - pos.entry_price) / initial_risk
            if profit_r < settings.PAPER_TRADING_TRAILING_ACTIVATION_R:
                return
            candidate = favorable_price - settings.PAPER_TRADING_TRAILING_ATR_MULTIPLE * pos.atr_at_entry
            pos.trailing_stop = candidate if pos.trailing_stop is None else max(pos.trailing_stop, candidate)
        else:
            profit_r = (pos.entry_price - favorable_price) / initial_risk
            if profit_r < settings.PAPER_TRADING_TRAILING_ACTIVATION_R:
                return
            candidate = favorable_price + settings.PAPER_TRADING_TRAILING_ATR_MULTIPLE * pos.atr_at_entry
            pos.trailing_stop = candidate if pos.trailing_stop is None else min(pos.trailing_stop, candidate)

    def _effective_stop(self, pos: PaperPosition) -> float:
        """Tighter of the original fixed stop and any active trailing stop
        — trailing only ever protects further, never loosens the original
        risk the position was sized against."""
        if pos.trailing_stop is None:
            return pos.stop_loss
        return max(pos.stop_loss, pos.trailing_stop) if pos.direction == "long" else min(pos.stop_loss, pos.trailing_stop)

    def _check_and_close_unlocked(self, current_prices: dict[str, float]) -> None:
        """Live mode: check open positions against a single latest price per symbol.

        Records the OBSERVED price that triggered the exit, not just the
        theoretical SL/TP level, so the gap between them can be measured.

        Why this matters more than it looks: `breakeven_under_slippage.py`
        established that this system's backtested edge dies somewhere around
        0.51% stop slippage, while the only real measurement we have puts
        live stop fills 0.25-0.60% past their trigger. That band straddles
        the break-even line, so whether the strategy makes or loses money is
        currently unknown -- and unmeasurable, because until now a stop exit
        was booked at exactly its trigger price and the overshoot was thrown
        away. Each stop exit now carries the numbers needed to settle it.

        IMPORTANT about what this does and does not measure. The gap
        captured here is the price move between the breach and our next
        poll. That is a LOWER BOUND on real execution slippage, not the
        whole of it -- real fills additionally pay order-book depth, market
        impact and exchange latency, none of which paper trading can see.
        Useful anyway, and in the conservative direction: if the lower bound
        alone approaches 0.5%, the question is already answered.

        Fill semantics differ by exit type, deliberately:
          - stop  -> booked at the observed price. A breached stop becomes a
                     MARKET order, so the real fill is at the prevailing
                     price (or worse), never back at the trigger.
          - target-> still booked at the target. That is a LIMIT order; it
                     fills at its price when touched and does not capture
                     any favourable overshoot past it.
        """
        for key, pos in list(self.open_positions.items()):
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            # Liquidation (derivative only) takes priority — it's the more
            # severe, exchange-forced outcome if it's ever actually reached.
            if pos.liquidation_price is not None:
                hit_liq = price <= pos.liquidation_price if pos.direction == "long" else price >= pos.liquidation_price
                if hit_liq:
                    self._close(key, pos, price, "loss", closed_at=None, liquidated=True,
                                exit_type="liquidation", trigger_price=pos.liquidation_price,
                                observed_price=price)
                    continue

            self._maybe_update_trailing_stop(pos, price)
            effective_stop = self._effective_stop(pos)

            if pos.direction == "long":
                hit_sl, hit_tp = price <= effective_stop, price >= pos.target
            else:  # short
                hit_sl, hit_tp = price >= effective_stop, price <= pos.target

            if hit_sl or hit_tp:
                trigger = effective_stop if hit_sl else pos.target
                # Stop = market order, so book the price we actually saw.
                # Target = limit order, so book the limit itself.
                exit_price = price if hit_sl else pos.target
                self._close(key, pos, exit_price, self._outcome(pos, trigger), closed_at=None,
                            exit_type="stop" if hit_sl else "target",
                            trigger_price=trigger, observed_price=price)

    def _check_and_close_ohlc_unlocked(self, symbol: str, low: float, high: float, closed_at: str, timeframe: str | None = None) -> None:
        """
        Backtest mode: check a position against a full candle's [low, high]
        range instead of a single price, since intrabar price action isn't
        known from OHLC alone. If both SL and TP fall inside the same
        candle's range, conservatively assume SL was hit first — avoids an
        optimistic look-ahead bias in the backtested win rate. Same
        approximation extends to trailing: the trail is updated using the
        candle's OWN favorable extreme (high for longs, low for shorts)
        before checking the opposite extreme against it, which could in
        principle overstate how far a trail tightened within one bar —
        an inherent limit of OHLC-only data, same category as the SL/TP
        simultaneity assumption above, not a new one.

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
                        self._close(key, pos, pos.liquidation_price, "loss", closed_at=closed_at, liquidated=True,
                                    exit_type="liquidation")
                        continue

                self._maybe_update_trailing_stop(pos, high if pos.direction == "long" else low)
                effective_stop = self._effective_stop(pos)

                if pos.direction == "long":
                    hit_sl, hit_tp = low <= effective_stop, high >= pos.target
                else:  # short
                    hit_sl, hit_tp = high >= effective_stop, low <= pos.target

                if hit_sl or hit_tp:
                    exit_price = effective_stop if hit_sl else pos.target
                    self._close(key, pos, exit_price, self._outcome(pos, exit_price), closed_at=closed_at,
                                exit_type="stop" if hit_sl else "target")

    def _outcome(self, pos: PaperPosition, exit_price: float) -> str:
        """"win"/"loss" by actual price movement, not just "which boundary
        was hit" — with a fixed stop these are always the same thing (a
        stop-loss is by definition on the bad side of entry), but a
        trailing stop can end up on the GOOD side of entry (it only ever
        activates after real profit), so an exit "via the stop path" isn't
        automatically a loss once trailing is active."""
        sign = 1 if pos.direction == "long" else -1
        return "win" if (exit_price - pos.entry_price) * sign > 0 else "loss"

    def _close(
        self, key: tuple[str, str, str], pos: PaperPosition, exit_price: float, outcome: str,
        closed_at: str | None, liquidated: bool = False, exit_type: str = "target",
        trigger_price: float | None = None, observed_price: float | None = None,
    ) -> None:
        # A stop fills past its trigger (market order); a target does not
        # (limit order). See the stop_slippage_pct note in __init__.
        slip = self.slippage_pct
        if exit_type in ("stop", "liquidation"):
            slip += self.stop_slippage_pct
        if slip:
            # Adverse fill on the way out too: a long sells for less, a
            # short buys back for more than the theoretical SL/TP/liq price.
            exit_price *= (1 - slip) if pos.direction == "long" else (1 + slip)

        # Measured overshoot past the trigger, signed so that POSITIVE always
        # means adverse (filled worse than the level), whatever the direction.
        # Only meaningful on the live path, which is the only caller that
        # knows an observed price; OHLC backtests leave it None rather than
        # inventing one from a bar extreme.
        observed_slippage_pct = None
        if trigger_price and observed_price and trigger_price > 0:
            adverse = (trigger_price - observed_price) if pos.direction == "long" else (observed_price - trigger_price)
            observed_slippage_pct = adverse / trigger_price * 100

        sign = 1 if pos.direction == "long" else -1
        gross_pnl_dollars = pos.quantity * (exit_price - pos.entry_price) * sign
        # Round-trip fee on both entry and exit notional — "win" / "loss"
        # still reflects whether the SL or TP was hit (signal correctness);
        # pnl_dollars reflects the realistic net economics after costs.
        fee_dollars = pos.quantity * (pos.entry_price + exit_price) * settings.PAPER_TRADING_FEE_PCT
        pnl_dollars = gross_pnl_dollars - fee_dollars
        pnl_pct = (pnl_dollars / (pos.quantity * pos.entry_price)) if pos.entry_price else 0.0

        self.balance += pnl_dollars
        self.peak_balance = max(self.peak_balance, self.balance)
        self.closed_trades.append({
            "symbol": pos.symbol, "market": pos.market, "direction": pos.direction,
            "timeframe": pos.timeframe,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "pnl_dollars": round(pnl_dollars, 4), "pnl_pct": round(pnl_pct, 4),
            "fee_dollars": round(fee_dollars, 4),
            "outcome": outcome, "liquidated": liquidated, "exit_type": exit_type,
            "trigger_price": trigger_price, "observed_price": observed_price,
            "observed_slippage_pct": None if observed_slippage_pct is None else round(observed_slippage_pct, 4),
            "opened_at": pos.opened_at,
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
            "PAPER CLOSE %s %s %s @ %.6f | %s | PnL=$%.2f (%.2f%%) | balance=$%.2f%s",
            pos.market, pos.direction.upper(), pos.symbol, exit_price,
            tag, pnl_dollars, pnl_pct * 100, self.balance,
            "" if observed_slippage_pct is None
            else f" | {exit_type} slip={observed_slippage_pct:+.3f}% (trigger={trigger_price:.6f})",
        )

    # ── Continual learning hook ────────────────────────────────────────────

    def _get_unlearned_trades_unlocked(self) -> list[dict]:
        """Closed trades since the last GNN learning update."""
        return self.closed_trades[self.last_learned_count:]

    def _mark_learned_unlocked(self) -> None:
        self.last_learned_count = len(self.closed_trades)

    # ── Reporting ──────────────────────────────────────────────────────────

    def _stats_unlocked(self) -> dict:
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

    # ── Thread-safety wrappers ─────────────────────────────────────────────
    # The real implementations are the _*_unlocked methods above/below. These
    # thin wrappers are the public API and hold self._lock for the whole
    # operation, so the fast SL/TP monitor thread and the 60s decision loop
    # cannot interleave inside a mutation. See the _lock comment in __init__.

    def save(self) -> None:
        with self._lock:
            return self._save_unlocked()

    def maybe_open(self, *args, **kwargs):
        with self._lock:
            return self._maybe_open_unlocked(*args, **kwargs)

    def check_and_close(self, current_prices: dict[str, float]) -> None:
        with self._lock:
            return self._check_and_close_unlocked(current_prices)

    def check_and_close_ohlc(self, symbol: str, low: float, high: float,
                             closed_at: str, timeframe: str | None = None) -> None:
        with self._lock:
            return self._check_and_close_ohlc_unlocked(symbol, low, high, closed_at, timeframe)

    def get_unlearned_trades(self) -> list[dict]:
        with self._lock:
            return self._get_unlearned_trades_unlocked()

    def mark_learned(self) -> None:
        with self._lock:
            return self._mark_learned_unlocked()

    def stats(self) -> dict:
        with self._lock:
            return self._stats_unlocked()
