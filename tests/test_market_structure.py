"""
Regression tests for the SmartMoneyAnalyzer.market_structure freeze bug
(found live, 2026-08-21, while extending the regime x direction study to
Binance's full 2019-2026 history -- the study showed zero bearish
observations across the entire 7.5-year run, which turned out to be this
bug, not a market fact).

Bug: _update_market_structure() only updated last_high inside the
bullish-flip branch and last_low inside the bearish-flip branch, so once
trend first flipped, the OTHER reference froze at its value from that
moment and could never update again short of price revisiting that exact
original level. For an asset that's appreciated ~20x since 2019 (BTC),
that made a bearish flip structurally almost impossible after the first
year of any long-running replay/analyzer.

Fixed via Option A (see README Contributing section for the full design
discussion): last_high/last_low are now the most recent CONFIRMED swing
pivot (via _find_confirmed_swing_pivots(), a causal 2-candle-each-side
primitive shared with liquidity-zone detection), refreshed on every call
regardless of current trend. Trend then derives from BOS (close breaks the
relevant level WITH the current trend -- continuation) vs ChoCH (breaks
AGAINST it -- reversal), matching standard SMC semantics rather than a
simpler always-updating rolling window.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from strategy.smart_money import SmartMoneyAnalyzer, Candle


def _zigzag_candles(turning_points: list[float], bars_per_leg: int,
                     start: datetime = datetime(2020, 1, 1), step: timedelta = timedelta(hours=1)) -> list[Candle]:
    """Piecewise-linear zigzag through `turning_points`, `bars_per_leg`
    candles per leg. Each turning point is a genuine local extreme
    (confirmable by _find_confirmed_swing_pivots() once enough subsequent
    bars close), as long as bars_per_leg >= 3. Flat candles (open=high=low=
    close) -- only the path's shape matters for this logic."""
    candles: list[Candle] = []
    t = start
    for leg in range(len(turning_points) - 1):
        p0, p1 = turning_points[leg], turning_points[leg + 1]
        for b in range(bars_per_leg):
            price = p0 + (p1 - p0) * (b / bars_per_leg)
            candles.append(Candle(timestamp=t, open=price, high=price, low=price, close=price, volume=100.0))
            t += step
    candles.append(Candle(timestamp=t, open=turning_points[-1], high=turning_points[-1],
                           low=turning_points[-1], close=turning_points[-1], volume=100.0))
    return candles


def _replay(analyzer: SmartMoneyAnalyzer, candles: list[Candle]) -> None:
    for c in candles:
        analyzer.update(c)


class TestCrashCanFlipTrendBearish:
    """The bug's headline symptom, made permanent: a real crash after a
    sustained uptrend must be able to flip trend bearish, even with a large
    lookback that still keeps the ORIGINAL pre-uptrend low technically
    visible (isolates the actual fix -- last_high/last_low always derived
    from the NEWEST confirmed pivot -- from the unrelated effect of old
    candles simply aging out of a small deque)."""

    def test_synthetic_btc_shaped_crash_flips_bearish(self):
        # chop near 3700 -> staircase uptrend to 69000 WITH pullbacks (real
        # bull runs aren't one smooth line -- each pullback confirms a
        # fresh, higher last_low reference, exactly what should happen live
        # too) -> crash to 15500. The crash should break below the LATEST
        # (45000) reference well before reaching bottom, not need to wait
        # for 15500 itself to confirm as a pivot. Mirrors BTC's real
        # 2019-2022 trajectory that originally surfaced this bug.
        points = [3700, 3650, 3700,      # initial chop, confirms first low ~3650
                   20000, 18000,          # uptrend leg, pullback confirms low @18000
                   30000, 27000,          # uptrend leg, pullback confirms low @27000
                   50000, 45000,          # uptrend leg, pullback confirms low @45000 (the LATEST one)
                   69000,                 # final push to peak
                   15500, 16000]          # crash + small bounce
        candles = _zigzag_candles(points, bars_per_leg=8)

        analyzer = SmartMoneyAnalyzer(lookback=1000)  # large: old 3650-ish low stays visible too
        _replay(analyzer, candles)

        assert analyzer.market_structure is not None
        assert analyzer.market_structure.trend == "bearish", (
            f"trend={analyzer.market_structure.trend}, last_low={analyzer.market_structure.last_low} "
            "-- crash failed to flip trend bearish, the exact bug this fix addresses"
        )
        # last_low must reflect a recent (post-uptrend) reference, not the
        # ~3650 low frozen at the very start under the old bug.
        assert analyzer.market_structure.last_low > 10000


class TestBOSvsChoCH:
    """The user-specified core scenario: a genuine higher low forming, a
    modest pullback that must NOT flip trend, then a real break below that
    SAME confirmed low that must flip it. Proves the fix uses the latest
    structural reference, not just "some last_low that happens to move"."""

    def test_modest_pullback_does_not_flip_bullish_trend(self):
        # Establish bullish structure: low -> high -> higher low -> higher high.
        points = [100, 90, 100, 95, 100, 110]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=200)
        _replay(analyzer, candles)
        assert analyzer.market_structure.trend == "bullish"
        confirmed_low_before_pullback = analyzer.market_structure.last_low

        # Modest pullback: a plain decline that stays clearly above the
        # confirmed higher low (95) and doesn't reverse -- so it can't
        # itself confirm as a NEW pivot (confirmation needs 2 candles
        # after a local extreme), isolating "does a break happen" from
        # "does a new reference legitimately form".
        pullback = _zigzag_candles([110, 105], bars_per_leg=6,
                                    start=candles[-1].timestamp + timedelta(hours=1))
        _replay(analyzer, pullback)

        assert analyzer.market_structure.trend == "bullish", "modest pullback incorrectly flipped trend"
        assert analyzer.market_structure.last_low == confirmed_low_before_pullback

    def test_break_below_latest_confirmed_low_flips_to_bearish(self):
        points = [100, 90, 100, 95, 100, 110]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=200)
        _replay(analyzer, candles)
        assert analyzer.market_structure.trend == "bullish"

        # Real break: closes below the confirmed higher low (95).
        breakdown = _zigzag_candles([110, 80], bars_per_leg=6,
                                     start=candles[-1].timestamp + timedelta(hours=1))
        _replay(analyzer, breakdown)

        assert analyzer.market_structure.trend == "bearish"

    def test_modest_rally_does_not_flip_bearish_trend(self):
        # Mirror: establish bearish structure (high -> low -> lower high -> lower low).
        points = [100, 110, 100, 105, 100, 90]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=200)
        _replay(analyzer, candles)
        assert analyzer.market_structure.trend == "bearish"
        confirmed_high_before_rally = analyzer.market_structure.last_high

        # Modest rally: a plain rise that stays clearly below the confirmed
        # lower high (105) and doesn't reverse -- so it can't itself
        # confirm as a new pivot.
        rally = _zigzag_candles([90, 95], bars_per_leg=6,
                                 start=candles[-1].timestamp + timedelta(hours=1))
        _replay(analyzer, rally)

        assert analyzer.market_structure.trend == "bearish", "modest rally incorrectly flipped trend"
        assert analyzer.market_structure.last_high == confirmed_high_before_rally

    def test_break_above_latest_confirmed_high_flips_to_bullish(self):
        points = [100, 110, 100, 105, 100, 90]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=200)
        _replay(analyzer, candles)
        assert analyzer.market_structure.trend == "bearish"

        breakout = _zigzag_candles([90, 120], bars_per_leg=6,
                                    start=candles[-1].timestamp + timedelta(hours=1))
        _replay(analyzer, breakout)

        assert analyzer.market_structure.trend == "bullish"


class TestNoPermanentLock:
    def test_choppy_structure_flips_repeatedly_not_locked(self):
        # Alternating swings well beyond each prior confirmed extreme in
        # both directions -- must be able to flip more than once.
        points = [100, 130, 70, 140, 60, 150, 50]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=500)

        trends_seen = []
        for c in candles:
            analyzer.update(c)
            if analyzer.market_structure and analyzer.market_structure.trend not in (trends_seen[-1:] or [None]):
                trends_seen.append(analyzer.market_structure.trend)

        directional = [t for t in trends_seen if t != "ranging"]
        assert len(directional) >= 3, f"expected multiple flips in a choppy sequence, got {trends_seen}"


class TestReferencesStayFresh:
    def test_last_low_advances_with_new_confirmed_higher_lows_during_sustained_uptrend(self):
        # Three consecutive higher lows during one long uptrend -- last_low
        # must track the NEWEST one, not freeze at the first.
        points = [100, 90, 105, 95, 115, 100, 130]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=500)

        seen_lows = []
        for c in candles:
            analyzer.update(c)
            if analyzer.market_structure and analyzer.market_structure.last_low is not None:
                if not seen_lows or analyzer.market_structure.last_low != seen_lows[-1]:
                    seen_lows.append(analyzer.market_structure.last_low)

        assert len(seen_lows) >= 2, f"last_low never advanced past its first value: {seen_lows}"
        assert seen_lows == sorted(seen_lows), f"last_low should only ever advance upward here: {seen_lows}"


class TestInitialState:
    def test_no_directional_label_before_any_confirmed_pivot(self):
        analyzer = SmartMoneyAnalyzer(lookback=100)
        # Enough candles to initialize (>=10) but perfectly flat, so no
        # strict-inequality pivot can ever confirm.
        candles = _zigzag_candles([100, 100], bars_per_leg=20)
        _replay(analyzer, candles)

        assert analyzer.market_structure is not None
        assert analyzer.market_structure.trend == "ranging"
        assert analyzer.market_structure.last_high is None
        assert analyzer.market_structure.last_low is None


class TestRestartInvariance:
    def test_continuous_run_matches_fresh_restart_on_recent_window(self):
        """The bug's other symptom: behavior silently depended on how long
        the analyzer had been running. Fixed structure should depend only
        on the currently-visible (lookback-bounded) window, not on
        accumulated history -- so a long continuous run and a fresh
        analyzer warmed up only on the tail of the same candle sequence
        must converge to the same last_high/last_low/trend."""
        lookback = 60
        points = [100, 90, 100, 95, 100, 110, 105, 108, 106, 112]
        full_history = _zigzag_candles(points, bars_per_leg=6)

        continuous = SmartMoneyAnalyzer(lookback=lookback)
        _replay(continuous, full_history)

        # "Restart": fresh analyzer, replayed only from the point where the
        # continuous analyzer's own deque (maxlen=lookback) would start.
        restarted = SmartMoneyAnalyzer(lookback=lookback)
        _replay(restarted, full_history[-lookback:])

        assert continuous.market_structure.last_high == restarted.market_structure.last_high
        assert continuous.market_structure.last_low == restarted.market_structure.last_low
        assert continuous.market_structure.trend == restarted.market_structure.trend


class TestLiquidityZonesUnchangedBySharedPrimitive:
    """The refactor moved _detect_liquidity_zones()'s pivot logic into a
    shared method -- this proves that extraction didn't change its
    behavior: swing highs/lows still become buy_side/sell_side zones."""

    def test_swing_points_still_produce_liquidity_zones(self):
        points = [100, 90, 100, 95, 100, 110]
        candles = _zigzag_candles(points, bars_per_leg=6)
        analyzer = SmartMoneyAnalyzer(lookback=200)
        _replay(analyzer, candles)

        assert any(z.type == "buy_side" for z in analyzer.liquidity_zones)
        assert any(z.type == "sell_side" for z in analyzer.liquidity_zones)
