"""
Regression tests for the FVGZone.created_at wall-clock bug (found live,
2026-08-21, while building the signal-age/decay experiment).

Bug: FVGZone was constructed with created_at=datetime.utcnow() -- the
real system clock -- instead of the candle actually being processed. In
live streaming this is harmless (a live candle's timestamp already IS
approximately "now"), but during a historical backtest replay it means
every zone's created_at is "whenever the backtest happened to run",
completely disconnected from the simulated timeline being replayed.
cleanup_old_fvgs()'s age cutoff had the same bug.

Fixed: created_at now uses curr.timestamp (the middle candle of the
3-candle FVG pattern, already in scope) -- matches the pattern
smart_money.py's OrderBlock already used correctly. cleanup_old_fvgs()
now accepts an optional current_time, defaulting to datetime.utcnow()
so live-mode behavior (unchanged, since nothing currently calls it) is
preserved.

Also proves the audit claim made before this fix was applied, not just
asserts it: confirmed via full-repo search that cleanup_old_fvgs() is
never called anywhere, so zone age currently has zero effect on which
zones remain usable -- TestZoneAgeDoesNotGateUsability makes that a
regression-tested fact instead of a one-time grep result that could go
stale silently.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle


def _candles_forming_bullish_gap(start: datetime, step: timedelta) -> list[Candle]:
    """3 candles with a clean bullish FVG: prev.high < next.low."""
    return [
        Candle(timestamp=start, open=100.0, high=101.0, low=99.5, close=100.5, volume=1000.0),
        Candle(timestamp=start + step, open=100.5, high=103.0, low=100.0, close=102.5, volume=1500.0),
        Candle(timestamp=start + step * 2, open=102.5, high=104.0, low=102.0, close=103.5, volume=1200.0),
    ]


class TestCreatedAtUsesCandleTimestamp:
    def test_historical_replay_timestamp_not_wallclock(self):
        old_time = datetime(2020, 1, 1, 12, 0, 0)  # deliberately far from any plausible "now"
        candles = _candles_forming_bullish_gap(old_time, timedelta(minutes=15))

        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        fvg = analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)

        assert fvg is not None
        # Must be the middle candle's own timestamp -- not today's real date.
        assert fvg.created_at == candles[1].timestamp
        assert fvg.created_at.year == 2020
        assert abs((fvg.created_at - datetime.utcnow()).days) > 365  # nowhere near real "now"

    def test_age_reflects_simulated_elapsed_time_not_real_elapsed_time(self):
        """The whole point of the fix: computing age from created_at against
        a LATER simulated candle time must give the simulated gap, not
        something close to zero (which is what wall-clock created_at would
        produce if the whole replay runs in a few real seconds)."""
        old_time = datetime(2020, 1, 1, 0, 0, 0)
        candles = _candles_forming_bullish_gap(old_time, timedelta(minutes=15))
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        fvg = analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)

        simulated_now = old_time + timedelta(hours=30)  # 30 simulated hours later
        age_hours = (simulated_now - fvg.created_at).total_seconds() / 3600.0
        assert 29.5 < age_hours < 30.5  # roughly 30h, not ~0 (which wall-clock created_at would give)


class TestCleanupOldFvgsCurrentTime:
    def test_expires_relative_to_passed_current_time_not_wallclock(self):
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        old_time = datetime(2020, 1, 1, 0, 0, 0)
        candles = _candles_forming_bullish_gap(old_time, timedelta(minutes=15))
        analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)
        assert len(analyzer.fvgs) == 1

        # 200 simulated hours later, past the 168h default -- must expire,
        # even though real wall-clock "now" is nowhere near this zone's age.
        far_future = old_time + timedelta(hours=200)
        analyzer.cleanup_old_fvgs(max_age_hours=168, current_time=far_future)
        assert len(analyzer.fvgs) == 0

    def test_does_not_expire_when_simulated_age_under_threshold(self):
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        old_time = datetime(2020, 1, 1, 0, 0, 0)
        candles = _candles_forming_bullish_gap(old_time, timedelta(minutes=15))
        analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)

        near_future = old_time + timedelta(hours=10)  # well under 168h
        analyzer.cleanup_old_fvgs(max_age_hours=168, current_time=near_future)
        assert len(analyzer.fvgs) == 1  # not expired

    def test_defaults_to_wallclock_when_current_time_omitted(self):
        """Live-mode backward compatibility: a freshly-created zone (real
        created_at, since nothing overrides it here) must survive a
        default call exactly as before this fix -- no live behavior change."""
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        now = datetime.utcnow()
        candles = _candles_forming_bullish_gap(now - timedelta(minutes=30), timedelta(minutes=15))
        analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)

        analyzer.cleanup_old_fvgs(max_age_hours=168)  # no current_time -- defaults to real utcnow()
        assert len(analyzer.fvgs) == 1  # freshly created, must not have been expired


class TestZoneAgeDoesNotGateUsability:
    """Regression-tests the audit claim made before deploying this fix:
    cleanup_old_fvgs() is never called anywhere in the codebase (confirmed
    via full-repo search), so an ancient, un-mitigated zone must still be
    just as usable as a fresh one -- age currently has zero effect on
    trade eligibility. If this test ever starts failing, something wired
    age-based expiry into the live decision path without updating this
    file, and the impact assessment behind this fix needs revisiting.
    """

    def test_ancient_unmitigated_zone_is_still_returned_as_nearest(self):
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        ancient = datetime(2000, 1, 1, 0, 0, 0)
        candles = _candles_forming_bullish_gap(ancient, timedelta(minutes=15))
        fvg = analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)
        assert fvg is not None
        assert (datetime.utcnow() - fvg.created_at).days > 9000  # decades old

        nearest = analyzer.get_nearest_fvg(current_price=102.0, fvg_type="bullish")
        assert nearest is not None
        assert nearest.id == fvg.id

    def test_ancient_unmitigated_zone_still_produces_a_trading_signal(self):
        analyzer = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
        ancient = datetime(2000, 1, 1, 0, 0, 0)
        candles = _candles_forming_bullish_gap(ancient, timedelta(minutes=15))
        analyzer.detect_fvg(candles=candles, index=1, current_price=103.5, volume=1500.0)
        analyzer.update_all(current_price=102.0)

        signal = analyzer.get_trading_signal(current_price=102.0, trend="bullish")
        assert signal["direction"] == "long"
        assert signal["entry_zone"] is not None
