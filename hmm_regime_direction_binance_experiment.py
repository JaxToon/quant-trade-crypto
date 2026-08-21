"""
Regime x direction decomposition, rebuilt on Binance's full historical klines
(2026-08-21) -- extends hmm_regime_direction_experiment.py's 700-day yfinance
result, which found turbulent+bearish consistently worst across all horizons
but rested on a thin, bull-skewed bearish sample (n=198 vs n=7,558 bullish at
24h, over a window that was ~90% bullish SMC structure). That result was
committed as a milestone, not discarded -- see this repo's git history and
the TL;DR entry for hmm_regime_direction_experiment.py.

This is a deliberate FROM-SCRATCH rerun on one consistent data source, not an
append of Binance data onto the yfinance run -- yfinance and Binance can
differ on exchange/price, candle-boundary/timezone semantics, missing
candles, and historical revisions, and mixing sources mid-study would
contaminate exactly the comparison this experiment needs to be clean.

Source: data.binance.vision's public monthly kline archives (BTCUSDT, 1h),
2019-01 through the most recent complete month -- reachable from this
sandbox even though the live api.binance.com REST endpoint is not (verified
directly, not assumed). Chosen deliberately for REGIME DIVERSITY, not candle
count: 2019 covers late bear/recovery, 2020 the COVID crash + recovery, 2021
a prolonged bull, 2022 the Terra/FTX bear, 2023 recovery, 2024-2026 another
bull run -- exactly the mix of prolonged bull / sharp crash / prolonged bear
/ recovery / volatile-sideways the 700-day window couldn't provide.

Found and handled during the build, not assumed: Binance's kline CSV
timestamp unit is NOT consistent across this span -- files switch from
millisecond to microsecond epoch timestamps at some point in the archive
(confirmed directly: 2019-01's first column is 13 digits, 2026-07's is 16).
Detected per-row by magnitude rather than trusting one file's format for the
whole fetch.

Regime and outcome methodology is IMPORTED, not re-typed, from
hmm_regime_direction_experiment.py -- same HMM config (n_states,
covariance_type, n_iter, random_state), same TRAIN_DAYS/REFIT_STEP_HOURS/
DECODE_WINDOW_HOURS causal walk-forward discipline, same SMC
market_structure.trend direction signal, same 4h/12h/24h horizons, same
MAE/MFE boundary convention (classify at close[t], evaluate from t+1
onward). Only the candle source and the reporting (era-sliced stability +
deeper tail metrics) are new -- this is explicitly a replication, not a
re-tuned experiment.

New in this script, per the follow-up discussion after the 700-day result:
  - Era-sliced stability check: discovery (2019-2023) / validation (2024) /
    recent holdout (2025-2026), NOT to retrain anything, but to ask whether
    "turbulent+bearish is worst" survives as the market regime itself
    changes, rather than trusting one pooled multi-year number.
  - Deeper tail metrics beyond P10 MAE: P90/P95/P99 (worst-1%) adverse
    excursion magnitude, and probability(adverse excursion beyond fixed
    thresholds) -- because the veto is a tail-risk control, not a return-
    maximization one, so the relevant question is "how bad can it get",
    not just "what's the average".

Purely observational. Does not modify risk/hmm_regime.py,
risk/circuit_breaker.py, model_checkpoints/, config/settings.py, or any
live/Path 1 code. No trades are opened.

Usage: python hmm_regime_direction_binance_experiment.py [--smoke-test]
  --smoke-test fetches only the most recent 6 months, for fast correctness
  checking before committing to the full multi-year run.
"""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import date, datetime

import numpy as np
import requests

from hmm_regime_direction_experiment import (
    DECODE_WINDOW_HOURS,
    DIRECTIONS,
    HORIZONS,
    REFIT_STEP_HOURS,
    REGIME_LABELS,
    ROUND_TRIP_FEE_PCT,
    TRAIN_DAYS,
    causal_walk_forward_regime,
    direction_series,
    forward_outcomes,
    order_remap,
    retrospective_regime,
    sign_for_direction,
)
from strategy.smart_money import Candle

SYMBOL = "BTCUSDT"
START_YEAR_MONTH = (2019, 1)
# Most recent fully-complete month as of this run (today is well past its end).
END_YEAR_MONTH = (2026, 7)
SMOKE_TEST_MONTHS = 6

ERAS = [
    ("discovery_2019_2023", date(2019, 1, 1), date(2023, 12, 31)),
    ("validation_2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("recent_holdout_2025_2026", date(2025, 1, 1), date(2027, 1, 1)),
]

TAIL_PROB_THRESHOLDS_PCT = [3.0, 5.0, 10.0]  # fixed before running


# ── Data: Binance monthly kline archive ─────────────────────────────────────

def _month_range(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _parse_kline_row(row: str) -> Candle | None:
    parts = row.strip().split(",")
    if len(parts) < 6 or not parts[0].lstrip("-").isdigit():
        return None
    ts_raw = int(parts[0])
    # Binance switched kline CSV timestamps from ms to us epoch at some point
    # in this archive's history -- detect per-row by magnitude (ms values in
    # this date range are ~13 digits/~1e12-1e13; us values are ~16 digits/
    # ~1e15-1e16; 1e14 sits cleanly between the two).
    ts_seconds = ts_raw / 1_000_000 if ts_raw > 10 ** 14 else ts_raw / 1000
    return Candle(
        timestamp=datetime.utcfromtimestamp(ts_seconds),
        open=float(parts[1]), high=float(parts[2]), low=float(parts[3]),
        close=float(parts[4]), volume=float(parts[5]),
    )


def fetch_binance_monthly_klines(symbol: str, start: tuple[int, int], end: tuple[int, int]) -> list[Candle]:
    candles: list[Candle] = []
    months = list(_month_range(start, end))
    for idx, (y, m) in enumerate(months):
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{y:04d}-{m:02d}.zip"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    month_candles = [c for line in f if (c := _parse_kline_row(line.decode())) is not None]
            candles.extend(month_candles)
            if (idx + 1) % 12 == 0 or idx == len(months) - 1:
                print(f"  fetched {y:04d}-{m:02d} ({idx + 1}/{len(months)} months, {len(candles)} candles so far)")
        except Exception as exc:
            print(f"  {y:04d}-{m:02d}: fetch/parse failed ({exc}) -- skipping month")
    candles.sort(key=lambda c: c.timestamp)
    return candles


# ── Fear & Greed: full history, not a trailing-N-days lookback ────────────

def fetch_fear_greed_all_history() -> dict:
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0}, timeout=20)
        resp.raise_for_status()
        history = {}
        for entry in resp.json().get("data", []):
            day = date.fromtimestamp(int(entry["timestamp"]))
            history[day] = float(entry["value"]) / 100.0
        return history
    except Exception as exc:
        print(f"Fear & Greed full-history fetch failed: {exc} -- proceeding with neutral (0.5) fill.")
        return {}


# ── Feature engineering (same formula as hmm_regime_direction_experiment.py's
# build_features -- reimplemented, not imported, only because that version
# fetches F&G internally via a trailing-days-from-now window unsuited to a
# fixed historical span; the vol/abs_change math is identical) ────────────

def build_features(candles: list, fg_history: dict) -> tuple[np.ndarray, list]:
    closes = [c.close for c in candles]
    log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])
    vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
    abs_change = np.array([
        abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
        for i in range(len(log_returns))
    ])
    timestamps = [c.timestamp for c in candles[1:]]  # X[i] <-> candles[i+1]
    fg_feature = np.array([fg_history.get(ts.date(), 0.5) for ts in timestamps])
    X = np.column_stack([vol, abs_change, fg_feature])
    return X, timestamps


# ── Cell aggregation, era-aware ────────────────────────────────────────────

def build_cells(regime_labels: np.ndarray, directions: list[str], candles: list, era_filter=None) -> dict:
    cells = {h: {(r, d): [] for r in REGIME_LABELS.values() for d in DIRECTIONS} for h in HORIZONS}
    n = len(candles)
    for i in range(len(regime_labels)):
        t_idx = i + 1
        if t_idx >= n:
            continue
        if era_filter is not None and not era_filter(candles[t_idx].timestamp.date()):
            continue
        regime = regime_labels[i]
        if regime < 0:
            continue
        regime_name = REGIME_LABELS.get(int(regime))
        if regime_name is None:
            continue
        direction = directions[t_idx]

        outcomes = forward_outcomes(candles, t_idx)
        if outcomes is None:
            continue
        for h in HORIZONS:
            raw_return, raw_mfe, raw_mae = outcomes[h]
            signed_return, signed_mfe, signed_mae = sign_for_direction(direction, raw_return, raw_mfe, raw_mae)
            cells[h][(regime_name, direction)].append((signed_return, signed_mfe, signed_mae))
    return cells


def print_table(cells: dict, title: str) -> None:
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
    for h in HORIZONS:
        print(f"\n-- Forward horizon: {h}h --")
        header = (f"{'Regime':<11}{'Direction':<10}{'n':>6}{'WinRate%':>10}{'Mean%':>8}{'Median%':>9}"
                   f"{'P90Adv%':>9}{'P95Adv%':>9}{'P99Adv%':>9}"
                   + "".join(f"{'P(adv>' + str(t) + '%)':>12}" for t in TAIL_PROB_THRESHOLDS_PCT)
                   + f"{'FeeAdjMean%':>13}")
        print(header)
        print("-" * len(header))
        for regime_name in REGIME_LABELS.values():
            for direction in DIRECTIONS:
                rows = cells[h][(regime_name, direction)]
                n = len(rows)
                if n == 0:
                    print(f"{regime_name:<11}{direction:<10}{0:>6}  (no observations)")
                    continue
                returns = np.array([r[0] for r in rows])
                maes = np.array([r[2] for r in rows])
                adverse = np.maximum(0.0, -maes) * 100  # positive magnitude, % terms

                win_rate = (returns > 0).mean() * 100
                mean_ret = returns.mean() * 100
                median_ret = np.median(returns) * 100
                p90 = np.percentile(adverse, 90)
                p95 = np.percentile(adverse, 95)
                p99 = np.percentile(adverse, 99)
                probs = [(adverse > t).mean() * 100 for t in TAIL_PROB_THRESHOLDS_PCT]
                fee_adj_mean = (returns.mean() - ROUND_TRIP_FEE_PCT) * 100
                flag = "" if direction != "neutral" else "  (no implied trade -- informational)"

                prob_cols = "".join(f"{p:>12.1f}" for p in probs)
                print(f"{regime_name:<11}{direction:<10}{n:>6}{win_rate:>10.1f}{mean_ret:>8.2f}{median_ret:>9.2f}"
                      f"{p90:>9.2f}{p95:>9.2f}{p99:>9.2f}{prob_cols}{fee_adj_mean:>13.2f}{flag}")


def main() -> None:
    smoke_test = "--smoke-test" in sys.argv
    start = START_YEAR_MONTH
    if smoke_test:
        y, m = END_YEAR_MONTH
        for _ in range(SMOKE_TEST_MONTHS - 1):
            m -= 1
            if m < 1:
                m, y = 12, y - 1
        start = (y, m)
        print(f"SMOKE TEST: fetching only {start} -> {END_YEAR_MONTH}\n")

    print(f"Fetching {SYMBOL} 1h klines from Binance archive, {start} -> {END_YEAR_MONTH}...")
    candles = fetch_binance_monthly_klines(SYMBOL, start, END_YEAR_MONTH)
    if len(candles) < 3000:
        print(f"Only {len(candles)} candles fetched -- not enough for a meaningful run. Aborting.")
        return
    print(f"  total: {len(candles)} candles, {candles[0].timestamp} -> {candles[-1].timestamp}\n")

    print("Fetching full Fear & Greed history (limit=0, covers 2018-02 onward)...")
    fg_history = fetch_fear_greed_all_history()
    print(f"  {len(fg_history)} days\n")

    print("Building HMM features (vol, abs_change, fear&greed)...")
    X, timestamps = build_features(candles, fg_history)
    print(f"  {len(X)} feature rows\n")

    print("Replaying SMC market_structure.trend (direction, causal by construction)...")
    directions = direction_series(candles)

    print("Fitting retrospective (descriptive, full-history) regime decode...")
    retro_labels = retrospective_regime(X)
    print(f"  retrospective turbulent fraction: {(retro_labels == 1).mean() * 100:.1f}%\n")

    print(f"Running causal walk-forward (weekly refit on trailing {TRAIN_DAYS}d, decision-grade) regime decode...")
    causal_labels = causal_walk_forward_regime(X)
    labeled = causal_labels[causal_labels >= 0]
    print(f"  causal turbulent fraction: {(labeled == 1).mean() * 100:.1f}%")
    agree = (causal_labels[causal_labels >= 0] == retro_labels[causal_labels >= 0])
    print(f"  causal vs retrospective agreement (where both defined): {agree.mean() * 100:.1f}%\n")

    pooled_causal = build_cells(causal_labels, directions, candles)
    print_table(pooled_causal, f"POOLED, FULL RANGE {candles[0].timestamp.date()} -> {candles[-1].timestamp.date()} "
                                 "(causal walk-forward, decision-grade)")

    print(f"\n{'#' * 118}\nERA-SLICED STABILITY CHECK (causal walk-forward labels only)\n{'#' * 118}")
    for era_name, era_start, era_end in ERAS:
        era_filter = lambda d, s=era_start, e=era_end: s <= d <= e
        era_cells = build_cells(causal_labels, directions, candles, era_filter=era_filter)
        print_table(era_cells, f"ERA: {era_name} ({era_start} -> {era_end})")

    print(f"\n{'=' * 118}")
    print("This is a market-conditional observational study only. Nothing here is")
    print("deployed to risk/circuit_breaker.py, risk/hmm_regime.py, or any live")
    print("decision path regardless of which cell or era looks best. No trades were")
    print("opened/closed -- these are hypothetical WITH-direction excursions from")
    print("close[t], not simulated strategy P&L.")


if __name__ == "__main__":
    main()
