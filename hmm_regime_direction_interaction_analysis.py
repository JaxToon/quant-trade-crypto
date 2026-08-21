"""
Interaction + tail-exceedance follow-up on the (now-fixed) regime x direction
study (2026-08-21) -- closes the research round started after the
market_structure freeze bug fix, per the explicit decision NOT to commit a
direction-aware veto and NOT to add new cuts (SMA filters, drift-correction
thresholds, etc.) without first testing whether an actual INTERACTION exists.

The pooled cell comparison (hmm_regime_direction_binance_experiment.py,
rerun post-fix) showed a directional MAIN EFFECT (bullish win rate >
bearish win rate, consistent across discovery/validation/holdout) but did
NOT show turbulent+bearish as tail-risk-worse than turbulent+bullish (P99
adverse excursion was ~equal, 13.73% vs 13.80% pooled). That's a main
effect, not evidence for the interaction ("does turbulence specifically
make bearish more dangerous than bullish, beyond the direction effect that
already exists in calm conditions?") that would actually justify a
direction-aware circuit breaker.

This script tests that interaction directly and rigorously:

  direction_gap(regime) = metric(bullish, regime) - metric(bearish, regime)
  interaction = direction_gap(turbulent) - direction_gap(calm)

  A positive interaction (for win rate / mean return) means turbulence
  WIDENS the bullish-outperforms-bearish gap beyond what calm conditions
  already show -- i.e. turbulence specifically penalizes bearish beyond the
  baseline directional effect. Near-zero or inconsistent-sign interaction
  across eras means what's being observed is just the pre-existing
  directional asymmetry, not a turbulence x direction effect -- exactly the
  distinction the decision tree below hinges on.

Also adds tail EXCEEDANCE probabilities (P(adverse excursion > 5%/10%/15%))
per cell, with the same interaction framing -- preferred over comparing
single percentiles (P99 can look similar between two cells while the
distributions differ meaningfully around a specific loss threshold that
actually matters for a circuit breaker's real question: "how often does
this exposure produce a loss path I can't tolerate").

Confidence intervals use a BLOCK bootstrap over calendar weeks (matching the
HMM's own weekly refit cadence), not i.i.d. per-observation resampling --
the 4h/12h/24h forward windows overlap heavily, so individual observations
are not independent; resampling whole weeks preserves that structure.

Decision rule (agreed before running, not after seeing results):
  interaction strong + consistent across eras + tail exceedance materially
    worse for turbulent+bearish -> worth testing a direction-aware veto in
    paper/backtest (still not committing anything here)
  win-rate interaction present but tail interaction absent -> keep the
    symmetric veto
  interaction inconsistent across eras -> keep the symmetric veto, close
    the research question

Purely observational. Does not modify risk/circuit_breaker.py,
risk/hmm_regime.py, model_checkpoints/, config/settings.py, or any
live/Path 1 code.

Usage: python hmm_regime_direction_interaction_analysis.py
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from hmm_regime_direction_binance_experiment import (
    END_YEAR_MONTH,
    ERAS,
    START_YEAR_MONTH,
    SYMBOL,
    build_features,
    fetch_binance_monthly_klines,
    fetch_fear_greed_all_history,
)
from hmm_regime_direction_experiment import (
    HORIZONS,
    causal_walk_forward_regime,
    direction_series,
    forward_outcomes,
)

CACHE_PATH = "backtrack_logs/_regime_direction_dataset_cache.pkl"  # post-fix dataset only
N_BOOTSTRAP = 2000
EXCEEDANCE_THRESHOLDS_PCT = [5.0, 10.0, 15.0]
BLOCK_HOURS = 24 * 7  # weekly blocks, matches the HMM's own refit cadence


# ── Dataset prep (cached -- this is pure additional analysis on data already
# validated by hmm_regime_direction_binance_experiment.py, not a new pull) ──

def _load_or_build_dataset():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print(f"No cache found -- fetching {SYMBOL} 1h klines, {START_YEAR_MONTH} -> {END_YEAR_MONTH}...")
    candles = fetch_binance_monthly_klines(SYMBOL, START_YEAR_MONTH, END_YEAR_MONTH)
    print(f"  {len(candles)} candles")

    fg_history = fetch_fear_greed_all_history()
    X, timestamps = build_features(candles, fg_history)
    directions = direction_series(candles)

    print("Running causal walk-forward regime decode (383 weekly refit cycles -- this is the slow part)...")
    causal_labels = causal_walk_forward_regime(X)

    dataset = {"candles": candles, "causal_labels": causal_labels, "directions": directions}
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  cached to {CACHE_PATH}")
    return dataset


# ── Build per-hour observation table ───────────────────────────────────────

REGIME_NAMES = {0: "calm", 1: "turbulent"}
CELLS = [("calm", "bullish"), ("calm", "bearish"), ("turbulent", "bullish"), ("turbulent", "bearish")]


def build_observations(candles, causal_labels, directions, horizon: int) -> dict:
    """Returns arrays aligned by observation: regime, direction, block_id,
    signed_return, adverse_magnitude (positive %), era. One row per
    causal-labeled hour with enough forward data for this horizon."""
    n = len(candles)
    t0 = candles[1].timestamp  # X[i] <-> candles[i+1], block numbering anchored here

    regimes, dirs, block_ids, returns, adverse, eras_col = [], [], [], [], [], []
    for i in range(len(causal_labels)):
        t_idx = i + 1
        if t_idx >= n:
            continue
        regime = causal_labels[i]
        if regime < 0:
            continue
        regime_name = REGIME_NAMES.get(int(regime))
        if regime_name is None:
            continue
        direction = directions[t_idx]
        if direction not in ("bullish", "bearish"):
            continue  # neutral excluded -- effectively unobserved anyway (see prior run)

        outcomes = forward_outcomes(candles, t_idx)
        if outcomes is None or horizon not in outcomes:
            continue
        raw_return, raw_mfe, raw_mae = outcomes[horizon]
        if direction == "bullish":
            signed_return, signed_mae = raw_return, raw_mae
        else:
            signed_return, signed_mae = -raw_return, -raw_mfe

        ts = candles[t_idx].timestamp
        block_id = int((ts - t0).total_seconds() // 3600 // BLOCK_HOURS)
        era_name = None
        for name, start, end in ERAS:
            if start <= ts.date() <= end:
                era_name = name
                break

        regimes.append(regime_name)
        dirs.append(direction)
        block_ids.append(block_id)
        returns.append(signed_return)
        adverse.append(max(0.0, -signed_mae))
        eras_col.append(era_name)

    return {
        "regime": np.array(regimes),
        "direction": np.array(dirs),
        "block_id": np.array(block_ids),
        "return": np.array(returns),
        "adverse": np.array(adverse) * 100,  # percent
        "era": np.array(eras_col, dtype=object),
    }


# ── Block bootstrap ─────────────────────────────────────────────────────────
#
# Resampling whole weeks by re-masking the full observation array on every
# replicate does not scale to the full dataset (verified: a small 6-month
# smoke test with 200 replicates already took ~5s per report; 2000
# replicates over the full ~64k-row/380-block pooled range would extrapolate
# to hours). Instead, precompute per-(block, cell) SUMS once (n, win count,
# return sum, exceedance counts), then each replicate just fancy-indexes
# those small per-block arrays by the drawn block ids and sums -- O(num
# blocks) of vectorized numpy work per replicate instead of O(num
# observations) of masking.

class _BlockAggregates:
    """Per-(regime, direction) arrays indexed by block_id, holding count/
    win/return-sum/exceedance-count contributed by each block. A bootstrap
    replicate draws block ids with replacement and fancy-indexes + sums."""

    def __init__(self, obs: dict):
        self.unique_blocks = np.unique(obs["block_id"])
        max_block = int(obs["block_id"].max())
        self.n = {}
        self.win = {}
        self.ret_sum = {}
        self.exc = {}  # (regime, direction, threshold) -> array
        for regime, direction in CELLS:
            mask = (obs["regime"] == regime) & (obs["direction"] == direction)
            b, r, a = obs["block_id"][mask], obs["return"][mask], obs["adverse"][mask]
            n_a = np.zeros(max_block + 1)
            win_a = np.zeros(max_block + 1)
            ret_a = np.zeros(max_block + 1)
            np.add.at(n_a, b, 1)
            np.add.at(win_a, b, (r > 0).astype(float))
            np.add.at(ret_a, b, r)
            self.n[(regime, direction)] = n_a
            self.win[(regime, direction)] = win_a
            self.ret_sum[(regime, direction)] = ret_a
            for t in EXCEEDANCE_THRESHOLDS_PCT:
                exc_a = np.zeros(max_block + 1)
                np.add.at(exc_a, b, (a > t).astype(float))
                self.exc[(regime, direction, t)] = exc_a

    def metric(self, drawn_blocks: np.ndarray, regime: str, direction: str, metric: str, threshold_pct: float = None) -> float | None:
        n_total = self.n[(regime, direction)][drawn_blocks].sum()
        if n_total == 0:
            return None
        if metric == "win_rate":
            return self.win[(regime, direction)][drawn_blocks].sum() / n_total * 100
        if metric == "mean_return":
            return self.ret_sum[(regime, direction)][drawn_blocks].sum() / n_total * 100
        if metric == "exceedance":
            return self.exc[(regime, direction, threshold_pct)][drawn_blocks].sum() / n_total * 100
        raise ValueError(metric)


def _direction_gap(agg: _BlockAggregates, drawn_blocks: np.ndarray, regime: str, metric: str, threshold_pct: float = None) -> float | None:
    m_bull = agg.metric(drawn_blocks, regime, "bullish", metric, threshold_pct)
    m_bear = agg.metric(drawn_blocks, regime, "bearish", metric, threshold_pct)
    if m_bull is None or m_bear is None:
        return None
    return (m_bear - m_bull) if metric == "exceedance" else (m_bull - m_bear)


def _interaction_value(agg: _BlockAggregates, drawn_blocks: np.ndarray, metric: str, threshold_pct: float = None) -> float | None:
    gap_calm = _direction_gap(agg, drawn_blocks, "calm", metric, threshold_pct)
    gap_turb = _direction_gap(agg, drawn_blocks, "turbulent", metric, threshold_pct)
    if gap_calm is None or gap_turb is None:
        return None
    return gap_turb - gap_calm


def block_bootstrap_interaction(agg: _BlockAggregates, metric: str, threshold_pct: float = None, n_boot: int = N_BOOTSTRAP):
    """Sign convention (consistent across ALL metrics -- positive
    interaction ALWAYS means "turbulence specifically penalizes bearish
    beyond whatever directional gap calm conditions already show"):
      win_rate / mean_return (higher = better): gap = bullish - bearish.
      exceedance (higher = worse): gap = bearish - bullish (flipped), so a
        positive gap still means "bearish looks worse", same reading.
    """
    all_blocks = agg.unique_blocks
    point = _interaction_value(agg, all_blocks, metric, threshold_pct)

    rng = np.random.default_rng(42)
    replicates = []
    for _ in range(n_boot):
        drawn = rng.choice(all_blocks, size=len(all_blocks), replace=True)
        val = _interaction_value(agg, drawn, metric, threshold_pct)
        if val is not None:
            replicates.append(val)

    if len(replicates) < n_boot * 0.5:
        return point, None, None, len(replicates)
    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return point, lo, hi, len(replicates)


def block_bootstrap_cell(agg: _BlockAggregates, regime: str, direction: str, metric: str, threshold_pct: float = None, n_boot: int = N_BOOTSTRAP):
    all_blocks = agg.unique_blocks
    point = agg.metric(all_blocks, regime, direction, metric, threshold_pct)
    if point is None:
        return None, None, None

    rng = np.random.default_rng(43)
    replicates = []
    for _ in range(n_boot):
        drawn = rng.choice(all_blocks, size=len(all_blocks), replace=True)
        val = agg.metric(drawn, regime, direction, metric, threshold_pct)
        if val is not None:
            replicates.append(val)
    if len(replicates) < n_boot * 0.5:
        return point, None, None
    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return point, lo, hi


# ── Reporting ────────────────────────────────────────────────────────────

def report_for_scope(obs_all: dict, scope_name: str, horizon: int) -> None:
    print(f"\n{'=' * 110}\n{scope_name} -- {horizon}h horizon\n{'=' * 110}")
    agg = _BlockAggregates(obs_all)
    all_blocks = agg.unique_blocks

    print(f"\n-- Cell metrics (n, win rate %, mean return %) --")
    header = f"{'Regime':<11}{'Direction':<10}{'n':>7}{'WinRate% [95%CI]':>28}{'MeanRet% [95%CI]':>28}"
    print(header)
    print("-" * len(header))
    for regime, direction in CELLS:
        mask = (obs_all["regime"] == regime) & (obs_all["direction"] == direction)
        n = mask.sum()
        if n == 0:
            print(f"{regime:<11}{direction:<10}{0:>7}  (no observations)")
            continue
        wr, wr_lo, wr_hi = block_bootstrap_cell(agg, regime, direction, "win_rate")
        mr, mr_lo, mr_hi = block_bootstrap_cell(agg, regime, direction, "mean_return")
        wr_ci = f"[{wr_lo:.1f}, {wr_hi:.1f}]" if wr_lo is not None else "[n/a]"
        mr_ci = f"[{mr_lo:.2f}, {mr_hi:.2f}]" if mr_lo is not None else "[n/a]"
        print(f"{regime:<11}{direction:<10}{n:>7}{wr:>10.1f} {wr_ci:>17}{mr:>10.2f} {mr_ci:>17}")

    print(f"\n-- Tail exceedance probability P(adverse excursion > threshold), % of observations --")
    header2 = f"{'Regime':<11}{'Direction':<10}" + "".join(f"{'>' + str(t) + '%':>12}" for t in EXCEEDANCE_THRESHOLDS_PCT)
    print(header2)
    print("-" * len(header2))
    for regime, direction in CELLS:
        if agg.n[(regime, direction)][all_blocks].sum() == 0:
            continue
        vals = [agg.metric(all_blocks, regime, direction, "exceedance", t) for t in EXCEEDANCE_THRESHOLDS_PCT]
        print(f"{regime:<11}{direction:<10}" + "".join(f"{v:>12.2f}" for v in vals))

    print(f"\n-- Interaction: does turbulence widen the bearish-penalty beyond calm's baseline gap? --")
    print("   positive interaction, ALL rows = turbulence makes bearish look WORSE relative to bullish")
    print("   than calm conditions alone already show (this is the actual test for a direction-aware veto)")
    header3 = f"{'Metric':<28}{'Interaction':>14}{'95% CI':>22}{'gap: calm':>14}{'gap: turbulent':>16}"
    print(header3)
    print("-" * len(header3))
    for metric, label in [("win_rate", "win rate gap (pp)"), ("mean_return", "mean return gap (pp)")]:
        point, lo, hi, n_valid = block_bootstrap_interaction(agg, metric)
        gap_calm = _direction_gap(agg, all_blocks, "calm", metric)
        gap_turb = _direction_gap(agg, all_blocks, "turbulent", metric)
        if gap_calm is None or gap_turb is None or point is None:
            print(f"{label:<28}  (insufficient data)")
            continue
        ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[n/a]"
        print(f"{label:<28}{point:>14.2f}{ci:>22}{gap_calm:>14.2f}{gap_turb:>16.2f}")

    for t in EXCEEDANCE_THRESHOLDS_PCT:
        point, lo, hi, n_valid = block_bootstrap_interaction(agg, "exceedance", t)
        gap_calm = _direction_gap(agg, all_blocks, "calm", "exceedance", t)
        gap_turb = _direction_gap(agg, all_blocks, "turbulent", "exceedance", t)
        label = f"P(adv>{t}%) gap (pp, bear-bull)"
        if gap_calm is None or gap_turb is None or point is None:
            print(f"{label:<28}  (insufficient data)")
            continue
        ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[n/a]"
        print(f"{label:<28}{point:>14.2f}{ci:>22}{gap_calm:>14.2f}{gap_turb:>16.2f}")


def main() -> None:
    dataset = _load_or_build_dataset()
    candles, causal_labels, directions = dataset["candles"], dataset["causal_labels"], dataset["directions"]

    for horizon in HORIZONS:
        obs = build_observations(candles, causal_labels, directions, horizon)
        report_for_scope(obs, f"POOLED, FULL RANGE {candles[0].timestamp.date()} -> {candles[-1].timestamp.date()}", horizon)
        for era_name, era_start, era_end in ERAS:
            era_mask = obs["era"] == era_name
            if era_mask.sum() == 0:
                continue
            era_obs = {k: v[era_mask] for k, v in obs.items()}
            report_for_scope(era_obs, f"ERA: {era_name} ({era_start} -> {era_end})", horizon)

    print(f"\n{'=' * 110}")
    print("Block bootstrap: whole calendar weeks resampled with replacement (not i.i.d. bars),")
    print(f"{N_BOOTSTRAP} replicates, matching the HMM's own weekly refit cadence.")
    print("Purely observational. Nothing here is deployed to risk/circuit_breaker.py or any live path.")


if __name__ == "__main__":
    main()
