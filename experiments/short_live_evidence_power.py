"""
Is the accumulated live evidence enough to conclude SHORT loses money?

The user's question, and a fair one: the system has been running a while.
So rather than assume the sample is too small, measure whether it is.

The naive test (a binomial on win rate) is not good enough here for two
reasons. First, win rate is not the quantity that matters — expectancy is;
a 35% win rate with large winners could still be profitable. Second, trades
are NOT independent draws: they overlap in time, share symbols, and respond
to the same market moves, so a plain t-test understates the true standard
error.

So this reports:
  1. Expectancy per SHORT trade with a bootstrap confidence interval
  2. A MOVING BLOCK bootstrap, which resamples contiguous runs of trades
     and therefore preserves the autocorrelation a plain bootstrap destroys
  3. The same for LONG, as a control — if the method says LONG is
     significantly profitable and SHORT significantly unprofitable, both
     conclusions carry the same weight
  4. How many more SHORT trades would be needed to reach significance, if
     it is not there yet

Decision rule stated up front: SHORT can be called unprofitable only if the
95% block-bootstrap interval for its mean PnL lies entirely below zero. An
interval straddling zero means "not yet answered", not "probably bad" —
and turning SHORT off on that basis would be the curve-fitting this project
keeps rejecting.

Usage: python -m experiments.short_live_evidence_power
"""
from __future__ import annotations

import json
import random
import statistics

random.seed(7)
BOOT = 20_000
BLOCK = 5          # contiguous trades per resampled block


def _load(asset: str) -> list[dict]:
    try:
        d = json.load(open(f"backtrack_logs/paper_ledger_{asset}.json"))
    except FileNotFoundError:
        return []
    return [t for t in d.get("closed_trades", []) if t.get("closed_at")]


# A moving block bootstrap needs enough distinct windows to resample from.
# With n=7 and block=5 there are only 3 possible windows and nblocks=1, so
# every resample is 5 of the 7 values — the interval then describes a
# different quantity than the reported mean and can exclude it outright.
# That is exactly what produced a confident "stock short PROFITS" verdict on
# n=7 (mean $433, CI [$202, $289]). Refuse rather than report nonsense.
MIN_N_FOR_BOOTSTRAP = 5 * BLOCK


def _block_bootstrap(vals: list[float], block: int = BLOCK, n: int = BOOT) -> tuple[float, float]:
    """95% CI for the mean, resampling contiguous blocks so serial
    correlation between nearby trades survives the resample.

    Returns NaN when the sample is too small for the block size to give a
    meaningful resample — see MIN_N_FOR_BOOTSTRAP."""
    if len(vals) < max(block, MIN_N_FOR_BOOTSTRAP):
        return (float("nan"), float("nan"))
    nblocks = max(1, len(vals) // block)
    means = []
    for _ in range(n):
        sample = []
        for _ in range(nblocks):
            s = random.randrange(0, len(vals) - block + 1)
            sample.extend(vals[s:s + block])
        means.append(statistics.mean(sample))
    means.sort()
    return (means[int(n * 0.025)], means[int(n * 0.975)])


def _report(label: str, trades: list[dict]) -> tuple[float, float, float]:
    if len(trades) < BLOCK:
        print(f"{label:<22} n={len(trades)} — too few to analyse")
        return (0.0, 0.0, 0.0)
    pnl = [t["pnl_dollars"] for t in trades]
    mean = statistics.mean(pnl)
    wr = sum(1 for p in pnl if p > 0) / len(pnl) * 100
    lo, hi = _block_bootstrap(pnl)
    if lo != lo:   # NaN — sample too small for a meaningful block bootstrap
        print(f"{label:<22}n={len(trades):<5}WR={wr:>5.1f}%  mean=${mean:>+8.2f}  "
              f"{'':<26}  TOO FEW (need n>={MIN_N_FOR_BOOTSTRAP})")
        return (mean, float("nan"), float("nan"))
    verdict = ("LOSES (CI entirely below 0)" if hi < 0 else
               "PROFITS (CI entirely above 0)" if lo > 0 else
               "UNRESOLVED (CI straddles 0)")
    print(f"{label:<22}n={len(trades):<5}WR={wr:>5.1f}%  mean=${mean:>+8.2f}  "
          f"95% CI [${lo:>+8.2f}, ${hi:>+8.2f}]  {verdict}")
    return (mean, lo, hi)


def main() -> None:
    print("=" * 104)
    print("IS THE LIVE EVIDENCE ENOUGH TO JUDGE SHORT? — moving block bootstrap "
          f"(block={BLOCK}, {BOOT:,} resamples)")
    print("=" * 104)

    combined: dict[str, list[dict]] = {"long": [], "short": []}
    for asset in ("crypto", "stock"):
        tr = _load(asset)
        if not tr:
            continue
        print(f"\n--- {asset.upper()} (all live history, n={len(tr)}) ---")
        for dirn in ("long", "short"):
            g = [t for t in tr if t.get("direction") == dirn]
            combined[dirn].extend(g)
            _report(f"{dirn}", g)
        for dirn in ("long", "short"):
            for mkt in ("spot", "derivative"):
                g = [t for t in tr if t.get("direction") == dirn and t.get("market") == mkt]
                if len(g) >= BLOCK:
                    _report(f"  {dirn}/{mkt}", g)

    print(f"\n--- CRYPTO + STOCK COMBINED ---")
    stats = {}
    for dirn in ("long", "short"):
        stats[dirn] = _report(dirn, combined[dirn])

    short = [t["pnl_dollars"] for t in combined["short"]]
    print("\n" + "=" * 104)
    print("ANSWER")
    print("=" * 104)
    if not short:
        print("No live SHORT trades recorded.")
        return
    mean, lo, hi = stats["short"]
    if hi < 0:
        print(f"  SHORT's 95% interval lies entirely below zero (${lo:+.2f} to ${hi:+.2f}).")
        print("  The live evidence IS sufficient: SHORT is losing money, not unlucky.")
    elif lo > 0:
        print("  SHORT's interval lies entirely above zero — it is profitable live.")
    else:
        print(f"  SHORT's 95% interval is [${lo:+.2f}, ${hi:+.2f}] — it STRADDLES ZERO.")
        print(f"  With n={len(short)} the live evidence cannot yet distinguish 'SHORT loses'")
        print("  from 'SHORT is break-even and we got unlucky'. Not the same as evidence")
        print("  that SHORT is fine — it is an absence of evidence either way.")
        sd = statistics.pstdev(short)
        if mean < 0 and sd > 0:
            needed = (1.96 * sd / abs(mean)) ** 2
            print(f"\n  At the observed mean (${mean:+.2f}) and spread (SD ${sd:,.2f}),")
            print(f"  roughly n={needed:.0f} SHORT trades would be needed for significance")
            print(f"  — about {max(0, needed - len(short)):.0f} more than we have.")
            print("  (Block-bootstrapped, so this already accounts for trades not being")
            print("   independent draws. A plain t-test would flatter the number.)")
    print("=" * 104)


if __name__ == "__main__":
    main()
