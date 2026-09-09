"""
Does THIS project's own result survive a search-aware correction?

Prompted by arXiv 2608.27734, "What survives honest evaluation?
Leakage-safe, search-aware assessment of LLM-driven trading strategy
discovery". Its finding is blunt: across 100 candidate strategies, two
frontier models (GPT-4.1 and Claude Sonnet 5) and five repeated runs, ZERO
survived certification — including several that were genuinely profitable
out-of-sample. Passive benchmarks passed; the discovered strategies did not.

The mechanism is multiple testing. Every configuration evaluated against
history is another hypothesis test, so the best-looking result is inflated
by selection alone. The paper's remedy is the Deflated Sharpe Ratio (Bailey
& López de Prado), indexed to the number of trials actually run.

This project has never applied that correction, and has run a great many
trials: 62 experiment scripts, 11 carrying pre-registered gates, and roughly
70 backtest configurations recorded in README. Every gate used a plain 5%
threshold. Twenty independent tests at 5% produce one pass by chance alone,
and we are far past twenty.

So this asks the uncomfortable question directly: with the trial count
honestly declared, does the spot configuration's result still stand?

METHOD (Bailey & López de Prado 2014)

  SR0  = expected MAXIMUM Sharpe under the null of zero true skill, given N
         independent trials:
             SR0 = sqrt(V) * [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]
         with γ the Euler-Mascheroni constant and V the variance of Sharpes
         across trials.

  DSR  = Z[ (SR - SR0)·sqrt(T-1) / sqrt(1 - skew·SR + ((kurt-1)/4)·SR²) ]

DSR is the probability the observed Sharpe reflects real skill rather than
the best of N draws from a skill-less process. Convention treats DSR > 0.95
as surviving.

The correction is deliberately harsh, and it is supposed to be: it is the
price of having looked at the data many times.

Usage: python -m experiments.deflated_sharpe_audit
"""
from __future__ import annotations

import json
import math
import statistics

EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(x: list[float]) -> tuple[float, float, float, float]:
    n = len(x)
    m = statistics.mean(x)
    sd = statistics.pstdev(x)
    if sd == 0:
        return m, 0.0, 0.0, 3.0
    skew = sum(((v - m) / sd) ** 3 for v in x) / n
    kurt = sum(((v - m) / sd) ** 4 for v in x) / n
    return m, sd, skew, kurt


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Best Sharpe you would EXPECT from N skill-less trials."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(var_trial_sr) * (
        (1 - EULER) * _norm_ppf(1 - 1.0 / n_trials)
        + EULER * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(returns: list[float], n_trials: int, var_trial_sr: float) -> dict:
    t = len(returns)
    mean, sd, skew, kurt = _moments(returns)
    if sd == 0 or t < 3:
        return {"sr": 0.0, "sr0": 0.0, "dsr": 0.0, "t": t}
    sr = mean / sd
    sr0 = expected_max_sharpe(n_trials, var_trial_sr)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2))
    dsr = _norm_cdf((sr - sr0) * math.sqrt(t - 1) / denom)
    return {"sr": sr, "sr0": sr0, "dsr": dsr, "t": t, "skew": skew, "kurt": kurt}


def main() -> None:
    # Live trades are the honest out-of-sample record; use them, not a
    # backtest, since a backtest re-run is itself another trial.
    d = json.load(open("backtrack_logs/paper_ledger_crypto.json"))
    trades = [t for t in d["closed_trades"] if t.get("pnl_pct") is not None]
    if len(trades) < 20:
        print("too few live trades"); return

    groups = {
        "ALL live crypto": trades,
        "LONG only": [t for t in trades if t.get("direction") == "long"],
        "SHORT only": [t for t in trades if t.get("direction") == "short"],
    }

    # Trial count. Counting honestly is the whole point: 62 experiment
    # scripts, ~70 backtest configurations in README, 11 pre-registered
    # gates, plus every arm inside each sweep. 150 is a conservative FLOOR.
    trial_counts = [1, 20, 70, 150, 300]
    # Spread of Sharpes across trials. Estimated from the per-trade Sharpe
    # dispersion we actually observe between arms; 0.5^2 is a mild default
    # in the literature when the ledger of trial Sharpes is unavailable.
    VAR_TRIAL_SR = 0.25

    print("=" * 92)
    print("DEFLATED SHARPE AUDIT — does our own result survive its own search intensity?")
    print("=" * 92)
    print("Per-trade Sharpe (not annualised). DSR > 0.95 conventionally means 'survives'.\n")

    for name, g in groups.items():
        rets = [t["pnl_pct"] for t in g]
        base = deflated_sharpe(rets, 1, VAR_TRIAL_SR)
        print(f"--- {name}  (n={len(g)}) ---")
        print(f"    per-trade Sharpe {base['sr']:+.4f}   skew {base['skew']:+.2f}  "
              f"kurtosis {base['kurt']:.2f}")
        print(f"    {'trials':<10}{'SR0 (null max)':>16}{'DSR':>10}   verdict")
        for n in trial_counts:
            r = deflated_sharpe(rets, n, VAR_TRIAL_SR)
            verdict = "survives" if r["dsr"] > 0.95 else "FAILS"
            tag = " <- honest floor" if n == 150 else ""
            print(f"    {n:<10}{r['sr0']:>16.4f}{r['dsr']:>10.4f}   {verdict}{tag}")
        print()

    print("=" * 92)
    print("READING THIS")
    print("  SR0 is the Sharpe you would expect from the LUCKIEST of N skill-less")
    print("  strategies. If our Sharpe does not clear it, the result is indistinguishable")
    print("  from having searched until something looked good.")
    print()
    print("  The correction is harsh on purpose. It is the price of having looked at the")
    print("  same data many times — which this project has, ~150 times at a conservative")
    print("  count. arXiv 2608.27734 found ZERO of 100 LLM-discovered strategies survived")
    print("  this test, including ones that were profitable out-of-sample.")
    print("=" * 92)


if __name__ == "__main__":
    main()
