"""Performance metrics and the pre-registered statistical checks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAYS = 365.0


def summarize(r: pd.Series, exposure: pd.Series | None = None, turnover: pd.Series | None = None,
              costs: pd.Series | None = None, trades: pd.DataFrame | None = None) -> dict:
    r = r.astype(float)
    n = len(r)
    if n < 2:
        return {}
    eq = (1 + r).cumprod()
    years = n / DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * math.sqrt(DAYS)
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(DAYS) if r.std(ddof=1) > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / down.std(ddof=1) * math.sqrt(DAYS) if len(down) > 1 and down.std(ddof=1) > 0 else 0.0
    dd = 1 - eq / eq.cummax()
    maxdd = float(dd.max())
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out = {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()), "days": n,
        "total_return": float(eq.iloc[-1] - 1), "cagr": float(cagr), "vol": float(vol),
        "sharpe": float(sharpe), "sortino": float(sortino), "maxdd": maxdd,
        "calmar": float(cagr / maxdd) if maxdd > 0 else float("inf"),
        "pct_months_up": float((monthly > 0).mean()),
    }
    if exposure is not None:
        out["avg_exposure"] = float(exposure.loc[r.index].mean())
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.loc[r.index].sum() / years)
    if costs is not None:
        out["costs_per_year"] = float(costs.loc[r.index].sum() / years)
    if trades is not None and not trades.empty:
        t = trades
        wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
        gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
        out.update({
            "trades": int(len(t)),
            "trades_per_year": float(len(t) / years),
            "win_rate": float(len(wins) / len(t)),
            "avg_win_pct": float(wins["ret_on_equity"].mean()) if len(wins) else 0.0,
            "avg_loss_pct": float(losses["ret_on_equity"].mean()) if len(losses) else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "expectancy_pct": float(t["ret_on_equity"].mean()),
            "avg_hold_days": float(t["days"].mean()),
        })
    return out


def yearly(r: pd.Series) -> dict[int, float]:
    return {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}


def stationary_bootstrap_sharpe(r: pd.Series, n_boot: int = 5000, mean_block: float = 20.0,
                                seed: int = 7) -> tuple[float, float]:
    """95% CI of the annualised Sharpe (Politis & Romano stationary bootstrap)."""
    x = r.to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    pos = np.arange(n)
    out = np.empty(n_boot)
    for b in range(n_boot):
        # A new block starts where `jumps` is True; inside a block the index
        # walks forward (wrapping), exactly as in the sequential definition.
        jumps = rng.random(n) < p
        jumps[0] = True
        last = np.maximum.accumulate(np.where(jumps, pos, 0))
        starts = rng.integers(n, size=n)
        idx = (starts[last] + (pos - last)) % n
        s = x[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * math.sqrt(DAYS) if sd > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def shift_signal(sig: pd.DataFrame, eligible: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Random-timing null: rotate each coin's signal inside its eligible window."""
    out = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    for c in sig.columns:
        mask = eligible[c].to_numpy()
        idx = np.nonzero(mask)[0]
        if len(idx) < 130:
            continue
        vals = sig[c].to_numpy()[idx]
        off = int(rng.integers(60, len(idx) - 60))
        out.iloc[idx, out.columns.get_loc(c)] = np.roll(vals, off)
    return out


# ---- Deflated Sharpe (Bailey & Lopez de Prado 2014), per-period Sharpe

EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def deflated_sharpe(r: pd.Series, n_trials: int, var_trial_sr: float) -> dict:
    x = r.to_numpy(dtype=float)
    t = len(x)
    sd = x.std(ddof=0)
    sr = x.mean() / sd
    z = (x - x.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    sr0 = math.sqrt(var_trial_sr) * ((1 - EULER) * _norm_ppf(1 - 1 / n_trials)
                                     + EULER * _norm_ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4) * sr ** 2))
    return {"sr_daily": sr, "sr0_daily": sr0, "dsr": _norm_cdf((sr - sr0) * math.sqrt(t - 1) / denom),
            "skew": skew, "kurt": kurt, "n_trials": n_trials}
