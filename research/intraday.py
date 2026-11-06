"""Intraday regression-trend study, run exactly as research/PREREGISTRATION_INTRADAY.md says.

Usage: python -m research.intraday      (after python -m research.intraday_data)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from research import metrics
from research.data import SYMBOLS
from research.round2 import funding_daily

SRC = os.path.join("data", "klines_perp")
OUT = os.path.join("research", "results")
BARS_PER_DAY = {"15m": 96, "1h": 24}
DEV = ("2020-01-01", "2025-01-01")
HOLD = ("2025-01-01", "2026-10-10")
VOL_TARGET = 0.25
COSTS = {"base": 0.0010, "stress": 0.0015}


def load(tf: str) -> pd.DataFrame:
    return pd.DataFrame({s: pd.read_parquet(os.path.join(SRC, f"{s}_{tf}.parquet"))["close"]
                         for s in SYMBOLS}).sort_index()


def positions(close: pd.DataFrame, n: int, r_star: float) -> pd.DataFrame:
    """+1 / -1 / 0 per bar. Enter on |r| > R* in the trend's direction, flat when r crosses 0."""
    lp = np.log(close)
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    r = lp.rolling(n, min_periods=n).corr(t)
    long_ = pd.DataFrame(np.where(r > r_star, 1.0, np.where(r <= 0, 0.0, np.nan)),
                         index=close.index, columns=close.columns).ffill().fillna(0.0)
    short = pd.DataFrame(np.where(r < -r_star, 1.0, np.where(r >= 0, 0.0, np.nan)),
                         index=close.index, columns=close.columns).ffill().fillna(0.0)
    return (long_ - short).where(close.notna(), 0.0)


def backtest(close: pd.DataFrame, pos: pd.DataFrame, fund_bar: pd.DataFrame, bpd: int, cost: float) -> pd.Series:
    """Bar returns of the portfolio. Size fixed at entry: 1/8 x min(1, 25%/vol30d)."""
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(30 * bpd, min_periods=10 * bpd).std() * np.sqrt(365 * bpd)
    scale = (VOL_TARGET / vol).clip(upper=1.0)
    entry = pos.ne(pos.shift(1)) & pos.ne(0)
    scale_at_entry = scale.where(entry).ffill().where(pos.ne(0), 0.0).fillna(0.0)
    w = pos * scale_at_entry / len(close.columns)
    nxt = ret.shift(-1).fillna(0.0)
    gross = (w * nxt).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    funding = (w * fund_bar.shift(-1).fillna(0.0)).sum(axis=1)   # long pays positive funding
    out = gross - cost * turnover - funding
    return out.shift(1).fillna(0.0)   # book each bar's P&L on the bar it accrues


def daily(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def summary(r_day: pd.Series, a: str, b: str) -> dict:
    r = r_day[(r_day.index >= pd.Timestamp(a, tz="UTC")) & (r_day.index < pd.Timestamp(b, tz="UTC"))]
    s = metrics.summarize(r)
    wk = (1 + r).groupby(r.index.to_period("W")).prod() - 1
    return {"cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["maxdd"], "total_return": s["total_return"],
            "sr_daily": float(r.mean() / r.std()) if r.std() > 0 else 0.0,
            "week_median": float(wk.median()), "week_best": float(wk.max()), "week_worst": float(wk.min()),
            "weeks_over_40pct": int((wk > 0.40).sum()), "weeks": int(len(wk))}


def main() -> None:
    fund = pd.DataFrame({s: funding_daily(s) for s in SYMBOLS})
    results = {}
    for tf in ("1h", "15m"):
        close = load(tf)
        bpd = BARS_PER_DAY[tf]
        fund_bar = (fund.reindex(close.index.floor("D")).set_axis(close.index) / bpd).fillna(0.0)
        for n in (24, 48, 96):
            for r_star in (0.6, 0.8):
                pos = positions(close, n, r_star)
                name = f"RT {tf} N{n} R{r_star}"
                results[name] = {}
                for ck, c in COSTS.items():
                    rd = daily(backtest(close, pos, fund_bar, bpd, c))
                    results[name][ck] = {"dev": summary(rd, *DEV), "holdout": summary(rd, *HOLD)}
                trades = int((pos.diff().abs() > 0).sum().sum())
                d, h = results[name]["base"]["dev"], results[name]["stress"]["holdout"]
                print(f"{name:<18} trades {trades:>6} | DEV@0.10% Sharpe {d['sharpe']:5.2f} CAGR {d['cagr']:+7.1%} "
                      f"MaxDD {d['maxdd']:5.1%} | HOLD@0.15% Sharpe {h['sharpe']:5.2f} ret {h['total_return']:+7.1%} "
                      f"MaxDD {h['maxdd']:5.1%} | best week {d['week_best']:+.1%}, weeks>40%: {d['weeks_over_40pct']}",
                      flush=True)

    names = list(results)
    best = max(names, key=lambda k: results[k]["base"]["dev"]["sharpe"])
    var_sr = float(np.var([results[k]["base"]["dev"]["sr_daily"] for k in names], ddof=1))
    # Re-run the chosen variant's DEV daily returns for the deflated Sharpe.
    tf = best.split()[1]
    n, r_star = int(best.split()[2][1:]), float(best.split()[3][1:])
    close = load(tf)
    bpd = BARS_PER_DAY[tf]
    fund_bar = (fund.reindex(close.index.floor("D")).set_axis(close.index) / bpd).fillna(0.0)
    rd = daily(backtest(close, positions(close, n, r_star), fund_bar, bpd, COSTS["base"]))
    rdev = rd[(rd.index >= pd.Timestamp(DEV[0], tz="UTC")) & (rd.index < pd.Timestamp(DEV[1], tz="UTC"))]
    dsr = metrics.deflated_sharpe(rdev, len(names), var_sr)["dsr"]
    h = results[best]["stress"]["holdout"]
    checks = {"dsr_gt_0.95": dsr > 0.95, "holdout_return_gt_0": h["total_return"] > 0,
              "holdout_sharpe_gt_0.5": h["sharpe"] > 0.5, "holdout_maxdd_lt_30pct": h["maxdd"] < 0.30}
    verdict = "PASS: paper sleeve" if all(checks.values()) else "REJECTED"
    print(f"\nbest DEV variant: {best}; deflated Sharpe prob {dsr:.3f}; checks {checks}\nVERDICT: {verdict}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "intraday.json"), "w") as f:
        json.dump({"results": results, "best": best, "dsr": dsr, "checks": checks, "verdict": verdict}, f, indent=1)


if __name__ == "__main__":
    main()
