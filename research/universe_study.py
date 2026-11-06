"""Which coins should C4 trade? Runs research/PREREGISTRATION_UNIVERSE.md as written.

C4's rules are untouched (research/strategies.py signal and vol-target
sizing, research/engine.py execution). Only the coin list changes, and every
list is chosen with data available at the time.

Usage: python -m research.universe_study      (after python -m research.universe_data)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from research import engine, metrics, strategies
from research.data import SYMBOLS as U8

COST = 0.003822 + 0.0005
DEV = (pd.Timestamp("2018-08-17", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HOLD = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-10-10", tz="UTC"))  # last bar has no next open
SRC = os.path.join("data", "klines_universe")
OUT = os.path.join("research", "results")


def load() -> dict[str, pd.DataFrame]:
    frames = {f[:-11]: pd.read_parquet(os.path.join(SRC, f)) for f in os.listdir(SRC) if f.endswith("_1d.parquet")}
    return {k: pd.DataFrame({s: d[k] for s, d in frames.items()}).sort_index() for k in ("open", "close", "qv")}


def month_starts(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx[idx.day == 1]


def top_by_volume(qv: pd.DataFrame, eligible: pd.DataFrame, n: int) -> pd.DataFrame:
    """Membership mask: on each month start, the n eligible coins with the most
    trailing-90-day USDT volume (known at that close), held until the next."""
    vol90 = qv.rolling(90, min_periods=60).sum()
    mask = pd.DataFrame(np.nan, index=qv.index, columns=qv.columns)
    for d in month_starts(qv.index):
        row = vol90.loc[d].where(eligible.loc[d]).dropna()
        mask.loc[d] = 0.0
        mask.loc[d, row.nlargest(n).index] = 1.0
    return mask.ffill().fillna(0.0).astype(bool)


def weights(sig: pd.DataFrame, close: pd.DataFrame, member: pd.DataFrame, n: int | pd.Series) -> pd.DataFrame:
    """C4's vol-target weight (research.strategies.w_voltarget) inside a universe of size n."""
    el = strategies.eligible(close) & member
    scale = (strategies.VOL_TARGET / strategies.sigma90(close)).clip(upper=1.0)
    w = (sig * scale).where(el, 0.0)
    w = w.div(n, axis=0) if isinstance(n, pd.Series) else w / n
    return strategies._cap(w.fillna(0.0))


def run(open_, close, sig, tgt) -> engine.Result:
    return engine.run(open_, tgt, sig.where(tgt > 0, 0.0), COST, engine.mondays(close.index), 0.2,
                      track_trades=False)


def report(res: engine.Result, a, b) -> dict:
    r = res.returns[(res.returns.index >= a) & (res.returns.index < b)]
    s = metrics.summarize(r, res.exposure)
    return {k: s[k] for k in ("cagr", "sharpe", "maxdd", "total_return", "avg_exposure")}


def walk_forward_pick(standalone: pd.DataFrame, pool: pd.DataFrame, eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    """Each 1 January keep the k pool coins with the best trailing-365-day standalone return."""
    trail = (1 + standalone).rolling(365, min_periods=300).apply(np.prod, raw=True) - 1
    mask = pd.DataFrame(np.nan, index=standalone.index, columns=standalone.columns)
    for d in standalone.index[(standalone.index.month == 1) & (standalone.index.day == 1)]:
        ok = pool.loc[d] & eligible.loc[d]
        row = trail.loc[d].where(ok).dropna()
        mask.loc[d] = 0.0
        mask.loc[d, row.nlargest(k).index] = 1.0
    return mask.ffill().fillna(0.0).astype(bool)


def main() -> None:
    p = load()
    close, open_, qv = p["close"], p["open"], p["qv"]
    el_all = strategies.eligible(close)
    t10 = top_by_volume(qv, el_all, 10)
    t20 = top_by_volume(qv, el_all, 20)
    cols = sorted(set(U8) | set(t20.columns[t20.any()]))
    close, open_ = close[cols], open_[cols]
    t10, t20, el = t10[cols], t20[cols], el_all[cols]
    print(f"coins ever in a top-20: {int(t20.any().sum())}; computing C4 signals for {len(cols)} coins", flush=True)
    sig = strategies.sig_donchian(close).where(el, 0.0)

    # Standalone C4 per coin (no 1/N): what a coin's own trend-following returned.
    scale = (strategies.VOL_TARGET / strategies.sigma90(close)).clip(upper=1.0)
    standalone = pd.DataFrame({c: engine.run(open_[[c]], (sig[[c]] * scale[[c]]).fillna(0.0), sig[[c]], COST,
                                             engine.mondays(close.index), 0.2, track_trades=False).returns
                               for c in cols})

    u8 = pd.DataFrame(False, index=close.index, columns=cols)
    u8[U8] = True
    n_u8 = (el & u8).sum(axis=1).replace(0, np.nan)          # as the original C4: N = eligible U8 coins
    wf4 = walk_forward_pick(standalone, u8, el, 4)
    wf8 = walk_forward_pick(standalone, t20, el, 8)
    cands = {"U8": (u8, n_u8), "T10": (t10, 10), "T20": (t20, 20), "WF4": (wf4, 4), "WF8": (wf8, 8)}

    results, members = {}, {}
    for name, (m, n) in cands.items():
        tgt = weights(sig, close, m, n)
        res = run(open_, close, sig, tgt)
        results[name] = {"dev": report(res, *DEV), "holdout": report(res, *HOLD)}
        last = m.iloc[-1]
        members[name] = sorted(last.index[last])
        print(f"{name:<4} DEV Sharpe {results[name]['dev']['sharpe']:.2f} CAGR {results[name]['dev']['cagr']:+.1%} "
              f"MaxDD {results[name]['dev']['maxdd']:.1%} | HOLDOUT Sharpe {results[name]['holdout']['sharpe']:.2f} "
              f"ret {results[name]['holdout']['total_return']:+.1%} MaxDD {results[name]['holdout']['maxdd']:.1%}", flush=True)

    base = results["U8"]
    verdict = {}
    for name, r in results.items():
        if name == "U8":
            continue
        better = all(r[ph]["sharpe"] > base[ph]["sharpe"] for ph in ("dev", "holdout"))
        dd_ok = all(r[ph]["maxdd"] <= base[ph]["maxdd"] + 0.05 for ph in ("dev", "holdout"))
        verdict[name] = {"higher_sharpe_both": better, "maxdd_within_5pp": dd_ok, "passes": better and dd_ok}
    passing = [k for k, v in verdict.items() if v["passes"]]
    chosen = max(passing, key=lambda k: results[k]["dev"]["sharpe"]) if passing else "U8"
    print("verdict:", json.dumps(verdict), "\nCHOSEN:", chosen, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "universe.json"), "w") as f:
        json.dump({"results": results, "verdict": verdict, "chosen": chosen,
                   "current_members": members}, f, indent=1)


if __name__ == "__main__":
    main()
