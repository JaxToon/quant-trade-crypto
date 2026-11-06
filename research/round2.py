"""Research round 2, run exactly as research/PREREGISTRATION_ROUND2.md says.

R2-A  C4-LS: C4 plus a mirrored short side on perpetual futures (real funding).
R2-B  C4+GOLD: PAXG added to the U8 universe, C4 rules unchanged.

Usage: python -m research.round2
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from research import carry_analysis, engine, metrics, strategies
from research.data import SYMBOLS as U8, load as load_klines, panel

SPOT_COST = 0.003822 + 0.0005
PERP_COST = 0.0015            # fee + slippage per side, assumed above typical taker fees
PERP_COST_HIGH = 0.004322     # sensitivity only
DEV = ("2018-08-17", "2025-01-01")
HOLD = ("2025-01-01", "2026-10-10")
OUT = os.path.join("research", "results")


# ---------------------------------------------------------------- data

def funding_daily(sym: str) -> pd.Series:
    """Daily sum of USDT-M funding prints (positive = longs pay shorts). Empty if no perp."""
    frames = [df for ym in pd.period_range("2019-09", "2026-10", freq="M").astype(str)
              if (df := carry_analysis._month_file(sym, ym)) is not None and len(df)]
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rcol = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
    s = pd.Series(df[rcol].astype(float).to_numpy(), index=pd.to_datetime(df[tcol], unit="ms", utc=True))
    s = s[~s.index.duplicated()].sort_index()
    return s.groupby(s.index.floor("D")).sum()


# ---------------------------------------------------------------- signals

def donchian_short(close: pd.Series) -> pd.Series:
    """Mirror of strategies.donchian_ensemble: on below the L-day low, stop starts at the
    channel midpoint and only ratchets down, off when the close rises above it."""
    c = close.to_numpy(dtype=float)
    states = np.zeros((len(strategies.LOOKBACKS), len(c)))
    for j, L in enumerate(strategies.LOOKBACKS):
        hi = close.shift(1).rolling(L).max().to_numpy()
        lo = close.shift(1).rolling(L).min().to_numpy()
        on, stop = False, np.inf
        for t in range(len(c)):
            if np.isnan(c[t]) or np.isnan(hi[t]) or np.isnan(lo[t]):
                on, stop = False, np.inf
                continue
            mid = (hi[t] + lo[t]) / 2
            if not on:
                if c[t] < lo[t]:
                    on, stop = True, mid
            else:
                stop = min(stop, mid)
                if c[t] > stop:
                    on, stop = False, np.inf
            states[j, t] = 1.0 if on else 0.0
    return pd.Series(states.mean(axis=0), index=close.index)


def ls_weights(close: pd.DataFrame, perp_live: pd.DataFrame, short: bool = True):
    el = strategies.eligible(close)
    long_s = strategies.sig_donchian(close)
    short_s = pd.DataFrame({s: donchian_short(close[s]) for s in close.columns}) if short else long_s * 0
    short_s = short_s.where(perp_live, 0.0)
    net = (long_s - short_s).where(el, 0.0)
    n = el.sum(axis=1).replace(0, np.nan)
    scale = (strategies.VOL_TARGET / strategies.sigma90(close)).clip(upper=1.0)
    w = (net * scale).div(n, axis=0).fillna(0.0)
    gross = w.abs().sum(axis=1)
    w = w.mul((1.0 / gross).where(gross > 1.0, 1.0), axis=0)
    return net, w


# ---------------------------------------------------------------- engine (signed weights)

def run_ls(open_: pd.DataFrame, target: pd.DataFrame, signal: pd.DataFrame, funding: pd.DataFrame,
           spot_cost: float, perp_cost: float, band: float = 0.2) -> engine.Result:
    """research/engine.py with signed weights: long legs pay spot cost, short legs pay perp
    cost and collect (or pay) funding. Identical to engine.run when no weight is negative."""
    dates, syms = open_.index, list(open_.columns)
    O = open_.to_numpy(dtype=float)
    T = target.reindex(index=dates, columns=syms).fillna(0.0).to_numpy()
    S = signal.reindex(index=dates, columns=syms).fillna(0.0).to_numpy()
    F = funding.reindex(index=dates, columns=syms).fillna(0.0).to_numpy()
    reb = engine.mondays(dates).to_numpy(dtype=bool)
    n, m = O.shape
    R = np.zeros_like(O)
    with np.errstate(invalid="ignore", divide="ignore"):
        R[:-1] = O[1:] / O[:-1] - 1.0
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    w, last_sig = np.zeros(m), np.zeros(m)
    out_r, out_x, out_t, out_c = (np.zeros(n) for _ in range(4))
    for i in range(1, n - 1):
        t, s, h = T[i - 1], S[i - 1], w
        exit_ = (t == 0) & (h != 0)
        enter = (t != 0) & (h == 0)
        flip = (t != 0) & (h != 0) & (np.sign(t) != np.sign(h))
        sig_chg = (t != 0) & (h != 0) & (s != last_sig)
        drift = (t != 0) & (h != 0) & reb[i - 1] & (np.abs(t - h) > band * np.abs(t))
        trade = exit_ | enter | flip | sig_chg | drift
        new = np.where(trade, t, h)
        if np.abs(new).sum() > 1.0 + 1e-9:
            trade = (t != 0) | (h != 0)
            new = t.copy()
        last_sig = np.where(trade, s, last_sig)
        c = (spot_cost * np.abs(np.maximum(new, 0) - np.maximum(h, 0)).sum()
             + perp_cost * np.abs(np.minimum(new, 0) - np.minimum(h, 0)).sum())
        w = new
        g = float((w * R[i]).sum() + (np.maximum(-w, 0) * F[i]).sum())
        out_x[i], out_t[i], out_c[i] = np.abs(w).sum(), np.abs(new - h).sum(), c
        out_r[i] = (1.0 - c) * (1.0 + g) - 1.0
        price_g = float((w * R[i]).sum())
        if 1.0 + price_g != 0:
            w = w * (1.0 + R[i]) / (1.0 + price_g)
    idx = dates
    return engine.Result(pd.Series(out_r, idx), pd.Series(out_x, idx), pd.Series(out_t, idx), pd.Series(out_c, idx))


# ---------------------------------------------------------------- report

def rep(res: engine.Result, a: str, b: str, mult: float = 1.0) -> dict:
    r = res.returns[(res.returns.index >= a) & (res.returns.index < b)]
    s = metrics.summarize(r, res.exposure)
    y = metrics.yearly(r)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["maxdd"], "total_return": s["total_return"],
            "avg_exposure": s["avg_exposure"], "worst_year": min(y.values()), "best_year": max(y.values()),
            "yearly": y}


def passes(c: dict, base: dict) -> bool:
    return all(c[p]["sharpe"] > base[p]["sharpe"] and c[p]["maxdd"] <= base[p]["maxdd"] + 0.05
               for p in ("dev", "holdout"))


def main() -> None:
    p = panel("1d")
    gold = load_klines("PAXGUSDT", "1d")
    close9 = p["close"].join(gold["close"].rename("PAXGUSDT"), how="left")
    open9 = p["open"].join(gold["open"].rename("PAXGUSDT"), how="left")
    close9.index = open9.index = pd.DatetimeIndex(pd.to_datetime(close9.index, utc=True))
    syms9 = list(close9.columns)
    fund = pd.DataFrame({s: funding_daily(s) for s in syms9}).reindex(close9.index)
    perp_live = fund.notna().cummax()   # True from a coin's first funding print onward
    fund = fund.fillna(0.0)
    print("perp listing dates:", {s: str(fund.index[perp_live[s]][0].date()) if perp_live[s].any() else None
                                  for s in syms9}, flush=True)

    def c4(cols):
        close, open_ = close9[cols], open9[cols]
        sig = strategies.sig_donchian(close).where(strategies.eligible(close), 0.0)
        tgt = strategies.w_voltarget(sig, close)
        return engine.run(open_, tgt, sig, SPOT_COST, engine.mondays(close.index), 0.2, track_trades=False)

    def ls(cols, perp_cost=PERP_COST, short=True):
        close, open_ = close9[cols], open9[cols]
        net, w = ls_weights(close, perp_live[cols], short=short)
        return run_ls(open_, w, net, fund[cols], SPOT_COST, perp_cost)

    # Sanity: the signed engine with the short side off must reproduce research C4 exactly.
    a, b = c4(U8).returns, ls(U8, short=False).returns
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12), "signed engine does not reproduce C4"
    print("sanity: signed engine with shorts off == research C4 (all days)", flush=True)

    runs = {"C4 (now)": c4(U8), "C4-LS": ls(U8), "C4+GOLD": c4(U8 + ["PAXGUSDT"]),
            "C4-LS+GOLD": ls(U8 + ["PAXGUSDT"]),
            "C4-LS @0.43% perp (sensitivity)": ls(U8, PERP_COST_HIGH)}
    results = {k: {"dev": rep(v, *DEV), "holdout": rep(v, *HOLD)} for k, v in runs.items()}
    base = results["C4 (now)"]
    for k, r in results.items():
        d, h = r["dev"], r["holdout"]
        print(f"{k:<34} DEV Sharpe {d['sharpe']:5.2f} CAGR {d['cagr']:+6.1%} MaxDD {d['maxdd']:5.1%} "
              f"worst yr {d['worst_year']:+6.1%} expo {d['avg_exposure']:4.0%} | HOLD Sharpe {h['sharpe']:5.2f} "
              f"ret {h['total_return']:+6.1%} MaxDD {h['maxdd']:5.1%}", flush=True)

    verdict = {k: passes(results[k], base) for k in ("C4-LS", "C4+GOLD")}
    if verdict["C4-LS"] and verdict["C4+GOLD"] and passes(results["C4-LS+GOLD"], base):
        chosen = "C4-LS+GOLD"
    else:
        ok = [k for k, v in verdict.items() if v]
        chosen = max(ok, key=lambda k: results[k]["dev"]["sharpe"]) if ok else "C4 (now)"
    print("verdict:", verdict, "| combo passes:", passes(results["C4-LS+GOLD"], base), "\nCHOSEN:", chosen, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "round2.json"), "w") as f:
        json.dump({"results": results, "verdict": verdict, "chosen": chosen,
                   "perp_cost": PERP_COST, "spot_cost": SPOT_COST}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
