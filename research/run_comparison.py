"""The pre-registered comparison (research/PREREGISTRATION.md), in three phases.

  python -m research.run_comparison dev       C1-C7 + benchmarks on DEV. Holdout
                                               prices are never loaded.
  python -m research.run_comparison select    merge DEV results (incl. SMC files
                                               from research.smc_adapter) and
                                               apply the selection rule.
  python -m research.run_comparison holdout   run once on the full data; report
                                               every candidate, gate the selected.

Outputs go to research/results/.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

from research import engine, metrics, strategies
from research.data import panel

COST = 0.003822 + 0.0005        # Tokocrypto fee + slippage, each side
COST_LOW = 0.001 + 0.0005       # sensitivity only
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
N_NULL = 1000
OUT = os.path.join("research", "results")
BENCHMARKS = ("BH-BTC", "BH-EW")
SMC = {"C8 SMC-1H": "1h", "C9 SMC-4H": "4h", "C10 SMC-1D": "1d"}


def _daily(phase: str) -> dict[str, pd.DataFrame]:
    p = panel("1d")
    if phase == "dev":
        p = {k: v[v.index < DEV_END] for k, v in p.items()}
    return p


def _effective_start(close: pd.DataFrame) -> pd.Timestamp:
    first = strategies.eligible(close).any(axis=1).idxmax()
    return first + pd.Timedelta(days=1)       # first day a position can be held


def _run(name: str, p: dict, cost: float, track_trades: bool = True):
    close, open_ = p["close"], p["open"]
    if name == "BH-BTC":
        sig, tgt = strategies.bh_btc(close)
        return sig, tgt, engine.run(open_, tgt, sig, cost, pd.Series(False, index=close.index), 0.0, track_trades)
    if name == "BH-EW":
        sig, tgt = strategies.bh_equal(close)
        return sig, tgt, engine.run(open_, tgt, sig, cost, engine.month_starts(close.index), 0.0, track_trades)
    sig_fn, w_fn = strategies.CANDIDATES[name]
    sig = sig_fn(close).where(strategies.eligible(close), 0.0)
    tgt = w_fn(sig, close)
    return sig, tgt, engine.run(open_, tgt, sig, cost, engine.mondays(close.index), 0.2, track_trades)


def _window(res: engine.Result, start, end) -> tuple[pd.Series, pd.DataFrame]:
    r = res.returns[(res.returns.index >= start) & (res.returns.index < end)]
    r = r.iloc[:-1] if len(r) and r.index[-1] == res.returns.index[-1] else r  # last day has no next open
    tr = res.trades
    if not tr.empty:
        tr = tr[(tr["entry"] >= start) & (tr["entry"] < end)]
    return r, tr


def _report(name, res, start, end, with_ci=True) -> dict:
    r, tr = _window(res, start, end)
    s = metrics.summarize(r, res.exposure, res.turnover, res.costs, tr)
    s["yearly"] = metrics.yearly(r)
    if with_ci:
        s["sharpe_ci"] = metrics.stationary_bootstrap_sharpe(r)
    return s


def _smc_report(tf: str, phase: str, start, end) -> dict | None:
    path = os.path.join(OUT, f"smc_{tf}_{phase}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    r = pd.Series(d["returns"])
    r.index = pd.to_datetime(r.index, utc=True)
    r = r[(r.index >= start) & (r.index < end)]
    s = metrics.summarize(r)
    s["yearly"] = metrics.yearly(r)
    s["sharpe_ci"] = metrics.stationary_bootstrap_sharpe(r)
    eq = 10_000 * (1 + r).cumprod()
    t = pd.DataFrame(d["trades"])
    if not t.empty:
        t["opened"] = pd.to_datetime(t["opened_at"], utc=True, format="ISO8601").dt.normalize()
        t = t[(t["opened"] >= start) & (t["opened"] < end)]
        base = eq.shift(1).reindex(t["opened"]).fillna(10_000).to_numpy()
        t["ret"] = t["pnl_dollars"].to_numpy() / base
        wins, losses = t[t["pnl_dollars"] > 0], t[t["pnl_dollars"] <= 0]
        years = len(r) / 365.0
        s.update({
            "trades": int(len(t)), "trades_per_year": float(len(t) / years),
            "win_rate": float(len(wins) / len(t)) if len(t) else 0.0,
            "avg_win_pct": float(wins["ret"].mean()) if len(wins) else 0.0,
            "avg_loss_pct": float(losses["ret"].mean()) if len(losses) else 0.0,
            "profit_factor": float(wins["pnl_dollars"].sum() / -losses["pnl_dollars"].sum())
            if len(losses) and losses["pnl_dollars"].sum() < 0 else float("inf"),
            "expectancy_pct": float(t["ret"].mean()) if len(t) else 0.0,
            "fees_dollars": float(t["fee_dollars"].fillna(0).sum()),
        })
    return s


def _null(name: str, p: dict, sig: pd.DataFrame, start, end, seed=11) -> dict:
    close, open_ = p["close"], p["open"]
    el = strategies.eligible(close)
    _, w_fn = strategies.CANDIDATES[name]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_NULL):
        ss = metrics.shift_signal(sig, el, rng)
        tgt = w_fn(ss, close)
        res = engine.run(open_, tgt, ss, COST, engine.mondays(close.index), 0.2, track_trades=False)
        r, _ = _window(res, start, end)
        sd = r.std(ddof=1)
        out.append(float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0)
    return {"p95": float(np.percentile(out, 95)), "median": float(np.median(out)), "draws": out}


def _fmt(name, s):
    if not s:
        return f"{name:<14} (no data)"
    ci = s.get("sharpe_ci", (float("nan"), float("nan")))
    return (f"{name:<14}{s['cagr']*100:>8.1f}%{s['sharpe']:>8.2f}  [{ci[0]:>5.2f},{ci[1]:>5.2f}]"
            f"{s['maxdd']*100:>8.1f}%{s['calmar']:>7.2f}{s.get('win_rate', float('nan'))*100:>7.1f}%"
            f"{s.get('trades', 0):>7}{s.get('expectancy_pct', float('nan'))*100:>8.2f}%"
            f"{s.get('avg_exposure', float('nan'))*100:>7.0f}%")


HEADER = (f"{'':<14}{'CAGR':>9}{'Sharpe':>8}  {'95% CI':^13}{'MaxDD':>9}{'Calmar':>7}{'WR':>8}"
          f"{'trades':>7}{'exp/tr':>9}{'expo':>8}")


def phase_dev():
    p = _daily("dev")
    start, end = _effective_start(p["close"]), DEV_END
    res_all = {}
    print(f"DEV window {start.date()} -> {(end - pd.Timedelta(days=1)).date()} "
          f"(effective start = first day any coin has 365 bars)\n", flush=True)
    for name in list(BENCHMARKS) + list(strategies.CANDIDATES):
        sig, _, res = _run(name, p, COST)
        rep = _report(name, res, start, end)
        _, _, res_low = _run(name, p, COST_LOW)
        rep["low_cost"] = metrics.summarize(_window(res_low, start, end)[0])
        if name in strategies.CANDIDATES:
            nl = _null(name, p, sig, start, end)
            rep["null_p95"], rep["null_median"] = nl["p95"], nl["median"]
            rep["null_pct_rank"] = float((np.array(nl["draws"]) < rep["sharpe"]).mean())
        res_all[name] = rep
        print(_fmt(name, rep), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"window": [str(start.date()), str(end.date())], "cost": COST, "results": res_all},
              open(os.path.join(OUT, "dev_core.json"), "w"), indent=1, default=float)
    print("\nwrote research/results/dev_core.json")


def phase_select():
    core = json.load(open(os.path.join(OUT, "dev_core.json")))
    start = pd.Timestamp(core["window"][0], tz="UTC")
    res = dict(core["results"])
    for name, tf in SMC.items():
        rep = _smc_report(tf, "dev", start, DEV_END)
        if rep is None:
            sys.exit(f"missing research/results/smc_{tf}_dev.json - run research.smc_adapter first")
        res[name] = rep
    bh = res["BH-EW"]
    cands = [n for n in res if n not in BENCHMARKS]
    rows = []
    for n in cands:
        s = res[n]
        c1 = s["sharpe_ci"][0] > 0
        c2 = (s["sharpe"] > s["null_p95"]) if "null_p95" in s else None
        c3 = s["maxdd"] < 0.40
        c4 = s["calmar"] > bh["calmar"]
        ok = c1 and (c2 is not False) and c3 and c4
        rows.append((n, c1, c2, c3, c4, ok, s["sharpe"]))
    eligible = sorted([r for r in rows if r[5]], key=lambda r: -r[6])
    chosen = None
    if eligible:
        best = eligible[0]
        close = [r for r in eligible if best[6] - r[6] <= 0.05]
        chosen = min(close, key=lambda r: (strategies.N_PARAMS.get(r[0], 9), -r[6]))[0]
    # Deflated Sharpe of the chosen one: N = 10 trials, variance of DEV daily Sharpe across them.
    dsr = None
    if chosen:
        daily_sr = [res[n]["sharpe"] / math.sqrt(365) for n in cands]
        p = _daily("dev")
        if chosen in strategies.CANDIDATES:
            _, _, r = _run(chosen, p, COST)
            rr, _ = _window(r, start, DEV_END)
            dsr = metrics.deflated_sharpe(rr, len(cands), float(np.var(daily_sr, ddof=1)))
    print(f"{'candidate':<14}{'CI>0':>6}{'>null95':>9}{'DD<40%':>8}{'Calmar>BH':>11}{'eligible':>10}{'Sharpe':>8}")
    for n, c1, c2, c3, c4, ok, sh in rows:
        f = lambda v: "n/a" if v is None else ("yes" if v else "no")
        print(f"{n:<14}{f(c1):>6}{f(c2):>9}{f(c3):>8}{f(c4):>11}{f(ok):>10}{sh:>8.2f}")
    print(f"\nBH-EW Calmar on DEV: {bh['calmar']:.2f}")
    print(f"SELECTED: {chosen}")
    if dsr:
        print(f"Deflated Sharpe (N={dsr['n_trials']}): {dsr['dsr']:.4f}")
    json.dump({"selected": chosen, "rows": rows, "dsr": dsr, "dev_results": res},
              open(os.path.join(OUT, "selection.json"), "w"), indent=1, default=float)
    print("wrote research/results/selection.json")


def phase_holdout():
    sel_path = os.path.join(OUT, "selection.json")
    if not os.path.exists(sel_path):
        sys.exit("run the select phase first: the holdout is opened only after selection")
    sel = json.load(open(sel_path))
    p = _daily("holdout")
    start = DEV_END
    end = p["close"].index[-1] + pd.Timedelta(days=1)
    res_all = {}
    print(f"HOLDOUT window {start.date()} -> {p['close'].index[-2].date()}\n")
    print(HEADER)
    for name in list(BENCHMARKS) + list(strategies.CANDIDATES):
        _, _, res = _run(name, p, COST)
        rep = _report(name, res, start, end)
        _, _, res_low = _run(name, p, COST_LOW)
        rep["low_cost"] = metrics.summarize(_window(res_low, start, end)[0])
        res_all[name] = rep
        print(_fmt(name, rep), flush=True)
    for name, tf in SMC.items():
        rep = _smc_report(tf, "holdout", start, end)
        res_all[name] = rep
        print(_fmt(name, rep), flush=True)
    chosen = sel["selected"]
    verdict = None
    if chosen:
        s = res_all[chosen]
        verdict = "PASS" if (s["total_return"] > 0 and s["maxdd"] < 0.40) else "FAIL"
        print(f"\nSELECTED {chosen}: holdout return {s['total_return']*100:+.1f}%, "
              f"MaxDD {s['maxdd']*100:.1f}% -> {verdict}")
    else:
        print("\nNo candidate was selected on DEV; holdout shown for information only.")
    json.dump({"selected": chosen, "verdict": verdict, "results": res_all},
              open(os.path.join(OUT, "holdout.json"), "w"), indent=1, default=float)
    print("wrote research/results/holdout.json")


if __name__ == "__main__":
    {"dev": phase_dev, "select": phase_select, "holdout": phase_holdout}[sys.argv[1]]()
