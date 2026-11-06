"""Paper vs backtest vs BTC for the forward test (docs/FORWARD_TEST.md criteria 1 and 3).

    python -m fund.report            # print the report
    python -m fund.report --send     # also send it to Telegram

The backtest replays the SAME period with the fund's own decision code on every
day (including days the paper run missed, e.g. laptop off), so the gap between
paper and backtest shows whether the bot is executing the strategy correctly.
"""
from __future__ import annotations

import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from fund import config, core, ledger


def _panel() -> dict[str, pd.DataFrame]:
    frames = {}
    for s in config.UNIVERSE:
        p = os.path.join(config.KLINE_DIR, f"{s}_1d.parquet")
        if os.path.exists(p):
            frames[s] = pd.read_parquet(p)
    return {f: pd.DataFrame({s: d[f] for s, d in frames.items()}).sort_index() for f in ("open", "close")}


def compare(mode: str = "paper", panel: dict | None = None) -> dict | None:
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    if not os.path.exists(path):
        return None
    b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    if not b.runs:
        return None
    p = panel if panel is not None else _panel()
    syms = [s for s in config.UNIVERSE if s in p["close"].columns]
    sig, tgt = core.targets(p["close"][syms])
    opens = p["open"][syms].copy()
    # Today's bar is still open, so its open price only exists in the ledger's last run.
    for r in b.runs:
        d = pd.Timestamp(r["date"], tz="UTC")
        if d not in opens.index:
            opens.loc[d] = pd.Series(r["prices"])[syms]
    opens = opens.sort_index()

    start, end = pd.Timestamp(b.runs[0]["date"], tz="UTC"), pd.Timestamp(b.runs[-1]["date"], tz="UTC")
    qty, cash, last = {}, b.start_capital, np.zeros(len(syms))
    bt = {}
    for d in pd.date_range(start, end, freq="D"):
        dec = d - timedelta(days=1)
        if d not in opens.index or dec not in tgt.index:
            continue
        prices = opens.loc[d].to_dict()
        _, qty, cash, last, eq, _ = core.plan_orders(
            syms, qty, cash, prices, tgt.loc[dec].to_numpy(), sig.loc[dec].to_numpy(), last,
            dec.dayofweek == 0, min_order=config.MIN_ORDER_USDT)
        bt[str(d.date())] = eq
    paper = {r["date"]: r["equity"] for r in b.runs}
    btc0 = b.runs[0]["prices"]["BTCUSDT"]
    btc = {r["date"]: b.start_capital * r["prices"]["BTCUSDT"] / btc0 for r in b.runs}

    days = (end - start).days + 1
    last_day = b.runs[-1]["date"]
    gap = paper[last_day] / bt[last_day] - 1 if last_day in bt else None
    months = max(days / 30.4, 1.0)
    eq = pd.Series(paper)
    paper_dd = float((1 - eq / eq.cummax()).max())
    btc_s = pd.Series(btc)
    btc_dd = float((1 - btc_s / btc_s.cummax()).max())
    return {"mode": mode, "start": b.runs[0]["date"], "end": last_day, "days": days, "runs": len(b.runs),
            "start_capital": b.start_capital, "paper": paper, "backtest": bt, "btc": btc,
            "paper_return": paper[last_day] / b.start_capital - 1,
            "backtest_return": bt[last_day] / b.start_capital - 1 if last_day in bt else None,
            "btc_return": btc[last_day] / b.start_capital - 1,
            "gap": gap, "gap_per_month": gap / months if gap is not None else None,
            "paper_maxdd": paper_dd, "btc_maxdd": btc_dd}


def text(c: dict) -> str:
    if not c:
        return "Belum ada data fund untuk dilaporkan."
    ok1 = c["gap_per_month"] is not None and abs(c["gap_per_month"]) <= 0.01
    beats = c["paper_return"] >= c["btc_return"] or c["paper_maxdd"] < c["btc_maxdd"] / 2
    lines = [f"Laporan fund {c['mode'].upper()}: {c['start']} s/d {c['end']} (hari ke-{c['days']}, {c['runs']} run)",
             f"Paper     {c['paper_return']:+.2%}  (drawdown terdalam {c['paper_maxdd']:.1%})",
             f"Backtest  {c['backtest_return']:+.2%}  (periode sama, kode yang sama)" if c["backtest_return"] is not None else "Backtest  -",
             f"BTC hold  {c['btc_return']:+.2%}  (drawdown terdalam {c['btc_maxdd']:.1%})",
             "",
             f"Syarat 1 (bot jalan benar): selisih paper vs backtest {c['gap']:+.2%} "
             f"= {c['gap_per_month']:+.2%}/bulan -> {'OK' if ok1 else 'PERIKSA'} (batas 1%/bulan)" if c["gap"] is not None
             else "Syarat 1: belum bisa dihitung",
             f"Syarat 3 (lebih baik dari pegang BTC): {'ya' if beats else 'belum'} sejauh ini",
             f"Evaluasi resmi paling cepat 2027-02-06." + (" Data masih terlalu sedikit untuk menilai strategi." if c["days"] < 90 else "")]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--mode", default="paper")
    a = ap.parse_args()
    t = text(compare(a.mode))
    print(t)
    if a.send:
        from fund import notify
        notify.send(t)


if __name__ == "__main__":
    main()
