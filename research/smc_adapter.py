"""Candidates C8-C10: the project's own SMC/FVG engine on Binance candles.

Reuses SmartMoneyAnalyzer, FairValueGapAnalyzer, build_entry_insight and
PaperTradingLedger unchanged. The only difference from the live system is the
honest fill: entry is booked at the signal bar's close (a market order), not at
the FVG midpoint (see experiments/entry_fill_realism.py, MARKET arm). Spot
market only, so long-only like the other candidates.

Usage: python -m research.smc_adapter --tf 1h --phase dev|holdout
Writes research/results/smc_<tf>_<phase>.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from datetime import timezone

import pandas as pd

from config import settings
from research.data import SYMBOLS, load

FEE = 0.003822            # Tokocrypto USDT market, per side
STOP_SLIP = 0.00119       # measured live stop slippage, n=31
RISK, CAP = 0.0025, 1.0   # spot-viable sizing
START_BALANCE = 10_000.0
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
# dev: trade from the core candidates' first execution day; holdout: a fresh
# $10,000 account from the first holdout day, with a few months of warm-up so
# the analyzers start with history, exactly as a restarted live system would.
WINDOWS = {
    "dev": (pd.Timestamp("2018-05-01", tz="UTC"), pd.Timestamp("2018-08-18", tz="UTC")),
    "holdout": (pd.Timestamp("2024-09-01", tz="UTC"), DEV_END),
}


def run(tf: str, phase: str) -> dict:
    logging.getLogger("backtrack.paper_trading").setLevel(logging.ERROR)
    settings.PAPER_TRADING_FEE_PCT = FEE
    import backtrack.paper_trading as pt
    importlib.reload(pt)
    settings.PAPER_TRADING_RISK_PCT = RISK
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = CAP
    from strategy.entry_insight import build_entry_insight
    from strategy.fvg_analyzer import FairValueGapAnalyzer
    from strategy.smart_money import Candle, SmartMoneyAnalyzer

    WARMUP_FROM, TRADE_FROM = WINDOWS[phase]
    frames = {}
    for s in SYMBOLS:
        df = load(s, tf)
        df = df[df.index >= WARMUP_FROM]
        if phase == "dev":
            df = df[df.index < DEV_END]   # holdout data is never loaded in the dev phase
        frames[s] = df

    candles = {s: {ts: Candle(timestamp=ts.to_pydatetime(), open=r.open, high=r.high, low=r.low,
                              close=r.close, volume=r.volume)
                   for ts, r in df.iterrows()} for s, df in frames.items()}
    stamps = sorted(set().union(*[c.keys() for c in candles.values()]))

    path = os.path.join("research", "results", f"_smc_ledger_{tf}_{phase}.json")
    led = pt.PaperTradingLedger(path=path, starting_balance=START_BALANCE, stop_slippage_pct=STOP_SLIP)
    led.balance = led.starting_balance
    led.open_positions.clear()
    led.closed_trades.clear()
    led.max_closed_trades = 10_000_000

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in SYMBOLS}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in SYMBOLS}

    for ts in stamps:
        for sym in SYMBOLS:
            c = candles[sym].get(ts)
            if c is None:
                continue
            at = c.timestamp.isoformat()
            sig = smc[sym].update(c)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2, current_price=c.close, volume=c.volume)
            fvg[sym].update_all(c.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(
                current_price=c.close, trend=sig.get("market_structure", "unknown")), "timestamp": at}
            ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
            led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
            if ts < TRADE_FROM:
                continue
            rec = ins["spot"]
            if rec.get("actionable") and rec.get("entry_zone"):
                mrec = dict(rec)
                mrec["entry_zone"] = [c.close, c.close]   # market fill at the signal close
                led.maybe_open(sym, "spot", mrec, timeframe=tf, opened_at=at)

    trades = [t for t in led.closed_trades if t.get("pnl_dollars") is not None]
    if os.path.exists(path):
        os.remove(path)

    # Daily equity from realised trades (positions are short-lived; not marked to market).
    last_day = (stamps[-1].normalize() if stamps else TRADE_FROM)
    days = pd.date_range(TRADE_FROM, last_day, freq="D", tz=timezone.utc)
    pnl = pd.Series(0.0, index=days)
    for t in trades:
        d = pd.Timestamp(t["closed_at"]).tz_convert("UTC").normalize() if pd.Timestamp(t["closed_at"]).tzinfo \
            else pd.Timestamp(t["closed_at"]).tz_localize("UTC").normalize()
        if d in pnl.index:
            pnl.loc[d] += t["pnl_dollars"]
    # Ruin rule: the ledger sizes off the STARTING balance, so it keeps trading
    # after the account is gone. A real account cannot. The first day equity
    # reaches zero is a -100% day and every day after it is flat.
    equity = START_BALANCE + pnl.cumsum()
    prev = equity.shift(1).fillna(START_BALANCE)
    rets = (pnl / prev).fillna(0.0)
    ruined = equity <= 0
    if ruined.any():
        first = ruined.idxmax()
        rets.loc[first] = -1.0
        rets.loc[rets.index > first] = 0.0

    out = {
        "candidate": f"SMC-{tf.upper()}", "tf": tf, "phase": phase,
        "ruined_on": str(ruined.idxmax().date()) if ruined.any() else None,
        "returns": {str(k.date()): float(v) for k, v in rets.items()},
        "trades": [{"symbol": t["symbol"], "opened_at": t["opened_at"], "closed_at": t["closed_at"],
                    "pnl_dollars": t["pnl_dollars"], "pnl_pct": t.get("pnl_pct"),
                    "fee_dollars": t.get("fee_dollars"), "outcome": t.get("outcome")} for t in trades],
        "settings": {"fee": FEE, "stop_slippage": STOP_SLIP, "risk": RISK, "cap": CAP,
                     "min_stop_pct": pt.MIN_STOP_DISTANCE_PCT},
    }
    os.makedirs(os.path.join("research", "results"), exist_ok=True)
    with open(os.path.join("research", "results", f"smc_{tf}_{phase}.json"), "w") as f:
        json.dump(out, f)
    print(f"SMC-{tf} {phase}: {len(trades)} trades, final ledger equity {equity.iloc[-1]:,.2f}"
          f"{' (account ruined on ' + out['ruined_on'] + ')' if out['ruined_on'] else ''}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True, choices=["1h", "4h", "1d"])
    ap.add_argument("--phase", required=True, choices=["dev", "holdout"])
    a = ap.parse_args()
    run(a.tf, a.phase)
