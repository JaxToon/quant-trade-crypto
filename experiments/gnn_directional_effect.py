"""
Does the GNN signal specifically DEGRADE shorts while helping longs?

The matched-window comparison (live_vs_backtest_matched_window.py) found a
sign flip that only affects one direction:

                    LIVE            BACKTEST (gnn_signal=0.0)
  LONG    73.7% WR, +$10,879     61.1% WR, +$9,014
  SHORT   35.1% WR,  -$7,975     55.8% WR, +$2,299

Live is BETTER than backtest on longs and catastrophically worse on shorts.
A uniform degradation (slippage, latency, outages) cannot produce that
asymmetry — something is specifically hurting shorts.

One concrete difference between the two runs: live calls
`build_entry_insight(..., gnn_signals.get(symbol, 0.0))` with a REAL GNN
signal, while the backtest replay passes 0.0.

The project's GNN ablation concluded the signal "adds no edge" — but that
was measured on AGGREGATE win rate (57.5% both arms). If the GNN improves
longs and damages shorts by similar magnitudes, an aggregate would look
flat while hiding two opposing effects. That is exactly the shape of the
asymmetry above, and it was never checked per direction.

This runs the same window, 1h lockstep across all symbols (the joint
forward pass the GNN needs), with the signal ON and OFF, and splits the
result by direction.

IMPORTANT LOOK-AHEAD CAVEAT. The checkpoint being loaded was trained on
data that includes this window, so its "prediction" here is contaminated
and flatters the GNN. That biases the test toward finding GNN helpful — so
if GNN still shows up as HARMFUL to shorts, the finding is conservative
and the real effect is likely worse. It also means a positive result here
would prove nothing.

Usage: python -m experiments.gnn_directional_effect
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

TF = "1h"
TF_MS = 3_600_000
WARMUP_BARS = 150
LIVE_LEDGER = "backtrack_logs/paper_ledger_crypto.json"


def _fetch(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    out, cursor = [], start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{settings.BINANCE_REST_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": TF,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1000},
                             timeout=20)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            for b in batch:
                out.append(Candle(timestamp=datetime.utcfromtimestamp(b[0] / 1000),
                                  open=float(b[1]), high=float(b[2]), low=float(b[3]),
                                  close=float(b[4]), volume=float(b[5])))
            cursor = batch[-1][0] + TF_MS
            time.sleep(0.12)
        except Exception:
            break
    return out


def _summarise(trades, label):
    if not trades:
        return f"{label:<22} (no trades)"
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    wins = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] > 0]
    loss = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] <= 0]
    pf = (sum(wins) / abs(sum(loss))) if loss and sum(loss) != 0 else float("inf")
    return (f"{label:<22}{len(trades):<7}{w/len(trades)*100:>6.1f}%  "
            f"${pnl:>+11,.0f}  ${pnl/len(trades):>+9,.2f}  {pf:>6.2f}")


def run(candles: dict, window_start, trainer) -> list[dict]:
    led = PaperTradingLedger(path=f"backtrack_logs/_gnnd_{'on' if trainer else 'off'}.json",
                             starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    led.balance = led.starting_balance
    led.open_positions.clear(); led.closed_trades.clear()
    led.max_closed_trades = 1_000_000

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}
    closes = {s: [] for s in candles}
    if trainer is not None:
        from train_gnn import _rsi, _macd, _bollinger, _encode_market_structure

    n = min(len(c) for c in candles.values())
    for t in range(n):
        step, ctxs = {}, {}
        for sym, cs in candles.items():
            c = cs[t]
            step[sym] = c
            sig = smc[sym].update(c)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2, current_price=c.close, volume=c.volume)
            fvg[sym].update_all(c.close)
            ctxs[sym] = {**sig, **fvg[sym].get_trading_signal(
                current_price=c.close, trend=sig.get("market_structure", "unknown")),
                "timestamp": c.timestamp.isoformat()}
            closes[sym].append(c.close)

        if trainer is not None:
            feats = {}
            for sym, c in step.items():
                cl = closes[sym]
                rsi = _rsi(cl); macd, msig, mhist = _macd(cl); bu, bm, bl = _bollinger(cl)
                s = ctxs[sym]
                feats[sym] = [c.close, c.volume, rsi, macd, msig, mhist, bu, bm, bl,
                              s.get("smc_signal", 0.0),
                              _encode_market_structure(s.get("market_structure", "unknown")),
                              float(s.get("active_order_blocks", 0)), float(s.get("active_fvgs", 0)),
                              float(s.get("liquidity_zones", 0)),
                              1.0 if s.get("in_bullish_ob") else 0.0, 1.0 if s.get("in_bearish_ob") else 0.0,
                              1.0 if s.get("in_bullish_fvg") else 0.0, 1.0 if s.get("in_bearish_fvg") else 0.0]
            gnn = trainer.predict_from_features(feats)
        else:
            gnn = {s: 0.0 for s in step}

        for sym, c in step.items():
            at = c.timestamp.isoformat()
            led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=TF)
            if c.timestamp < window_start:
                continue
            ins = build_entry_insight(sym, c.close, ctxs[sym], gnn_signal=gnn.get(sym, 0.0))
            led.maybe_open(sym, "spot", ins["spot"], timeframe=TF, opened_at=at)
            led.maybe_open(sym, "derivative", ins["derivative"], timeframe=TF, opened_at=at)

    return list(led.closed_trades)


def main() -> None:
    d = json.load(open(LIVE_LEDGER))
    tr = [t for t in d["closed_trades"] if t.get("opened_at") and t.get("closed_at")]
    to_ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms, end_ms = to_ms(min(t["opened_at"] for t in tr)), to_ms(max(t["closed_at"] for t in tr))
    window_start = datetime.utcfromtimestamp(start_ms / 1000)
    print(f"Window: {window_start} -> {datetime.utcfromtimestamp(end_ms/1000)}\n", flush=True)

    candles = {}
    for sym in settings.WATCHLIST:
        c = _fetch(sym, start_ms - WARMUP_BARS * TF_MS, end_ms)
        if c:
            candles[sym] = c
            print(f"  {sym:<9}{len(c):>5} candles", flush=True)
    if not candles:
        print("no data"); return

    print("\nArm 1/2: GNN OFF...", flush=True)
    off = run(candles, window_start, None)

    print("Arm 2/2: GNN ON...", flush=True)
    from gnn.trainer import GNNTrainer
    trainer = GNNTrainer()
    if not trainer.load():
        print("No GNN checkpoint — cannot run the ON arm. Aborting.")
        return
    on = run(candles, window_start, trainer)

    print(f"\n{'=' * 80}")
    print(f"GNN DIRECTIONAL EFFECT — matched live window, {TF} lockstep")
    print(f"{'=' * 80}")
    print(f"{'Arm':<22}{'n':<7}{'WR':>7}  {'PnL':>12}  {'exp/trade':>10}  {'PF':>6}")
    print("-" * 80)
    print(_summarise(off, "GNN OFF  all"))
    print(_summarise(on,  "GNN ON   all"))
    print("-" * 80)
    for dirn in ("long", "short"):
        o = [t for t in off if t.get("direction") == dirn]
        n_ = [t for t in on if t.get("direction") == dirn]
        print(_summarise(o, f"GNN OFF  {dirn}"))
        print(_summarise(n_, f"GNN ON   {dirn}"))
        if o and n_:
            po, pn = sum(t["pnl_dollars"] for t in o), sum(t["pnl_dollars"] for t in n_)
            print(f"{'  delta':<22}{'':<7}{'':>7}  ${pn-po:>+11,.0f}")
        print("-" * 80)

    os_ = [t for t in off if t.get("direction") == "short"]
    ns_ = [t for t in on if t.get("direction") == "short"]
    if os_ and ns_:
        po, pn = sum(t["pnl_dollars"] for t in os_), sum(t["pnl_dollars"] for t in ns_)
        print("\nVERDICT:")
        if pn < po:
            print(f"  GNN makes SHORT worse (${po:+,.0f} -> ${pn:+,.0f}) — and remember the")
            print("  checkpoint was trained on this window, so the test is biased IN the")
            print("  GNN's favour. A harmful result here is conservative.")
        else:
            print(f"  GNN does not degrade SHORT here (${po:+,.0f} -> ${pn:+,.0f}). Given the")
            print("  look-ahead contamination this proves nothing positive — it only fails")
            print("  to support the hypothesis, which must then be looked for elsewhere.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
