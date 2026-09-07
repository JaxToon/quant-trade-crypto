"""
Is the GNN genuinely uninformative, or has it simply been gated out?

`gnn_directional_effect.py` found the live checkpoint emits, across 2,000
real (symbol, timestep) predictions:

    min -0.0508   max +0.1248   mean -0.0068   stdev 0.0173
    GNN_CONVICTION_THRESHOLD = 0.08  ->  22/2000 (1.10%) ever reach it

So `_apply_gnn_confluence` returns early on 98.9% of calls. The model is
not dead — it varies and differentiates between symbols (cross-symbol
spread 0.037 within a timestep) — but its output scale is ~4.6 standard
deviations short of the gate it has to clear.

That means the project's GNN ablation ("adds no edge", 57.5% win rate both
arms) never actually tested what it was read as testing. It measured a
signal that was gated out, not a signal that was uninformative. Same
result, entirely different conclusion, and the difference was invisible
because nobody compared the output distribution to the threshold.

This runs the ablation the earlier one was assumed to be running: the same
730d / 6-segment / 1h bar, with the gate placed where the signal actually
speaks, and split by direction — because an aggregate can hide a signal
that helps longs and hurts shorts in equal measure.

Thresholds are chosen from the MEASURED distribution, not tuned for
results: 0.08 (current, ~4.6 sigma), 0.035 (~2 sigma), 0.017 (~1 sigma),
0.0 (ungated).

PRE-REGISTERED DECISION RULE (fixed before running):
  A lowered threshold is worth adopting only if it beats GNN_OFF on total
  PnL AND in >= 5 of 6 segments AND does not make either direction worse
  in PnL. If none qualifies, the GNN is uninformative to Path 1 and can be
  retired with evidence instead of assumption — that is a perfectly good
  outcome and must be reported as such, not re-run with new thresholds
  until something passes.

LOOK-AHEAD CAVEAT, and it is not small: the checkpoint was trained on data
overlapping this window, so its predictions here are contaminated in the
GNN's favour. A NULL result is therefore strong evidence (it failed even
with the advantage). A POSITIVE result would be weak and would need
walk-forward retraining before anyone acts on it.

Usage: python -m experiments.gnn_threshold_ablation [--segments 6]
"""
from __future__ import annotations

import argparse
import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy import entry_insight as ei
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# name -> threshold; None means "GNN off entirely" (the control)
ARMS = {
    "GNN_OFF":       None,
    "THRESH_0.08":   0.08,    # current production gate, ~4.6 sigma
    "THRESH_0.035":  0.035,   # ~2 sigma
    "THRESH_0.017":  0.017,   # ~1 sigma
    "THRESH_0.0":    0.0,     # ungated — signal always applies
}


def _summarise(trades):
    if not trades:
        return (0, 0.0, 0.0)
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    return (len(trades), w / len(trades) * 100, pnl)


def run_segment(symbols_candles, timeframe, label, trainer):
    ledgers = {}
    for name in ARMS:
        led = PaperTradingLedger(path=f"backtrack_logs/_gth_{label}_{name}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}
    closes = {s: [] for s in symbols_candles}
    from train_gnn import _rsi, _macd, _bollinger, _encode_market_structure

    original_threshold = settings.GNN_CONVICTION_THRESHOLD
    n = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(n):
        step, ctxs = {}, {}
        for sym, cs in symbols_candles.items():
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

        # ONE forward pass serves every arm: the threshold changes how the
        # signal is consumed, never what the model emits.
        feats = {}
        for sym, c in step.items():
            cl = closes[sym]
            rsi = _rsi(cl); m, ms, mh = _macd(cl); bu, bm, bl = _bollinger(cl)
            s = ctxs[sym]
            feats[sym] = [c.close, c.volume, rsi, m, ms, mh, bu, bm, bl,
                          s.get("smc_signal", 0.0),
                          _encode_market_structure(s.get("market_structure", "unknown")),
                          float(s.get("active_order_blocks", 0)), float(s.get("active_fvgs", 0)),
                          float(s.get("liquidity_zones", 0)),
                          1.0 if s.get("in_bullish_ob") else 0.0, 1.0 if s.get("in_bearish_ob") else 0.0,
                          1.0 if s.get("in_bullish_fvg") else 0.0, 1.0 if s.get("in_bearish_fvg") else 0.0]
        gnn = trainer.predict_from_features(feats)

        for name, thr in ARMS.items():
            led = ledgers[name]
            for sym, c in step.items():
                at = c.timestamp.isoformat()
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=timeframe)
                if thr is None:
                    sig_val = 0.0
                else:
                    sig_val = gnn.get(sym, 0.0)
                    settings.GNN_CONVICTION_THRESHOLD = thr
                    ei.settings.GNN_CONVICTION_THRESHOLD = thr
                ins = build_entry_insight(sym, c.close, ctxs[sym], gnn_signal=sig_val)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=timeframe, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=timeframe, opened_at=at)
        settings.GNN_CONVICTION_THRESHOLD = original_threshold
        ei.settings.GNN_CONVICTION_THRESHOLD = original_threshold

    return {n_: list(l.closed_trades) for n_, l in ledgers.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--timeframe", default="1h")
    args = ap.parse_args()
    tf = args.timeframe
    days = YF_MAX_DAYS.get(tf, 730)

    from gnn.trainer import GNNTrainer
    trainer = GNNTrainer()
    if not trainer.load():
        print("No GNN checkpoint — aborting."); return
    print(f"GNN checkpoint loaded. Fetching {days}d of {tf}...\n", flush=True)

    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / args.segments) for i in range(args.segments + 1)]
    trades = {k: [] for k in ARMS}
    seg_pnl = {k: [] for k in ARMS}

    for i in range(args.segments):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for name, tr in run_segment(part, tf, f"s{i}", trainer).items():
            trades[name].extend(tr)
            seg_pnl[name].append(sum(t["pnl_dollars"] for t in tr))
        print(f"  seg{i} done", flush=True)

    base = seg_pnl["GNN_OFF"]
    base_pnl = sum(base)
    print(f"\n{'=' * 96}")
    print(f"GNN THRESHOLD ABLATION — {days}d, {args.segments} segments, {tf}, per direction")
    print(f"{'=' * 96}")
    print(f"{'Arm':<15}{'n':<7}{'WR':>7}  {'Total PnL':>13}  {'vs base':>9}  "
          f"{'LONG PnL':>12}  {'SHORT PnL':>12}  {'segs>base':>10}")
    print("-" * 96)
    for name in ARMS:
        cnt, wr, pnl = _summarise(trades[name])
        lp = _summarise([t for t in trades[name] if t.get("direction") == "long"])[2]
        sp = _summarise([t for t in trades[name] if t.get("direction") == "short"])[2]
        beats = sum(1 for a, c in zip(seg_pnl[name], base) if a > c)
        vs = "-" if name == "GNN_OFF" else f"{(pnl-base_pnl)/abs(base_pnl)*100:+.1f}%"
        bs = "-" if name == "GNN_OFF" else f"{beats}/{args.segments}"
        print(f"{name:<15}{cnt:<7}{wr:>6.1f}%  ${pnl:>+12,.0f}  {vs:>9}  "
              f"${lp:>+11,.0f}  ${sp:>+11,.0f}  {bs:>10}")
    print("-" * 96)

    print("\nPRE-REGISTERED GATE: beats GNN_OFF on total PnL, in >=5/6 segments,")
    print("and makes NEITHER direction worse in PnL.")
    base_l = _summarise([t for t in trades["GNN_OFF"] if t.get("direction") == "long"])[2]
    base_s = _summarise([t for t in trades["GNN_OFF"] if t.get("direction") == "short"])[2]
    winners = []
    for name in ARMS:
        if name == "GNN_OFF":
            continue
        pnl = sum(t["pnl_dollars"] for t in trades[name])
        lp = _summarise([t for t in trades[name] if t.get("direction") == "long"])[2]
        sp = _summarise([t for t in trades[name] if t.get("direction") == "short"])[2]
        beats = sum(1 for a, c in zip(seg_pnl[name], base) if a > c)
        a, bb, cc = pnl > base_pnl, beats >= 5, (lp >= base_l and sp >= base_s)
        ok = a and bb and cc
        if ok:
            winners.append(name)
        print(f"  {name:<15}(PnL){'PASS' if a else 'FAIL':<6}(segs){'PASS' if bb else 'FAIL':<6}"
              f"(dirs){'PASS' if cc else 'FAIL':<6} -> {'ADOPT' if ok else 'no'}")
    if winners:
        print(f"\nQualifying: {', '.join(winners)}")
        print("Remember the look-ahead contamination — a positive result here is WEAK and")
        print("needs walk-forward retraining before anyone acts on it.")
    else:
        print("\nNothing qualified. The GNN is uninformative to Path 1 at every threshold")
        print("tested, INCLUDING ungated — and it failed even with look-ahead contamination")
        print("working in its favour. It can now be retired on evidence rather than")
        print("assumption. Do not re-run with new thresholds hunting for a pass.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
