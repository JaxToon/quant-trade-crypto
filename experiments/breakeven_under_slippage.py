"""
The deploy gate for the breakeven finding: does its edge survive the one
cost we have actually MEASURED but never modelled?

Chain so far. `trailing_breakeven_robustness.py` found BE_0.5R beat
baseline 6/6 segments (+85%). Three falsification passes followed:
  - look-ahead (`breakeven_lookahead_check.py`): REJECTED as the cause --
    the lagged, honest version was better, not worse.
  - concurrent exposure (`breakeven_exposure_check.py`): FAILED its
    pre-registered gate -- notional p95 +47%, peak positions 4 -> 12.
  - cap honesty (`breakeven_cap_honest_check.py`): found why. Breakeven
    overwriting `pos.stop_loss` zeroes that position's term in the
    correlation cap, so breakeven'd positions became invisible to the
    portfolio risk limit while still holding full notional. Applying it
    through `pos.trailing_stop` instead keeps the cap honest and costs
    ~52% of the measured edge. What remained was +78% over baseline,
    6/6 segments, worst-segment drawdown 23.2% vs 55.0%.

That remainder is why this file exists rather than a deployment. Breakeven
works by exiting earlier and more often: 3,355 -> 5,501 exits, the majority
of them stop-type. Meanwhile 4 real live SHORT stop-outs (2026-09-03)
filled 0.25-0.60% PAST the recorded stop, because a breached stop is a
market order. The backtest priced that at exactly zero. So the candidate
multiplies exposure to the single cost the measurement says is real and
the simulation says is free -- and at these notionals that differential is
plausibly the same order as the surviving edge.

`slippage_pct` could not answer this: it is symmetric across every fill,
and would tax baseline's target exits just as hard as BE's stop exits,
flattering exactly the arm under suspicion. `stop_slippage_pct` (added
alongside this file) charges stop exits only, leaving target exits filling
at their limit as a limit order does.

Note the residual bias runs the OTHER way and is not corrected here:
target exits are assumed to fill whenever a candle merely touches them,
which favours whichever arm takes more target exits -- the baseline. A BE
arm that wins here is therefore winning against a flattered opponent.

PRE-REGISTERED DECISION RULE (fixed before running; do not renegotiate
after seeing the numbers):

  Slippage levels: 0.0% (reproduction), 0.25% (measured low),
                   0.40% (measured midpoint -> THE GATE), 0.60% (measured high)

  DEPLOY BE_0.5R_CAP_HONEST only if, at 0.40%, ALL of:
    (a) total PnL still beats BASELINE, and
    (b) it beats BASELINE in >= 5 of 6 segments, and
    (c) worst-segment drawdown is still below BASELINE's.

  REJECT if (a) fails at 0.40%.
  PARTIAL/INCONCLUSIVE if (a) holds but (b) or (c) fails -> do not deploy,
  report as unresolved rather than reaching for a weaker threshold.

  Sanity condition: the edge must DEGRADE monotonically as slippage rises.
  If it does not, the slippage model itself is suspect and no conclusion
  may be drawn from this run.

  0.60% is reported as a stress case only -- it is NOT a deploy gate, and
  failing there while passing 0.40% is not grounds for rejection.

Usage: python -m experiments.breakeven_under_slippage [--segments 6]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# Cap-honest arms only. The cap-bypass variants are already settled as a
# risk-limit artifact and are deliberately not re-run here.
ARMS: dict[str, float | None] = {
    "BASELINE": None,
    "BE_0.5R": 0.5,
    "BE_0.3R": 0.3,
}
SLIPPAGE_LEVELS = [0.0, 0.0025, 0.0040, 0.0060]
GATE = 0.0040


def _apply_breakeven(ledger: PaperTradingLedger, symbol: str, high: float, low: float, activation_r: float) -> None:
    """Cap-honest breakeven: moves pos.trailing_stop (which _effective_stop
    already honours for exits) and leaves pos.stop_loss alone, so the
    correlation cap keeps seeing this position's ORIGINAL risk."""
    for pos in ledger.open_positions.values():
        if pos.symbol != symbol:
            continue
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 0:
            continue
        if pos.direction == "long":
            if (high - pos.entry_price) / risk >= activation_r:
                pos.trailing_stop = pos.entry_price if pos.trailing_stop is None else max(pos.trailing_stop, pos.entry_price)
        else:
            if (pos.entry_price - low) / risk >= activation_r:
                pos.trailing_stop = pos.entry_price if pos.trailing_stop is None else min(pos.trailing_stop, pos.entry_price)


def _max_dd(trades: list[dict]) -> float:
    equity = peak = settings.PAPER_TRADING_STARTING_BALANCE
    worst = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak * 100)
    return worst


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[tuple[str, float], list[dict]]:
    ledgers: dict[tuple[str, float], PaperTradingLedger] = {}
    for name in ARMS:
        for slip in SLIPPAGE_LEVELS:
            led = PaperTradingLedger(
                path=f"backtrack_logs/_beslip_{label}_{name}_{slip}.json",
                starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                stop_slippage_pct=slip,
            )
            led.balance = led.starting_balance
            led.open_positions.clear()
            led.closed_trades.clear()
            led.max_closed_trades = 1_000_000
            ledgers[(name, slip)] = led

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}
    prev_bar: dict[str, tuple[float, float]] = {}  # last COMPLETED bar per symbol

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            hist = list(smc.candles)
            if len(hist) >= 3:
                fvg.detect_fvg(candles=hist, index=len(hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            prior = prev_bar.get(sym)

            for (name, slip), led in ledgers.items():
                be_r = ARMS[name]
                if be_r is not None and prior is not None:
                    _apply_breakeven(led, sym, prior[0], prior[1], be_r)
                led.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                led.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                led.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            prev_bar[sym] = (candle.high, candle.low)

    out = {}
    for key, led in ledgers.items():
        led.open_positions.clear()
        out[key] = list(led.closed_trades)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    tf = args.timeframe
    days = min(args.days if args.days is not None else YF_MAX_DAYS.get(tf, 730), YF_MAX_DAYS.get(tf, 730))

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — fetching {days}d of {tf}...\n", flush=True)
    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        c = fetch_candles(symbol, days, tf)
        if c:
            all_candles[symbol] = c
    if not all_candles:
        print("No data — aborting.")
        return
    print(f"{len(all_candles)} symbols loaded.\n", flush=True)

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]

    keys = [(name, slip) for name in ARMS for slip in SLIPPAGE_LEVELS]
    trades_by: dict[tuple, list[dict]] = {k: [] for k in keys}
    seg_pnl: dict[tuple, list[float]] = {k: [] for k in keys}
    seg_dd: dict[tuple, list[float]] = {k: [] for k in keys}

    for i in range(n_seg):
        seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()}
        res = run_segment(seg, tf, f"seg{i}")
        for k, trades in res.items():
            trades_by[k].extend(trades)
            seg_pnl[k].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[k].append(_max_dd(trades))
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 108}")
    print(f"BREAKEVEN UNDER MEASURED STOP SLIPPAGE — the deploy gate ({days}d, {n_seg} seg)")
    print(f"{'=' * 108}")
    print(f"{'Slip':<8}{'Arm':<12}{'Trades':<9}{'StopExit%':<12}{'WR':<8}{'Total PnL':<17}"
          f"{'WorstSegDD':<13}{'vs base':<10}{'Segs>base':<10}")
    print("-" * 108)

    for slip in SLIPPAGE_LEVELS:
        base_key = ("BASELINE", slip)
        base_pnl = sum(t["pnl_dollars"] for t in trades_by[base_key])
        base_seg = seg_pnl[base_key]
        for name in ARMS:
            k = (name, slip)
            ts = trades_by[k]
            if not ts:
                continue
            pnl = sum(t["pnl_dollars"] for t in ts)
            wins = sum(1 for t in ts if t["pnl_dollars"] > 0)
            stops = sum(1 for t in ts if t.get("exit_type") in ("stop", "liquidation"))
            beats = sum(1 for a, b in zip(seg_pnl[k], base_seg) if a > b)
            vs = "-" if name == "BASELINE" else f"{(pnl - base_pnl) / abs(base_pnl) * 100:+.0f}%"
            bs = "-" if name == "BASELINE" else f"{beats}/{n_seg}"
            print(f"{slip * 100:<8.2f}{name:<12}{len(ts):<9}{stops / len(ts) * 100:<12.1f}"
                  f"{wins / len(ts) * 100:<8.1f}${pnl:>+14,.0f}   {max(seg_dd[k]):<13.1f}{vs:<10}{bs:<10}")
        print("-" * 108)

    # ---- pre-registered verdict ----
    print(f"\nPRE-REGISTERED GATE at stop slippage {GATE * 100:.2f}%:")
    base_k = ("BASELINE", GATE)
    base_pnl = sum(t["pnl_dollars"] for t in trades_by[base_k])
    base_dd = max(seg_dd[base_k])
    for name in ("BE_0.5R", "BE_0.3R"):
        k = (name, GATE)
        pnl = sum(t["pnl_dollars"] for t in trades_by[k])
        beats = sum(1 for a, b in zip(seg_pnl[k], seg_pnl[base_k]) if a > b)
        dd = max(seg_dd[k])
        a_ok, b_ok, c_ok = pnl > base_pnl, beats >= 5, dd < base_dd
        verdict = "DEPLOY" if (a_ok and b_ok and c_ok) else ("REJECT" if not a_ok else "INCONCLUSIVE — do not deploy")
        print(f"  {name}: (a) PnL>base {a_ok} (${pnl:+,.0f} vs ${base_pnl:+,.0f})  "
              f"(b) segs>=5 {b_ok} ({beats}/{n_seg})  (c) DD<base {c_ok} ({dd:.1f}% vs {base_dd:.1f}%)")
        print(f"     -> {verdict}")

    print("\nSANITY — edge must degrade monotonically as slippage rises:")
    for name in ("BE_0.5R", "BE_0.3R"):
        edges = []
        for slip in SLIPPAGE_LEVELS:
            b = sum(t["pnl_dollars"] for t in trades_by[("BASELINE", slip)])
            e = sum(t["pnl_dollars"] for t in trades_by[(name, slip)])
            edges.append(e - b)
        mono = all(edges[i] >= edges[i + 1] for i in range(len(edges) - 1))
        print(f"  {name}: " + "  ".join(f"{s*100:.2f}%=${v:+,.0f}" for s, v in zip(SLIPPAGE_LEVELS, edges))
              + f"   monotonic={mono}")
    print(f"{'=' * 108}")


if __name__ == "__main__":
    main()
