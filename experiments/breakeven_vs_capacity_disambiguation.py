"""
Is BE_0.5R (breakeven stop at 0.5R) a genuine exit improvement, or just
another proxy for the correlation-cap capacity lever?

`experiments/trailing_breakeven_robustness.py` found BE_0.5R beats baseline
in 6/6 segments by +$246,931 (+85%) -- but with a telltale signature:
trade count roughly DOUBLES (682 -> 1288 in seg0, 554 -> 1273 in seg3)
while win rate CRATERS (59% -> 34%). That is the exact pattern
`correlation_cap_vs_short_sizing.py` already caught once: SHORT_20PCT
looked like a robust 6/6 winner, but the mechanism turned out to be
"smaller positions free up portfolio correlation-cap capacity, so more of
the same opportunity set gets captured" -- and raising
`PORTFOLIO_MAX_CORRELATED_RISK_PCT` directly beat it by 3.5-10x, proving
SHORT_20PCT "was a proxy for the real lever the whole time, not a finding
in its own right."

Breakeven stops plausibly do the same thing: they close positions EARLY
(at ~zero PnL, which is why win rate collapses -- a scratch counts as a
non-win), releasing correlation-cap budget sooner so more trades fit.

The disambiguation, same shape as the SHORT_20PCT one:

  BASELINE            cap 0.08, no breakeven
  BE_0.5R             cap 0.08, breakeven at 0.5R      (the candidate)
  CAP_1.5X            cap 0.12, no breakeven           (the known lever)
  CAP_2X              cap 0.16, no breakeven
  BE_0.5R_CAP_1.5X    cap 0.12, breakeven at 0.5R      (additive or not?)

If BE_0.5R alone ~= CAP_1.5X alone AND the combination isn't meaningfully
better than either, breakeven is the same lever wearing a different hat --
and the cap (already validated, already understood) is the honest way to
pull it. If the combination is clearly better than both, they're
independent and breakeven is a real, separate edge worth having.

Also reports max drawdown per arm: risk analysis is what flipped the
verdict on CAP_3X (best returns, 41.2% drawdown -> rejected), so a
returns-only comparison would repeat a mistake this project already made
and corrected.

Usage: python -m experiments.breakeven_vs_capacity_disambiguation [--segments 6]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# name -> (correlation cap, breakeven activation in R or None)
ARMS: dict[str, tuple] = {
    "BASELINE": (0.08, None),
    "BE_0.5R": (0.08, 0.5),
    "CAP_1.5X": (0.12, None),
    "CAP_2X": (0.16, None),
    "BE_0.5R_CAP_1.5X": (0.12, 0.5),
}


def _apply_breakeven(ledger: PaperTradingLedger, symbol: str, high: float, low: float, activation_r: float) -> None:
    for pos in ledger.open_positions.values():
        if pos.symbol != symbol:
            continue
        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if initial_risk <= 0:
            continue
        if pos.direction == "long":
            if (high - pos.entry_price) / initial_risk >= activation_r:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            if (pos.entry_price - low) / initial_risk >= activation_r:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)


def _metrics(trades: list[dict]) -> dict:
    """Same equity-curve drawdown walk correlation_cap_vs_short_sizing and
    backtest_concurrence_experiment use -- sizing is a fixed fraction of
    starting_balance, so these dollar figures are apples-to-apples."""
    if not trades:
        return {"trades": 0, "win_rate_pct": 0.0, "pnl": 0.0, "max_dd_pct": 0.0, "profit_factor": 0.0}
    wins = [t for t in trades if t["pnl_dollars"] > 0]
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in trades if t["pnl_dollars"] <= 0))
    equity = peak = settings.PAPER_TRADING_STARTING_BALANCE
    max_dd = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return {
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "pnl": sum(t["pnl_dollars"] for t in trades),
        "max_dd_pct": max_dd,
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
    }


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    ledgers = {}
    for name in ARMS:
        ledger = PaperTradingLedger(
            path=f"backtrack_logs/_bedisambig_{label}_{name}.json",
            starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
        )
        ledger.balance = ledger.starting_balance
        ledger.open_positions.clear()
        ledger.closed_trades.clear()
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

    orig_cap = settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT
    try:
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

                for name, (cap, be_r) in ARMS.items():
                    ledger = ledgers[name]
                    if be_r is not None:
                        _apply_breakeven(ledger, sym, candle.high, candle.low, be_r)
                    settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = cap
                    ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                    ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                    ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)
    finally:
        settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = orig_cap

    out = {}
    for name, ledger in ledgers.items():
        ledger.open_positions.clear()
        out[name] = _metrics(ledger.closed_trades)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — fetching {days}d of {timeframe}...\n", flush=True)
    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        c = fetch_candles(symbol, days, timeframe)
        if c:
            all_candles[symbol] = c
            print(f"  {symbol}: {len(c)} candles", flush=True)
    if not all_candles:
        print("No data — aborting.")
        return

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]

    agg = {name: {"pnl": 0.0, "trades": 0, "wins": 0.0, "seg_wins_vs_base": 0, "max_dd": 0.0} for name in ARMS}

    print(f"\n{'=' * 104}")
    print(f"BREAKEVEN vs CORRELATION-CAP CAPACITY — is BE_0.5R its own lever? ({n_seg} segments, {days}d)")
    print(f"{'=' * 104}", flush=True)

    for i in range(n_seg):
        seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()}
        res = run_segment(seg, timeframe, f"seg{i}")
        base_pnl = res["BASELINE"]["pnl"]
        print(f"\n--- seg{i} ---")
        print(f"{'Arm':<20}{'Trades':<9}{'WR':<9}{'PnL $':<16}{'MaxDD%':<10}{'PF':<8}")
        for name in ARMS:
            m = res[name]
            agg[name]["pnl"] += m["pnl"]
            agg[name]["trades"] += m["trades"]
            agg[name]["wins"] += m["win_rate_pct"] * m["trades"] / 100
            agg[name]["max_dd"] = max(agg[name]["max_dd"], m["max_dd_pct"])
            if name != "BASELINE" and m["pnl"] > base_pnl:
                agg[name]["seg_wins_vs_base"] += 1
            print(f"{name:<20}{m['trades']:<9}{m['win_rate_pct']:<9.1f}{m['pnl']:>+13,.0f}   {m['max_dd_pct']:<10.1f}{m['profit_factor']:<8.2f}", flush=True)

    print(f"\n{'=' * 104}")
    print(f"{'Arm':<20}{'Trades':<9}{'WR':<9}{'Total PnL':<18}{'WorstSegDD%':<14}{'Segs beat base':<16}")
    print("-" * 104)
    for name in ARMS:
        a = agg[name]
        wr = a["wins"] / a["trades"] * 100 if a["trades"] else 0
        beat = "-" if name == "BASELINE" else f"{a['seg_wins_vs_base']}/{n_seg}"
        print(f"{name:<20}{a['trades']:<9}{wr:<9.1f}${a['pnl']:>+14,.0f}    {a['max_dd']:<14.1f}{beat:<16}")
    print("-" * 104)
    print("READ THIS: if BE_0.5R ~= CAP_1.5X and BE_0.5R_CAP_1.5X isn't meaningfully better than")
    print("either alone, breakeven is the SAME correlation-cap-capacity lever in disguise -- exactly")
    print("what SHORT_20PCT turned out to be. Only a clearly additive combination makes it a real,")
    print("independent edge. And check MaxDD: CAP_3X was rejected on drawdown despite best returns.")
    print(f"{'=' * 104}")


if __name__ == "__main__":
    main()
