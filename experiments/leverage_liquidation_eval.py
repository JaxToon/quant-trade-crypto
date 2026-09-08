"""
Does the futures path survive its own liquidation risk? The evaluation that
decides spot vs futures.

CORRECTING MYSELF FIRST. I claimed liquidation was "completely unmodelled in
every backtest". That was wrong — PaperTradingLedger computes a
liquidation_price for derivative positions and checks it BEFORE the stop in
both exit paths. The accurate statement is narrower and more useful:

    liquidation_price = entry * (1 - 1/leverage)     # long

At LEVERAGE=1, which is what every backtest here has run, that puts the
liquidation price at ZERO for longs and 2x entry for shorts. The mechanism
exists but has never once fired: 0 liquidations across every backtest trade
sampled. So the risk is not unmodelled — it is UNTESTED, because we never
ran the leverage that makes it bite.

Futures is the only way to execute the SHORT half of this strategy from
Indonesia (spot cannot short, and Bittime's futures API is reachable and
lists all 8 watchlist symbols). So the question this answers is: at the
leverage futures actually implies, does liquidation eat the edge?

TWO MODEL IMPROVEMENTS over the production formula, both making it stricter:

  1. Maintenance margin. Real liquidation happens BEFORE equity reaches zero,
     when it falls under the maintenance requirement. Binance charges roughly
     0.4-0.5% on majors at low leverage tiers, so
         liq = entry * (1 - 1/leverage + MMR)
     The production formula omits MMR and is therefore optimistic.

  2. Intra-bar ordering. check_and_close_ohlc already tests liquidation
     before the stop within a bar, which is the conservative order.

Arms: leverage 1 (current), 3, 5, 10.

PRE-REGISTERED READING (fixed before running):
  - If liquidations stay ~0 and drawdown is not materially worse, leverage is
    survivable and futures is worth the integration work.
  - If liquidations appear at 3-5x, futures is only viable at low leverage,
    and the extra return has to justify a risk spot does not carry at all.
  - If they appear even at 3x, futures is off the table for this strategy and
    the spot/LONG-only plan stands.

Usage: python -m experiments.leverage_liquidation_eval
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SLIP = 0.00119
RISK, CAP = 0.0025, 1.0          # the spot-viable config, held constant
LEVERAGES = [1, 3, 5, 10]
MMR = 0.005                      # maintenance margin rate, ~Binance majors
START = None


def _liq_price(entry: float, direction: str, lev: float) -> float | None:
    """Stricter than production: includes maintenance margin, so liquidation
    lands slightly BEFORE equity would hit zero — as it really does."""
    if lev <= 1:
        return None
    frac = 1.0 / lev - MMR
    return entry * (1 - frac) if direction == "long" else entry * (1 + frac)


def _max_dd(trades):
    eq = peak = START
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def _ruined(trades):
    eq = START
    for t in trades:
        eq += t["pnl_dollars"]
        if eq <= 0:
            return True
    return False


def run_segment(candles, tf, label):
    leds = {}
    settings.PAPER_TRADING_RISK_PCT = RISK
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = CAP
    for lev in LEVERAGES:
        led = PaperTradingLedger(path=f"backtrack_logs/_lev_{label}_{lev}.json",
                                 starting_balance=START, stop_slippage_pct=SLIP)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        leds[lev] = led

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}

    n = min(len(c) for c in candles.values())
    for t in range(n):
        for sym, cs in candles.items():
            c = cs[t]
            sig = smc[sym].update(c)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                    current_price=c.close, volume=c.volume)
            fvg[sym].update_all(c.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(
                current_price=c.close, trend=sig.get("market_structure", "unknown")),
                "timestamp": c.timestamp.isoformat()}
            ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
            at = c.timestamp.isoformat()
            for lev, led in leds.items():
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
                for market in ("spot", "derivative"):
                    rec = dict(ins[market])
                    if market == "derivative":
                        rec["suggested_leverage"] = lev
                    led.maybe_open(sym, market, rec, timeframe=tf, opened_at=at)
                # Override the production liquidation price with the stricter
                # one, on any derivative position opened this step.
                for key, p in led.open_positions.items():
                    if p.market == "derivative" and p.liquidation_price is not None:
                        strict = _liq_price(p.entry_price, p.direction, lev)
                        if strict is not None:
                            p.liquidation_price = strict
    return {lev: list(l.closed_trades) for lev, l in leds.items()}


def main():
    global START
    START = settings.PAPER_TRADING_STARTING_BALANCE
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf} (risk {RISK*100:.2f}%, cap {CAP:.1f}x, MMR {MMR*100:.1f}%)...\n",
          flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]
    tot = {l: [] for l in LEVERAGES}
    seg_pnl = {l: [] for l in LEVERAGES}
    seg_dd = {l: [] for l in LEVERAGES}
    ruins = {l: 0 for l in LEVERAGES}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for lev, trades in run_segment(part, tf, f"s{i}").items():
            tot[lev].extend(trades)
            seg_pnl[lev].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[lev].append(_max_dd(trades))
            ruins[lev] += 1 if _ruined(trades) else 0
        print(f"  seg{i} done", flush=True)

    base = sum(seg_pnl[1])
    print(f"\n{'=' * 92}")
    print(f"LEVERAGE / LIQUIDATION — does futures survive its own risk? ({days}d, {n_seg} seg)")
    print(f"{'=' * 92}")
    print(f"{'lev':<7}{'n':<7}{'WR':>6}  {'Total PnL':>13}{'%base':>8}  "
          f"{'liquidations':>13}  {'worstDD':>9}{'medDD':>8}  {'ruined':>8}{'prof':>7}")
    print("-" * 92)
    for lev in LEVERAGES:
        ts = tot[lev]
        if not ts:
            continue
        w = sum(1 for t in ts if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in ts)
        liq = sum(1 for t in ts if t.get("liquidated"))
        prof = sum(1 for v in seg_pnl[lev] if v > 0)
        tag = f"{lev}x" + ("*" if lev == 1 else "")
        print(f"{tag:<7}{len(ts):<7}{w/len(ts)*100:>5.1f}%  ${pnl:>+12,.0f}{pnl/base*100:>7.0f}%  "
              f"{liq:>6} ({liq/len(ts)*100:>4.2f}%)  {max(seg_dd[lev]):>8.1f}%"
              f"{statistics.median(seg_dd[lev]):>7.1f}%  {ruins[lev]}/{n_seg:<6}{prof}/{n_seg}")
    print("-" * 92)
    print("* current setting. liquidation price includes maintenance margin, so this is")
    print("  STRICTER than the production formula, which omits it.")

    print(f"\n{'=' * 92}")
    print("VERDICT")
    worst_liq = max((sum(1 for t in tot[l] if t.get('liquidated')) for l in LEVERAGES if l > 1),
                    default=0)
    any_ruin = any(ruins[l] for l in LEVERAGES if l > 1)
    if worst_liq == 0 and not any_ruin:
        print("  No liquidations at any tested leverage, no segment ruined. Liquidation is")
        print("  not what stops the futures path — the remaining objections are the")
        print("  integration cost and the fact that SHORT is still statistically unproven.")
    elif any_ruin:
        print("  Some leverage RUINED at least one segment. Futures is off the table for")
        print("  this strategy at those levels; read the table for where it breaks.")
    else:
        print(f"  Liquidations occur ({worst_liq} at the worst level) without ruining a")
        print("  segment. Futures is viable only at the leverages where they stay near")
        print("  zero, and the extra return must justify a risk spot does not carry.")
    print(f"{'=' * 92}")


if __name__ == "__main__":
    main()
