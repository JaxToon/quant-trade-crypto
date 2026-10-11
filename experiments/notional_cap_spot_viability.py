"""
What does it cost to make this strategy fit inside a SPOT account?

`spot_viable_risk_sizing.py` showed the obvious lever does not work: cutting
risk per trade from 2.00% to 0.50% left peak total notional essentially
unchanged (17.2x -> 16.5x of the account), because
PORTFOLIO_MAX_CORRELATED_RISK_PCT caps RISK, and the freed budget simply
admitted more positions (3,364 -> 8,015 trades). Only 20-24% of replay steps
ever had everything open fitting inside the account.

Spot cannot borrow, so strategy/live_mirror.py must reject any decision whose
notional exceeds the free balance. The lever that actually binds capital is a
cap on TOTAL OPEN NOTIONAL, which this project never had. This measures what
imposing one costs.

  0    no cap (current behaviour)
  3.0  3x the account — futures territory, included for shape
  2.0  2x
  1.5  1.5x
  1.0  fully spot-executable: everything open fits inside the account

Reported per level: PnL, win rate, drawdown, and the share of replay steps
that were spot-executable. The honest question is how much return is
surrendered to make the strategy physically placeable on spot.

Usage: python -m experiments.notional_cap_spot_viability
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
# (risk_pct, notional_cap). Neither lever works alone: capping risk leaves
# total notional unchanged (more positions fill the gap), and capping notional
# at 1.0x blocks almost every position because a single one is typically >1x
# the account at the production 2% risk. The combination is the untested case
# — smaller positions that individually fit, with a cap bounding how many run
# at once.
COMBOS = [
    (0.020, 0.0),    # production, uncapped
    (0.020, 1.0),    # cap alone
    (0.010, 1.0),
    (0.005, 1.0),
    (0.0025, 1.0),
    (0.005, 0.0),    # risk alone
]
CAPS = COMBOS
START = None


def _max_dd(trades):
    eq = peak = START
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def _p(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0.0


def run_segment(candles, tf, label):
    leds, lev = {}, {}
    original = (settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE, settings.PAPER_TRADING_RISK_PCT)
    for c_ in CAPS:
        led = PaperTradingLedger(path=f"backtrack_logs/_ncs_{label}_{c_}.json",
                                 starting_balance=START, stop_slippage_pct=SLIP)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        leds[c_] = led
        lev[c_] = []

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}

    n = min(len(x) for x in candles.values())
    for t in range(n):
        for sym, cs in candles.items():
            cd = cs[t]
            sig = smc[sym].update(cd)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                    current_price=cd.close, volume=cd.volume)
            fvg[sym].update_all(cd.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(
                current_price=cd.close, trend=sig.get("market_structure", "unknown")),
                "timestamp": cd.timestamp.isoformat()}
            ins = build_entry_insight(sym, cd.close, ctx, gnn_signal=0.0)
            at = cd.timestamp.isoformat()
            for c_, led in leds.items():
                settings.PAPER_TRADING_RISK_PCT, settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = c_[0], c_[1]
                led.check_and_close_ohlc(sym, cd.low, cd.high, closed_at=at, timeframe=tf)
                led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=at)
        for c_, led in leds.items():
            lev[c_].append(sum(p.quantity * p.entry_price for p in led.open_positions.values()) / START)
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE, settings.PAPER_TRADING_RISK_PCT = original
    return {c_: (list(l.closed_trades), lev[c_]) for c_, l in leds.items()}


def main():
    global START
    START = settings.PAPER_TRADING_STARTING_BALANCE
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf}...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]
    tot = {c: [] for c in CAPS}
    seg_pnl = {c: [] for c in CAPS}
    seg_dd = {c: [] for c in CAPS}
    lev = {c: [] for c in CAPS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for c_, (trades, l) in run_segment(part, tf, f"s{i}").items():
            tot[c_].extend(trades)
            seg_pnl[c_].append(sum(t["pnl_dollars"] for t in trades))
            seg_dd[c_].append(_max_dd(trades))
            lev[c_].extend(l)
        print(f"  seg{i} done", flush=True)

    base = sum(seg_pnl[COMBOS[0]])
    print(f"\n{'=' * 96}")
    print(f"TOTAL-NOTIONAL CAP — cost of making the strategy spot-executable ({days}d, {n_seg} seg)")
    print(f"{'=' * 96}")
    print(f"{'cap':<9}{'n':<7}{'WR':>6}  {'Total PnL':>13}{'%base':>8}  "
          f"{'peak lev':>9}{'p95 lev':>9}  {'spot-OK':>9}  {'worstDD':>9}{'medDD':>8}  {'prof':>6}")
    print("-" * 96)
    for c_ in CAPS:
        ts = tot[c_]
        if not ts:
            continue
        w = sum(1 for t in ts if t["pnl_dollars"] > 0)
        pnl = sum(t["pnl_dollars"] for t in ts)
        ok = sum(1 for v in lev[c_] if v <= 1.0) / len(lev[c_]) * 100
        prof = sum(1 for v in seg_pnl[c_] if v > 0)
        tag = f"{c_[0]*100:.2f}%/" + ("none" if c_[1] == 0.0 else f"{c_[1]:.1f}x")
        print(f"{tag:<9}{len(ts):<7}{w/len(ts)*100:>5.1f}%  ${pnl:>+12,.0f}{pnl/base*100:>7.0f}%  "
              f"{max(lev[c_]):>9.2f}{_p(lev[c_],0.95):>9.2f}  {ok:>8.1f}%  "
              f"{max(seg_dd[c_]):>8.1f}%{statistics.median(seg_dd[c_]):>7.1f}%  {prof}/{n_seg}")
    print("-" * 96)
    print("* current behaviour.  lev = TOTAL open notional / account.  "
          "spot-OK = steps where it all fit.")

    print("\nSPOT-VIABLE CANDIDATES (spot-OK >= 95%, profitable >= 5/6 segments):")
    found = False
    for c_ in CAPS:
        if not tot[c_]:
            continue
        ok = sum(1 for v in lev[c_] if v <= 1.0) / len(lev[c_]) * 100
        prof = sum(1 for v in seg_pnl[c_] if v > 0)
        if ok >= 95 and prof >= 5:
            found = True
            print(f"  risk {c_[0]*100:.2f}% + cap {c_[1]:.1f}x -> "
                  f"{sum(seg_pnl[c_])/base*100:.0f}% of production return, "
                  f"{len(tot[c_])} trades, medDD {statistics.median(seg_dd[c_]):.1f}%, "
                  f"worstDD {max(seg_dd[c_]):.1f}%")
    if not found:
        print("  NONE — constraining this strategy to spot is not a tuning problem.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
