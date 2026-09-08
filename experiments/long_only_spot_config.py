"""
Can we deploy LONG-ONLY right now, and stop waiting for the SHORT question?

The blunt version of the real-money question. Live evidence already says:

  LONG   n=82  +$166.85/trade  95% CI [+$91.49, +$275.83]  PROVEN profitable
  SHORT  n=44  -$112.36/trade  95% CI [-$399.28, +$66.87]  UNRESOLVED

Standard practice is to deploy what is validated and leave out what is not.
That is not the same as "turn SHORT off because it looks bad" — it is
"don't deploy an unvalidated component with real money".

The one legitimate objection: live LONG covers ~25 days of a SINGLE regime
(the logs show HIGH_STRESS_TURBULENT with the bear veto active throughout).
A long-only book looking good in one favourable stretch is exactly what a
regime-fragile strategy produces. RESEARCH_NOTES also records a 2026-08-30
decision to KEEP short because "LONG_ONLY robustly loses" — but that was
measured with the GNN signal active, which has since been shown inert
(gated out on 98.9% of calls), and before stop slippage was modelled at all.

So: re-test it properly. Full 730d / 6-segment bar, realistic slippage
(0.119%, the measured live figure), per-segment so regime fragility is
visible rather than averaged away.

PRE-REGISTERED DECISION RULE (fixed before running):

  DEPLOY LONG_ONLY if it is profitable in >= 5 of 6 segments AND its worst
  segment is no worse than BASELINE's worst segment. Total PnL may be lower
  than baseline — that is an acceptable price for dropping an unvalidated
  component.

  REJECT if it loses in >= 2 segments — that would mean long-only really is
  regime-fragile, the 2026-08-30 decision stands on its own merits, and
  waiting for SHORT data is genuinely the only path.

Note this is a BACKTEST, and the whole reason this question is open is that
the backtest does not describe live SHORT behaviour. It describes LONG
adequately though — matched-window live LONG (73.7% WR) actually beat
backtest LONG (61.1%), so the backtest is if anything pessimistic about
longs. That makes it a fair, conservative instrument for THIS question.

Usage: python -m experiments.long_only_viability
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SLIP = 0.00119          # measured live mean stop slippage
# Re-run under the SPOT configuration. The original run used production
# sizing (2% risk, no notional cap) and found LONG_ONLY ruinous at 126.7%
# worst-segment drawdown. But LiveOrderMirror can only place spot longs —
# 60.8% of live trades — so mirroring IS long-only in practice, and the
# question has to be re-asked at the sizing that spot can actually hold.
SPOT_RISK, SPOT_CAP = 0.0025, 1.0
ARMS = ("BASELINE", "LONG_ONLY", "SHORT_ONLY")   # short-only as a reference point


def _max_dd(trades):
    eq = peak = settings.PAPER_TRADING_STARTING_BALANCE
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def run_segment(candles, tf, label):
    leds = {}
    settings.PAPER_TRADING_RISK_PCT = SPOT_RISK
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = SPOT_CAP
    for a in ARMS:
        led = PaperTradingLedger(path=f"backtrack_logs/_lov_{label}_{a}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                                 stop_slippage_pct=SLIP)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        leds[a] = led

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
            for a, led in leds.items():
                led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
                for market in ("spot", "derivative"):
                    rec = ins[market]
                    d = rec.get("direction")
                    if a == "LONG_ONLY" and d != "long":
                        continue
                    if a == "SHORT_ONLY" and d != "short":
                        continue
                    led.maybe_open(sym, market, rec, timeframe=tf, opened_at=at)
    return {a: list(l.closed_trades) for a, l in leds.items()}


def main():
    tf = "1h"
    days = YF_MAX_DAYS.get(tf, 730)
    n_seg = 6
    print(f"Fetching {days}d of {tf} (slippage {SLIP*100:.3f}%)...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]
    seg_pnl = {a: [] for a in ARMS}
    seg_dd = {a: [] for a in ARMS}
    tot = {a: [] for a in ARMS}

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        for a, tr in run_segment(part, tf, f"s{i}").items():
            seg_pnl[a].append(sum(t["pnl_dollars"] for t in tr))
            seg_dd[a].append(_max_dd(tr))
            tot[a].extend(tr)
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 88}")
    print(f"LONG-ONLY VIABILITY UNDER THE SPOT CONFIG — {days}d, {n_seg} segments, {tf}, slippage {SLIP*100:.3f}%")
    print(f"{'=' * 88}")
    print(f"{'Arm':<13}{'n':<7}{'WR':>7}  {'Total PnL':>13}  {'worstDD':>9}  {'profitable':>11}")
    print("-" * 88)
    for a in ARMS:
        tr = tot[a]
        if not tr:
            print(f"{a:<13}(no trades)"); continue
        w = sum(1 for t in tr if t["pnl_dollars"] > 0)
        prof = sum(1 for v in seg_pnl[a] if v > 0)
        print(f"{a:<13}{len(tr):<7}{w/len(tr)*100:>6.1f}%  ${sum(t['pnl_dollars'] for t in tr):>+12,.0f}  "
              f"{max(seg_dd[a]):>8.1f}%  {prof}/{n_seg:<11}")
    print("-" * 88)
    print(f"\n{'Segment PnL':<13}" + "".join(f"{'seg'+str(i):>13}" for i in range(n_seg)))
    for a in ARMS:
        print(f"{a:<13}" + "".join(f"${v:>+12,.0f}" for v in seg_pnl[a]))

    print(f"\n{'=' * 88}")
    print("PRE-REGISTERED GATE: LONG_ONLY profitable in >=5/6 segments AND its worst")
    print("segment no worse than BASELINE's worst segment.")
    lp, bp = seg_pnl["LONG_ONLY"], seg_pnl["BASELINE"]
    prof = sum(1 for v in lp if v > 0)
    a_ok, b_ok = prof >= 5, min(lp) >= min(bp)
    print(f"  (a) profitable {prof}/6 -> {'PASS' if a_ok else 'FAIL'}")
    print(f"  (b) worst segment ${min(lp):+,.0f} vs baseline ${min(bp):+,.0f} -> {'PASS' if b_ok else 'FAIL'}")
    if a_ok and b_ok:
        print("\n  -> DEPLOY LONG_ONLY. The proven component stands on its own across")
        print("     regimes; there is no need to wait on the SHORT question to start.")
    elif prof <= 4:
        print("\n  -> REJECT. Long-only is regime-fragile, the 2026-08-30 decision to keep")
        print("     SHORT stands on its own merits, and waiting for SHORT data really is")
        print("     the only honest path.")
    else:
        print("\n  -> INCONCLUSIVE. Do not deploy on a partial pass; report it as unresolved.")
    print(f"{'=' * 88}")


if __name__ == "__main__":
    main()
