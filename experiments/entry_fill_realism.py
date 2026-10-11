"""
How much of this strategy's profit comes from booking entries at a price the
market never offered?

THE SUSPECTED FLAW, found by reading the code (2026-10-11):

  strategy/fvg_analyzer.py — a bullish FVG spans [low, high], gap G. Its
  mitigation is measured from the top:  mitigation = (high - price) / G.
  get_trading_signal() requires is_valid_entry(): price inside the gap AND
  mitigation <= 50%. So a LONG signal can only fire while price sits in the
  UPPER half of the gap: price in [mid, high]. (Bearish mirrors it: a SHORT
  fires only with price in the LOWER half.)

  backtrack/paper_trading.py:281 — the ledger then books the entry at
      entry_price = sum(entry_zone) / 2      # the gap MIDPOINT

  So every long is booked at or BELOW the market price available when the
  signal fired, and every short at or ABOVE it. The ledger buys cheaper and
  sells dearer than the market allowed, on every trade, in both directions.

  Meanwhile strategy/live_mirror.py sends a MARKET order. Real money would
  fill at the market price — not at the midpoint the paper ledger records.

Back-of-envelope: if price is spread across the valid half-gap, the average
gift is ~0.25G against a booked risk of 0.7G (stop sits 0.2G beyond the
gap), i.e. ~0.36R per trade. On a strategy whose edge per trade is small,
that could be the whole edge. That is arithmetic, not measurement — so this
script measures it.

ARMS (identical signals, identical stops/targets, only the fill differs):

  AS_IS          booked at the gap midpoint (what every result so far used)
  MARKET         filled at the signal bar's close — what LiveOrderMirror
                 actually does with a market order
  LIMIT_AT_MID   a resting limit at the midpoint that fills ONLY if a later
                 bar actually trades through it, expiring after 3 bars —
                 the honest version of what the ledger claims to do

Run on the configuration a real deployment would use: Tokocrypto's real fee
(0.3822%/side), the spot config (risk 0.25%, notional cap 1.0x), measured
stop slippage (0.119%).

PRE-REGISTERED DECISION RULE (fixed before running):

  - If MARKET is unprofitable, or profitable in fewer than 4 of 6 segments,
    then the strategy as LiveOrderMirror would execute it has no
    demonstrated edge, and the mirror must not be armed with real money in
    its current form.
  - If LIMIT_AT_MID survives where MARKET does not, the deployable fix is
    resting limit orders, and the cost is the unfilled signals.
  - If both survive, the bias is real but not fatal; report its size.

Usage: python -m experiments.entry_fill_realism [--fee 0.001]

--fee sets the per-side fee (default 0.003822, Tokocrypto's published USDT
schedule). Added 2026-10-11 to ask whether a cheap venue rescues the
realistic-fill arms.
"""
from __future__ import annotations

import argparse
import importlib
import statistics

from config import settings

FEE = 0.003822            # Tokocrypto USDT market, per side, published schedule
SLIP = 0.00119            # measured live stop slippage, n=31
RISK, CAP = 0.0025, 1.0   # the spot-viable configuration
LIMIT_EXPIRY_BARS = 3
ARMS = ("AS_IS", "MARKET", "LIMIT_AT_MID")


def _max_dd(trades, start):
    eq = peak = start
    worst = 0.0
    for t in trades:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100)
    return worst


def main():
    # Fee feeds MIN_STOP_DISTANCE_PCT at import time — reload so the floor
    # moves with it, exactly as it would on the real venue.
    settings.PAPER_TRADING_FEE_PCT = FEE
    import backtrack.paper_trading as pt
    importlib.reload(pt)
    settings.PAPER_TRADING_RISK_PCT = RISK
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = CAP

    from backtest import YF_MAX_DAYS, fetch_candles
    from strategy.entry_insight import build_entry_insight
    from strategy.fvg_analyzer import FairValueGapAnalyzer
    from strategy.smart_money import SmartMoneyAnalyzer

    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    start = settings.PAPER_TRADING_STARTING_BALANCE
    print(f"Fetching {days}d of {tf} (fee {FEE*100:.4f}%/side, minStop "
          f"{pt.MIN_STOP_DISTANCE_PCT*100:.2f}%)...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]

    agg = {a: {"trades": [], "seg_pnl": [], "seg_dd": []} for a in ARMS}
    bias_r = []            # AS_IS: how far the booked entry beat the market, in R
    signals = filled = 0   # LIMIT_AT_MID fill rate

    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        leds = {}
        for a in ARMS:
            led = pt.PaperTradingLedger(path=f"backtrack_logs/_efr_{a}_{i}.json",
                                        starting_balance=start, stop_slippage_pct=SLIP)
            led.balance = led.starting_balance
            led.open_positions.clear(); led.closed_trades.clear()
            led.max_closed_trades = 1_000_000
            leds[a] = led
        pending = {}   # (sym, market) -> (rec_at_mid, expires_at_t, mid, direction)

        smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in part}
        fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in part}
        m = min(len(c) for c in part.values())
        for t in range(m):
            for sym, cs in part.items():
                c = cs[t]
                at = c.timestamp.isoformat()

                # Resolve pending limits against THIS bar before anything else.
                lim = leds["LIMIT_AT_MID"]
                for mkt in ("spot", "derivative"):
                    key = (sym, mkt)
                    if key not in pending:
                        continue
                    rec, expires, mid, dirn = pending[key]
                    touched = c.low <= mid if dirn == "long" else c.high >= mid
                    if touched:
                        before = len(lim.open_positions)
                        lim.maybe_open(sym, mkt, rec, timeframe=tf, opened_at=at)
                        if len(lim.open_positions) > before:
                            filled += 1
                        del pending[key]
                    elif t >= expires:
                        del pending[key]

                sig = smc[sym].update(c)
                hist = list(smc[sym].candles)
                if len(hist) >= 3:
                    fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                        current_price=c.close, volume=c.volume)
                fvg[sym].update_all(c.close)
                ctx = {**sig, **fvg[sym].get_trading_signal(
                    current_price=c.close, trend=sig.get("market_structure", "unknown")),
                    "timestamp": at}
                ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)

                for a, led in leds.items():
                    led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)

                for mkt in ("spot", "derivative"):
                    rec = ins[mkt]
                    if not rec.get("actionable") or not rec.get("entry_zone"):
                        continue
                    lo, hi = rec["entry_zone"]
                    mid = (lo + hi) / 2
                    dirn = rec.get("direction", "long")

                    # AS_IS — the existing behaviour, measuring its gift as it goes
                    as_is = leds["AS_IS"]
                    k = (sym, mkt, tf)
                    was_open = k in as_is.open_positions
                    as_is.maybe_open(sym, mkt, rec, timeframe=tf, opened_at=at)
                    if not was_open and k in as_is.open_positions:
                        p = as_is.open_positions[k]
                        risk = abs(p.entry_price - p.stop_loss)
                        if risk > 0:
                            sign = 1 if dirn == "long" else -1
                            bias_r.append((c.close - p.entry_price) * sign / risk)

                    # MARKET — fill at this bar's close, same stop and target
                    mrec = dict(rec)
                    mrec["entry_zone"] = [c.close, c.close]
                    leds["MARKET"].maybe_open(sym, mkt, mrec, timeframe=tf, opened_at=at)

                    # LIMIT_AT_MID — park a resting order; it fills only if traded
                    if (sym, mkt, tf) not in leds["LIMIT_AT_MID"].open_positions \
                            and (sym, mkt) not in pending:
                        lrec = dict(rec)
                        lrec["entry_zone"] = [mid, mid]
                        pending[(sym, mkt)] = (lrec, t + LIMIT_EXPIRY_BARS, mid, dirn)
                        signals += 1

        for a, led in leds.items():
            tr = list(led.closed_trades)
            agg[a]["trades"].extend(tr)
            agg[a]["seg_pnl"].append(sum(x["pnl_dollars"] for x in tr))
            agg[a]["seg_dd"].append(_max_dd(tr, start))
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 92}")
    print(f"ENTRY FILL REALISM — {days}d, {n_seg} segments, fee {FEE*100:.4f}%/side, "
          f"spot config")
    print(f"{'=' * 92}")
    if bias_r:
        print(f"AS_IS booked entry vs the market price available at the signal, in R:")
        print(f"  mean {statistics.mean(bias_r):+.3f}R   median {statistics.median(bias_r):+.3f}R   "
              f"positive (booked better than market) on "
              f"{sum(1 for v in bias_r if v > 0)/len(bias_r)*100:.0f}% of {len(bias_r)} entries")
    if signals:
        print(f"LIMIT_AT_MID fill rate: {filled}/{signals} signals actually traded through "
              f"the midpoint within {LIMIT_EXPIRY_BARS} bars ({filled/signals*100:.0f}%)")
    print()
    print(f"{'arm':<15}{'n':<7}{'WR':>6}  {'Total PnL':>13}  {'exp/trade':>10}  "
          f"{'worstDD':>8}  {'profitable segs':>16}")
    print("-" * 92)
    for a in ARMS:
        tr = agg[a]["trades"]
        if not tr:
            print(f"{a:<15}(no trades)"); continue
        w = sum(1 for x in tr if x["pnl_dollars"] > 0)
        pnl = sum(x["pnl_dollars"] for x in tr)
        prof = sum(1 for v in agg[a]["seg_pnl"] if v > 0)
        print(f"{a:<15}{len(tr):<7}{w/len(tr)*100:>5.1f}%  ${pnl:>+12,.0f}  ${pnl/len(tr):>+9.2f}  "
              f"{max(agg[a]['seg_dd']):>7.1f}%  {prof}/{n_seg:<15}")
    print("-" * 92)
    print("Segment PnL:")
    for a in ARMS:
        print(f"  {a:<13}" + "".join(f"${v:>+10,.0f}" for v in agg[a]["seg_pnl"]))

    print(f"\n{'=' * 92}")
    print("PRE-REGISTERED VERDICT")
    mk = agg["MARKET"]
    mk_pnl = sum(x["pnl_dollars"] for x in mk["trades"])
    mk_prof = sum(1 for v in mk["seg_pnl"] if v > 0)
    lm = agg["LIMIT_AT_MID"]
    lm_pnl = sum(x["pnl_dollars"] for x in lm["trades"])
    lm_prof = sum(1 for v in lm["seg_pnl"] if v > 0)
    mk_ok = mk_pnl > 0 and mk_prof >= 4
    lm_ok = lm_pnl > 0 and lm_prof >= 4
    print(f"  MARKET       PnL ${mk_pnl:+,.0f}, profitable {mk_prof}/6 -> "
          f"{'survives' if mk_ok else 'FAILS'}")
    print(f"  LIMIT_AT_MID PnL ${lm_pnl:+,.0f}, profitable {lm_prof}/6 -> "
          f"{'survives' if lm_ok else 'FAILS'}")
    if not mk_ok and not lm_ok:
        print("  -> Neither realistic fill survives. The strategy's measured profit was")
        print("     produced by booking prices the market never offered. No demonstrated")
        print("     edge remains; LiveOrderMirror must NOT be armed with real money.")
    elif not mk_ok and lm_ok:
        print("  -> Market orders destroy the edge; resting limits keep it. The deployable")
        print("     fix is limit entries at the midpoint, at the cost of unfilled signals.")
        print("     LiveOrderMirror (market orders) must NOT be armed as it stands.")
    elif mk_ok:
        print("  -> The edge survives realistic fills. The bias is real but not fatal;")
        print("     every prior PnL figure is still overstated by the AS_IS-vs-MARKET gap.")
    print(f"{'=' * 92}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--fee", type=float, default=FEE, help="per-side fee as a fraction")
    FEE = _ap.parse_args().fee
    main()
