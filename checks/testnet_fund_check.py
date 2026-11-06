"""Drive the fund's live path end to end on Binance TESTNET (play money).

Arms fund.broker.ExchangeBroker against testnet.binance.vision, runs one
fund day through fund.run_daily.run("live") with a throwaway ledger (so the
real data/fund_ledger_live.json stays untouched), then checks that the
exchange balances moved the way the plan said.

Usage: python -m checks.testnet_fund_check
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

from fund import broker as fbroker
from fund import config, run_daily


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    b = fbroker.ExchangeBroker(venue="binance_testnet", enabled=True, confirm_real_money=False)
    assert b.is_testnet, "refusing: not a testnet endpoint"
    syms = config.UNIVERSE
    listed = []
    for s in syms:
        try:
            b._filter(s)
            listed.append(s)
        except Exception as e:
            print(f"  {s}: not usable on testnet ({e})")
    q0, cash0 = b.positions(syms)
    print(f"before: USDT {cash0:,.2f}  " + "  ".join(f"{s}={q0[s]:g}" for s in syms))

    config.LIVE_LEDGER = os.path.join(tempfile.mkdtemp(), "fund_ledger_testnet.json")
    config.LIVE_CAPITAL = 1000.0  # the fund manages $1,000; the rest of the testnet account is not its money
    rec = run_daily.run("live", broker=b)
    print(f"plan: {len(rec['orders'])} orders, equity {rec['equity']:,.2f}")
    for o in rec["orders"]:
        print(f"  {o['side']} {o['symbol']} qty={o['qty']:.6f} ~${o['notional']:,.2f}")

    q1, cash1 = b.positions(syms)
    print(f"after:  USDT {cash1:,.2f}  " + "  ".join(f"{s}={q1[s]:g}" for s in syms))
    bad = []
    for o in rec["orders"]:
        moved = q1[o["symbol"]] - q0[o["symbol"]]
        want = o["qty"] if o["side"] == "BUY" else -o["qty"]
        step = b._filter(o["symbol"])[0]  # quantities are floored to the lot step
        if o["notional"] >= config.MIN_ORDER_USDT and abs(moved - want) > max(0.02 * abs(want), step * 1.01):
            bad.append((o["symbol"], want, moved))
    print("RESULT:", "OK, balances moved as planned" if not bad else f"MISMATCH {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
