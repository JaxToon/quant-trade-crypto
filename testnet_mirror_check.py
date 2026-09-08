"""
Does the live mirror reproduce a real PaperTradingLedger decision on the
exchange, exactly?

The point of `strategy/live_mirror.py` is that live behaviour must be
IDENTICAL to what was validated. Path 1's evidence (profitable 6/6 segments;
live LONG n=82, +$166.85/trade, 95% CI [+$91.49, +$275.83]) describes a
system whose stops come from SMC/FVG structure, whose targets come from the
measured gap, and whose size comes from risk/stop-distance with conviction
scaling and a correlation cap. Any execution layer that recomputes those
from its own settings is trading a different strategy than the one measured.

So this test does not hand-build an order. It drives a real
`PaperTradingLedger` to produce a real `PaperPosition`, mirrors THAT, and
then checks the exchange agrees on quantity, stop and target to the tick.

Testnet only — the mirror refuses to arm against a non-testnet endpoint
unless LIVE_TRADING_CONFIRM_REAL_MONEY is also set.

Usage: python testnet_mirror_check.py
"""
from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from backtrack.paper_trading import PaperTradingLedger  # noqa: E402
from strategy.live_mirror import LiveOrderMirror  # noqa: E402

SYM = "BTCUSDT"


def main() -> None:
    mirror = LiveOrderMirror(enabled=True)
    if not mirror.enabled:
        print("mirror refused to arm — check BINANCE_TESTNET / credentials.")
        return
    url = mirror._client.API_URL
    print(f"endpoint: {url}")
    if "testnet" not in url:
        print("REFUSING: not testnet."); return

    price = float(mirror._client.get_symbol_ticker(symbol=SYM)["price"])
    print(f"{SYM} last {price:,.2f}\n")

    # ---- let the REAL ledger decide, with all its own guards -------------
    # Size the paper account to the REAL one. Otherwise the ledger sizes off a
    # $10k notion of capital, produces a notional the actual balance cannot
    # cover (spot has no leverage), and the test only ever proves the balance
    # guard fires. Matching them is also what a real deployment must do.
    real_balance = mirror._balance()
    print(f"testnet USDT balance: {real_balance:,.2f} - sizing the paper ledger to match")
    led = PaperTradingLedger(path=os.path.join(tempfile.mkdtemp(), "l.json"),
                             starting_balance=real_balance)
    rec = {
        "actionable": True,
        "direction": "long",
        "entry_zone": [price, price],
        # A structural stop ~2.5% away and a target ~5% away, i.e. the shape
        # entry_insight actually produces — NOT the flat 3%/6% ExecutionEngine
        # would have imposed.
        "stop_loss": round(price * 0.975, 2),
        "target": round(price * 1.05, 2),
        "confidence": 60.0,
        "suggested_leverage": 1,
    }
    reason = led.maybe_open(SYM, "spot", rec, timeframe="1h")
    pos = led.open_positions.get((SYM, "spot", "1h"))
    if pos is None:
        print(f"ledger declined to open: {reason}"); return

    print("[ledger decided]")
    print(f"    qty={pos.quantity:.8f}  entry={pos.entry_price:.2f}  "
          f"SL={pos.stop_loss:.2f}  TP={pos.target:.2f}")
    print(f"    notional ${pos.quantity * pos.entry_price:,.2f}, "
          f"risk ${pos.quantity * abs(pos.entry_price - pos.stop_loss):,.2f}")

    print("\n[mirroring to the exchange] ...")
    res = mirror.mirror_open(pos)
    print(f"    ok={res.ok}  reason={res.reason or '-'}")
    if res.ok:
        print(f"    entry_order_id={res.entry_order_id}  fill={res.fill_price}")
        print(f"    protection ids={res.protection_order_ids}")

    # ---- does the exchange agree with the ledger? -----------------------
    step, tick, _mn = mirror._symbol_filters(SYM)
    live = mirror._client.get_open_orders(symbol=SYM)
    print(f"\n[exchange state] {len(live)} resting order(s)")
    for o in live:
        print(f"    {o['type']:<18} side={o['side']:<5} qty={o['origQty']} "
              f"stop={o.get('stopPrice','-')} price={o['price']}")

    if res.ok:
        want_qty = mirror._fmt(pos.quantity, step)
        want_sl = mirror._fmt(pos.stop_loss, tick)
        want_tp = mirror._fmt(pos.target, tick)
        got_qty = {o["origQty"].rstrip("0").rstrip(".") for o in live}
        stops = {o.get("stopPrice", "").rstrip("0").rstrip(".") for o in live if o.get("stopPrice")}
        prices = {o["price"].rstrip("0").rstrip(".") for o in live}
        norm = lambda v: v.rstrip("0").rstrip(".")
        print("\n[agreement check]")
        print(f"    quantity  ledger {norm(want_qty):<12} exchange {sorted(got_qty)}")
        print(f"    stop      ledger {norm(want_sl):<12} exchange {sorted(stops)}")
        print(f"    target    ledger {norm(want_tp):<12} exchange {sorted(prices)}")
        ok = (norm(want_qty) in got_qty and norm(want_sl) in stops and norm(want_tp) in prices)
        print(f"\nRESULT: {'PASS — the exchange holds exactly what the ledger decided' if ok else 'MISMATCH — see above'}")

    print("\n[cleanup] ...", end=" ")
    n = 0
    for o in mirror._client.get_open_orders(symbol=SYM):
        try:
            mirror._client.cancel_order(symbol=SYM, orderId=o["orderId"]); n += 1
        except Exception:
            pass
    if res.ok:
        try:
            mirror._client.create_order(symbol=SYM, side="SELL", type="MARKET",
                                        quantity=mirror._fmt(pos.quantity, step))
            print(f"cancelled {n}, sold back {mirror._fmt(pos.quantity, step)}")
        except Exception as e:
            print(f"cancelled {n}, sell-back failed: {e}")
    else:
        print(f"cancelled {n}")


if __name__ == "__main__":
    main()
