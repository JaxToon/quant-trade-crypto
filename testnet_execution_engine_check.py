"""
Drive the REAL ExecutionEngine against testnet — not a hand-written script.

`testnet_order_path_check.py` and `testnet_stop_order_check.py` proved the
exchange accepts our order shapes, but both used code I wrote for the test.
The code that would actually hold real money is `strategy/execution.py`, and
it has never run against a live exchange in this project's history — Path 2
has produced 3,050+ debate decisions and executed exactly zero trades.

This builds a synthetic TradeDecision and pushes it through
`ExecutionEngine.execute()` unchanged, so every real code path runs:
balance lookup, price lookup, quantity sizing, SL/TP price computation, the
market entry, both protective orders, and Position bookkeeping.

Two defects are already visible by reading, and this is here to confirm them
against the exchange rather than argue about them:

  1. SL/TP prices are computed with `round(price * x, 4)` — 4 decimal places,
     NOT the symbol's PRICE_FILTER tickSize. BTCUSDT's tick is 0.01, so a
     price like 76874.1234 is not a valid multiple and should be rejected.
  2. `fill_price = float(order.get("fills", [{}])[0].get("price", price))`
     reads only the FIRST fill. A market order that walks several price
     levels records a wrong entry price, which then corrupts the SL/TP
     distances and the recorded PnL.

Safety: LONG only by default. A SHORT here would issue a spot SELL of an
asset the account may not hold — `ExecutionEngine`'s own docstring says it
expects USDT-margined futures, and testnet spot is not that. Pass --short
deliberately if you want to see how that fails.

Everything is cleaned up at the end: protective orders cancelled, and the
position reported so it can be closed manually.

Usage: python testnet_execution_engine_check.py [--symbol BTCUSDT] [--short]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--short", action="store_true", help="try a SHORT (expected to fail on spot)")
    args = ap.parse_args()

    from binance.client import Client
    from strategy.agents import AgentDebate
    from strategy.consensus import TradeDecision
    from strategy.execution import ExecutionEngine

    engine = ExecutionEngine()
    print(f"engine endpoint: {engine._client.API_URL}")
    if "testnet" not in engine._client.API_URL:
        print("REFUSING TO RUN: engine is not pointed at testnet.")
        return

    sym = args.symbol
    step, tick, min_notional = engine._symbol_filters(sym)
    price = engine._get_price(sym)
    balance = engine._get_account_balance()
    print(f"balance: {balance:,.2f} USDT   {sym} price {price:,.2f}")
    print(f"filters: step={step} tick={tick} minNotional={min_notional}\n")

    qty = engine._compute_quantity(sym, balance, price)
    print(f"[sizing] _compute_quantity -> {qty!r}")
    if not qty:
        print("  engine refused to size this trade; nothing further to test.")
        return
    print(f"  notional at entry {float(qty)*price:,.2f}, "
          f"at stop leg {float(qty)*price*(1-settings.STOP_LOSS_PCT):,.2f}")

    # ---- inspect the SL/TP prices the engine WILL use, before sending ----
    direction = "SHORT" if args.short else "LONG"
    if direction == "LONG":
        sl = round(price * (1 - settings.STOP_LOSS_PCT), 4)
        tp = round(price * (1 + settings.TAKE_PROFIT_PCT), 4)
    else:
        sl = round(price * (1 + settings.STOP_LOSS_PCT), 4)
        tp = round(price * (1 - settings.TAKE_PROFIT_PCT), 4)
    on_tick = lambda v: abs(v / tick - round(v / tick)) < 1e-9
    print(f"\n[prices] engine will send SL={sl} TP={tp}  (tickSize={tick})")
    print(f"  SL is a valid multiple of tickSize: {on_tick(sl)}")
    print(f"  TP is a valid multiple of tickSize: {on_tick(tp)}")
    if not (on_tick(sl) and on_tick(tp)):
        print("  -> expect a PRICE_FILTER rejection; the engine rounds to 4dp, not to tick.")

    decision = TradeDecision(
        symbol=sym, p_trade=0.99, execute=True, direction=direction,
        debate=AgentDebate(symbol=sym, bull_score=0.9, bull_reasoning="synthetic",
                           bear_score=0.1, bear_reasoning="synthetic",
                           c_risk=0.0, critic_reasoning="synthetic"),
        gnn_signal=0.0,
    )

    print(f"\n[execute] pushing a synthetic {direction} through ExecutionEngine.execute() ...")
    pos = engine.execute(decision)
    if pos is None:
        print("  RESULT: execute() returned None — see the logged error above.")
    else:
        print(f"  RESULT: Position opened  entry={pos.entry_price} qty={pos.quantity} "
              f"SL={pos.sl_price} TP={pos.tp_price} order_id={pos.order_id}")
        print(f"  quantity type: {type(pos.quantity).__name__}")

    # ---- what actually landed on the exchange? --------------------------
    print("\n[verify] open orders on the exchange now:")
    open_orders = engine._client.get_open_orders(symbol=sym)
    if not open_orders:
        print("  NONE — if a position was opened, it is UNPROTECTED.")
    for o in open_orders:
        print(f"  {o['type']:<20} side={o['side']:<5} qty={o['origQty']} "
              f"stop={o.get('stopPrice')} price={o['price']} status={o['status']}")

    acct = engine._client.get_account()
    base = sym.replace("USDT", "")
    held = {b["asset"]: float(b["free"]) for b in acct["balances"]}
    print(f"\n[verify] balances now: {base}={held.get(base,0):.8f}  USDT={held.get('USDT',0):,.2f}")

    print("\n[cleanup] cancelling protective orders ...", end=" ", flush=True)
    for o in open_orders:
        try:
            engine._client.cancel_order(symbol=sym, orderId=o["orderId"])
        except Exception:
            pass
    print(f"cancelled {len(open_orders)}")
    if pos is not None:
        print(f"  NOTE: the entry FILLED, so {base} was actually bought/sold on testnet.")
        print("  That is expected — this test deliberately exercises a real fill.")


if __name__ == "__main__":
    main()
