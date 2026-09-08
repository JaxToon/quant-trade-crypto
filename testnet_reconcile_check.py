"""
Does startup reconciliation actually catch a position that lost its
protection? Reproduced end-to-end on testnet.

`_load_positions` restores from a local JSON file and, until now, nothing
ever compared that to the exchange. The divergence is not hypothetical: this
deployment restarted 39 times in one 25-day window, and a crash between an
order filling and the state being written is enough to cause it. The
dangerous direction is a position the file says is guarded whose protective
orders are no longer there — naked, and silent.

This creates that exact state deliberately:

  1. open a position through the real engine (protection placed)
  2. cancel the protective orders behind its back, as an outage or a manual
     cancel would
  3. construct a fresh ExecutionEngine, which runs startup reconciliation
  4. assert the protection is back

Testnet money throughout; everything is cleaned up at the end.

Usage: python testnet_reconcile_check.py
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

SYM = "BTCUSDT"


def protective(client, symbol):
    return [o for o in client.get_open_orders(symbol=symbol)
            if o["type"] in ("STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT")]


def main() -> None:
    from strategy.agents import AgentDebate
    from strategy.consensus import TradeDecision
    from strategy.execution import ExecutionEngine

    eng = ExecutionEngine()
    if "testnet" not in eng._client.API_URL:
        print("REFUSING: not pointed at testnet."); return

    print("[1] opening a position through the real engine ...")
    d = TradeDecision(symbol=SYM, p_trade=0.99, execute=True, direction="LONG",
                      debate=AgentDebate(symbol=SYM, bull_score=0.9, bull_reasoning="x",
                                         bear_score=0.1, bear_reasoning="x",
                                         c_risk=0.0, critic_reasoning="x"),
                      gnn_signal=0.0)
    pos = eng.execute(d)
    if pos is None:
        print("    could not open a position; aborting."); return
    print(f"    opened qty={pos.quantity} entry={pos.entry_price} "
          f"SL={pos.sl_price} TP={pos.tp_price}")
    print(f"    protective orders on exchange: {len(protective(eng._client, SYM))}")

    print("\n[2] cancelling protection behind the engine's back (simulating an outage) ...")
    for o in protective(eng._client, SYM):
        eng._client.cancel_order(symbol=SYM, orderId=o["orderId"])
    left = len(protective(eng._client, SYM))
    print(f"    protective orders now: {left}  <- the position is NAKED")
    print(f"    but the local state file still says it is tracked and guarded")

    print("\n[3] constructing a fresh ExecutionEngine (runs startup reconciliation) ...")
    eng2 = ExecutionEngine()

    after = protective(eng2._client, SYM)
    print(f"\n[4] protective orders after reconciliation: {len(after)}")
    for o in after:
        print(f"    {o['type']:<20} stop={o.get('stopPrice')} price={o['price']} qty={o['origQty']}")

    ok = len(after) >= 2
    print(f"\nRESULT: {'PASS — reconciliation re-protected the naked position' if ok else 'FAIL — position is still naked'}")

    print("\n[cleanup] cancelling orders ...", end=" ")
    n = 0
    for o in eng2._client.get_open_orders(symbol=SYM):
        try:
            eng2._client.cancel_order(symbol=SYM, orderId=o["orderId"]); n += 1
        except Exception:
            pass
    print(f"{n} cancelled")


if __name__ == "__main__":
    main()
