"""
What happens when orders FAIL? The paths that have never been exercised.

Every check so far has tested the happy path. But the failure observed on
2026-09-08 — entry filled, protection rejected, exception swallowed,
position left naked and unrecorded — was a failure path, and it was found
only by accident. The others have still never run.

Each case below asks the same two questions:

  1. Does the failure surface, or is it swallowed?
  2. Is the state CLEAN afterwards — no orphan orders resting, no asset
     bought that nothing is tracking?

Question 2 is the one that matters with real money. A rejected order is
harmless; a half-completed sequence is not.

Cases:
  A  quantity below the exchange's minimum       (should refuse pre-flight)
  B  notional the balance cannot cover           (should refuse pre-flight)
  C  price far off the tick grid                 (exchange-side rejection)
  D  unknown symbol                              (exchange-side rejection)
  E  protection rejected after entry fills       (must roll back — the 09-08 bug)
  F  cancelling an order that no longer exists   (idempotence)

Testnet only.

Usage: python testnet_failure_paths_check.py
"""
from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from strategy.live_mirror import LiveOrderMirror  # noqa: E402

SYM = "BTCUSDT"


@dataclass
class _Pos:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    target: float
    quantity: float


def _state(mirror, label):
    """Anything left resting, and how much base asset is held."""
    orders = mirror._client.get_open_orders(symbol=SYM)
    base = SYM.replace("USDT", "")
    held = mirror._balance(base)
    print(f"      state after: {len(orders)} resting order(s), {base} free={held:.8f}")
    return len(orders)


def main() -> None:
    mirror = LiveOrderMirror(enabled=True)
    if not mirror.enabled or "testnet" not in mirror._client.API_URL:
        print("REFUSING: not armed against testnet."); return

    price = float(mirror._client.get_symbol_ticker(symbol=SYM)["price"])
    step, tick, min_notional = mirror._symbol_filters(SYM)
    base_held_before = mirror._balance(SYM.replace("USDT", ""))
    print(f"{SYM} {price:,.2f}  step={step} tick={tick} minNotional={min_notional}")
    print(f"starting BTC free={base_held_before:.8f}\n")

    results = []

    def case(name, expectation, fn):
        print(f"[{name}] {expectation}")
        try:
            res = fn()
            ok = res.ok if hasattr(res, "ok") else bool(res)
            reason = getattr(res, "reason", "")
            print(f"      -> ok={ok}  {reason[:100]}")
        except Exception as exc:
            print(f"      -> RAISED {type(exc).__name__}: {str(exc)[:90]}")
            ok, reason = None, str(exc)
        left = _state(mirror, name)
        results.append((name, ok, left))
        print()

    # A — below minimum quantity
    case("A below-min qty", "should refuse BEFORE touching the exchange",
         lambda: mirror.mirror_open(_Pos(SYM, "long", price, price * 0.975,
                                         price * 1.05, step)))

    # B — notional larger than the balance
    free = mirror._balance()
    case("B over-balance", "should refuse BEFORE touching the exchange",
         lambda: mirror.mirror_open(_Pos(SYM, "long", price, price * 0.975,
                                         price * 1.05, (free * 5) / price)))

    # C — a stop price nowhere near the tick grid
    qty = max(min_notional / (price * 0.975) * 1.3, step)
    case("C off-tick stop", "exchange rejects; entry must not be left naked",
         lambda: mirror.mirror_open(_Pos(SYM, "long", price,
                                         price * 0.975 + 0.001234567,
                                         price * 1.05 + 0.007654321, qty)))

    # D — a symbol that does not exist
    case("D unknown symbol", "should fail cleanly, no state change",
         lambda: mirror.mirror_open(_Pos("NOTREALUSDT", "long", 100.0, 97.5, 105.0, 1.0)))

    # E — protection rejected after the entry fills (the 2026-09-08 bug)
    print("[E protection-fails-after-entry] the failure that left a naked position")
    original = mirror._client.create_oco_order
    mirror._client.create_oco_order = lambda **kw: (_ for _ in ()).throw(
        RuntimeError("APIError(code=-1013): Filter failure: PRICE_FILTER"))
    try:
        res = mirror.mirror_open(_Pos(SYM, "long", price, price * 0.975, price * 1.05, qty))
        print(f"      -> ok={res.ok}  {res.reason[:100]}")
    finally:
        mirror._client.create_oco_order = original
    after = mirror._balance(SYM.replace("USDT", ""))
    left = _state(mirror, "E")
    rolled_back = abs(after - base_held_before) < float(mirror._fmt(qty, step)) * 0.5
    print(f"      BTC before={base_held_before:.8f} after={after:.8f}  "
          f"ROLLED BACK: {rolled_back}")
    results.append(("E protection-fails", False, left))
    print()

    # F — cancelling something that is already gone
    print("[F cancel-nonexistent] should not raise or corrupt state")
    try:
        mirror._client.cancel_order(symbol=SYM, orderId=999999999)
        print("      -> returned without error")
    except Exception as exc:
        print(f"      -> raised {type(exc).__name__} (acceptable if handled by callers)")
    _state(mirror, "F")

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY — the question that matters is the LAST column")
    print("=" * 72)
    print(f"{'case':<28}{'refused/failed':>16}{'orders left':>14}")
    print("-" * 72)
    clean = True
    for name, ok, left in results:
        if left:
            clean = False
        print(f"{name:<28}{str(ok):>16}{left:>14}")
    print("-" * 72)
    final = mirror._balance(SYM.replace("USDT", ""))
    print(f"BTC free: {base_held_before:.8f} -> {final:.8f}")
    verdict = ("CLEAN: no orphan orders left by any failure path" if clean
               else "DIRTY: a failure path left resting orders behind")
    print("")
    print(verdict)
    if abs(final - base_held_before) > 1e-8:
        print("WARNING: base-asset balance changed across the run — a failed path")
        print("         may have left an untracked position.")
    print("=" * 72)


if __name__ == "__main__":
    main()
