"""
Do exchange-held protective orders actually work? The safety-critical test.

`testnet_order_path_check.py` proved the basic lifecycle (authenticate ->
place -> read -> cancel). This tests the part the real-money plan actually
depends on.

Why it matters more than the basic path: this deployment has a MEASURED
history of multi-day outages — 2026-09-04 to 09-07, 2.5 days down, during
which two shorts breached their stops and were only resolved retroactively
by replaying OHLC. In paper trading that is a bookkeeping exercise. With
real money and stops living only inside a Python process on a laptop, those
positions would have run completely unprotected for the whole outage, with
losses bounded by nothing.

The fix is for the stop to rest ON the exchange, so protection survives the
process dying. Binance supports STOP_LOSS_LIMIT / TAKE_PROFIT_LIMIT / OCO,
and Tokocrypto's docs list the same set — which is exactly why Indodax was
disqualified as a venue (it rejects them outright with
`stop_limit_not_available`).

So: prove these order types are actually accepted and behave, rather than
assuming it from documentation.

  1. STOP_LOSS_LIMIT   sell-stop well BELOW market (won't trigger)
  2. TAKE_PROFIT_LIMIT sell-TP well ABOVE market (won't trigger)
  3. OCO               both at once, and confirm cancelling one kills both

Everything is priced far from the market so nothing executes, and every
order is cancelled at the end. Testnet money throughout.

Usage: python testnet_stop_order_check.py [--symbol BTCUSDT]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()
    sym = args.symbol

    from binance.client import Client
    from binance.exceptions import BinanceAPIException

    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, testnet=True)
    client.API_URL = Client.API_TESTNET_URL   # see execution.py — the flag alone doesn't do this

    acct = client.get_account()
    base = sym.replace("USDT", "")
    bal = {b["asset"]: float(b["free"]) for b in acct["balances"]}
    print(f"endpoint {client.API_URL}")
    print(f"balances: {base}={bal.get(base, 0):.8f}  USDT={bal.get('USDT', 0):,.2f}\n")

    info = client.get_symbol_info(sym)
    filt = {f["filterType"]: f for f in info["filters"]}
    tick = float(filt["PRICE_FILTER"]["tickSize"])
    step = float(filt["LOT_SIZE"]["stepSize"])
    min_notional = float(filt.get("NOTIONAL", filt.get("MIN_NOTIONAL", {})).get("minNotional", 0))
    pprec = max(0, len(f"{tick:.10f}".rstrip("0").split(".")[1]))
    qprec = max(0, len(f"{step:.10f}".rstrip("0").split(".")[1]))
    px = lambda v: f"{round(v / tick) * tick:.{pprec}f}"

    price = float(client.get_symbol_ticker(symbol=sym)["price"])
    print(f"{sym} last {price:,.2f}   allowed order types: {','.join(info['orderTypes'])}\n")

    # Size against the LOWEST price any leg will use (the SL limit at 0.68x),
    # not the current price — otherwise the stop leg fails MIN_NOTIONAL even
    # though the same quantity clears it at market. This is a real trap for
    # small accounts: the entry passes and the protective order silently
    # cannot be placed, leaving the position naked.
    lowest_leg_price = price * 0.68
    qty_f = max(min_notional / lowest_leg_price * 1.2, step)
    # MUST be a fixed-point string: a small float like 8e-05 serialises to
    # "8e-05" and the exchange rejects it with -1100 illegal characters.
    qty = f"{qty_f:.{qprec}f}"
    have_base = bal.get(base, 0) >= qty_f

    placed = []

    def attempt(label, **kw):
        print(f"  {label:<34}", end="", flush=True)
        try:
            o = client.create_order(symbol=sym, **kw)
            oid = o.get("orderId")
            placed.append(oid)
            print(f"PLACED   id={oid} status={o.get('status')}")
            return o
        except BinanceAPIException as e:
            print(f"REJECTED APIError {e.code}: {e.message}")
            return None
        except Exception as e:
            print(f"ERROR    {type(e).__name__}: {str(e)[:70]}")
            return None

    print("Order types the exchange advertises are listed above. Now actually place them:\n")

    if not have_base:
        print(f"  (no {base} balance — sell-side stops need the asset to sell.")
        print(f"   Validating shapes with create_test_order instead of placing.)\n")
        for label, kw in [
            ("STOP_LOSS_LIMIT (sell)", dict(side="SELL", type="STOP_LOSS_LIMIT", timeInForce="GTC",
                                            quantity=qty, price=px(price * 0.68), stopPrice=px(price * 0.70))),
            ("TAKE_PROFIT_LIMIT (sell)", dict(side="SELL", type="TAKE_PROFIT_LIMIT", timeInForce="GTC",
                                              quantity=qty, price=px(price * 1.32), stopPrice=px(price * 1.30))),
        ]:
            print(f"  {label:<34}", end="", flush=True)
            try:
                client.create_test_order(symbol=sym, **kw)
                print("ACCEPTED (shape valid)")
            except BinanceAPIException as e:
                print(f"REJECTED APIError {e.code}: {e.message}")
    else:
        attempt("STOP_LOSS_LIMIT (sell)", side="SELL", type="STOP_LOSS_LIMIT", timeInForce="GTC",
                quantity=qty, price=px(price * 0.68), stopPrice=px(price * 0.70))
        attempt("TAKE_PROFIT_LIMIT (sell)", side="SELL", type="TAKE_PROFIT_LIMIT", timeInForce="GTC",
                quantity=qty, price=px(price * 1.32), stopPrice=px(price * 1.30))

        # OCO — the one that matters most: SL and TP as a linked pair, so a fill
        # on either side automatically cancels the other. Without it, a filled TP
        # leaves an orphaned stop that can later sell an asset you no longer hold.
        print(f"  {'OCO (SL + TP linked)':<34}", end="", flush=True)
        try:
            # Binance moved OCO to above*/below* parameters; the old
            # price/stopPrice form now returns -1102 'aboveType was not sent'.
            oco = client.create_oco_order(
                symbol=sym, side="SELL", quantity=qty,
                aboveType="LIMIT_MAKER",
                abovePrice=px(price * 1.30),                  # TP leg
                belowType="STOP_LOSS_LIMIT",
                belowStopPrice=px(price * 0.70),              # SL trigger
                belowPrice=px(price * 0.68),                  # SL limit
                belowTimeInForce="GTC",
            )
            ids = [r["orderId"] for r in oco.get("orderReports", [])]
            print(f"PLACED   listOrderStatus={oco.get('listOrderStatus')} legs={ids}")
            print(f"  {'  -> cancel one leg':<34}", end="", flush=True)
            client.cancel_order(symbol=sym, orderId=ids[0])
            import time
            time.sleep(0.5)
            still = [o["orderId"] for o in client.get_open_orders(symbol=sym) if o["orderId"] in ids]
            print("BOTH legs gone (correct)" if not still else f"LEFTOVER {still} — orphan risk!")
        except BinanceAPIException as e:
            print(f"REJECTED APIError {e.code}: {e.message}")
        except Exception as e:
            print(f"ERROR    {type(e).__name__}: {str(e)[:70]}")

    # cleanup
    print("\ncleanup ...", end=" ", flush=True)
    open_now = client.get_open_orders(symbol=sym)
    for o in open_now:
        try:
            client.cancel_order(symbol=sym, orderId=o["orderId"])
        except Exception:
            pass
    print(f"cancelled {len(open_now)} remaining open order(s)")
    print(f"open orders now: {len(client.get_open_orders(symbol=sym))}")


if __name__ == "__main__":
    main()
