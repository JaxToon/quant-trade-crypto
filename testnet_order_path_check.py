"""
End-to-end check of the real order path against Binance testnet.

This project has never sent a single order — not one, on any venue. That is
a hard blocker on real money regardless of how good the strategy looks:
fills, partial fills, rejections, precision/step-size rules and exchange
errors are all completely unexercised code paths.

RESEARCH_NOTES records the credentials failing with APIError -2015 on
2026-09-04. This re-checks that, and if authentication now works, walks the
full lifecycle against testnet money:

  1. authenticate                (GET /api/v3/account)
  2. read balances
  3. read symbol filters         (MIN_NOTIONAL, LOT_SIZE, PRICE_FILTER)
  4. VALIDATE an order           (POST /api/v3/order/test — places nothing)
  5. place a real testnet order  (a limit order far from the market so it
                                  rests instead of filling)
  6. read it back
  7. cancel it

Everything runs against testnet.binance.vision with fake funds. Step 5 is
deliberately a resting limit order priced well away from the market, so
nothing executes and the state is easy to clean up.

Usage: python testnet_order_path_check.py [--symbol BTCUSDT] [--place]
       (without --place it stops after step 4, the read-only checks)
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--place", action="store_true",
                    help="also place, read back and cancel a real resting testnet order")
    args = ap.parse_args()

    from binance.client import Client
    from binance.exceptions import BinanceAPIException

    if not settings.BINANCE_API_KEY or not settings.BINANCE_API_SECRET:
        print("FAIL: no API key/secret configured in .env")
        return

    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, testnet=True)
    # python-binance's testnet flag does NOT redirect spot REST (its docstring
    # says "only available for vanilla options at the moment"), so pin it —
    # same fix as strategy/execution.py. Without this the testnet key is sent
    # to production and comes back -2015, which looks exactly like a bad key.
    client.API_URL = Client.API_TESTNET_URL
    print(f"endpoint: {client.API_URL}")
    print(f"testnet : {settings.BINANCE_TESTNET}\n")

    # ---- 1. authenticate -------------------------------------------------
    print("[1] authenticate ...", end=" ", flush=True)
    try:
        acct = client.get_account()
        print("OK")
    except BinanceAPIException as e:
        print(f"FAIL  APIError {e.code}: {e.message}")
        if e.code == -2015:
            print("\n  -2015 means the key is invalid, or not a TESTNET key, or its IP")
            print("  restriction excludes this machine. Testnet keys are generated at")
            print("  https://testnet.binance.vision (login with GitHub) and are separate")
            print("  from production keys — a production key will always fail here.")
        return
    except Exception as e:
        print(f"FAIL  {type(e).__name__}: {str(e)[:120]}")
        return

    print(f"    canTrade={acct.get('canTrade')}  accountType={acct.get('accountType')}")

    # ---- 2. balances -----------------------------------------------------
    print("[2] balances ...")
    funded = [b for b in acct.get("balances", []) if float(b["free"]) + float(b["locked"]) > 0]
    if not funded:
        print("    (no funded assets — testnet accounts start empty; the faucet is on the")
        print("     testnet site. Order placement will fail with insufficient balance.)")
    for b in sorted(funded, key=lambda x: -float(x["free"]))[:6]:
        print(f"    {b['asset']:<6} free={float(b['free']):>14,.8f}  locked={float(b['locked']):>12,.8f}")

    # ---- 3. symbol filters ----------------------------------------------
    print(f"[3] filters for {args.symbol} ...")
    try:
        info = client.get_symbol_info(args.symbol)
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {str(e)[:100]}")
        return
    if not info:
        print(f"    {args.symbol} not listed on testnet")
        return
    filt = {f["filterType"]: f for f in info["filters"]}
    tick = float(filt["PRICE_FILTER"]["tickSize"])
    step = float(filt["LOT_SIZE"]["stepSize"])
    min_notional = float(filt.get("NOTIONAL", filt.get("MIN_NOTIONAL", {})).get("minNotional", 0))
    print(f"    tickSize={tick}  stepSize={step}  minNotional={min_notional}")
    print(f"    -> this is the real constraint on small capital: every order must clear")
    print(f"       ${min_notional:,.2f} notional and snap to those increments.")

    price = float(client.get_symbol_ticker(symbol=args.symbol)["price"])
    print(f"    last price: {price:,.2f}")

    # A resting BUY far below market: won't fill, easy to cancel.
    limit_price = round(price * 0.70 / tick) * tick
    qty = max(min_notional / limit_price * 1.1, step)
    qty = round(qty / step) * step
    prec = max(0, len(f"{step:.10f}".rstrip("0").split(".")[1]))
    qty = float(f"{qty:.{prec}f}")
    pprec = max(0, len(f"{tick:.10f}".rstrip("0").split(".")[1]))
    limit_price = float(f"{limit_price:.{pprec}f}")
    print(f"    test order: BUY {qty} {args.symbol} @ {limit_price:,.2f} "
          f"(notional ${qty*limit_price:,.2f}, 30% below market so it rests)")

    # ---- 4. validate without placing -------------------------------------
    print("[4] validate order (no order created) ...", end=" ", flush=True)
    try:
        client.create_test_order(symbol=args.symbol, side="BUY", type="LIMIT",
                                 timeInForce="GTC", quantity=qty, price=f"{limit_price:.{pprec}f}")
        print("OK — the exchange accepts this order shape")
    except BinanceAPIException as e:
        print(f"FAIL  APIError {e.code}: {e.message}")
        return
    except Exception as e:
        print(f"FAIL  {type(e).__name__}: {str(e)[:120]}")
        return

    if not args.place:
        print("\nRead-only checks passed. Re-run with --place to actually place, read back")
        print("and cancel a resting order.")
        return

    # ---- 5/6/7. place, read back, cancel ---------------------------------
    print("[5] place resting order ...", end=" ", flush=True)
    try:
        order = client.create_order(symbol=args.symbol, side="BUY", type="LIMIT",
                                    timeInForce="GTC", quantity=qty,
                                    price=f"{limit_price:.{pprec}f}")
        oid = order["orderId"]
        print(f"OK  orderId={oid}  status={order['status']}")
    except BinanceAPIException as e:
        print(f"FAIL  APIError {e.code}: {e.message}")
        if e.code == -2010:
            print("    (insufficient testnet balance — fund the account from the faucet)")
        return

    print("[6] read it back ...", end=" ", flush=True)
    try:
        got = client.get_order(symbol=args.symbol, orderId=oid)
        print(f"OK  status={got['status']}  executedQty={got['executedQty']}")
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e)[:100]}")

    print("[7] cancel ...", end=" ", flush=True)
    try:
        cx = client.cancel_order(symbol=args.symbol, orderId=oid)
        print(f"OK  status={cx['status']}")
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e)[:100]}")

    print("\nFull order lifecycle exercised against testnet: authenticate -> filters ->")
    print("validate -> place -> read -> cancel. That is the path real money would use.")


if __name__ == "__main__":
    main()
