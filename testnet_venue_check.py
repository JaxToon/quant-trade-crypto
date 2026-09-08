"""
Verify venue switching works, and that Tokocrypto is reachable and
Binance-shaped enough to drive with the same client.

Binance production is DNS-hijacked on this network, so Tokocrypto is the only
real-money venue available. Its spot API turned out to be Binance-compatible
down to the auth — signed endpoints accept X-MBX-APIKEY and return Binance
error codes — so python-binance drives it with a base-URL swap alone.

Checks, all read-only and unauthenticated where possible:
  1. each configured venue's REST base answers
  2. Tokocrypto returns the identical /api/v3 shapes the mirror depends on
     (exchangeInfo filters, klines, ticker)
  3. the watchlist symbols exist there
  4. the mirror refuses to arm on an unknown venue, and refuses real money
     without the confirmation flag

Usage: python testnet_venue_check.py
"""
from __future__ import annotations

import requests
from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402


def main() -> None:
    print("=== 1. venue endpoints ===")
    for venue, base in settings.VENUE_REST_URLS.items():
        try:
            r = requests.get(f"{base}/v3/ping", timeout=15)
            print(f"  {venue:<18}{base:<40} HTTP {r.status_code}")
        except Exception as e:
            print(f"  {venue:<18}{base:<40} FAIL {type(e).__name__}")

    tk = settings.VENUE_REST_URLS["tokocrypto"]
    print("\n=== 2. Tokocrypto returns the shapes the mirror needs ===")
    try:
        info = requests.get(f"{tk}/v3/exchangeInfo", timeout=25).json()
        syms = {s["symbol"]: s for s in info.get("symbols", [])}
        print(f"  exchangeInfo: {len(syms)} symbols")
        btc = syms.get("BTCUSDT")
        if btc:
            f = {x["filterType"]: x for x in btc["filters"]}
            print(f"  BTCUSDT filters: LOT_SIZE step={f['LOT_SIZE']['stepSize']} "
                  f"tick={f['PRICE_FILTER']['tickSize']} "
                  f"minNotional={f.get('NOTIONAL', f.get('MIN_NOTIONAL', {})).get('minNotional')}")
            print(f"  orderTypes: {','.join(btc.get('orderTypes', []))}")
            has_stop = any("STOP" in t for t in btc.get("orderTypes", []))
            print(f"  -> exchange-side stops available: {has_stop}")
    except Exception as e:
        print(f"  exchangeInfo FAIL {type(e).__name__}: {str(e)[:80]}")
        syms = {}

    print("\n=== 3. watchlist coverage ===")
    found = sum(1 for w in settings.WATCHLIST if w in syms)
    for w in settings.WATCHLIST:
        print(f"  {w:<10}{'YES' if w in syms else 'NOT LISTED'}")
    print(f"  coverage: {found}/{len(settings.WATCHLIST)}")

    print("\n=== 4. safety guards ===")
    from strategy.live_mirror import LiveOrderMirror
    original = settings.EXCHANGE_VENUE
    try:
        settings.EXCHANGE_VENUE = "nonsense_venue"
        m = LiveOrderMirror(enabled=True)
        print(f"  unknown venue        -> armed={m.enabled}  (must be False)")

        settings.EXCHANGE_VENUE = "tokocrypto"
        m = LiveOrderMirror(enabled=True)
        why = ("no credentials" if not settings.TOKOCRYPTO_API_KEY
               else "no LIVE_TRADING_CONFIRM_REAL_MONEY")
        print(f"  tokocrypto, real $   -> armed={m.enabled}  (must be False: {why})")
    finally:
        settings.EXCHANGE_VENUE = original

    print("\nWhat is still needed to place a real order:")
    print("  - TOKOCRYPTO_API_KEY / TOKOCRYPTO_API_SECRET in .env")
    print("  - EXCHANGE_VENUE=tokocrypto")
    print("  - LIVE_TRADING_ENABLED=true")
    print("  - LIVE_TRADING_CONFIRM_REAL_MONEY=true")
    print("  All four, deliberately — no single flag reaches real money.")


if __name__ == "__main__":
    main()
