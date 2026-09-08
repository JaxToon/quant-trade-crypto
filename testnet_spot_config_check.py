"""
Verify the spot-viable configuration end-to-end on testnet, and find the
minimum capital it actually needs.

The configuration (from experiments/notional_cap_spot_viability.py):

    PAPER_TRADING_RISK_PCT          = 0.0025   (0.25%)
    PORTFOLIO_MAX_NOTIONAL_MULTIPLE = 1.0

Backtested at 730d/6 segments: 2,656 trades, 57.9% WR, profitable 6/6,
worst-segment drawdown 9.7%, and 100% of replay steps spot-executable.

Three things to check that a backtest cannot:

  1. Does a ledger decision under this config actually PLACE on the exchange,
     with protection, at the exact quantity/stop/target the ledger chose?
  2. Does the notional cap really bind? It is the whole reason the config is
     spot-viable, so it must refuse the position that would overcommit rather
     than relying on the exchange to reject it.
  3. What is the MINIMUM account size? Notional per position is
     balance x risk% / stop%, which shrinks with the account — below some
     balance it falls under the exchange's MIN_NOTIONAL and nothing can be
     placed at all. That number is what a first deployment actually needs to
     know, and it is easy to get wrong by hand.

Testnet only.

Usage: python testnet_spot_config_check.py
"""
from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from config import settings  # noqa: E402

# Apply the candidate configuration for this process only.
settings.PAPER_TRADING_RISK_PCT = 0.0025
settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = 1.0

from backtrack.paper_trading import PaperTradingLedger  # noqa: E402
from strategy.live_mirror import LiveOrderMirror  # noqa: E402

SYM = "BTCUSDT"
STOP_PCT, TARGET_PCT = 0.025, 0.05


def _rec(price):
    return {"actionable": True, "direction": "long", "entry_zone": [price, price],
            "stop_loss": round(price * (1 - STOP_PCT), 2),
            "target": round(price * (1 + TARGET_PCT), 2),
            "confidence": 60.0, "suggested_leverage": 1}


def main() -> None:
    mirror = LiveOrderMirror(enabled=True)
    if not mirror.enabled or "testnet" not in mirror._client.API_URL:
        print("REFUSING: mirror not armed against testnet."); return

    price = float(mirror._client.get_symbol_ticker(symbol=SYM)["price"])
    step, tick, min_notional = mirror._symbol_filters(SYM)
    print(f"endpoint {mirror._client.API_URL}")
    print(f"{SYM} {price:,.2f}   minNotional ${min_notional:,.2f}")
    print(f"config: risk {settings.PAPER_TRADING_RISK_PCT*100:.2f}%  "
          f"notional cap {settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE:.1f}x\n")

    # ---- 3. minimum viable account size ---------------------------------
    print("[minimum capital] notional per position = balance x risk% / stop%")
    print(f"{'balance':>10}{'risk $':>10}{'notional $':>13}   verdict")
    print("-" * 52)
    smallest_ok = None
    for bal in (20, 50, 100, 200, 500, 1000, 5000):
        risk = bal * settings.PAPER_TRADING_RISK_PCT
        notional = risk / STOP_PCT
        ok = notional >= min_notional
        if ok and smallest_ok is None:
            smallest_ok = bal
        print(f"{bal:>10,}{risk:>10.2f}{notional:>13.2f}   "
              f"{'OK' if ok else f'BELOW ${min_notional:.0f} minimum — unplaceable'}")
    print("-" * 52)
    need = min_notional * STOP_PCT / settings.PAPER_TRADING_RISK_PCT
    print(f"exact minimum for ONE position: ${need:,.2f}")
    print(f"the cap allows several at once, so a usable account wants a multiple of that\n")

    # ---- 1. does a ledger decision actually place? ----------------------
    real_balance = mirror._balance()
    print(f"[placement] testnet USDT {real_balance:,.2f} — sizing the ledger to match")
    led = PaperTradingLedger(path=os.path.join(tempfile.mkdtemp(), "l.json"),
                             starting_balance=real_balance)
    reason = led.maybe_open(SYM, "spot", _rec(price), timeframe="1h")
    pos = led.open_positions.get((SYM, "spot", "1h"))
    if pos is None:
        print(f"  ledger declined: {reason}"); return
    notional = pos.quantity * pos.entry_price
    print(f"  ledger: qty={pos.quantity:.8f} SL={pos.stop_loss:.2f} TP={pos.target:.2f}")
    print(f"          notional ${notional:,.2f} = {notional/real_balance:.3f}x the account")

    res = mirror.mirror_open(pos)
    print(f"  mirror: ok={res.ok}  {res.reason or ''}")
    if res.ok:
        live = mirror._client.get_open_orders(symbol=SYM)
        print(f"  exchange: {len(live)} protective order(s)")
        for o in live:
            print(f"    {o['type']:<18} qty={o['origQty']} stop={o.get('stopPrice','-')} "
                  f"price={o['price']}")
        norm = lambda v: v.rstrip("0").rstrip(".")
        agree = (norm(mirror._fmt(pos.quantity, step)) in {norm(o["origQty"]) for o in live}
                 and norm(mirror._fmt(pos.stop_loss, tick)) in
                     {norm(o.get("stopPrice", "")) for o in live})
        print(f"  MATCHES LEDGER: {agree}")

    # ---- 2. does the notional cap bind? ---------------------------------
    print("\n[cap] filling the book until the cap refuses a position ...")
    opened, refused = 1, None
    for i, sym2 in enumerate(["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
                              "DOTUSDT", "AVAXUSDT"]):
        try:
            p2 = float(mirror._client.get_symbol_ticker(symbol=sym2)["price"])
        except Exception:
            continue
        r = led.maybe_open(sym2, "spot", _rec(p2), timeframe="1h")
        if led.open_positions.get((sym2, "spot", "1h")) is not None:
            opened += 1
        else:
            refused = (sym2, r)
            break
    total = sum(p.quantity * p.entry_price for p in led.open_positions.values())
    print(f"  {opened} position(s) open, total notional ${total:,.2f} "
          f"= {total/real_balance:.3f}x the account")
    if refused:
        print(f"  refused {refused[0]}: {refused[1]}")
        print("  -> the cap binds, which is what makes this spot-executable")
    else:
        print("  cap never triggered with the symbols tried")

    # ---- cleanup --------------------------------------------------------
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
        except Exception:
            pass
    print(f"cancelled {n}, position closed")


if __name__ == "__main__":
    main()
