"""Run on a new server before moving the fund there. Read-only, places no orders.

    python3 deploy/check_access.py
"""
import json
import urllib.request

CHECKS = {
    "Binance market data (prices)": "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT",
    "Tokocrypto API (orders go here)": "https://www.tokocrypto.site/api/v3/ticker/price?symbol=BTCUSDT",
    "Telegram (alerts)": "https://api.telegram.org",
    "Public IP of this server": "https://api.ipify.org?format=json",
}
ok = True
for name, url in CHECKS.items():
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            body = r.read(300).decode(errors="replace")
        print(f"OK    {name}: {body[:120]}")
    except Exception as e:
        ok = ok and name.startswith("Telegram") and "404" in str(e)  # bare Telegram host answers 404 = reachable
        print(f"{'OK' if 'Telegram' in name and '404' in str(e) else 'FAIL'}  {name}: {e}")
print("\nAll reachable: the fund can run here." if ok else "\nSomething is blocked: do not move the fund here yet.")
