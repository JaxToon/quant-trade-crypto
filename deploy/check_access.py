"""Run on a new server before moving the fund there. Read-only, places no orders.

    python3 deploy/check_access.py
"""
import urllib.error
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
        print(f"OK    {name}: {body[:80]!r}")
    except urllib.error.HTTPError as e:  # the host answered, so it is reachable
        print(f"OK    {name}: HTTP {e.code}")
    except Exception as e:
        ok = False
        print(f"FAIL  {name}: {e}")
print("\nAll reachable: the fund can run here." if ok else "\nSomething is blocked: do not move the fund here yet.")
