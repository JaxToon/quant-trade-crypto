"""Telegram commands for the fund, from the owner's phone.

    python -m fund.telegram_bot

Long-polls Telegram (no public IP or open port needed) and answers ONLY the chat
in FUND_TELEGRAM_CHAT_ID; every other sender is ignored. Commands:

    /status   equity, drawdown, exposure, positions, last run
    /stop     no new buys from the next run on (exits still happen); asks to confirm
    /lanjut   lift the stop and a drawdown/kill halt; asks to confirm
    /help     this list
"""
from __future__ import annotations

import logging
import os
import sys
import time

import requests

from fund import config, ledger

log = logging.getLogger("fund.telegram")
API = "https://api.telegram.org/bot{token}/{method}"
CONFIRM_SECONDS = 120


def _call(method: str, **params):
    r = requests.post(API.format(token=config.TELEGRAM_TOKEN, method=method), json=params, timeout=40)
    return r.json()


def _say(text: str) -> None:
    _call("sendMessage", chat_id=config.TELEGRAM_CHAT_ID, text=text)


def status_text(mode: str = "paper") -> str:
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    if not os.path.exists(path):
        return f"Belum ada ledger {mode}."
    b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    last = b.runs[-1] if b.runs else {}
    prices = last.get("prices", {})
    pos = {k: q * prices.get(k, 0.0) for k, q in b.qty.items() if q > 0}
    equity = b.cash + sum(pos.values())
    dd = 1 - equity / b.peak_equity if b.peak_equity else 0.0
    lines = [f"Fund {mode.upper()} (C4 ×{config.RISK_MULTIPLIER:g} + emas)",
             f"Ekuitas ${equity:,.2f} ({equity / b.start_capital - 1:+.2%} sejak {b.runs[0]['date'] if b.runs else '-'})",
             f"Drawdown {dd:.1%} (berhenti beli di {config.MAX_DRAWDOWN_HALT:.0%}) | di pasar {1 - b.cash / equity:.0%}",
             f"Run terakhir: {last.get('date', '-')}"]
    lines += [f"  {k.replace('USDT', '')}: ${v:,.2f}" for k, v in sorted(pos.items(), key=lambda kv: -kv[1])] or ["  Kas penuh"]
    if b.halted:
        lines.append(f"BERHENTI: {b.halt_reason}")
    elif os.path.exists(config.KILL_FILE):
        lines.append("Stop pembelian aktif (berlaku di run berikutnya).")
    return "\n".join(lines)


def do_stop() -> str:
    with open(config.KILL_FILE, "w") as f:
        f.write("stopped from Telegram\n")
    return "Stop pembelian aktif. Mulai run berikutnya tidak ada pembelian baru; posisi tetap dijual sesuai sinyal. /lanjut untuk membatalkan."


def do_resume() -> str:
    if os.path.exists(config.KILL_FILE):
        os.remove(config.KILL_FILE)
    for mode, path in (("paper", config.PAPER_LEDGER), ("live", config.LIVE_LEDGER)):
        if os.path.exists(path):
            b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
            if b.halted:
                b.halted, b.halt_reason = False, ""
                ledger.save(b, path)
    return "Bot dilanjutkan: pembelian kembali normal mulai run berikutnya."


HELP = ("/status - ekuitas, drawdown, posisi\n"
        "/stop - hentikan pembelian baru\n"
        "/lanjut - lanjutkan pembelian\n"
        "/help - daftar perintah")


def handle(text: str, pending: dict) -> str:
    """Pure command logic (testable). `pending` holds a command awaiting confirmation."""
    cmd = text.strip().split()[0].lower().split("@")[0] if text.strip() else ""
    now = time.time()
    if cmd in ("ya", "yes", "/ya") and pending.get("cmd") and now - pending.get("t", 0) < CONFIRM_SECONDS:
        c = pending.pop("cmd")
        return do_stop() if c == "/stop" else do_resume()
    pending.clear()
    if cmd == "/status":
        return status_text("paper") + ("\n\n" + status_text("live") if os.path.exists(config.LIVE_LEDGER) else "")
    if cmd in ("/stop", "/lanjut"):
        pending.update(cmd=cmd, t=now)
        what = "MENGHENTIKAN pembelian baru" if cmd == "/stop" else "MELANJUTKAN pembelian"
        return f"Yakin {what}? Balas 'ya' dalam 2 menit untuk konfirmasi."
    return HELP


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(config.DATA_DIR, "telegram_bot.log"), encoding="utf-8")]
                        + ([logging.StreamHandler(sys.stdout)] if sys.stdout else []))
    if not (config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID):
        log.error("FUND_TELEGRAM_TOKEN / FUND_TELEGRAM_CHAT_ID not set")
        return 1
    owner = str(config.TELEGRAM_CHAT_ID)
    offset, pending = None, {}
    # Skip anything sent while the bot was down, so an old /stop is not replayed.
    old = _call("getUpdates", timeout=0)
    if old.get("result"):
        offset = old["result"][-1]["update_id"] + 1
    _call("setMyCommands", commands=[{"command": "status", "description": "Ekuitas, drawdown, posisi"},
                                     {"command": "stop", "description": "Hentikan pembelian baru"},
                                     {"command": "lanjut", "description": "Lanjutkan pembelian"},
                                     {"command": "help", "description": "Daftar perintah"}])
    log.info("telegram bot listening")
    while True:
        try:
            up = _call("getUpdates", timeout=30, **({"offset": offset} if offset else {}))
            for u in up.get("result", []):
                offset = u["update_id"] + 1
                m = u.get("message") or {}
                if str((m.get("chat") or {}).get("id")) != owner:
                    continue  # only the owner may command the fund
                reply = handle(m.get("text", ""), pending)
                log.info("cmd %r", m.get("text", "")[:20])
                _say(reply)
        except Exception as e:
            log.warning("poll failed: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
