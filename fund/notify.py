"""Telegram notifications, optional. Without credentials it only logs."""
from __future__ import annotations

import logging

import requests

from fund import config

log = logging.getLogger("fund")


def send(text: str) -> bool:
    log.info("NOTIFY: %s", text.replace("\n", " | "))
    if not (config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        return r.ok
    except Exception as e:  # a notification must never break a trading run
        log.warning("telegram send failed: %s", e)
        return False
