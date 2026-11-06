"""Every fund setting in one place. Env vars override where a user would tune."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

STRATEGY = "C4 DONCH-VT"                     # research/strategies.py CANDIDATES key
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT"]  # = research.data.SYMBOLS
BAND = 0.2                                   # Monday drift band, as pre-registered
FEE = 0.003822                               # Tokocrypto per side (settings.VENUE_FEE_PCT)
SLIPPAGE = 0.0005                            # per side, as in the research cost model
COST = FEE + SLIPPAGE

DATA_DIR = "data"
KLINE_DIR = os.path.join(DATA_DIR, "klines_live")   # never the pinned research cache
PAPER_LEDGER = os.path.join(DATA_DIR, "fund_ledger_paper.json")
LIVE_LEDGER = os.path.join(DATA_DIR, "fund_ledger_live.json")
KILL_FILE = os.path.join(DATA_DIR, "FUND_KILL")     # exists => no new buys, alert

PAPER_START_CAPITAL = float(os.getenv("FUND_PAPER_CAPITAL", "1000"))
MIN_ORDER_USDT = 5.0                         # Tokocrypto minimum notional (measured)

# Guards (FIA 2024 best practice, scaled to a once-a-day bot)
MAX_DRAWDOWN_HALT = float(os.getenv("FUND_MAX_DRAWDOWN_HALT", "0.25"))
# Backtest worst drawdown was 13.7%. Past 25% something is wrong (data, code,
# or a regime the strategy never saw): stop buying and tell the owner.
MAX_ORDER_FRACTION = 0.5                     # no single order above 50% of equity

# Real money: three switches, all required, mirrors strategy/live_mirror.py.
LIVE_ENABLED = os.getenv("FUND_LIVE_ENABLED", "false").lower() == "true"
LIVE_CONFIRM_REAL_MONEY = os.getenv("FUND_CONFIRM_REAL_MONEY", "false").lower() == "true"
LIVE_VENUE = os.getenv("FUND_VENUE", "binance_testnet")
# USDT the fund may use on its first live day. The fund manages only this
# money and the coins it bought itself, never other balances in the account.
LIVE_CAPITAL = float(os.getenv("FUND_LIVE_CAPITAL", "0"))
RECONCILE_TOLERANCE = 0.01                   # exchange may hold 1% less than the book, no more

TELEGRAM_TOKEN = os.getenv("FUND_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("FUND_TELEGRAM_CHAT_ID", "")
