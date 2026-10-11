# Launch environment shared by start_system.sh and watchdog.sh (sourced, not
# executed).
#
# Define it here and nowhere else. Until 2026-10-11 watchdog.sh hard-coded its
# own copy without SPOT_CFG or MIRROR_CFG, so any process it restarted came
# back on default sizing with the testnet mirror off, without saying so.

# SPOT-VIABLE SIZING, live from 2026-09-08 (paper ledger history splits there:
# everything before is the leveraged configuration).
#
# The default sizing (2% risk, no notional cap) needs up to 17x the account in
# total open notional, and spot cannot borrow. Risk 0.25% per trade plus total
# open notional capped at 1.0x the account is 100% spot-executable. Neither
# half works alone: capping risk leaves total notional unchanged because the
# freed budget admits more positions, and capping notional at 1.0x blocks
# everything because one default-sized position already exceeds the account.
# See experiments/notional_cap_spot_viability.py. Its backtested returns were
# booked at the FVG midpoint and are optimistic (README TL;DR, 2026-10-11).
SPOT_CFG="PAPER_TRADING_RISK_PCT=0.0025 PORTFOLIO_MAX_NOTIONAL_MULTIPLE=1.0"

# LiveOrderMirror copies each paper decision onto Binance TESTNET (fake money)
# so the order path runs against a real exchange. Real money would also need
# EXCHANGE_VENUE=tokocrypto, Tokocrypto API keys and
# LIVE_TRADING_CONFIRM_REAL_MONEY=true. None of them are set here, and the
# README says not to set them: there is no demonstrated edge.
MIRROR_CFG="LIVE_TRADING_ENABLED=true"

# Path 2 (LLM debate) stays off: 3,050+ recorded decisions, zero trades, three
# OpenAI calls per symbol per cycle. Set ENABLE_LLM_CONSENSUS=true to re-enable.
#
# Stock gets the same sizing but no mirror: no reachable venue lists NASDAQ
# symbols.
CRYPTO_ENV="ENABLE_LLM_CONSENSUS=false $SPOT_CFG $MIRROR_CFG"
STOCK_ENV="ENABLE_LLM_CONSENSUS=false $SPOT_CFG ASSET_TYPE=stock MARKET_SOURCE=yfinance"
