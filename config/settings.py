import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Binance ────────────────────────────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ── Asset Selection & Blue Chip Configuration ────────────────────────────
ASSET_TYPE = os.getenv("ASSET_TYPE", "crypto")  # "crypto" or "stock"

# Binance has no stock symbols, so IntegratedMarketStream's "auto" source
# (main.py's default) must not default to "binance" for the stock asset
# class — that silently 404s every symbol with APIError(-1121) "Invalid
# symbol" and never falls back. Same reasoning applies to the TradingView
# exchange/screener below — "BINANCE"/"crypto" are meaningless for NASDAQ
# tickers. All three still honor an explicit env override; only the
# fallback default is asset-aware now, matching WATCHLIST below.
MARKET_SOURCE = os.getenv("MARKET_SOURCE", "tradingview" if ASSET_TYPE == "stock" else "binance")  # "binance" or "tradingview"

# Symbols to track
# Crypto Blue Chips (Binance futures pairs)
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT"]

# Stock Blue Chips (NASDAQ)
STOCK_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO"]

# Active watchlist (determined by ASSET_TYPE)
WATCHLIST = CRYPTO_WATCHLIST if ASSET_TYPE == "crypto" else STOCK_WATCHLIST

# TradingView configuration
TRADINGVIEW_EXCHANGE = os.getenv("TRADINGVIEW_EXCHANGE", "NASDAQ" if ASSET_TYPE == "stock" else "BINANCE")  # BINANCE, NASDAQ, NYSE
TRADINGVIEW_SCREENER = os.getenv("TRADINGVIEW_SCREENER", "america" if ASSET_TYPE == "stock" else "crypto")  # crypto, america
TRADINGVIEW_INTERVAL = os.getenv("TRADINGVIEW_INTERVAL", "1m")  # 1m, 5m, 15m, 1h, 4h, 1d

# ── OpenAI (multi-agent LLM consensus) ────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BULL_BEAR_MODEL = "gpt-4o-mini"   # fast + cheap for debaters
CRITIC_MODEL = "gpt-4o"           # deeper reasoning for critic
# Path 2 (Bull/Bear/Critic debate -> Binance testnet) kill switch. Default
# on to preserve existing behavior; set to "false" to skip the whole debate
# cycle in main.py's decision_loop (saves OpenAI credits) while leaving
# Path 1 (SMC/FVG paper trading) and open-position monitoring untouched.
# Disabled 2026-08-30: 3,050+ recorded debate decisions (1,050 crypto +
# 2,000 stock) across this project's history, zero ever executed a trade
# -- see RESEARCH_NOTES.md.
ENABLE_LLM_CONSENSUS = os.getenv("ENABLE_LLM_CONSENSUS", "true").lower() == "true"

# ── Trading risk parameters ────────────────────────────────────────────────
TRADE_THRESHOLD = 0.55      # minimum P(Trade) to open a position
MAX_POSITION_PCT = 0.02     # max 2% of portfolio per trade
STOP_LOSS_PCT = 0.03        # hard stop-loss at -3% from entry
TAKE_PROFIT_PCT = 0.06      # take-profit at +6% from entry
DAILY_DRAWDOWN_LIMIT = 0.05 # halt trading if daily loss > 5%
LEVERAGE = 1                # 1x = spot-equivalent; increase carefully

# ── Multi-agent weights ────────────────────────────────────────────────────
W1_BULL = 0.6   # weight on bull score
W2_BEAR = 0.4   # weight on bear score
# P(Trade) = W1_BULL * S_Bull - W2_BEAR * S_Bear - C_Risk

# ── GNN hyperparameters ────────────────────────────────────────────────────
GNN_HIDDEN_CHANNELS = 128
GNN_NUM_HEADS = 4
GNN_NUM_LAYERS = 2
GNN_OUT_CHANNELS = 1        # single scalar signal per token node
GNN_LR = 1e-3
GNN_EPOCHS = 50
GNN_BATCH_SIZE = 32
MODEL_CHECKPOINT_DIR = "model_checkpoints"

# ── Backtrack / continual learning ────────────────────────────────────────
BACKTRACK_LAMBDA = 0.5      # λ in contrastive loss formula
BACKTRACK_LOG_DIR = "backtrack_logs"

# ── HMM regime detection ──────────────────────────────────────────────────
# Was 3 states (0=low-vol, 1=normal, 2=high-stress-turbulent), but states 0
# and 1 were consistently near-identical in mean volatility (~0.3%
# separation, vs ~74-80% between the calm pair and the turbulent state) —
# a genuine 2-regime process at this resolution, not 3. Confirmed via a
# direct A/B: 2-state vs 3-state produce IDENTICAL turbulent/calm
# classification for every trade tested (circuit_breaker.py already
# treated states 0/1 identically, so this changes nothing behaviorally —
# housekeeping/correctness, not an optimization). See
# hmm_regime_ab_experiment.py for the comparison.
HMM_N_STATES = 2            # 0=low-vol/calm, 1=high-stress-turbulent
HMM_TURBULENT_STATE = 1
HMM_RISK_REDUCTION = 0.70   # slash risk 70% in turbulent regime

# detect_regime() used to classify a SINGLE observation at a time, which
# for a length-1 sequence reduces Viterbi to argmax(log(startprob) +
# log(emission)) -- startprob_ on the live checkpoint puts ~100% of its
# mass on one state, so single-frame predict() returned that same state
# for virtually any input (verified directly against the deployed
# checkpoint: 0.00% turbulent flagged across a 60+ day walk-forward test,
# vs 23.6% by a retrospective full-history decode). Fixed by decoding a
# rolling window of recent observations instead of one point, so
# transmat_ and emission likelihoods actually drive the decision past the
# window's first step. 336h (14d) was the smallest window that reaches
# the agreement plateau against retrospective ground truth (84.7%, same
# as 500h+) with ~1.6h average detection lag and sub-millisecond decode
# latency -- see hmm_windowed_decode_experiment.py for the validation.
HMM_DECODE_WINDOW_HOURS = 336

# ── Temporal graph decay ──────────────────────────────────────────────────
TEMPORAL_DECAY_GAMMA = 0.001  # γ: edge weight decay per day

# At this gamma, decay alone is too slow to bound growth in practice — 90
# days only knocks a starting confidence of 0.8 down to ~0.73
# (exp(-0.001*90) ≈ 0.914), nowhere near GRAPH_MIN_EDGE_CONFIDENCE. The
# real growth-control lever is age-based deletion: Macro_Event nodes have
# no cap or expiry otherwise (confirmed — one new node per qualifying
# article, forever), and their IMPACTS edges grow right alongside them.
# Runs as a scheduled daily job (see main.py), not on every tick — this is
# maintenance, not something that needs sub-day freshness.
GRAPH_MACRO_EVENT_MAX_AGE_DAYS = 90   # delete Macro_Event nodes (+ their edges) older than this
GRAPH_MIN_EDGE_CONFIDENCE = 0.05      # secondary cleanup: drop any edge decayed below this floor

# ── RSS news feeds (crypto) ────────────────────────────────────────────────
RSS_FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://cointelegraph.com/rss",
    "https://cryptonews.com/news/feed/",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://www.theblock.co/rss.xml",
    "https://cryptopotato.com/feed/",
    "https://ambcrypto.com/feed/",
    "https://u.today/rss",
]
NEWS_POLL_INTERVAL_SECONDS = 300  # poll every 5 minutes

# ── Technical indicator windows ────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0

# ── Smart Money Concept (SMC) Configuration ───────────────────────────────
SMC_ENABLED = os.getenv("SMC_ENABLED", "true").lower() == "true"
SMC_LOOKBACK_CANDLES = 100  # How many candles to analyze for SMC patterns

# Real OHLC candle timeframe SMC/FVG operate on (Binance kline interval string).
# Order Blocks / FVG / market structure need real candle bodies+wicks — using
# tick-level data here degenerates every "candle" to open==high==low==close.
# Cap on TOTAL open notional across all positions, as a multiple of the
# account. 0 disables (the historical behaviour).
#
# PORTFOLIO_MAX_CORRELATED_RISK_PCT caps RISK, not capital committed, and
# those come apart badly. Measured over 730d/6 segments: cutting risk per
# trade from 2.00% to 0.50% left peak total notional essentially unchanged
# (17.2x -> 16.5x of the account) because the freed-up risk budget simply
# admitted more positions (3,364 -> 8,015 trades). Only ~20-24% of replay
# steps ever had everything open fitting inside the account.
#
# That matters because spot cannot borrow: strategy/live_mirror.py must
# reject any decision whose notional exceeds the free balance. Capping risk
# does not make the strategy spot-executable; capping notional is the lever
# that actually does.
PORTFOLIO_MAX_NOTIONAL_MULTIPLE = float(os.getenv("PORTFOLIO_MAX_NOTIONAL_MULTIPLE", "0"))

# ── Live order mirroring (Path 1 -> real exchange) ────────────────────────
# Path 1 (SMC/FVG -> PaperTradingLedger) is the path with actual evidence
# behind it, and until now it could only ever write to a paper ledger.
# strategy/live_mirror.py can reproduce its decisions as real orders.
#
# THREE independent switches guard real money, on purpose. One forgotten
# environment variable should never be the only thing separating a test run
# from a funded account:
#   1. LIVE_TRADING_ENABLED must be true at all
#   2. BINANCE_TESTNET must be false to leave testnet
#   3. LIVE_TRADING_CONFIRM_REAL_MONEY must ALSO be true
# The mirror refuses to arm against a non-testnet endpoint unless (3) is set.
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
LIVE_TRADING_CONFIRM_REAL_MONEY = os.getenv("LIVE_TRADING_CONFIRM_REAL_MONEY", "false").lower() == "true"

# Halt NEW live entries once the day is this far down, in percent of the
# balance at the day's first check. 0 disables.
#
# Path 1 has never had a loss limit of any kind: DAILY_DRAWDOWN_LIMIT is
# enforced only in strategy/execution.py, which is Path 2 and disabled. So a
# losing streak on the path that would actually hold money has run unchecked
# by design. Open positions keep their exchange-held stops when this trips;
# only new entries stop.
LIVE_TRADING_DAILY_LOSS_LIMIT_PCT = float(os.getenv("LIVE_TRADING_DAILY_LOSS_LIMIT_PCT", "5.0"))

# Base URL for Binance REST (klines) calls.
#
# Defaults to data-api.binance.vision, NOT api.binance.com, because the
# latter is DNS-hijacked on at least one network this project runs from:
# every *.binance.com name resolves to 202.51.102.114 (traefik54.iforte.net.id,
# an Indonesian ISP sinkhole), and the TLS handshake then dies with
# `TLSV1_UNRECOGNIZED_NAME` since that host has no Binance certificate.
# Diagnosed 2026-09-07 after the failure appeared as three different-looking
# symptoms — a connect timeout, an SSL error, and a silent yfinance
# fallback firing every ~80s in risk.hmm_regime.
#
# data-api.binance.vision is Binance's own public market-data endpoint,
# serves the identical /api/v3/klines contract, resolves to real AWS IPs,
# and is not on the block list. It is read-only market data (no auth, no
# trading), which is exactly what every REST caller here needs.
#
# NOTE this does NOT make Binance TRADING reachable from such a network —
# order placement would still need api.binance.com. That is a live blocker
# for any real-money deployment here and is tracked in README.md.
# Size positions off CURRENT equity instead of the original starting balance.
#
# The existing behaviour (False) computes risk_amount, the correlation cap
# and the notional cap all from `starting_balance`, which never changes. So
# a $10,000 account risking 2% risks $200 per trade forever — and when a
# drawdown takes it to $3,150, that same $200 is 6.3% of what's left. The
# risk FRACTION grows exactly as the account can least afford it, which
# accelerates drawdowns instead of damping them.
#
# True sizes off current balance, so losses shrink the next position
# automatically. Backtested drawdown at measured slippage was worst-segment
# 68.5% / median 42.7% under the old behaviour -- see
# experiments/drawdown_at_real_slippage.py and the sizing experiment.
#
# Default False: existing live ledgers and every historical backtest keep
# behaving exactly as before unless a caller opts in.
PAPER_TRADING_EQUITY_BASED_SIZING = os.getenv("PAPER_TRADING_EQUITY_BASED_SIZING", "false").lower() == "true"

# Stop opening NEW positions once equity falls this far below its peak
# (percent). 0.0 disables the halt entirely, which is the current state.
#
# Worth being explicit about why this is needed: DAILY_DRAWDOWN_LIMIT (5%)
# is enforced only in strategy/execution.py -- Path 2, currently disabled.
# Path 1, the path with the actual backtested evidence and the one any
# real-money deployment would use, has no loss limit of any kind: no daily
# halt, no drawdown stop, no kill switch. A losing streak runs unchecked.
#
# Existing positions are never force-closed by this -- they keep their own
# stops. Only new entries are blocked. Trading resumes once drawdown
# recovers to half the threshold (hysteresis), so the system doesn't flap
# on and off at the boundary.
PAPER_TRADING_MAX_DRAWDOWN_HALT_PCT = float(os.getenv("PAPER_TRADING_MAX_DRAWDOWN_HALT_PCT", "0.0"))

# How often the dedicated SL/TP monitor thread checks open positions
# against live price, in seconds.
#
# Until 2026-09-07 there was no such thread: check_and_close ran once per
# 60s decision_loop cycle, queued behind a regime update (a network fetch)
# and full GNN inference. Prices arrived every tick over the WebSocket, but
# the stop check only looked once a minute — so a breached stop could sit
# unnoticed for up to a minute of real market movement.
#
# That gap is very likely the dominant term in the 0.25-0.60% stop slippage
# measured live, which matters because the strategy's backtested break-even
# is ~0.51%. Measured 1-minute candle ranges across the watchlist: median
# 0.065%, p90 0.197%, p99 0.410%, max 1.273% — the same order of magnitude
# as the observed slippage, which is what you would expect if the check
# interval were the cause. Stops break during fast moves, i.e. the tail.
#
# 1s is deliberately conservative rather than per-tick: it shrinks the
# window ~60x while keeping the loop cheap (a dict read plus a comparison
# per open position, no network, no GNN, no entry logic).
#
# TRADE-OFF, to be measured and not assumed: checking more often also means
# reacting to transient spikes that a 60s check would have slept through
# and survived. Expect stop-exit COUNT to rise even as per-exit slippage
# falls. Judge the change on slippage_report.py plus net PnL, never on
# either number alone.
PAPER_TRADING_SLTP_CHECK_SECONDS = float(os.getenv("PAPER_TRADING_SLTP_CHECK_SECONDS", "1.0"))

BINANCE_REST_BASE = os.getenv("BINANCE_REST_BASE", "https://data-api.binance.vision")

BINANCE_KLINE_INTERVAL = os.getenv("BINANCE_KLINE_INTERVAL", "5m")

# SMC/FVG runs independently on each of these timeframes ("layers"), not just
# the primary one above. Why: FVG gap size scales with candle range, so a 5m
# gap is often too tight to clear PAPER_TRADING's fee-viability stop floor
# (measured live: 0.09-0.27% vs a 0.67% minimum) — a 1h gap on the same
# symbol is naturally wider in %, and clears that bar far more often, without
# loosening the fee-viability filter itself. Technical indicators (RSI/MACD/BB)
# still only run on the first/primary entry — see structured_stream.py.
BINANCE_SMC_TIMEFRAMES = [BINANCE_KLINE_INTERVAL, "15m", "1h"]

# Order Block settings
OB_MIN_STRENGTH = 30.0  # Minimum strength score (0-100) to consider valid
OB_MAX_AGE_HOURS = 168  # 1 week max age

# Fair Value Gap settings
FVG_MIN_GAP_PERCENT = 0.1  # Minimum gap size as % of price (filter noise)
FVG_MITIGATION_THRESHOLD = 50.0  # % of gap filled before considering mitigated
FVG_MAX_AGE_HOURS = 168  # 1 week max age

# Market Structure settings
MS_SWING_LOOKBACK = 10  # Candles to look back for swing highs/lows

# SMC signal weights in final decision
SMC_SIGNAL_WEIGHT = 0.25  # 25% weight on SMC signal vs GNN
SMC_CONFLUENCE_BOOST = 0.15  # Boost when SMC + GNN agree
SMC_DIVERGENCE_PENALTY = 0.20  # Penalty when SMC + GNN disagree

# Blue Chip Filter settings
BLUE_CHIP_ENABLED = os.getenv("BLUE_CHIP_ENABLED", "true").lower() == "true"
BLUE_CHIP_MIN_MARKET_CAP = 10_000_000_000  # $10B for crypto
BLUE_CHIP_MIN_VOLUME_24H = 1_000_000_000  # $1B daily volume
BLUE_CHIP_MAX_VOLATILITY = 15.0  # 15% max daily volatility

# ── GNN signal thresholds (shared across decision_loop + entry_insight) ───
GNN_CONVICTION_THRESHOLD = 0.08  # below this, treat the GNN signal as "no opinion"

# ── Entry-point insight (dashboard only — does NOT place any orders) ──────
# Leverage used purely to estimate a rough liquidation price shown on the
# dashboard's derivative insight card. Independent of LEVERAGE above, which
# governs real position sizing for spot execution.
INSIGHT_DERIVATIVE_LEVERAGE = 3

# Below this confidence, a GNN/SMC divergence vetoes the recommendation
# entirely (actionable=False) rather than just lowering its confidence.
ENTRY_INSIGHT_MIN_CONFIDENCE = 15.0

# ── Dashboard (embedded directly in main.py — single command to run both) ──
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
# Asset-type-conditional default so a crypto+stock restart never collides
# on 8080 by omission — found live (2026-08-19): restarting just the stock
# process to deploy a fix and forgetting to pass DASHBOARD_PORT explicitly
# silently knocked its dashboard offline for ~90s (bind conflict with the
# crypto process already on 8080), while the trading loop itself kept
# running fine in the background thread. Explicit DASHBOARD_PORT still
# wins when set — this only changes what happens when it's omitted.
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8081" if ASSET_TYPE == "stock" else "8080"))

# ── Paper trading (scores entry_insight recommendations, no real orders) ──
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
PAPER_TRADING_STARTING_BALANCE = float(os.getenv("PAPER_TRADING_STARTING_BALANCE", "10000"))
PAPER_TRADING_RISK_PCT = 0.02  # base risk unit: 2% of STARTING balance per simulated trade
PAPER_TRADING_FEE_PCT = 0.001  # 0.1% per side (0.2% round trip) — Binance spot taker fee, no BNB discount

# Reward higher-conviction signals with a bigger (but still bounded) bet:
# risk scales linearly from 1x the base unit at confidence=0 up to this
# multiplier at confidence=100. Deliberately NOT allowed to reach anything
# like real "125x leverage" territory — at this system's own 0.67% minimum
# stop distance, 125x-scaled risk would put the liquidation buffer inside
# normal price noise, guaranteeing stop-outs before the plan ever executes.
PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER = 2.5

# Portfolio-level correlated-risk cap: found via backtest analysis that 60%
# of trades had at least one concurrent position in a correlated symbol
# (avg correlation 0.62 among simultaneously-open pairs — basically the same
# as the unconditional average, meaning positions weren't specifically
# avoiding correlated overlap), with a peak of 7 simultaneous positions and
# an estimated 14-35% of starting balance at risk at once if every
# correlated position moved the same way together. "2% risk per trade" only
# describes an ISOLATED trade — this caps the aggregate, correlation-
# weighted exposure across everything currently open. See strategy/correlation.py.
PORTFOLIO_CORRELATION_THRESHOLD = 0.5   # only count exposure from pairs correlated >= this
# Env-overridable (like PAPER_TRADING_STARTING_BALANCE) so this can be tuned
# without a code edit — 0.08 (8%, $800 on a $10k account) is the long-standing
# default. Backtest evidence (RESEARCH_NOTES.md, "Correlation cap vs short
# sizing", 2026-08-27): raising it to 0.12 (1.5x) improved 2yr backtested
# return +65% with max drawdown slightly BETTER than default across 6
# market regimes; 0.16 (2x) improved return further but roughly doubled the
# worst-single-day dollar loss; 0.24 (3x) was rejected outright — drawdown
# 47% worse, worst-day loss near-wiped the starting balance. Not yet
# deployed at a new default; override via .env to experiment.
PORTFOLIO_MAX_CORRELATED_RISK_PCT = float(os.getenv("PORTFOLIO_MAX_CORRELATED_RISK_PCT", "0.08"))

# ATR-based trailing stop: once a position has moved favorably by
# PAPER_TRADING_TRAILING_ACTIVATION_R multiples of its own initial risk
# (entry-to-stop distance), the stop starts trailing behind price at
# PAPER_TRADING_TRAILING_ATR_MULTIPLE x ATR (computed at entry — see
# strategy/smart_money.py's SmartMoneyAnalyzer._compute_atr, a standard
# 14-period Average True Range reusing the same candle history OB/FVG
# detection already maintains) instead of just waiting for the original
# fixed target. Never loosens once activated. Defaults OFF — backtest
# before enabling, same discipline as every other risk-affecting change
# this project has made; a plausible-sounding trailing-stop rule is not
# automatically a profitable one.
PAPER_TRADING_TRAILING_STOP_ENABLED = os.getenv("PAPER_TRADING_TRAILING_STOP_ENABLED", "false").lower() == "true"
PAPER_TRADING_TRAILING_ACTIVATION_R = 1.0    # activate once profit >= 1x initial risk
PAPER_TRADING_TRAILING_ATR_MULTIPLE = 2.0    # trail this many ATRs behind the best price seen

# ── Historical backtest (backtest.py) ──────────────────────────────────────
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "30"))

# ── Continual learning from realized paper-trade outcomes ─────────────────
# Not formal reinforcement learning (no reward/policy loop) — a contrastive
# fine-tuning step (backtrack/continual_learning.py) triggered periodically
# once enough new closed paper trades have accumulated to be worth a GNN
# gradient step on. See main.py's scheduled `run_paper_trade_learning`.
LEARNING_INTERVAL_HOURS = 4
LEARNING_MIN_NEW_TRADES = 5

