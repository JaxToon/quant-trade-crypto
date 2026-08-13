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
MARKET_SOURCE = os.getenv("MARKET_SOURCE", "binance")  # "binance" or "tradingview"

# Symbols to track
# Crypto Blue Chips (Binance futures pairs)
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT"]

# Stock Blue Chips (NASDAQ)
STOCK_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO"]

# Active watchlist (determined by ASSET_TYPE)
WATCHLIST = CRYPTO_WATCHLIST if ASSET_TYPE == "crypto" else STOCK_WATCHLIST

# TradingView configuration
TRADINGVIEW_EXCHANGE = os.getenv("TRADINGVIEW_EXCHANGE", "BINANCE")  # BINANCE, NASDAQ, NYSE
TRADINGVIEW_SCREENER = os.getenv("TRADINGVIEW_SCREENER", "crypto")  # crypto, america
TRADINGVIEW_INTERVAL = os.getenv("TRADINGVIEW_INTERVAL", "1m")  # 1m, 5m, 15m, 1h, 4h, 1d

# ── OpenAI (multi-agent LLM consensus) ────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BULL_BEAR_MODEL = "gpt-4o-mini"   # fast + cheap for debaters
CRITIC_MODEL = "gpt-4o"           # deeper reasoning for critic

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
HMM_N_STATES = 3            # 0=low-vol, 1=normal, 2=high-stress-turbulent
HMM_TURBULENT_STATE = 2
HMM_RISK_REDUCTION = 0.70   # slash risk 70% in turbulent regime

# ── Temporal graph decay ──────────────────────────────────────────────────
TEMPORAL_DECAY_GAMMA = 0.001  # γ: edge weight decay per day

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
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))

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

# ── Historical backtest (backtest.py) ──────────────────────────────────────
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "30"))

# ── Continual learning from realized paper-trade outcomes ─────────────────
# Not formal reinforcement learning (no reward/policy loop) — a contrastive
# fine-tuning step (backtrack/continual_learning.py) triggered periodically
# once enough new closed paper trades have accumulated to be worth a GNN
# gradient step on. See main.py's scheduled `run_paper_trade_learning`.
LEARNING_INTERVAL_HOURS = 4
LEARNING_MIN_NEW_TRADES = 5

