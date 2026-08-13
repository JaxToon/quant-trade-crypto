# Quant-Trade for blue chip stock and crypto

Advanced quantitative trading system with **Smart Money Concepts (SMC)**, **Fair Value Gap (FVG)** analysis, and **Blue Chip stock/crypto filtering**.

## 🚀 Features

### Core Trading System
- **Graph Neural Network (GNN)** for price prediction using knowledge graph embeddings
- **Multi-Agent LLM Consensus** (Bull, Bear, Critic agents) for trade decisions
- **HMM Regime Detection** with circuit breakers for turbulent markets
- **Neo4j Knowledge Graph** for fundamental relationships and news integration
- **Continual Learning** with backtrack analysis from past losses

### 🆕 Smart Money Concepts (SMC)
Institutional order flow analysis with:
- **Order Blocks (OB)**: Last opposing candle before strong moves (bullish/bearish)
- **Break of Structure (BOS)**: Market structure shifts (higher highs, lower lows)
- **Change of Character (ChoCH)**: Trend reversals
- **Liquidity Zones**: Stop hunt areas (swing highs/lows)
- **Market Structure Analysis**: Real-time trend detection

### 🆕 Fair Value Gap (FVG) Analysis
Imbalance/inefficiency detection:
- **Bullish FVG**: Gap-up patterns (support zones)
- **Bearish FVG**: Gap-down patterns (resistance zones)
- **Mitigation Tracking**: Monitor gap-fill percentage
- **FVG Clusters**: Confluence zones with multiple FVGs
- **Trading Signals**: Entry, stop-loss, and target calculations

### 🆕 Blue Chip Filter
High-quality asset selection:
- **Crypto Blue Chips**: BTC, ETH, BNB, SOL, ADA, AVAX, DOT, XRP
- **NASDAQ Blue Chips**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO
- **Quality Scoring**: Market cap, volume, volatility, age metrics
- **Auto-filtering**: Configurable thresholds for institutional-grade assets

### Data Sources
- **Binance**: Real-time crypto WebSocket (default)
- **TradingView**: Multi-market support (crypto, stocks, forex)
- **Multi-timeframe**: 1m, 5m, 15m, 1h, 4h, 1d

---

## 📦 Installation

### Requirements
- Python 3.10+
- Neo4j 5.0+
- Binance API keys (for crypto trading)
- OpenAI API key (for LLM agents)

### Setup

1. **Clone and install dependencies:**
```bash
git clone <repo-url>
cd TraderStuff
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Neo4j:**
```bash
# Start Neo4j (Docker)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.0
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

Example `.env`:
```ini
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Binance
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
BINANCE_TESTNET=true

# OpenAI (for LLM multi-agent consensus)
OPENAI_API_KEY=your_openai_key

# Asset Selection
ASSET_TYPE=crypto  # "crypto" or "stock"
MARKET_SOURCE=binance  # "binance" or "tradingview"

# TradingView (if using TradingView as source)
TRADINGVIEW_EXCHANGE=BINANCE  # BINANCE, NASDAQ, NYSE
TRADINGVIEW_SCREENER=crypto   # crypto, america
TRADINGVIEW_INTERVAL=1m        # 1m, 5m, 15m, 1h, 4h, 1d

# Smart Money Concepts
SMC_ENABLED=true
BLUE_CHIP_ENABLED=true
```

4. **Install spaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

---

## 🎯 Usage

### Start Everything (one command)

```bash
python main.py
```

`main.py` now starts the trading orchestrator **and** the monitoring dashboard together — no separate terminal needed. The system will:
1. Initialize Neo4j schema
2. Serve the dashboard at `http://localhost:8080`
3. Fit HMM regime model (60 days of data)
4. Load/build GNN model
5. Start real-time data streams (Binance or TradingView)
6. Run SMC/FVG analysis on every tick
7. Execute trades based on multi-agent consensus + SMC signals

Set `DASHBOARD_ENABLED=false` in `.env` to skip the embedded dashboard, or run it standalone in its own terminal with `python run_dashboard.py` (still supported).

Dashboard shows:
- Real-time P(Trade) scores
- GNN signals
- SMC market structure
- Active Order Blocks and FVGs
- Current regime (low-vol, normal, turbulent)
- Recent trades
- **Entry Point Insight** — read-only spot vs. derivative recommendations (direction, entry zone, SL/TP, and a rough liquidation estimate for derivatives) computed from the live SMC/FVG signals. This is informational only — the system never auto-trades derivatives; only spot orders are ever placed (see `strategy/execution.py`).

---

## 🧠 System Architecture

### Trading Decision Flow

```
Market Data (Binance/TradingView)
         ↓
SMC Analyzer (Order Blocks, FVG, Market Structure)
         ↓
Technical Indicators (RSI, MACD, BB)
         ↓
GNN Inference (knowledge graph embedding)
         ↓
Multi-Agent Debate (Bull, Bear, Critic)
         ↓
P(Trade) = W1*S_Bull - W2*S_Bear - C_Risk + SMC_Boost
         ↓
Execute if P(Trade) > threshold
```

### Smart Money Integration

SMC signals influence trading decisions through:

1. **Confluence Boost** (+15%): When SMC signal agrees with GNN
   - Bullish GNN + Bullish Order Block/FVG = Higher conviction
   
2. **Divergence Penalty** (-20%): When SMC contradicts GNN
   - Bullish GNN + Bearish Order Block = Lower conviction

3. **Zone Boost** (+10%): Price inside high-quality institutional zones
   - Entry at bullish OB/FVG for longs
   - Entry at bearish OB/FVG for shorts

### LLM Agent Prompts (Enhanced)

**Bull Agent** now understands:
- "Price in bullish Order Block = institutional demand"
- "Bullish FVG acting as support"
- "Market structure: higher highs/higher lows"

**Bear Agent** now identifies:
- "Bearish Order Block = institutional supply"
- "Liquidity sweeps before reversals"
- "Market structure breakdown"

---

## ⚙️ Configuration

### Key Settings (`config/settings.py`)

#### Asset Selection
```python
ASSET_TYPE = "crypto"  # or "stock"
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", ...]
STOCK_WATCHLIST = ["AAPL", "MSFT", "GOOGL", ...]
```

#### Smart Money Concepts
```python
SMC_ENABLED = True
SMC_LOOKBACK_CANDLES = 100

# Order Block settings
OB_MIN_STRENGTH = 30.0  # 0-100
OB_MAX_AGE_HOURS = 168  # 1 week

# Fair Value Gap settings
FVG_MIN_GAP_PERCENT = 0.1  # Filter noise
FVG_MITIGATION_THRESHOLD = 50.0  # % filled
FVG_MAX_AGE_HOURS = 168

# SMC signal weights
SMC_CONFLUENCE_BOOST = 0.15
SMC_DIVERGENCE_PENALTY = 0.20
```

#### Blue Chip Filter
```python
BLUE_CHIP_ENABLED = True
BLUE_CHIP_MIN_MARKET_CAP = 10_000_000_000  # $10B
BLUE_CHIP_MIN_VOLUME_24H = 1_000_000_000   # $1B
BLUE_CHIP_MAX_VOLATILITY = 15.0  # 15%
```

#### Risk Management
```python
TRADE_THRESHOLD = 0.55      # Min P(Trade) to open
MAX_POSITION_PCT = 0.02     # Max 2% per trade
STOP_LOSS_PCT = 0.03        # -3% hard stop
TAKE_PROFIT_PCT = 0.06      # +6% target
LEVERAGE = 1                # 1x (spot-equivalent)
```

---

## 📊 SMC Indicators Explained

### Order Blocks (OB)

**What**: Last opposing candle before a strong directional move.

**Bullish OB**: Bearish candle followed by strong bullish breakout
- Acts as **demand zone** (institutional buying)
- Entry: Price returns to OB zone

**Bearish OB**: Bullish candle followed by strong bearish breakdown
- Acts as **supply zone** (institutional selling)
- Entry: Price returns to OB zone

**Strength Score** (0-100):
- Volume of OB candle
- Momentum of follow-through move
- Higher = stronger institutional interest

### Fair Value Gaps (FVG)

**What**: 3-candle imbalance where price "gaps" without trading in a zone.

**Bullish FVG**: `candle[i-1].high < candle[i+1].low`
- Gap up → price moved too fast
- Acts as **support** (institutions may buy the dip)

**Bearish FVG**: `candle[i-1].low > candle[i+1].high`
- Gap down → price moved too fast
- Acts as **resistance** (institutions may sell the rally)

**Mitigation**: When price returns to fill the gap
- 0-50% filled: Gap still active
- 50-100% filled: Gap closing/closed

### Market Structure

**Bullish Structure**: Series of Higher Highs (HH) + Higher Lows (HL)
- **BOS** (Break of Structure): Price breaks previous high
- **ChoCH** (Change of Character): Structure shifts from bearish to bullish

**Bearish Structure**: Series of Lower Highs (LH) + Lower Lows (LL)
- **BOS**: Price breaks previous low
- **ChoCH**: Structure shifts from bullish to bearish

### Liquidity Zones

**What**: Swing highs/lows where stop losses cluster.

**Buy-Side Liquidity**: Above recent highs
- Retail long stop losses
- Smart money may **sweep** then reverse down

**Sell-Side Liquidity**: Below recent lows
- Retail short stop losses
- Smart money may **sweep** then reverse up

---

## 🔧 Advanced Usage

### Switch to Stock Trading

```bash
# Edit .env
ASSET_TYPE=stock
MARKET_SOURCE=tradingview
TRADINGVIEW_EXCHANGE=NASDAQ
TRADINGVIEW_SCREENER=america
TRADINGVIEW_INTERVAL=5m
```

Restart `main.py` → now trading NASDAQ blue chips.

### Custom Watchlist

Edit `config/settings.py`:
```python
# Your custom crypto list
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]

# Your custom stock list
STOCK_WATCHLIST = ["TSLA", "NVDA", "PLTR", "COIN"]
```

### Train GNN Model

```bash
python train_gnn.py
```

Trains on:
- Historical price data
- Neo4j graph structure
- Past trade outcomes (continual learning)

### Run Post-Mortem Analysis

```bash
python -c "from backtrack.post_mortem import run_nightly_postmortem; run_nightly_postmortem()"
```

Analyzes:
- Today's winning/losing trades
- Clusters failed scenarios
- Updates critic agent knowledge

---

## 📈 Performance & Monitoring

### Logs

- `trader.log`: Main system log
- `backtrack_logs/`: Nightly post-mortem analyses
- `dashboard/status.json`: Real-time system state

### Neo4j Queries

**View active SMC zones:**
```cypher
MATCH (t:Token)
WHERE t.smc_in_bullish_ob = true OR t.smc_in_bullish_fvg = true
RETURN t.symbol, t.smc_market_structure, t.smc_signal
ORDER BY t.smc_fvg_confidence DESC
```

**View recent trades:**
```cypher
MATCH (t:Trade)
WHERE t.timestamp > timestamp() - 86400000  // Last 24h
RETURN t.symbol, t.direction, t.pnl_pct
ORDER BY t.timestamp DESC
LIMIT 20
```

---

## 🧪 Testing

Run system in testnet mode (default):
```ini
BINANCE_TESTNET=true
```

This uses Binance testnet with fake money for safe testing.

---

## 🛠️ Troubleshooting

### "tradingview_ta not installed"
```bash
pip install tradingview-ta
```

### "Neo4j connection failed"
Check Neo4j is running:
```bash
docker ps | grep neo4j
# Or restart: docker restart neo4j
```

### "OpenAI rate limit"
LLM agents are cached for 5 minutes. If hitting limits:
- Use `gpt-4o-mini` (cheaper, faster)
- Increase cache TTL in `strategy/agents.py`

### SMC not detecting patterns
- Ensure enough candles: `SMC_LOOKBACK_CANDLES = 100`
- Lower thresholds: `FVG_MIN_GAP_PERCENT = 0.05`
- Check logs for detection events

---

## 📚 Project Structure

```
TraderStuff/
├── main.py                    # Main orchestrator
├── config/
│   └── settings.py            # Configuration (SMC, Blue Chip, etc.)
├── strategy/
│   ├── smart_money.py         # SMC analyzer (OB, BOS, ChoCH)
│   ├── fvg_analyzer.py        # FVG detection & signals
│   ├── blue_chip_filter.py    # Asset quality filter
│   ├── agents.py              # LLM multi-agent (SMC-aware)
│   ├── consensus.py           # P(Trade) computation (SMC-enhanced)
│   └── execution.py           # Order execution
├── ingestion/
│   ├── structured_stream.py   # Binance WebSocket
│   ├── tradingview_stream.py  # TradingView data
│   └── integrated_stream.py   # Unified stream + SMC
├── gnn/
│   ├── model.py               # GAT model
│   ├── trainer.py             # Training loop
│   └── dataset.py             # Graph dataset
├── graph/
│   ├── neo4j_client.py        # Neo4j operations (SMC data)
│   └── schema.py              # Graph schema
├── risk/
│   ├── hmm_regime.py          # HMM regime detector
│   └── circuit_breaker.py     # Risk control
└── backtrack/
    ├── post_mortem.py         # Nightly analysis
    └── continual_learning.py  # GNN fine-tuning
```

---

## 🤝 Contributing

PRs welcome! Areas for improvement:
- Additional SMC patterns (Wyckoff, ICT concepts)
- More data sources (Polygon.io, Alpaca)
- Backtesting engine
- Multi-exchange arbitrage

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- Smart Money Concepts inspired by ICT (Inner Circle Trader)
- GNN architecture based on PyTorch Geometric
- Multi-agent framework inspired by AutoGen
