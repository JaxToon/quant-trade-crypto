# TraderStuff — Quant Trading System for Crypto & Blue-Chip Stocks

Combines **Smart Money Concepts (SMC)**, **Fair Value Gap (FVG)** analysis, a **Graph Neural Network (GNN)**, and **multi-agent LLM debate** into a single research/paper-trading system for crypto (default) or NASDAQ blue chips.

## TL;DR

> **Status 2026-10-11: no demonstrated edge. Do not trade real money with this system.**

- **What it is:** an SMC/FVG signal engine with a $10k virtual paper ledger (Path 1), plus a disabled LLM-debate path (Path 2), a retired GNN and a Neo4j knowledge graph. `./start_system.sh` (or `python main.py`) starts it; dashboard at `http://localhost:8080`. Requires Neo4j running first.
- **Why there is no edge (measured 2026-10-11):** the paper ledger books every entry at the FVG **midpoint** (`backtrack/paper_trading.py:281`), but `is_valid_entry()` only lets a signal fire while price is in the half of the gap that is *worse* for the trade (mitigation ≤ 50%). So every long was booked below the market price and every short above it: on average **+0.52R per trade in the backtest** (better than market on 100% of 334 entries) and **+0.53% of price live** (98 of 103 entries).
  - Backtest at realistic fills (730d, 6 segments, Tokocrypto fees): midpoint booking +$7,065 (4/6 segments profitable); **market fill −$26,573 (0/6)**; resting limit at the midpoint −$8,198 (0/6). See `experiments/entry_fill_realism.py`.
  - Live paper record re-priced at the real market fill: **booked +$4,375 becomes −$18,653.** LONG 95% CI [−$98.74, +$73.87], unresolved. SHORT [−$697.24, −$236.03], a loss.
  - The market-fill arm wins **63.1%** of its trades and still loses money. Win rate is the wrong objective; what counts is expectancy after costs.
- **Other weaknesses, each measured:** about 150 trials run on one rolling window with no untouched holdout, and the live LONG result fails the deflated Sharpe test at that trial count. The live month was a +28.7% rally. Real venue fees are 0.3822% per side, 3.8x the modelled 0.1%. The 2-year multi-regime backtests cover 1h only (yfinance caps 5m/15m at 60 days), while live also traded 5m and 15m, and live 15m lost money. The GNN was gated out on 98.9% of calls and is retired. Path 2 produced zero trades in 3,050+ decisions.
- **What still holds:** signal detection is causal (no look-ahead). The execution plumbing is real and testnet-proven: exchange-held OCO protection, startup reconciliation against the exchange, rollback when protection fails, a thread-safe ledger, and live slippage instrumentation (stop slippage mean 0.119%, n=31).
- **Real money:** `strategy/live_mirror.py` can place Tokocrypto spot orders (long only) but **must stay unarmed**. `start_system.sh` enables the mirror against Binance **testnet** only. Real money would additionally need Tokocrypto API keys, `EXCHANGE_VENUE=tokocrypto` and `LIVE_TRADING_CONFIRM_REAL_MONEY=true`, and none of those are set. Binance itself is DNS-blocked on the development network.
- **If research continues, in this order:** (1) fix the ledger's fill model; (2) re-run one baseline on Binance-native data (5m/15m history back to 2022 is available from `data-api.binance.vision`) with the newest months locked away as a holdout; (3) make targets several times the 0.76% round-trip cost, e.g. on higher timeframes; (4) benchmark against buy-and-hold and against random entries with the same exits; (5) count trials and report deflated Sharpe instead of p < 0.05.
- **Every figure recorded before 2026-10-11 was booked at the midpoint and is optimistic.** The full history of findings is kept in [`docs/FINDINGS_LOG.md`](docs/FINDINGS_LOG.md), every experiment script is indexed in [`experiments/README.md`](experiments/README.md), and the full write-ups are in `RESEARCH_NOTES.md` (local, not pushed).

---

## 🚀 Features

### Core Trading System
- **Graph Neural Network (GNN)** — Heterogeneous Graph Transformer (`gnn/model.py: HGTSignalModel`) for price-signal prediction over the knowledge graph
- **Multi-Agent LLM Consensus** (Bull, Bear, Critic agents) for the separate testnet-execution path
- **HMM Regime Detection** with a circuit breaker (bear veto + risk cut in turbulent regime) — feeds the LLM-consensus path only. 2-state (low-vol/turbulent) as of this writing — was 3-state, but the middle "normal" state was consistently near-identical to "low-vol" (~0.3% mean-vol separation vs ~74-80% against the genuinely distinct turbulent state), a real 2-regime process at this resolution, not 3. Confirmed via A/B that this changes nothing behaviorally. Decodes a rolling window of recent observations rather than one point at a time — see the bug note in the TL;DR.
- **Neo4j Knowledge Graph** for token/protocol relationships and news-derived edges. Entity resolution/keyword detection covers the full 16-symbol watchlist (crypto + stock) — previously only 5 of 16 had alias coverage, a silent gap in what NLP extraction could even recognize. Bounded growth: `Macro_Event` nodes (one per qualifying news article, otherwise unbounded) are pruned past `GRAPH_MACRO_EVENT_MAX_AGE_DAYS` and decayed edges below `GRAPH_MIN_EDGE_CONFIDENCE` are dropped, on a daily scheduled job — temporal decay alone (see Continual Learning below) is far too slow to functionally prune anything on its own
- **Continual Learning** — two independent loops: contrastive fine-tuning from realized paper-trade outcomes (`backtrack/continual_learning.py`), and nightly post-mortem causal discovery from real testnet losses (`backtrack/post_mortem.py`, dormant until the LLM-consensus path actually closes a trade)

### Smart Money Concepts (SMC)
Institutional order flow analysis with:
- **Order Blocks (OB)**: Last opposing candle before strong moves (bullish/bearish)
- **Break of Structure (BOS)**: Market structure shifts (higher highs, lower lows)
- **Change of Character (ChoCH)**: Trend reversals
- **Liquidity Zones**: Stop hunt areas (swing highs/lows)
- **Market Structure Analysis**: Real-time trend detection, run independently across 5m/15m/1h layers simultaneously per symbol

### Fair Value Gap (FVG) Analysis
Imbalance/inefficiency detection:
- **Bullish FVG**: Gap-up patterns (support zones)
- **Bearish FVG**: Gap-down patterns (resistance zones)
- **Mitigation Tracking**: Monitor gap-fill percentage
- **FVG Clusters**: Confluence zones with multiple FVGs
- **Trading Signals**: Entry, stop-loss, and target calculations

### $10k Virtual Paper-Trading Ledger
- Fixed-fractional risk sizing (2%–5% of starting balance per trade, scaled by signal confidence), sized off *starting* balance to avoid compounding drift
- Fee-aware stop-distance filtering — rejects setups too tight to survive round-trip fees
- **Already-invalidated-setup guard** — rejects opening a position whose current live price has already crossed its own stop or target, so a stale signal can't open a trade that's mathematically already lost (see the phantom-trade bug note in the TL;DR)
- **Portfolio-level correlated-risk cap** — caps aggregate correlation-weighted exposure across open positions at 8% of starting balance, so several correlated bets can't silently stack into one oversized bet
- Persists across restarts (`backtrack_logs/paper_ledger_{crypto,stock}.json`)

### Blue Chip Filter
High-quality asset selection:
- **Crypto Blue Chips**: BTC, ETH, BNB, SOL, ADA, AVAX, DOT, XRP
- **NASDAQ Blue Chips**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO
- **Quality Scoring**: Market cap, volume, volatility, age metrics

### Data Sources
- **Binance**: Real-time crypto WebSocket (default), klines on 5m/15m/1h simultaneously
- **TradingView**: Multi-market support (crypto, stocks, forex) — an option for the stock path since Binance doesn't list equities, but its free tier rate-limits after ~1 poll cycle regardless of pacing (documented, unresolved external constraint)
- **yfinance**: Historical backtesting data source (`backtest.py`, `train_gnn.py`, up to 60 days at 5m/15m and 730 days at 1h) *and* a live-ish polling stream for the stock path (`ingestion/yfinance_stream.py`) — regular session only (prepost=False; extended-hours 1m data is frequently zero-volume noise), same 5m/15m/1h multi-timeframe layering as Binance. Verified live that repeated polling against genuinely static (market-closed) data produces zero duplicate/fake ticks, and that day/weekend session boundaries are handled correctly with no fabricated bars.

---

## 📦 Installation

### Requirements
- Python 3.10+
- Neo4j 5.0+ (Neo4j Desktop or Docker)
- Binance API keys (testnet is fine — see below)
- OpenAI API key (for the LLM-consensus path)

### Setup

1. **Clone and install dependencies:**
```bash
git clone <repo-url>
cd TraderStuff
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start Neo4j** (Neo4j Desktop — start the DBMS from the app; or Docker):
```bash
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

# Asset Selection — a global switch, not two paths running at once.
# Only one of crypto/stock is active per run; see "Switch to Stock Trading" below.
ASSET_TYPE=crypto  # "crypto" or "stock"
# MARKET_SOURCE default is asset-aware (binance for crypto, tradingview for
# stock) if left unset — only set this explicitly to override that default,
# e.g. MARKET_SOURCE=yfinance for the stock path's alternative live poller.
MARKET_SOURCE=binance  # "binance", "tradingview", or "yfinance"

# TradingView (only used when MARKET_SOURCE=tradingview — Binance doesn't list equities)
TRADINGVIEW_EXCHANGE=BINANCE  # BINANCE, NASDAQ, NYSE — asset-aware default if unset
TRADINGVIEW_SCREENER=crypto   # crypto, america — asset-aware default if unset
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

`main.py` starts the trading orchestrator **and** the monitoring dashboard together — no separate terminal needed. The system will:
1. Initialize Neo4j schema
2. Serve the dashboard at `http://localhost:8080`
3. Fit the HMM regime model (60 days of data) — refits weekly
4. Load/build the GNN model (per-asset-type checkpoint: `model_checkpoints/hgt_latest_{crypto,stock}.pt`)
5. Start real-time data streams (Binance, TradingView, or yfinance — see `MARKET_SOURCE`) across all configured timeframes
6. Run SMC/FVG analysis on every tick, feed the paper-trading ledger
7. Run the LLM-consensus debate + testnet execution path every 60s (skipped while bear veto is active)

Set `DASHBOARD_ENABLED=false` in `.env` to skip the embedded dashboard, or run it standalone in its own terminal with `python run_dashboard.py`.

**Running both crypto and stock together:** `python main.py` alone only starts one asset type (`ASSET_TYPE` is a global switch — see "Switch to Stock Trading" below). To run both at once, use:
```bash
bash start_system.sh   # launches crypto (default env) + stock (ASSET_TYPE=stock MARKET_SOURCE=yfinance), backgrounded
bash stop_system.sh    # gracefully stops both (SIGTERM, waits for the ledger to save), reports final balances
```
`start_system.sh` refuses to double-launch if either process is already running, and runs `reconcile_downtime.py` first — so a position that genuinely hit its SL/TP while the system was down gets closed for real before live monitoring resumes, rather than sitting stale. See "Reconciling the paper ledger after downtime" in Troubleshooting below for what that does on its own.

Dashboard shows:
- **Entry Point Insight** — read-only spot vs. derivative recommendations (direction, entry zone, SL/TP, and a rough liquidation estimate) computed from live SMC/FVG signals, per symbol per timeframe layer
- **Paper-trading stats** — balance, win rate, breakdown by market (spot/derivative) and by timeframe layer, plus any rejection reasons (fee-too-tight, correlation cap hit) so a filtered signal is never silently invisible
- Equity curve and price-movement charts (persisted across restarts)
- Real-time GNN signals, current regime (low-vol/turbulent), bear-veto state
- **LLM-consensus decision history** (`consensus_decisions` in `/api/status`) — every P(Trade) evaluation from the debate path, executed or not, so its threshold can eventually be calibrated with real evidence instead of guesswork

### Historical backtesting

```bash
python backtest.py                                    # default: in-sample/out-of-sample split, primary timeframe
python backtest.py --timeframe 1h --days 730 --segments 6   # multi-regime: does the edge hold across 2 years, not one window?
python backtest.py --with-gnn                          # controlled ablation: does GNN confluence add real value over SMC alone?
python backtest.py --multi-timeframe                   # replays 5m+15m+1h concurrently per symbol — matches main.py's real live condition
python backtest.py --slippage-sweep                     # finds the slippage level at which the backtested edge turns net-negative
```

All modes replay real historical candles (yfinance) through the same `entry_insight` pipeline the dashboard uses live, in true chronological lockstep across symbols — not one-symbol-fully-then-the-next, which would hide the portfolio correlation cap's real cross-symbol behavior. Results save to `backtrack_logs/backtest_ledger_*.json`, separate from the live paper ledger.

**Related validation tooling** — all one-off research/validation scripts live in `experiments/` (a Python package; run from the repo root with `python -m experiments.<name>`, not `python experiments/<name>.py`, so their `from backtest import ...`-style imports resolve). Quantitative-only — none of these change live strategy code unless a result explicitly earns it:
```bash
python -m experiments.backtest_target_matrix --timeframe 15m       # target/exit variant comparison (gap 1.0-2.0x, fixed 2R/3R/5R) vs the live baseline, by expectancy + drawdown, not just win rate
python -m experiments.hmm_regime_ab_experiment                      # 2-state vs 3-state HMM cluster quality + outcome relevance
python -m experiments.hmm_windowed_decode_experiment                 # validates the rolling-window regime-decode fix against retrospective ground truth
python -m experiments.hmm_regime_filter_backtest                     # regime-filtered Path 1 A/B (hard/soft filter vs baseline) — result: rejected, see TL;DR
python -m experiments.hmm_regime_direction_experiment                # regime x direction decomposition (Path 2 bear-veto) — 700d yfinance, superseded by the Binance rerun below
python -m experiments.hmm_regime_direction_binance_experiment         # same study, full 2019-2026 Binance history, post-bugfix — see TL;DR (CLOSED)
python -m experiments.hmm_regime_direction_interaction_analysis       # interaction + tail-exceedance follow-up (block bootstrap) — closed the bear-veto question, see TL;DR
python -m experiments.market_structure_fix_ab_backtest                # pre-deploy A/B for the market_structure fix, single-timeframe
python -m experiments.market_structure_fix_ab_multitimeframe          # same A/B, joint multi-timeframe replay (higher fidelity) — see RESEARCH_NOTES.md
python -m experiments.smc_confidence_matrix_backtest                 # respect-count / ATR-relative confidence bonuses vs baseline — result: mixed, not deployed, see TL;DR
python -m experiments.backtest_stoploss_matrix --timeframe 15m       # stop-loss placement variants (wider FVG-anchored, ATR-based) vs baseline — ATR rejected, gap mixed, see TL;DR
python -m experiments.backtest_momentum_coverage_experiment --timeframe 15m   # momentum overlay for SMC/FVG's structural coverage gap — result: rejected 9/9, see TL;DR
python -m experiments.vwap_confluence_ab_experiment --timeframe 1h --days 90       # VWAP directional-confluence veto — result: rejected, see TL;DR
python -m experiments.ta_full_survey_experiment --phase 1 --timeframe 1h          # screens 10 TA indicators individually before combining any (phase 2)
python -m experiments.full_ta_robustness_validation --segments 6 --timeframe 1h   # all 11 filters, multi-regime (2yr) — 0 regime-robust, see TL;DR
python -m experiments.combo_multiregime_robustness --asset crypto --segments 6    # combinations of the best individual filters — still 0 robust
python -m experiments.standalone_ta_strategies_robustness --segments 6            # SMC/FVG vs 5 classic standalone TA strategies with no SMC involvement
```

---

## 🧠 System Architecture

There are **two independent decision paths** — don't conflate their results:

### Path 1: SMC/FVG entry-insight → paper-trading ledger (dashboard, read-only + simulated)

```
Market Data (Binance, multi-timeframe)
         ↓
SMC Analyzer + FVG Analyzer (per timeframe layer)
         ↓
GNN confluence check (boost if agrees, penalize/veto if disagrees)
         ↓
entry_insight (spot + derivative recommendation, confidence score)
         ↓
Paper ledger: fee-viability filter → conviction-scaled sizing → portfolio correlation cap → open/reject
```

### Path 2: LLM-consensus → Binance testnet execution

```
Market Data (primary timeframe only)
         ↓
Technical Indicators (RSI, MACD, BB) + GNN signal
         ↓
Multi-Agent Debate (Bull, Bear, Critic) — skipped entirely while bear veto is active
         ↓
P(Trade) = W1_BULL * S_Bull - W2_BEAR * S_Bear - C_Risk  (max possible: W1_BULL = 0.6)
         ↓
Execute if P(Trade) >= TRADE_THRESHOLD (0.55) — real testnet order via strategy/execution.py
```

These paths share the GNN and Neo4j graph, but have separate ledgers, separate risk logic, and — as of this writing — very different track records: Path 1 has real backtested + live evidence behind it; Path 2 has never closed a single trade (see TL;DR).

### Smart Money Integration (both paths)

1. **Confluence Boost**: When SMC/FVG signal agrees with the GNN signal, confidence is boosted
2. **Divergence Penalty / Veto**: When SMC contradicts GNN, confidence is penalized — and vetoed entirely below `ENTRY_INSIGHT_MIN_CONFIDENCE` (Path 1) or when the Bear agent also scores high during a bear-veto regime (Path 2)

---

## ⚙️ Configuration

### Key Settings (`config/settings.py`)

#### Asset Selection
```python
ASSET_TYPE = "crypto"  # or "stock" — global switch, see .env
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", ...]
STOCK_WATCHLIST = ["AAPL", "MSFT", "GOOGL", ...]
```

#### Smart Money Concepts
```python
SMC_ENABLED = True
SMC_LOOKBACK_CANDLES = 100
BINANCE_SMC_TIMEFRAMES = ["5m", "15m", "1h"]  # all run concurrently per symbol

# Order Block settings
OB_MIN_STRENGTH = 30.0  # 0-100
OB_MAX_AGE_HOURS = 168  # 1 week

# Fair Value Gap settings
FVG_MIN_GAP_PERCENT = 0.1  # Filter noise
FVG_MITIGATION_THRESHOLD = 50.0  # % filled
FVG_MAX_AGE_HOURS = 168
```

#### Paper Trading (Path 1)
```python
PAPER_TRADING_STARTING_BALANCE = 10000
PAPER_TRADING_RISK_PCT = 0.02                    # base risk unit: 2% of starting balance
PAPER_TRADING_MAX_CONFIDENCE_MULTIPLIER = 2.5    # up to 5% at max confidence
PORTFOLIO_CORRELATION_THRESHOLD = 0.5
PORTFOLIO_MAX_CORRELATED_RISK_PCT = 0.08         # aggregate correlated-risk cap
```

#### LLM-Consensus Risk Management (Path 2)
```python
TRADE_THRESHOLD = 0.55      # Min P(Trade) to open — see TL;DR re: reachability
W1_BULL = 0.6                # formula ceiling is W1_BULL itself (0.6), not 1.0
W2_BEAR = 0.4
MAX_POSITION_PCT = 0.02     # Max 2% per trade
STOP_LOSS_PCT = 0.03        # -3% hard stop
TAKE_PROFIT_PCT = 0.06      # +6% target
LEVERAGE = 1                # 1x (spot-equivalent)
```

#### HMM Regime Detection
```python
HMM_N_STATES = 2            # 0=low-vol/calm, 1=high-stress-turbulent
HMM_TURBULENT_STATE = 1
HMM_RISK_REDUCTION = 0.70   # slash MAX_POSITION_PCT by 70% in turbulent regime
HMM_DECODE_WINDOW_HOURS = 336  # rolling-window decode, not single-observation — see TL;DR
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

Runs end-to-end (yfinance → backfill → SMC/FVG → GNN → `entry_insight` → paper ledger, all 8 symbols across 5m/15m/1h) and now has its first real live trades (4 closed, 5 open as of this writing) — still far too small a sample to judge. Don't treat early results as validation; the discipline from the target-matrix experiment applies here too — expectancy/win-rate/drawdown/rejection-distribution over enough samples, not a quick look at balance.

Two data-source options — `MARKET_SOURCE=yfinance` is the one actually validated live this session (TradingView's free tier rate-limits after ~1 poll cycle regardless of pacing):

```bash
# Edit .env — recommended (yfinance)
ASSET_TYPE=stock
MARKET_SOURCE=yfinance
```

```bash
# Edit .env — alternative (TradingView; may hit rate limits)
ASSET_TYPE=stock
MARKET_SOURCE=tradingview
TRADINGVIEW_EXCHANGE=NASDAQ
TRADINGVIEW_SCREENER=america
TRADINGVIEW_INTERVAL=5m
```

```bash
python train_gnn.py   # trains a SEPARATE checkpoint for stocks — must run before main.py, there is no pretrained stock model
python main.py
```

`ASSET_TYPE` is a global switch — crypto and stock don't run in the same process. To observe both at once (e.g. while validating stock), run two separate processes — `DASHBOARD_PORT` now defaults asset-aware (8080 crypto / 8081 stock) so you don't need to set it explicitly unless you want a different port; each asset type already gets its own checkpoint, paper ledger, and HMM checkpoint file, so they don't collide. Starting both at once used to be able to crash on a Neo4j schema-init race — fixed (see TL;DR) and verified safe through a multi-hour concurrent run.

Example second-process launch (stock, alongside a crypto process already running on the defaults):
```bash
ASSET_TYPE=stock MARKET_SOURCE=yfinance python -u main.py
```

Note: `yfinance`-sourced live data is regular-session-only (no pre/post-market — NASDAQ hours are 9:30–16:00 ET) — outside that window the poller idles by design, and open positions simply sit unchanged (no fresh price ticks to check them against) until the next session opens. Not a bug if you check in and nothing's moved; see `ingestion/yfinance_stream.py`.

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

Trains from scratch each run (fresh weights avoid stale-checkpoint mismatches) on historical candles replayed through the real SMC analyzer — not zeroed placeholder features.

### Run Post-Mortem Analysis

```bash
python -c "
from backtrack.post_mortem import run_nightly_postmortem
from gnn.trainer import GNNTrainer
gnn = GNNTrainer()
gnn.load()
run_nightly_postmortem(gnn)
"
```

Reads real *closed, losing* trades from Neo4j (Path 2 only — the LLM-consensus/testnet path), clusters recent news, asks an LLM to deduce a missing causal relationship, and adds it to the graph. Runs automatically at 00:00 UTC via `main.py`'s scheduler. Currently a no-op until Path 2 closes its first trade.

---

## 📈 Performance & Monitoring

### ⚠️ Execution cost is the binding constraint — read before trusting any backtest

> **2026-10-11:** the larger execution cost turned out to be on the ENTRY side. The ledger books every entry at the FVG midpoint, about 0.5R better than the market price available at the signal, and with that removed no fill model is profitable (see the TL;DR). Every dollar figure in this section was computed at midpoint booking and a 0.1% fee; the stop-slippage measurements themselves are unaffected.

Measured 2026-09-04 (`experiments/breakeven_under_slippage.py`, 730d /
6 segments / 1h, full write-up in `RESEARCH_NOTES.md`).

A breached stop becomes a **market order** and fills past its trigger. The
backtest priced that at zero. Charging it to stop exits only (targets are
limit orders and don't slip adversely) gives:

| Stop slippage | Baseline total PnL | Worst-segment DD |
|---|---|---|
| 0.00% | +$288,902 | 55.0% |
| 0.25% | +$148,114 | 85.3% |
| 0.40% | +$63,640 | **116.9%** |
| 0.60% | −$48,991 | 253.4% |

**The strategy crosses into loss at roughly 0.51% slippage.** A
worst-segment drawdown above 100% means the account went past zero: at
0.40% the baseline suffers **ruin in at least one of the six segments**
despite positive total PnL.

**How close are we actually? Not close — corrected 2026-09-07.** An
earlier version of this section claimed 0.51% was "inside the band we have
already measured", citing 0.25-0.60%. That figure came from **4 stop-outs
examined by hand** — and those 4 were not a sample, they were the SHORT
losses under investigation at the time, i.e. drawn from the bad tail.
Reconstructed properly from **31 live-detected stop exits** using Binance
1-second klines (`experiments/reconstruct_historical_slippage.py`):

| | claimed (n=4) | actual (n=31) |
|---|---|---|
| Stop slippage | 0.25-0.60% | **mean 0.119%, median 0.068%** |

So the live system sits ~4x further below break-even than previously
stated, and was never on a knife's edge. The 0.51% threshold still matters
as a design constraint — it is why venue liquidity and execution latency
get checked at all — but it is not a live emergency. (Caveat on the
reconstruction: the pre-fix system acted on the last tick RECEIVED, which
under a 60s loop could be up to a minute older than the recorded
`closed_at`, so the OLD column carries real error bars. What survives is
the order of magnitude: ~0.1%, not ~0.5%.)

Consequences for how this repo should be used:

- **Backtest PnL run at `stop_slippage_pct=0.0` is not a viability
  estimate.** Any new candidate must clear the 0.40% gate before it is
  considered real. The breakeven-stop rule looked like +78% at 0% and was
  rejected outright at 0.40%.
- Use `stop_slippage_pct` (stop exits only), **not** the symmetric
  `slippage_pct`, whenever a change alters the mix of stop vs target
  exits — a flat rate taxes target exits too and flatters exactly the
  rule under test.
- **Gap CLOSED 2026-09-07.** `check_and_close` used to record
  `exit_price = effective_stop` — the theoretical level, not the price
  actually polled when the breach was detected — so every live stop-out
  computed the overshoot and discarded it, leaving ~90 live trades with
  zero usable slippage observations. The live path now books a stop at the
  OBSERVED price and records `trigger_price`, `observed_price` and
  `observed_slippage_pct` (positive = adverse, both directions). Targets
  still book at the target: a limit order fills at its price and does not
  capture favourable overshoot. `outcome` still derives from the trigger,
  so the win rate stays a statement about signal correctness rather than
  quietly becoming a cost metric.
- **Read it with `python slippage_report.py`.** It refuses to conclude
  below n=30; target ~100 stop exits. Historical trades cannot be
  backfilled — only exits after 2026-09-07 carry the measurement.
- **What that number is and is not:** it measures the price move between
  breach and next poll, which is a **lower bound** on real execution
  slippage (real fills also pay order-book depth, market impact and
  exchange latency). True cost is that figure or worse, never better —
  conservative in the useful direction, since a lower bound already near
  0.51% settles the question on its own. (Reconstruction from 31 historical
  stop exits since put the real figure at ~0.119% mean — see the correction
  above.)
- Reducing slippage (limit-based stop handling, liquidity-aware sizing)
  now outranks every signal-side improvement measured in this project.

### Logs

- `trader.log`: Main system log, accumulates across restarts (append mode) — the reliable source for "what happened since the system started," unlike...
- `backtrack_logs/main_run.log`: gets truncated on every restart if you're launching via a redirect (`python main.py > backtrack_logs/main_run.log`)
- `backtrack_logs/`: paper-trading ledgers (`paper_ledger_{crypto,stock}.json`) and execution position state. Raw experiment output lives in `backtrack_logs/experiment_outputs/` (indexed by its `INDEX.md`); old run logs, old backtest ledgers and the archived candidate datasets are gzip-compressed under `backtrack_logs/archive/`. All of it is gitignored.
- `dashboard/status.json`, `dashboard/history.json`, `dashboard/consensus_history.json`: real-time and historical dashboard state, persisted across restarts

### Neo4j Queries

**View active SMC zones:**
```cypher
MATCH (t:Token)
WHERE t.smc_in_bullish_ob = true OR t.smc_in_bullish_fvg = true
RETURN t.symbol, t.smc_market_structure, t.smc_signal
ORDER BY t.smc_fvg_confidence DESC
```

**View recent (Path 2 / testnet) trades:**
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

This uses Binance testnet with fake money — Path 2's execution engine only ever places testnet orders under this config. Path 1 (paper-trading ledger) never touches a real exchange at all, testnet or otherwise; it's a purely internal simulation scored against live price feeds.

---

## 🛠️ Troubleshooting

### "tradingview_ta not installed"
```bash
pip install tradingview-ta
```

### "Neo4j connection failed"
Neo4j Desktop: open the app and start the DBMS manually — it doesn't auto-start with `main.py`. Docker:
```bash
docker ps | grep neo4j
# Or restart: docker restart neo4j
```

### "OpenAI rate limit"
LLM agents are cached for 5 minutes. If hitting limits:
- Use `gpt-4o-mini` (cheaper, faster) — already the default for Bull/Bear
- Increase cache TTL in `strategy/agents.py`

### SMC not detecting patterns
- Ensure enough candles: `SMC_LOOKBACK_CANDLES = 100`
- Lower thresholds: `FVG_MIN_GAP_PERCENT = 0.05`
- Check logs for detection events

### Reconciling the paper ledger after downtime (shutdown, crash, etc.)
```bash
python reconcile_downtime.py --asset-type both   # or crypto / stock
```
Run this after any shutdown, before treating a restart as clean — standing practice, not a one-off. `check_and_close_ohlc()` (`backtrack/paper_trading.py`) works from any real OHLC candle stream, not just a live one, so `reconcile_downtime.py` replays a ledger's open positions against real historical candles covering exactly the [ledger's last-save time, now] window (Binance klines for crypto, yfinance fallback if that's unreachable — Binance's REST API has been flaky this session, TLS handshake failures rather than just timeouts; yfinance-only for stock, skipped entirely if the whole window was outside NASDAQ hours) using the exact same close logic live trading uses — not guessed or fabricated. Backs up the ledger file before writing. Precondition: only run while the process(es) being reconciled are actually stopped, to avoid a save race with a still-running main.py. Verified live on 2026-08-20 after a ~1-hour shutdown (3 open crypto positions checked against real 1m candles, none had crossed SL/TP) and again against safe copies of both real ledgers before this tool was committed.

### Dashboard shows a trade but the LLM-consensus path never executes
This is expected right now — see TL;DR. `TRADE_THRESHOLD=0.55` sits close to the P(Trade) formula's own mathematical ceiling (0.6), and real observed debate scores haven't gotten close. Check `dashboard/consensus_history.json` (or the `consensus_decisions` field in `/api/status`) to see the live P(Trade) distribution.

---

## 📚 Project Structure

```
TraderStuff/
├── main.py                    # Main orchestrator — runs both decision paths + dashboard
├── checks/                 # Live-exchange verification scripts (talk to a real venue;
│                           #   run as `python -m checks.NAME`). tests/ is the offline suite.
├── backtest.py                # Historical validation: is/oos, multi-regime, GNN ablation, multi-timeframe concurrency, slippage sweep
├── reconcile_downtime.py      # Reconciles paper-ledger open positions against real price history after a shutdown
├── start_system.sh / stop_system.sh   # One-command start/stop for both crypto + stock processes
├── system_env.sh              # Launch environment (sizing, testnet mirror) shared by start_system.sh and watchdog.sh
├── train_gnn.py                # GNN training from historical candles (separate checkpoint per ASSET_TYPE)
├── DEPLOYMENT.md               # Persistence contract, cloud/AuraDB migration notes
├── slippage_report.py         # Reads live stop-slippage observations; refuses conclusions below n=30
├── docs/
│   └── FINDINGS_LOG.md         # Chronological record of every finding (the README's former TL;DR, verbatim)
├── experiments/                # Research/validation scripts, run via `python -m experiments.<name>`
│   └── README.md               # Index: one line per script, what question it asks
├── config/
│   └── settings.py            # Configuration (SMC, paper trading, LLM-consensus risk, HMM regime, etc.)
├── strategy/
│   ├── smart_money.py         # SMC analyzer (OB, BOS, ChoCH)
│   ├── fvg_analyzer.py        # FVG detection & signals
│   ├── entry_insight.py       # Path 1: dashboard recommendation builder
│   ├── correlation.py         # Static crypto/stock correlation matrices for the portfolio risk cap
│   ├── blue_chip_filter.py    # Asset quality filter
│   ├── agents.py              # Path 2: LLM multi-agent debate (SMC-aware)
│   ├── consensus.py           # Path 2: P(Trade) computation
│   ├── execution.py           # Path 2: Binance testnet order execution (persisted open-position state)
│   └── live_mirror.py         # Path 1: mirrors paper positions as real orders (testnet; Tokocrypto spot, unarmed)
├── backtrack/
│   ├── paper_trading.py       # Path 1: $10k virtual ledger, sizing, correlation cap
│   ├── post_mortem.py         # Path 2: nightly causal-discovery analysis (dormant until Path 2 has real losses)
│   └── continual_learning.py  # Path 1: GNN fine-tuning from realized paper-trade outcomes
├── ingestion/
│   ├── structured_stream.py   # Binance WebSocket, multi-timeframe klines
│   ├── tradingview_stream.py  # TradingView data (stock path option)
│   ├── yfinance_stream.py     # yfinance polling stream (stock path option, regular-session-only)
│   └── integrated_stream.py   # Unified stream + per-timeframe SMC (binance/tradingview/yfinance)
├── gnn/
│   ├── model.py               # HGTSignalModel (Heterogeneous Graph Transformer)
│   ├── trainer.py             # Training + live/backtest inference
│   └── dataset.py             # Graph dataset builder from Neo4j
├── graph/
│   ├── neo4j_client.py        # Neo4j operations, incl. daily Macro_Event/edge pruning (bounded graph growth)
│   └── schema.py              # Graph schema, seed tokens (crypto + stock) — concurrent-start-safe (see TL;DR)
└── risk/
    ├── hmm_regime.py          # HMM regime detector (feeds Path 2's circuit breaker only) — 2-state, rolling-window decode
    └── circuit_breaker.py     # Bear veto + risk reduction in turbulent regime
```

---

## 🤝 Contributing

> **2026-10-11:** the list below predates the entry-fill finding and is kept for the record. The current priority order is the "If research continues" bullet in the TL;DR. Several items below (the target matrix, the TA survey, standalone TA strategies) were measured with midpoint booking and need re-testing under realistic fills before their verdicts are trusted.

PRs welcome! Known gaps worth tackling, roughly in order of value:
- Let the stock/NASDAQ path (now running live in observation mode — see Advanced Usage) accumulate enough real samples for a genuine Stage 4 comparison against the crypto baseline (expectancy/win-rate/drawdown/rejection-distribution, not raw P&L — different universe, hours, and opportunity count)
- Calibrate `TRADE_THRESHOLD`/the P(Trade) formula for Path 2 once it has real outcome data to work from
- Investigate whether GNN confluence can be made to add real value, or whether SMC/FVG alone is the more honest baseline (see `backtest.py --with-gnn`)
- A genuinely bigger reward lever than target width — the target/exit matrix closed that door (wider targets are strictly worse); the more promising direction flagged by the regime-filter experiment, and reinforced by HTF bias, is entry *selection* ("which trades should never have been taken"), not exit sizing. Agreed research ordering for pursuing that (external research memo + review, 2026-08-21):
  1. ✅ **Persist rejection reasons historically** — done and live. Every rejection (fee-too-tight, correlation-cap, stale-signal) is now appended to a bounded history file instead of being overwritten every cycle. Pure infrastructure — decision path provably unchanged (`tests/test_rejection_history.py`). See `RESEARCH_NOTES.md` for detail.
  2. ✅ **Timeframe concurrence, crypto first — research candidate, not deployed.** Purely observational — full agreement across 5m/15m/1h showed the strongest, most consistent performance; full conflict stayed profitable but with much higher drawdown. See `experiments/backtest_concurrence_experiment.py`, detail in `RESEARCH_NOTES.md`.
  3. ✅ **Signal age/decay, crypto 15m — research candidate, stronger than step 2, not yet a strategy finding.** Found and fixed a real backtest-fidelity bug first (see bug-history above). 0-2h old zones consistently worst, 8-24h consistently strongest — deliberately not called "decay" or "older is better," since persistence/survivorship/regime confounds aren't yet separated. See `experiments/backtest_signal_age_experiment.py`, detail in `RESEARCH_NOTES.md`.
  4. **Only after both reveal repeated mechanics, extract shared experiment infrastructure** — not a generic harness built speculatively in advance of having two real experiments to generalize from (that would repeat the premature-abstraction mistake in miniature).
  5. **Then** index/sector context, then the company/macro/sector news program (`experiments/audit_news_coverage.py` already quantified the stock gap at 0 live mentions — see TL;DR; treat "news coverage" and "macro-event coverage" as separate axes, ablation ladder baseline → +company → +macro → +sector → combinations, scored on trade *selection* changes not just aggregate P&L). Re-test GNN's "negligible contribution" finding only after the stock graph is actually populated — that finding was measured against crypto's much denser graph, not stock's, so it doesn't automatically transfer.
  6. ✅ **Full TA confluence-filter survey — CLOSED.** 11 filters (every major category), their combinations, and 5 standalone non-SMC strategies, all tested across a full 2-year/6-regime bar — none beat baseline; standalone strategies lose money outright. Strongest evidence yet the SMC/FVG structural approach itself is the edge, not a filterable input to it. See `experiments/full_ta_robustness_validation.py`, `experiments/standalone_ta_strategies_robustness.py`, detail in `RESEARCH_NOTES.md`.
  Standing rule for all of the above: **crypto establishes or disproves the hypothesis; stock only tests whether it transfers** — prevents the thinner stock dataset from becoming the accidental discovery environment.
- Local/self-hosted LLM option for Path 2's debate (feasible via an OpenAI-compatible local server; tradeoff is reasoning quality vs. removing the API dependency)
- Multi-exchange arbitrage

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- Smart Money Concepts inspired by ICT (Inner Circle Trader)
- GNN architecture based on PyTorch Geometric
- Multi-agent framework inspired by AutoGen
