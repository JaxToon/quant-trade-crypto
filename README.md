# TraderStuff — Quant Trading System for Crypto & Blue-Chip Stocks

Combines **Smart Money Concepts (SMC)**, **Fair Value Gap (FVG)** analysis, a **Graph Neural Network (GNN)**, and **multi-agent LLM debate** into a single research/paper-trading system for crypto (default) or NASDAQ blue chips.

## TL;DR

- **Run it:** `python main.py` — one command starts everything (data streams, GNN, SMC/FVG analysis, dashboard at `http://localhost:8080`). Requires Neo4j running first.
- **No real money at risk right now.** There are two separate decision paths, both simulated:
  1. **SMC/FVG entry-insight + $10k virtual paper-trading ledger** — read-only dashboard recommendations, scored against live price with a fixed-risk simulated account. This is the path with real backtested evidence behind it.
  2. **Multi-agent LLM consensus (Bull/Bear/Critic) + Binance testnet orders** — a separate, independent path (`strategy/agents.py` → `consensus.py` → `execution.py`). Places real orders, but only on Binance's testnet (fake funds). **If this path is ever pointed at a real account:** it already places real, exchange-side `STOP_LOSS_LIMIT`/`TAKE_PROFIT_LIMIT` orders (GTC) at entry time — protection lives on Binance's own matching engine, not in this process, so a laptop shutdown does not leave a position unprotected. Position state also persists to disk across restarts. Unexercised in practice, though — it has never actually closed a real trade, even on testnet (see below).
- **What's actually validated:** a 2-year backtest (`backtest.py`) shows win rate holding in a 55–61% band across 6 different market regimes — not just one lucky window. Position sizing is fee-aware and capped by a portfolio-level correlation risk cap. A separate target/exit backtest matrix (`backtest_target_matrix.py`) confirmed the live FVG-gap-based target is already the best of 7 tested variants at 5m/15m on expectancy *and* drawdown — wider targets (1-2x the gap, or fixed 2R/3R/5R) are strictly worse, monotonically, the wider they get.
- **What's not yet proven:**
  - **GNN's contribution is measured as negligible** in a properly controlled ablation (SMC-only vs SMC+GNN: identical win rate, identical trade count, on real data).
  - **The stock/NASDAQ path now runs end-to-end and has its first real live samples**: 4 closed trades (75% win rate, +12.3% on the $10k paper balance) plus 5 open as of this writing — still far too small to judge, but no longer a zero-sample path. See "Switch to Stock Trading" below.
  - **The LLM-consensus/testnet path's decision quality is unaudited** — it has literally never closed a trade, and its execute threshold appears very hard to reach given the formula's own math (see `TRADE_THRESHOLD` in Configuration below).
  - **Live paper-trading sample size is still small** — crypto has 20 closed trades (70% win rate, +28.6%), stock has 4; treat both live figures and any per-timeframe/per-market breakdown as directional, not statistically meaningful yet — 1-8 trades per bucket is noise, not signal.
  - **A regime-based entry filter for Path 1 was tested and rejected**: skipping trades while the HMM benchmark reads "turbulent" made expectancy and win rate *worse* at 5m/15m, and even where it looked better at 1h, drawdown got worse. See `hmm_regime_filter_backtest.py` — not deployed, kept only as documented negative evidence.
  - **SMC confidence-weighting (respect-count / ATR-relative bonuses) was tested and not deployed**: mixed result, one variant only won at a single timeframe (the same "not enough" bar the target-matrix experiment already established). See `smc_confidence_matrix_backtest.py`.
  - **A stop-loss placement matrix was tested — one clear rejection, one inconclusive result**: ATR-based stops (volatility-relative, replacing the live FVG-anchored buffer entirely) are decisively rejected — negative expectancy and negative return in all 9 variant×timeframe combinations tested, drawdown ballooning to 100–234% at wider multiples. Wider FVG-anchored stops (same method, bigger buffer) are mixed — better win rate and trade count everywhere, but only won on expectancy at 5m, not 15m/1h. Neither deployed. See `backtest_stoploss_matrix.py`.
  - **A momentum/breakout overlay was tested and decisively rejected**: raised by a real observation (a strong, low-pullback BTC rally produced almost no SMC/FVG signal for hours — not a bug, just outside that strategy's structural opportunity set) but tested as a strict hypothesis before touching anything live, per this project's own established discipline. A causal-only momentum signal (directional consistency + ATR-normalized move over a lookback, computed strictly from past bars) lost money 9/9 across 5m/15m/1h (29.6–33.3% win rate, -$77 to -$125/trade expectancy) and also degraded SMC's own trades in the same run via the shared portfolio-correlation cap. This rejects this specific lagging-confirmation design, not momentum trading as a category. See `backtest_momentum_coverage_experiment.py`.
  - **A higher-timeframe (HTF) bias filter — research candidate, regime-dependent, not deployment-ready.** Researching TA methods that could complement SMC/FVG surfaced a real gap: `BINANCE_SMC_TIMEFRAMES` (5m/15m/1h) run as fully independent parallel signal generators with zero cross-timeframe filtering, unlike ICT/SMC's standard "higher timeframe has authority" principle. Evidence got progressively more calibrated as evaluation resolution increased, kept here as three separate stages rather than overwriting earlier ones:
    - **Discovery** (30d vs 60d, same trailing window): looked strong, but wasn't actually independent — 30d ending "now" is fully contained inside 60d ending "now", same data measured twice.
    - **Walk-forward** (3 genuinely non-overlapping ~20-day segments, cold-started analyzer state each): gating 15m entries on 1h structure was positive in 3/3 segments; gating 5m alone was positive in only 1/3 — the edge looked concentrated in 15m specifically, not "hierarchical alignment helps everywhere."
    - **Finer-resolution robustness check** (6 non-overlapping ~10-day segments, same 60d source data — a granularity check on the walk-forward finding, not a fourth independent sample): 15m→1h expectancy was positive in only 4/6 segments, drawdown did *not* improve consistently (the earlier "nearly halved" result doesn't hold as a general conclusion — it was likely an artifact of a bad sub-period getting averaged into a good one at coarser 20-day resolution), and segment 2 is a concrete counterexample: baseline was strong there (+$74.34 expectancy, 58.6% win rate) and the filter turned it negative — meaning the filter can also remove genuinely profitable trades, not just weak ones. Read "4/6 positive" as a diagnostic, not a probability — segment sample sizes vary a lot (one segment had only 4 filtered 15m trades).
    - The open research question shifted accordingly, from *"is 15m→1h better than baseline"* to **"what conditions make 1h bias help 15m, versus over-filter it"** — not yet answered, and deliberately not being patched with an ad-hoc regime/volatility exception for segment 2 right now, since that would be tuned to the one failure already observed (post-hoc, not validated).
    - `htf_both` (gating both 5m and 15m) is treated as the less honest candidate of the two despite a higher positive-segment count (5/6) — the walk-forward ablation already showed 5m's own contribution doesn't hold up independently, and finer slicing doesn't change that.
    - See `backtest_htf_bias_experiment.py`.
  - **The stock path has almost no news/macro-event coverage — a real source gap, quantified, not yet fixed.** The graph has infrastructure for news-derived edges (Macro_Event nodes, IMPACTS/COMPETES_WITH/etc.), but `RSS_FEEDS` (`config/settings.py`) is 9 crypto-only news sites, and the SEC EDGAR fetch (`ingestion/unstructured_stream.py`) is itself scoped to `"cryptocurrency"`/`"digital asset"` keyword filings rather than general company news — there is currently no general stock/equity news source anywhere in the system. `audit_news_coverage.py` quantified it live: across 176 articles fetched in one snapshot (166 RSS + 10 EDGAR), stock's live-mention count was **0** — zero, across all 8 watchlist companies — while crypto's funnel showed entity resolution and the sentiment gate both working correctly (BTC: 59 mentions → 37 passed the sentiment gate → 159 in the graph over time). The gap is upstream of resolution/gating, at the fetch stage itself. Not yet fixed — deliberately not bolting on new feeds without the same backtest-before-deploy discipline as everything else here: the plan is baseline vs. +company-news vs. +macro-calendar vs. +company+macro+sector, measuring expectancy/drawdown/which-trades-changed, not just coverage. Also worth checking before trusting it: the earlier "GNN's contribution is negligible" finding was measured on the graph as it existed — if that graph is this sparse for stock specifically, GNN's contribution there could look artificially small for lack of information to exploit, not because GNN itself doesn't help; this hasn't been separately tested yet.
- **A real regime-detection bug was found and fixed in an earlier session**: `detect_regime()` classified one observation at a time, which for a length-1 HMM sequence is dominated by the model's learned `startprob_` almost regardless of input — verified directly against the deployed checkpoint that the bear-veto circuit breaker had likely never fired from real market conditions (0.00% turbulent flagged across a 60+ day walk-forward test). Fixed via a rolling-window decode; see `risk/hmm_regime.py` and `hmm_windowed_decode_experiment.py`.
- **A real paper-trading bug was found live and fixed this session**: `entry_insight` recommends off the SMC/FVG order-block zone without checking current price, so a stale-but-still-"actionable" signal (price already broke through the zone) kept re-firing the same entry/SL/TP after it was no longer reachable — the ledger would open at the zone price and get instantly stopped out on the very next check, always a loss. Hit live: the stock ledger lost ~28% in 30 minutes to phantom NVDA trades before being caught. Fixed via a `current_price` guard in `maybe_open()` that rejects a setup already past its own stop/target; see `backtrack/paper_trading.py`.
- **A real Neo4j crash was found and fixed**: starting crypto and stock at once raced on `CREATE INDEX/CONSTRAINT ... IF NOT EXISTS`, which only protects single-call idempotency, not true concurrent creation from two separate processes — one process would crash with `EquivalentSchemaRuleAlreadyExists`. Fixed in `graph/schema.py`; running both asset types concurrently (see "Switch to Stock Trading" below) is now verified safe, including through a multi-hour unattended overnight run.
- **A real bug was found in the Macro_Event pruning added earlier this session**: its coalesce compared an INTEGER (`created_at`) against a ZONED DATETIME (`recovered_at`, written natively by the forensic reconstruction script) — Cypher's `<` on mismatched temporal/numeric types returns NULL rather than erroring, so all 273 forensically-recovered nodes (71% of the Macro_Event graph) were silently un-prunable at any age, the exact opposite of the feature's purpose. Caught by directly testing the live method against the real database rather than trusting the earlier "doesn't delete anything it shouldn't" validation, which couldn't have caught this since it never checked the query would eventually delete what it should. Fixed via `.epochMillis` normalization; see `graph/neo4j_client.py`.
- **trader.log now rotates** instead of growing unbounded forever (found at 30MB after ~1 week of intermittent runtime) — 50MB per file, 5 backups kept. Same class of fix as the Macro_Event pruning, just for the plain-text log.
- **Where to look next:** `DEPLOYMENT.md` for persistence/cloud-deployment notes, `python backtest.py --help`-style flags below for validation tooling, `dashboard/consensus_history.json` (via `/api/status`) for the LLM-consensus path's live decision history.

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

**Related validation tooling** (all quantitative-only — none of these change live strategy code unless a result explicitly earns it):
```bash
python backtest_target_matrix.py --timeframe 15m       # target/exit variant comparison (gap 1.0-2.0x, fixed 2R/3R/5R) vs the live baseline, by expectancy + drawdown, not just win rate
python hmm_regime_ab_experiment.py                      # 2-state vs 3-state HMM cluster quality + outcome relevance
python hmm_windowed_decode_experiment.py                 # validates the rolling-window regime-decode fix against retrospective ground truth
python hmm_regime_filter_backtest.py                     # regime-filtered Path 1 A/B (hard/soft filter vs baseline) — result: rejected, see TL;DR
python smc_confidence_matrix_backtest.py                 # respect-count / ATR-relative confidence bonuses vs baseline — result: mixed, not deployed, see TL;DR
python backtest_stoploss_matrix.py --timeframe 15m       # stop-loss placement variants (wider FVG-anchored, ATR-based) vs baseline — ATR rejected, gap mixed, see TL;DR
python backtest_momentum_coverage_experiment.py --timeframe 15m   # momentum overlay for SMC/FVG's structural coverage gap — result: rejected 9/9, see TL;DR
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

### Logs

- `trader.log`: Main system log, accumulates across restarts (append mode) — the reliable source for "what happened since the system started," unlike...
- `backtrack_logs/main_run.log`: gets truncated on every restart if you're launching via a redirect (`python main.py > backtrack_logs/main_run.log`)
- `backtrack_logs/`: paper-trading ledgers, backtest results, execution position state
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
├── backtest.py                # Historical validation: is/oos, multi-regime, GNN ablation, multi-timeframe concurrency, slippage sweep
├── backtest_target_matrix.py  # Target/exit variant comparison (Path 1 only) — result: live baseline already optimal
├── hmm_regime_ab_experiment.py       # 2-state vs 3-state HMM cluster quality + outcome relevance
├── hmm_windowed_decode_experiment.py # Validates the rolling-window regime-decode fix vs retrospective ground truth
├── hmm_regime_filter_backtest.py     # Regime-filtered Path 1 A/B — result: rejected, not deployed
├── smc_confidence_matrix_backtest.py # SMC confidence-weighting bonuses — result: mixed, not deployed
├── backtest_stoploss_matrix.py       # Stop-loss placement variants — ATR rejected, wider FVG-anchored mixed, not deployed
├── backtest_momentum_coverage_experiment.py  # Momentum overlay for SMC/FVG's coverage gap — result: rejected 9/9, not deployed
├── backtest_htf_bias_experiment.py    # Higher-timeframe bias filter — research candidate, regime-dependent, see TL;DR
├── audit_news_coverage.py             # Quantifies news/Macro_Event coverage per symbol (graph state + live funnel snapshot) — found stock has ~0 coverage, see TL;DR
├── reconcile_downtime.py              # Reconciles paper-ledger open positions against real price history after a shutdown
├── start_system.sh / stop_system.sh   # One-command start/stop for both crypto + stock processes
├── train_gnn.py                # GNN training from historical candles (separate checkpoint per ASSET_TYPE)
├── DEPLOYMENT.md               # Persistence contract, cloud/AuraDB migration notes
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
│   └── execution.py           # Path 2: Binance testnet order execution (persisted open-position state)
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

PRs welcome! Known gaps worth tackling, roughly in order of value:
- Let the stock/NASDAQ path (now running live in observation mode — see Advanced Usage) accumulate enough real samples for a genuine Stage 4 comparison against the crypto baseline (expectancy/win-rate/drawdown/rejection-distribution, not raw P&L — different universe, hours, and opportunity count)
- Calibrate `TRADE_THRESHOLD`/the P(Trade) formula for Path 2 once it has real outcome data to work from
- Investigate whether GNN confluence can be made to add real value, or whether SMC/FVG alone is the more honest baseline (see `backtest.py --with-gnn`)
- A genuinely bigger reward lever than target width — the target/exit matrix closed that door (wider targets are strictly worse); the more promising direction flagged by the regime-filter experiment, and reinforced by HTF bias, is entry *selection* ("which trades should never have been taken"), not exit sizing. Agreed research ordering for pursuing that (external research memo + review, 2026-08-21 — not yet started, no live code touched):
  1. **Persist rejection reasons historically** — pure infrastructure, no strategy behavior change. `paper_trade_rejections` (`main.py`) is currently recomputed and overwritten every cycle in the live status file, never logged — so "rejection distribution" analysis has no data to run against yet. Log timestamp, symbol, timeframe, rejection reason, candidate signal identity, and relevant price/signal context per event. Prerequisite for the portfolio-correlation-cap attribution question below.
  2. **Timeframe concurrence, crypto first.** Does 5m/15m/1h agreement (vs. partial concurrence vs. full conflict) predict outcome better than the HTF experiment's hard hierarchy veto? Reuses `backtest_htf_bias_experiment.py`'s existing scaffolding (the causally-correct merged event queue, per-timeframe state tracking) rather than new infrastructure. Crypto is the discovery dataset — far more sample depth (60+ days vs. stock's regular-session-only limits) — stock only runs afterward, as a transfer/replication test, not the first place the effect is looked for.
  3. **Signal age/decay, crypto first.** Does an OB/FVG's predictive value change as it ages? Same narrow-script pattern, same crypto-discovery / stock-transfer-test ordering. Establish whether the relationship exists and whether it's monotonic before ever weighting anything by age.
  4. **Only after both reveal repeated mechanics, extract shared experiment infrastructure** — not a generic harness built speculatively in advance of having two real experiments to generalize from (that would repeat the premature-abstraction mistake in miniature).
  5. **Then** index/sector context, then the company/macro/sector news program (`audit_news_coverage.py` already quantified the stock gap at 0 live mentions — see TL;DR; treat "news coverage" and "macro-event coverage" as separate axes, ablation ladder baseline → +company → +macro → +sector → combinations, scored on trade *selection* changes not just aggregate P&L). Re-test GNN's "negligible contribution" finding only after the stock graph is actually populated — that finding was measured against crypto's much denser graph, not stock's, so it doesn't automatically transfer.
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
