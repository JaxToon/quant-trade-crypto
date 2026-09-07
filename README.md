# TraderStuff — Quant Trading System for Crypto & Blue-Chip Stocks

Combines **Smart Money Concepts (SMC)**, **Fair Value Gap (FVG)** analysis, a **Graph Neural Network (GNN)**, and **multi-agent LLM debate** into a single research/paper-trading system for crypto (default) or NASDAQ blue chips.

## TL;DR

- **Run it:** `python main.py` — one command starts everything (data streams, GNN, SMC/FVG analysis, dashboard at `http://localhost:8080`). Requires Neo4j running first.
- **No real money at risk right now.** There are two separate decision paths, both simulated:
  1. **SMC/FVG entry-insight + $10k virtual paper-trading ledger** — read-only dashboard recommendations, scored against live price with a fixed-risk simulated account. This is the path with real backtested evidence behind it.
  2. **Multi-agent LLM consensus (Bull/Bear/Critic) + Binance testnet orders** — a separate, independent path (`strategy/agents.py` → `consensus.py` → `execution.py`). Places real orders, but only on Binance's testnet (fake funds). **If this path is ever pointed at a real account:** it already places real, exchange-side `STOP_LOSS_LIMIT`/`TAKE_PROFIT_LIMIT` orders (GTC) at entry time — protection lives on Binance's own matching engine, not in this process, so a laptop shutdown does not leave a position unprotected. Position state also persists to disk across restarts. Unexercised in practice, though — it has never actually closed a real trade, even on testnet (see below).
- **What's actually validated:** a 2-year backtest (`backtest.py`) shows win rate holding in a 55–61% band across 6 different market regimes — not just one lucky window. Position sizing is fee-aware and capped by a portfolio-level correlation risk cap. A separate target/exit backtest matrix (`experiments/backtest_target_matrix.py`) confirmed the live FVG-gap-based target is already the best of 7 tested variants at 5m/15m on expectancy *and* drawdown — wider targets (1-2x the gap, or fixed 2R/3R/5R) are strictly worse, monotonically, the wider they get.
  - **~70 backtest configurations across 11 technical-analysis confluence filters (VWAP, EMA, RSI, MACD, Stochastic, CCI, Bollinger, ADX, volume, OBV, Donchian), their combinations, and 5 standalone crossover/breakout strategies — none beat the SMC/FVG baseline once tested across a full 2-year, 6-regime window.** Standalone TA strategies with no SMC/FVG involvement lose money outright where baseline earns $200-290K — the strongest evidence yet that the structural entry logic is doing real work. See `experiments/full_ta_robustness_validation.py`, `experiments/combo_multiregime_robustness.py`, `experiments/standalone_ta_strategies_robustness.py`, full detail in `RESEARCH_NOTES.md`.
- **What's not yet proven:**
  - **GNN's contribution to Path 1 was measured as negligible, and re-confirmed (2026-08-30) with the fixed/retrained checkpoint over a long multi-regime window.** Original ablation (SMC-only vs SMC+GNN, identical win rate/trade count) carried an asterisk after the checkpoint-corruption bug (below) raised the possibility it was silently measuring a degraded zero-signal checkpoint. Re-ran `backtest.py --with-gnn` against the now-stable, fully-retrained checkpoint over 730d/1h (139,824 candle-replays): **57.5% win rate both arms, −$969 PnL delta on ~$297K — still negligible, asterisk resolved.** GNN confluence measurably adds no edge to Path 1 with either a broken or a healthy checkpoint. Not a reason to remove GNN from the architecture (Path 2's LLM debate still reads it), but Path 1's actual driver is SMC/FVG, not GNN. Full detail in `RESEARCH_NOTES.md`.
  - **Real-money-readiness audit (Fase 3-4, 2026-09-04): Path 2 is unfixable-as-is without user action, and live paper trading has a real, quantified slippage gap.** Path 2's testnet API credentials don't authenticate at all (`APIError -2015`, isolated to credentials not connectivity) — needs the user to generate valid testnet keys before any further work there; separately, `TRADE_THRESHOLD=0.55` is 91.7% of the P(Trade) formula's hard 0.6 ceiling, meaning even a perfectly-unbiased debate would rarely cross it. Recommendation (not acted on, user's call): decommission Path 2, focus on Path 1. Live paper trading's SL exits WERE recorded at the exact theoretical stop price, not the real price that triggered the check — tested against real market data on 4 recent live SHORT stop-outs and found 0.25-0.60% real overshoot every time (1.25-3x the modeled round-trip fee). **Instrumented and fixed 2026-09-07**: stop exits now book the observed price and record the measured overshoot, so this cost is accumulating as data instead of being discarded (`python slippage_report.py`). Full detail in `RESEARCH_NOTES.md` ("Path 2 audit, consolidated" and "Fase 4" sections).
  - **A structural GNN checkpoint-corruption bug was found and permanently fixed (2026-08-28).** The model's architecture was built from whichever Neo4j edge types happened to have >=1 edge at retrain time — the daily graph-pruning job fully emptying a low-count type (SUPPLIES, REGULATES) made the next retrain produce an incompatibly-shaped model, silently falling back to a zero signal until noticed. Root-fixed in `gnn/dataset.py`/`gnn/trainer.py` — the model's parameter shape is now independent of which edge types are currently populated. Both checkpoints retrained against the stable architecture and verified loading cleanly. Full forensics in `RESEARCH_NOTES.md`.
  - **Real 1-minute stock candles were tested as a training-data upgrade and REJECTED — no measured benefit (2026-09-03).** A HuggingFace 1m OHLCV dataset (Finnhub-sourced) was pulled for all 8 stock tickers (2022-2026, ~5.6M rows, via duckdb querying remote parquet directly — no 87.7GB download) and A/B-tested against the current yfinance-1h production path over the same 12-month window with equalized training compute. **The 1h baseline scored highest** (74.4% held-out directional accuracy / IC 0.701) vs the 1-minute arms' 71.1-71.9% — all differences inside noise, and no arm showed the collapse-to-constant failure the extra data was meant to fix (already solved by the earlier feature-normalization work). Not wired into `train_gnn.py`; the fetch/AB scripts are kept in `experiments/` and re-runnable. Full write-up in `RESEARCH_NOTES.md`.
  - **Neo4j now runs on the user's local Neo4j Desktop instance, not Docker.** Docker Desktop's own reliability (crashes/pauses, especially after laptop sleep) was the dominant operational failure mode for weeks; switched 2026-08-28 with a verified, lossless data merge. See `DEPLOYMENT.md`.
  - **A portfolio correlation-risk cap increase (0.08→0.12, crypto only) was backtest-validated and briefly deployed, then reverted** pending an unresolved live/backtest discrepancy: live crypto SHORT trades underperformed a backtest replay of the identical historical window dramatically (30% vs 61% win rate) — raising the cap risked amplifying whatever's behind that gap, since its own mechanism is "let more correlated exposure through, including more shorts." Open research question, not yet closed. Full arc in `RESEARCH_NOTES.md`.
  - **Re-tested whether banning SHORT trades entirely fixes the above — re-confirmed with the real, current GNN signal (2026-08-30), not just `gnn_signal=0.0`.** User raised a valid methodology concern: every backtest in this project's research line used `gnn_signal=0.0` for consistency, never the GNN's actual point-in-time predictions, and the checkpoint itself had since gone through a full corruption/refix/retrain cycle. Re-ran `experiments/directional_sizing_robustness.py --with-gnn` (real, freshly-retrained crypto checkpoint driving every step via `GNNTrainer.predict_from_features()`, same pattern as `backtest.py`): **LONG_ONLY (full ban) still loses 0/6 segments, −$106,816 vs baseline — essentially identical to the `gnn_signal=0.0` result (−$103,191).** SHORT-side de-risking (10-25% confidence multiplier) still wins 6/6. The verdict was never a stale/zero-GNN-signal artifact. The live/backtest SHORT discrepancy itself remains unresolved — this only rules out one candidate explanation. Full detail in `RESEARCH_NOTES.md`.
  - **A real methodology bug was found and fixed that's a strong candidate for the live/backtest SHORT gap — not yet confirmed live (2026-09-03).** Live's 60s decision loop could open new positions off a still-forming candle's transient `peek_signals()` state (a live-only code path, not something `backtest.py`'s replay — which only ever evaluates fully-closed candles — can produce), instead of always waiting for a settled, closed candle like backtest does. Fixed in `main.py` (`_latest_closed_indicators`, gated to `kline_closed=True`); SL/TP monitoring is untouched and still reacts to every tick, only new-position decisions were gated. Regression-tested, 179/179 passing. Needs a real batch of live SHORT trades post-fix before treating the SHORT gap as resolved — full forensics in `RESEARCH_NOTES.md`.
  - **The stock/NASDAQ path now runs end-to-end and has its first real live samples**: 4 closed trades (75% win rate, +12.3% on the $10k paper balance) plus 5 open as of this writing — still far too small to judge, but no longer a zero-sample path. See "Switch to Stock Trading" below.
  - **The LLM-consensus/testnet path (Path 2) is now disabled by default (2026-08-30), to save OpenAI cost.** Its decision quality was never audited, and it has literally never closed a trade — a check of its own decision history found 3,050+ recorded debate cycles (1,050 crypto + 2,000 stock, the latter already at its history cap) with **zero** `execute=True` outcomes ever, while each cycle that runs still costs 3 real OpenAI calls (`gpt-4o-mini` Bull, `gpt-4o-mini` Bear, `gpt-4o` Critic). `ENABLE_LLM_CONSENSUS=false` is now the default in `start_system.sh` (set to `true` to re-enable) — Path 1 (SMC/FVG paper trading) and open-position monitoring are completely unaffected by this switch. Its execute threshold also appears very hard to reach given the formula's own math (see `TRADE_THRESHOLD` in Configuration below), a separate reason it's never fired even when it was running.
  - **Live paper-trading sample size is still small** — treat live figures and any per-timeframe/per-market breakdown as directional, not statistically meaningful yet — 1-8 trades per bucket is noise, not signal.
  - **A regime-based entry filter for Path 1 was tested and rejected** (worse expectancy/win rate at 5m/15m, worse drawdown at 1h). Not deployed. See `experiments/hmm_regime_filter_backtest.py`.
  - **A real live bug was found and fixed: `market_structure.trend` could get permanently stuck**, feeding stale direction into Path 1, Path 2's LLM context, FVG confidence, and GNN features. Fixed (confirmed swing pivots + BOS/ChoCH), regression-tested (147/147 passing), A/B-validated before deploying, and live since 2026-08-21. `circuit_breaker.py` untouched. See `strategy/smart_money.py`, `tests/test_market_structure.py` — full forensics in `RESEARCH_NOTES.md` (local, gitignored).
  - **Regime × direction research for Path 2's bear-veto — CLOSED.** SMC direction carries real directional information (bullish consistently outperforms bearish), but a rigorous interaction + tail-exceedance test (block bootstrap over calendar weeks) found no basis to make the volatility circuit breaker direction-aware — the effect is a main effect, not a turbulence-specific interaction. Symmetric veto retained, `circuit_breaker.py` untouched throughout. Full arc (including a real bug found and fixed along the way) in `RESEARCH_NOTES.md`.
  - **SMC confidence-weighting was tested and not deployed** (mixed result). See `experiments/smc_confidence_matrix_backtest.py`.
  - **A stop-loss placement matrix was tested**: ATR-based stops decisively rejected, wider FVG-anchored stops mixed. Neither deployed. See `experiments/backtest_stoploss_matrix.py`.
  - **A momentum/breakout overlay was tested and decisively rejected** (9/9 losing across 5m/15m/1h). See `experiments/backtest_momentum_coverage_experiment.py`.
  - **A higher-timeframe (HTF) bias filter is a research candidate, not deployment-ready** — evidence got more calibrated as evaluation resolution increased, but the open question ("what conditions make 1h bias help 15m vs. over-filter it") isn't answered yet. See `experiments/backtest_htf_bias_experiment.py`.
  - **The stock path has almost no news/macro-event coverage** — a real, quantified gap (0 live mentions across all 8 companies), not yet fixed. See `experiments/audit_news_coverage.py`.
- **A handful of real bugs were found and fixed across this project's history** (regime-detection single-observation decode, a paper-trading phantom-trade guard, a Neo4j concurrent-startup race, a Macro_Event pruning type mismatch, unbounded log growth, FVG zone timestamp fidelity, the GNN checkpoint-corruption bug above, a silent multi-hour stock-stream stall from a blocking `yf.download()` call inside an async loop that let a real stop-loss breach go unresolved, cross-session process management in `start_system.sh`/`stop_system.sh` that made "is it running?" unreliable from a different terminal, and an unhandled Binance WebSocket drop (`ReadLoopClosed`) that killed the whole crypto process with no reconnect logic — real live outage 2026-09-02) — each traced to its actual blast radius before fixing, each covered by tests. See `RESEARCH_NOTES.md` for the full write-ups, or `git log` for the commits.
- **Where to look next:** `DEPLOYMENT.md` for persistence/cloud-deployment notes, `python backtest.py --help`-style flags below for validation tooling, `dashboard/consensus_history.json` (via `/api/status`) for the LLM-consensus path's live decision history, `RESEARCH_NOTES.md` (local, gitignored) for full research/bug forensics.

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

Measured 2026-09-04 (`experiments/breakeven_under_slippage.py`, 730d /
6 segments / 1h, full write-up in `RESEARCH_NOTES.md`).

A breached stop becomes a **market order** and fills past its trigger. On
4 real live SHORT stop-outs that overshoot was **0.25-0.60%**. The
backtest priced it at zero. Charging it to stop exits only (targets are
limit orders and don't slip adversely) gives:

| Stop slippage | Baseline total PnL | Worst-segment DD |
|---|---|---|
| 0.00% | +$288,902 | 55.0% |
| 0.25% | +$148,114 | 85.3% |
| 0.40% | +$63,640 | **116.9%** |
| 0.60% | −$48,991 | 253.4% |

**The strategy crosses into loss at roughly 0.51% slippage — inside the
band we have already measured.** Worse, a worst-segment drawdown above
100% means the account went past zero: at 0.40% the baseline suffers
**ruin in at least one of the six segments** despite positive total PnL.

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
  0.51% settles the question on its own.
- Reducing slippage (limit-based stop handling, liquidity-aware sizing)
  now outranks every signal-side improvement measured in this project.

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
├── reconcile_downtime.py      # Reconciles paper-ledger open positions against real price history after a shutdown
├── start_system.sh / stop_system.sh   # One-command start/stop for both crypto + stock processes
├── train_gnn.py                # GNN training from historical candles (separate checkpoint per ASSET_TYPE)
├── DEPLOYMENT.md               # Persistence contract, cloud/AuraDB migration notes
├── experiments/                # One-off research/validation scripts — quantitative-only, run via `python -m experiments.<name>` (see above)
│   ├── backtest_target_matrix.py  # Target/exit variant comparison (Path 1 only) — result: live baseline already optimal
│   ├── hmm_regime_ab_experiment.py       # 2-state vs 3-state HMM cluster quality + outcome relevance
│   ├── hmm_windowed_decode_experiment.py # Validates the rolling-window regime-decode fix vs retrospective ground truth
│   ├── hmm_regime_filter_backtest.py     # Regime-filtered Path 1 A/B — result: rejected, not deployed
│   ├── hmm_regime_direction_experiment.py # Regime x direction decomposition (Path 2 bear-veto), 700d yfinance — superseded, see below
│   ├── hmm_regime_direction_binance_experiment.py # Same study, full 2019-2026 Binance history, post-bugfix — see TL;DR (CLOSED)
│   ├── hmm_regime_direction_interaction_analysis.py # Interaction + tail-exceedance follow-up — closed the bear-veto question
│   ├── market_structure_fix_ab_backtest.py        # Pre-deploy A/B for the market_structure fix
│   ├── market_structure_fix_ab_multitimeframe.py  # Same A/B, joint multi-timeframe replay (higher fidelity)
│   ├── smc_confidence_matrix_backtest.py # SMC confidence-weighting bonuses — result: mixed, not deployed
│   ├── backtest_stoploss_matrix.py       # Stop-loss placement variants — ATR rejected, wider FVG-anchored mixed, not deployed
│   ├── backtest_momentum_coverage_experiment.py  # Momentum overlay for SMC/FVG's coverage gap — result: rejected 9/9, not deployed
│   ├── backtest_htf_bias_experiment.py    # Higher-timeframe bias filter — research candidate, see TL;DR
│   ├── backtest_concurrence_experiment.py # Timeframe concurrence (observational, crypto) — full_agreement strongest, see Contributing
│   ├── backtest_signal_age_experiment.py  # Signal age/decay (observational, crypto 15m) — 0-2h weakest, 8-24h strongest, see Contributing
│   ├── backtest_meta_filter_experiment.py # Regime/concurrence/signal-age combined as a meta pre-filter — see RESEARCH_NOTES.md
│   ├── audit_news_coverage.py             # Quantifies news/Macro_Event coverage per symbol — found stock has ~0 coverage, see TL;DR
│   ├── ta_indicators.py                   # Shared causal (no-lookahead) indicator library — RSI/MACD/Bollinger/Stochastic/CCI/OBV/Donchian/ADX/ATR series
│   ├── vwap_confluence_ab_experiment.py   # VWAP directional-confluence veto — result: rejected, see TL;DR
│   ├── confluence_matrix_ab_experiment.py # EMA/RSI/volume 3-filter combination matrix — see TL;DR
│   ├── ta_full_survey_experiment.py       # Two-phase screen (10 indicators individually, then combine survivors)
│   ├── ema50_robustness_validation.py     # EMA50-specific multi-regime/parameter/concentration/significance validation
│   ├── full_ta_robustness_validation.py   # All 11 filters, multi-regime (2yr, 6 segments) — 0 regime-robust, see TL;DR
│   ├── combo_multiregime_robustness.py    # Combinations of the best individual filters, same bar — still 0 robust
│   └── standalone_ta_strategies_robustness.py # SMC/FVG vs 5 classic standalone TA strategies with no SMC involvement — see TL;DR
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
