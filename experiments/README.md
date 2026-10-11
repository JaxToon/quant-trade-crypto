# Experiments index

Every script here is a self-contained research question, run as
`python -m experiments.<name>` from the repo root. The second column is the
script's own docstring (first paragraph), not a verdict: results and verdicts
live in the script's output, in `docs/FINDINGS_LOG.md`, and in
`RESEARCH_NOTES.md` (local).

> **Before trusting any result here:** until 2026-10-11 every experiment ran on
> the paper ledger that books entries at the FVG midpoint, about 0.5R better
> than the market price available at the signal. Absolute PnL is optimistic,
> and A/B comparisons that changed the number of trades favoured the arm with
> more trades. `entry_fill_realism.py` measures the bias. Most scripts also
> used a 0.1% fee instead of the real 0.3822% (`real_fee_viability.py`).
>
> Nearly all backtests fetch a rolling `period=<days>d` window from yfinance
> (1h up to 730 days, 5m/15m up to 60 days), so re-running one today reads a
> different window than the one it originally reported on. The exceptions that
> read Binance klines are listed by `grep -l klines experiments/*.py`.

| Script | Question it asks |
|---|---|
| `audit_news_coverage` | News/event coverage audit -- diagnoses the stock-vs-crypto Macro_Event imbalance found live (2026-08-21): 255 crypto Token<-Macro_Event edges vs. 4 stock ones, despite equal token counts (8 crypto, 8 stock). |
| `backtest_concurrence_experiment` | Timeframe concurrence experiment — Path 1 (SMC/FVG paper trading) ONLY. Step 2 of the agreed entry-selection research roadmap (see README), run crypto-first per the standing rule: crypto establishes or disproves the ... |
| `backtest_htf_bias_experiment` | Higher-timeframe (HTF) bias filter experiment — Path 1 (SMC/FVG paper trading) ONLY. |
| `backtest_meta_filter_experiment` | Cheap gate before Stage 2 (v3 track) -- tests whether a SIMPLE RULE combining already-validated signals (full timeframe concurrence, direction) beats the current baseline (v2.5, unfiltered), before committing to a ... |
| `backtest_momentum_coverage_experiment` | Momentum-coverage-gap experiment — Path 1 (SMC/FVG paper trading) ONLY. |
| `backtest_signal_age_experiment` | Signal age/decay experiment — Path 1 (SMC/FVG paper trading) ONLY. Step 3 of the agreed entry-selection research roadmap (see README), crypto-first per the standing rule. |
| `backtest_stoploss_matrix` | Stop-loss placement backtest matrix — Path 1 (SMC/FVG paper trading) ONLY. |
| `backtest_target_matrix` | Target/exit backtest matrix — Path 1 (SMC/FVG paper trading) ONLY. |
| `breakeven_cap_honest_check` | Is the BE_0.5R breakeven "edge" real, or intra-candle look-ahead bias? |
| `breakeven_deep_validation` | Deep validation of the BE_0.5R breakeven-stop finding before deploying it. |
| `breakeven_exposure_check` | Final deploy gate for the breakeven-stop finding: does it quietly increase CONCURRENT portfolio exposure? |
| `breakeven_lookahead_check` | Is the BE_0.5R breakeven "edge" real, or intra-candle look-ahead bias? |
| `breakeven_under_slippage` | The deploy gate for the breakeven finding: does its edge survive the one cost we have actually MEASURED but never modelled? |
| `breakeven_vs_capacity_disambiguation` | Is BE_0.5R (breakeven stop at 0.5R) a genuine exit improvement, or just another proxy for the correlation-cap capacity lever? |
| `combo_multiregime_robustness` | Extends the full 11-filter multi-regime robustness pass to COMBINATIONS, using the exact same "must win the segment on PnL, across all 6 segments of ~2 years of real data" bar established in ... |
| `confluence_matrix_ab_experiment` | Pre-registered A/B/combination-matrix experiment: does layering established, non-ML technical confluence filters on top of the existing SMC/FVG entry pipeline improve trade quality — individually or combined? Run ... |
| `correlation_cap_drawdown_sweep` | Can a TIGHTER correlation cap bring drawdown into survivable territory? |
| `correlation_cap_vs_short_sizing` | Competing-hypothesis test flagged by short_halfsize_deep_validation.py's mechanism analysis: SHORT_20PCT's edge turned out to come almost entirely from freeing portfolio correlation-cap capacity (779 extra SHORT ... |
| `deflated_sharpe_audit` | Does THIS project's own result survive a search-aware correction? |
| `directional_sizing_robustness` | Tests the research question RESEARCH_NOTES.md's regime x direction arc explicitly "parked, not pursued": SMC direction carries a real directional main effect (bullish consistently outperforms bearish, confirmed ... |
| `drawdown_at_real_slippage` | What drawdown does the live strategy actually carry at its MEASURED slippage — and would a real account have survived it? |
| `drawdown_control_ab` | Can this strategy's drawdown be brought into survivable territory without destroying its returns? |
| `drawdown_methodology_check` | Is every drawdown number this project has reported understated? |
| `ema50_robustness_validation` | Robustness validation for the EMA(50) crypto entry filter before any live deployment — closes four specific gaps the original 90-day, single-parameter test could not rule out: |
| `entry_fill_realism` | How much of this strategy's profit comes from booking entries at a price the market never offered? |
| `equity_curve_shape` | What does the equity curve actually LOOK like — and does a 68% drawdown mean losing your capital, or giving back profits? |
| `fetch_hf_stock_ohlcv` | One-off data-engineering script: pulls our 8 STOCK_WATCHLIST tickers' 1-minute OHLCV history from the mito0o852/OHLCV-1m HuggingFace dataset (https://huggingface.co/datasets/mito0o852/OHLCV-1m, Finnhub-sourced, ... |
| `forming_candle_entry_effect` | Does entering off a FORMING candle specifically damage shorts? |
| `full_ta_robustness_validation` | Extends ema50_robustness_validation.py's multi-regime test to EVERY technical-analysis filter tested so far in this project — not just EMA50. |
| `gnn_directional_effect` | Does the GNN signal specifically DEGRADE shorts while helping longs? |
| `gnn_threshold_ablation` | Is the GNN genuinely uninformative, or has it simply been gated out? |
| `hmm_regime_ab_experiment` | Controlled A/B experiment: does a 2-state HMM regime classifier separate market conditions any better than the live 3-state one, and — the question that actually matters — does either model's "turbulent" flag ... |
| `hmm_regime_direction_binance_experiment` | Regime x direction decomposition, rebuilt on Binance's full historical klines (2026-08-21) -- extends hmm_regime_direction_experiment.py's 700-day yfinance result, which found turbulent+bearish consistently worst ... |
| `hmm_regime_direction_experiment` | Regime x direction decomposition study (2026-08-21). |
| `hmm_regime_direction_interaction_analysis` | Interaction + tail-exceedance follow-up on the (now-fixed) regime x direction study (2026-08-21) -- closes the research round started after the market_structure freeze bug fix, per the explicit decision NOT to commit ... |
| `hmm_regime_filter_backtest` | Regime-filtered Path 1 backtest — controlled A/B/C, nothing deployed. |
| `hmm_windowed_decode_experiment` | Validates the windowed-decode fix for detect_regime()'s single-observation Viterbi bug BEFORE touching risk/hmm_regime.py or any live checkpoint. |
| `intracandle_evaluation_effect` | Does re-evaluating entries many times per candle change WHICH setups get taken — the last standing explanation for the live/backtest SHORT gap? |
| `leverage_liquidation_eval` | Does the futures path survive its own liquidation risk? The evaluation that decides spot vs futures. |
| `live_vs_backtest_matched_window` | Does the backtest actually describe the live system? Matched-window, per-direction comparison. |
| `long_only_spot_config` | Can we deploy LONG-ONLY right now, and stop waiting for the SHORT question? |
| `long_only_viability` | Can we deploy LONG-ONLY right now, and stop waiting for the SHORT question? |
| `market_structure_alignment_filter` | New hypothesis from live-trade forensics (2026-08-28): the only 6 crypto SHORT trades matchable against the candidate log (which logs SMC's own market_structure reading at decision time) were ALL taken while ... |
| `market_structure_fix_ab_backtest` | A/B backtest for the market_structure.trend fix (2026-08-21) -- answers a narrower, more honest question than "is the fix good" (it is; it's a correctness fix, see strategy/smart_money.py and ... |
| `market_structure_fix_ab_multitimeframe` | A/B backtest for the market_structure fix, take 2 (2026-08-22) -- fixes a real methodology flaw in market_structure_fix_ab_backtest.py: that script replayed each timeframe (5m, 15m, 1h) through its OWN separate ... |
| `mfe_mae_exit_analysis` | MFE/MAE (Maximum Favorable / Adverse Excursion) analysis, split by trade direction — asks whether this system's SHORT problem is an EXIT-geometry problem rather than the entry-selection problem every prior experiment ... |
| `notional_cap_spot_viability` | What does it cost to make this strategy fit inside a SPOT account? |
| `per_timeframe_layer_analysis` | Is one of the three concurrent SMC/FVG timeframe layers (5m / 15m / 1h) a net drag on the system? |
| `real_fee_viability` | Does the strategy survive Tokocrypto's REAL fees? |
| `reconstruct_historical_slippage` | Reconstructs the stop slippage our live system ACTUALLY paid, from trades already in the ledger — and simulates what the 1s SL/TP monitor would have paid on those same trades. |
| `restart_fvg_memory_effect` | Does restarting the process truncate FVG memory enough to change WHICH setups get taken — and does that hit shorts hardest? |
| `short_by_regime` | Is the live/backtest SHORT gap just REGIME? |
| `short_halfsize_deep_validation` | Deep validation for SHORT_20PCT (the directional-sizing finding from directional_sizing_robustness.py -- robust 6/6 segments, broad plateau 5-25%, crypto-only) before it's trusted enough to deploy. Applies the same ... |
| `short_live_evidence_power` | Is the accumulated live evidence enough to conclude SHORT loses money? |
| `short_setup_attribution` | Is the live SHORT gap a SELECTION problem or an EXECUTION problem? |
| `small_capital_projection` | What would $20 actually do in one month — and is it even tradeable? |
| `smc_confidence_matrix_backtest` | SMC confidence-weighting backtest matrix — Path 1 only, quantitative experiment. Does NOT modify strategy/fvg_analyzer.py, entry_insight.py, paper_trading.py, or touch any live process. Reuses the same real ... |
| `spot_viable_risk_sizing` | What risk-per-trade makes this strategy actually executable on SPOT, and what does it cost? |
| `standalone_ta_strategies_robustness` | Every prior filter test in this research line (VWAP, the 3-filter matrix, the 10-indicator survey, all robustness passes) ONLY vetoes the existing SMC/FVG baseline's own signals — it can never generate a trade the ... |
| `stock_training_data_ab` | Does sourcing the STOCK GNN's training set from real 1-minute candles (data/stock_ohlcv_1m/, fetched by experiments/fetch_hf_stock_ohlcv.py) actually beat the current production path (yfinance 1h/730d)? |
| `ta_full_survey_experiment` | Full technical-analysis confluence survey, two phases, per user request to test "semua model technical analysis yang ada" (all TA models) rather than just VWAP/EMA/RSI/VOL. |
| `ta_indicators` | Causal (no-lookahead) technical-indicator series, shared by every AB/matrix experiment script in this repo that needs more than the single-latest-value helpers in train_gnn.py (_rsi, _ema, _macd, _bollinger only ... |
| `trailing_breakeven_robustness` | Tests dynamic stop movement (breakeven + ATR trailing) against baseline — the one exit-side lever this project has never evaluated. |
| `vwap_confluence_ab_experiment` | Pre-registered A/B experiment: does adding a VWAP directional-confluence filter to the existing SMC/FVG entry pipeline improve trade quality? |
