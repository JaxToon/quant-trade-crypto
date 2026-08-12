---
description: Analyzes why TraderStuff isn't trading and suggests concrete config changes to settings.py to improve signal quality, lower thresholds, or reduce veto sensitivity. Use when the system has been running for hours but still hasn't placed a trade.
allowed-tools: Read, Grep, Edit
---

# /trader-tune

Diagnose why TraderStuff isn't trading and apply targeted parameter fixes. This is not a general review — focus only on changes that directly unblock trades or improve GNN signal quality.

## Steps

1. **Read current config** — read `d:\Project\TraderStuff\config\settings.py` in full. Note these key values:
   - TRADE_THRESHOLD
   - W1_BULL, W2_BEAR
   - MAX_POSITION_PCT
   - HMM_TURBULENT_STATE, HMM_N_STATES
   - GNN_HIDDEN_CHANNELS, GNN_NUM_LAYERS

2. **Read recent log** — read last 80 lines of `d:\Project\TraderStuff\trader.log`. Identify:
   - Is bear veto always active? (HMM stuck turbulent)
   - Are GNN signals flat (all near 0 or all -1)?
   - Are there any "P(Trade)" log lines? What values?
   - Is the regime ever anything other than HIGH_STRESS_TURBULENT?

3. **Read graph edge count** — run `python check_graph.py`. Get TOTAL EDGES.

4. **Diagnose and apply fixes** — based on what you find, apply ONLY the changes that are clearly warranted:

   **If HMM is always TURBULENT and yfinance is the data source:**
   - The HMM was trained on Binance data but is now fed yfinance data — scale might differ
   - Consider: in `risk/hmm_regime.py`, the `detect_regime` fallback behavior
   - Do NOT retrain HMM (it requires Binance data which is geo-blocked)
   - Instead: in `config/settings.py`, check if HMM_TURBULENT_STATE is correctly set to 2

   **If GNN signals are all flat (near 0.0):**
   - Lower the signal filter in `main.py` decision_loop from 0.15 to 0.05
   - This lets weak signals through to the debate stage

   **If TRADE_THRESHOLD is >= 0.55 and no trades ever fire:**
   - Lower TRADE_THRESHOLD to 0.45 in settings.py (temporarily, for testnet)
   - This is safe in testnet mode

   **If W1_BULL=0.6 and W2_BEAR=0.4 and market is generally bullish (SOL/ETH positive):**
   - Shift to W1_BULL=0.65, W2_BEAR=0.35 to let bullish signals through more easily

   **If graph edges < 30:**
   - Do NOT change GNN architecture yet — too few edges to benefit from tuning
   - Just report: wait for more edges

5. **Apply changes** using the Edit tool. Show each change as a before/after diff inline in your response.

6. **Tell user what to do** — after edits: "Restart main.py to apply changes" or "No restart needed for X change."

Only make changes you are confident will help. If you are unsure, explain the tradeoff and ask the user to decide. Never change STOP_LOSS_PCT or TAKE_PROFIT_PCT without explicit user approval.
