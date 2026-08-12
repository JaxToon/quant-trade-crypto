---
description: Full health check of the TraderStuff trading system — shows regime, bear veto, GNN signals, graph edge count, and last log lines. Use when the user wants to know if the system is healthy or why it's not trading.
allowed-tools: Bash, Read, Grep
---

# /trader-status

Run a full system health check on the TraderStuff trading system. Do all steps sequentially and present a clean summary.

## Steps

1. **Graph health** — run `python check_graph.py` from `d:\Project\TraderStuff` with the venv active. Show node and edge counts.

2. **Log tail** — read the last 60 lines of `d:\Project\TraderStuff\trader.log`. Look for:
   - Current regime label (HIGH_STRESS_TURBULENT / NORMAL / LOW_VOLATILITY)
   - Whether "Bear veto active" appears (means no trades are firing)
   - GNN signal values for each symbol
   - Any ERROR lines
   - Any Edge write confirmations from the graph populator

3. **Summary table** — output a markdown table with these rows:

| Check | Status | Detail |
|---|---|---|
| Neo4j edges | 🔴/🟡/🟢 | count (red <20, yellow 20-50, green >50) |
| Regime | 🔴/🟡/🟢 | label |
| Bear veto | 🔴/🟢 | active / inactive |
| GNN signals | range | min to max across symbols |
| Errors last hour | count | — |

4. **Verdict** — one sentence: what is the single biggest thing blocking the first trade from firing, and what should the user do next (wait for graph to grow / retrain / check internet / market is calm enough).

Be concise. No code blocks except for the table. Do not repeat raw log lines verbatim.
