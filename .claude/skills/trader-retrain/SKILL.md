---
description: Checks if the Neo4j graph has enough edges, then retrains the GNN model and reports whether signal quality improved. Use when the user wants to retrain the model or improve trading signals.
allowed-tools: Bash, Read, PowerShell
---

# /trader-retrain

Retrain the HGT GNN model for the TraderStuff trading system. Follow these steps exactly.

## Steps

1. **Check graph readiness** — run `python check_graph.py` from `d:\Project\TraderStuff` (venv active). Read the TOTAL EDGES count.
   - If TOTAL EDGES < 20: tell the user the graph is still too sparse to retrain meaningfully. Tell them to wait at least 1-2 more hours and check again with `/trader-status`. STOP here.
   - If TOTAL EDGES >= 20: proceed.

2. **Run training** — run `python train_gnn.py` from `d:\Project\TraderStuff`. Capture full output.

3. **Evaluate result** — after training completes, check the final loss and the labels printed:
   - Loss < 0.3: excellent — signals are meaningful
   - Loss 0.3–0.6: decent — model converging, retrain again tomorrow
   - Loss > 0.6: still sparse — report this and explain more edges needed

4. **Signal diversity check** — look at the printed labels. Are they spread across both positive and negative values? If all labels are near 0 or all the same sign, explain why that's a weak training signal.

5. **Recommend next action**:
   - If loss improved vs last run (compare to any previous loss values in the conversation): say "good progress, restart main.py to apply the new model"
   - If loss got worse: explain why (new edge types added → model rebuilt from scratch, which is normal) and say this will self-correct as more edges accumulate
   - Always tell user: `python main.py` to restart and apply the new weights

Output a brief 3-line summary: graph edges / final loss / what to do next.
