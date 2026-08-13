# Deployment & Persistence

This file exists to answer one question precisely: **if I run this somewhere
other than my own machine, what state needs to survive a restart or a
redeploy, and what's safe to lose?**

## What MUST persist (the model's accumulated learning)

| Path | What it is | Lost if not persisted |
|---|---|---|
| `model_checkpoints/hgt_latest.pt` | GNN weights — bootstrapped by `train_gnn.py`, then fine-tuned automatically every `LEARNING_INTERVAL_HOURS` by `main.py`'s scheduled paper-trade learning job | All GNN training, including everything it's learned from live paper-trade outcomes since bootstrap |
| `model_checkpoints/hmm_regime.pkl` | HMM regime detector, fit once from 60 days of history | Regime detection falls back to refitting on next start (slow, not catastrophic) |
| `backtrack_logs/paper_ledger.json` | Live paper-trading ledger: balance, every closed trade, win rate, and `last_learned_count` (the GNN learning job's bookmark) | The entire win-rate track record, and the GNN learning job would treat all history as new |
| **Neo4j database itself** | All Token/Protocol/Macro_Event nodes and edges — the knowledge graph the GNN reads | Every relationship learned from news enrichment, LLM causal discovery, and the base seed graph |

`backtest_ledger*.json` and `dashboard/status.json` are **not** in this list —
they're regenerated from scratch each run and aren't part of what the system
has "learned."

## What SHOULD persist (dashboard continuity, not model learning)

| Path | What it is | Lost if not persisted |
|---|---|---|
| `dashboard/history.json` | Rolling window (~3.5 days at the 60s decision-loop cadence) feeding the dashboard's equity-curve and price-movement charts | Charts reset to blank on every restart instead of showing continuity across runs — cosmetic only, doesn't affect the GNN or trading decisions |

This is a separate category on purpose: it makes the dashboard *feel*
continuous across restarts, but the model doesn't read it for anything. If
you're deploying under real storage constraints and have to choose, the
table above (checkpoints, ledger, Neo4j) is what actually matters.

## Why this repo doesn't just commit those files to git

`.gitignore` excludes `model_checkpoints/`, `backtrack_logs/`, and
`dashboard/status.json` / `dashboard/history.json` on purpose. Git tracks
source code; these are
frequently-changing runtime state (the checkpoint gets rewritten every few
hours, the ledger grows every trade). Committing them would bloat the repo's
history with binary diffs and doesn't actually solve persistence — a fresh
`git clone` on a new machine would still start from whatever was last
committed, not from live state.

**The actual fix is a persistent volume/disk at deploy time**, mounted at
those two directories, independent of git. GitHub is for versioning the
*code* — it was never going to solve this by itself, which is also why
GitHub Actions (ephemeral VM per run, no disk between runs, 6h max runtime)
isn't a viable host for `main.py` as a 24/7 process.

## Neo4j: moving off `localhost`

Nothing in the code hardcodes `bolt://localhost:7687` — `graph/neo4j_client.py`
passes `NEO4J_URI` straight to the driver, so any valid URI works, including
Neo4j AuraDB's `neo4j+s://` scheme. To move to Aura (managed, free tier,
persists automatically on Neo4j's side — no volume to manage yourself):

1. Create a free instance at https://neo4j.com/product/auradb/
2. Copy the connection URI it gives you (`neo4j+s://<id>.databases.neo4j.io`)
3. Update `.env`: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
4. Run `python check_graph.py` to confirm it connects. `initialize_schema()`
   (called at `main.py` startup) will reseed the base tokens/protocols via
   `MERGE`, so a fresh Aura instance self-bootstraps — no manual import
   needed, though you'll lose whatever edges accumulated in your old local
   graph unless you export it first (`neo4j-admin database dump`).

## When you do pick a host

Whatever you choose (VPS, Fly.io, Railway, home server, etc.), the hard
requirement is: **mount a persistent volume at `model_checkpoints/` and
`backtrack_logs/`** (the model's actual learning), relative to wherever
`main.py` runs from. Include `dashboard/` too if you want the charts to
carry over across redeploys — optional, doesn't affect the model. Everything
else in this doc stays true regardless of the specific platform — ask again
when you're ready to pick one and this can be made concrete (Dockerfile,
systemd unit, restart policy, etc.).

## Until then

Running `python main.py` still requires *some* machine to stay on — for now
that's your laptop, same as before. This document just makes sure that
whenever you do move it off your laptop, the learning isn't silently reset
to zero.
