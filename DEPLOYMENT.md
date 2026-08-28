# Deployment & Persistence

This file exists to answer one question precisely: **if I run this somewhere
other than my own machine, what state needs to survive a restart or a
redeploy, and what's safe to lose?**

## What MUST persist (the model's accumulated learning)

Crypto and stock are two entirely separate models — see "Crypto vs stock"
below. Every path in this table is suffixed with the active `ASSET_TYPE`
(`crypto` or `stock`) for exactly that reason.

| Path | What it is | Lost if not persisted |
|---|---|---|
| `model_checkpoints/hgt_latest_<asset_type>.pt` | GNN weights — bootstrapped by `train_gnn.py`, then fine-tuned automatically every `LEARNING_INTERVAL_HOURS` by `main.py`'s scheduled paper-trade learning job | All GNN training for that asset class, including everything it's learned from live paper-trade outcomes since bootstrap |
| `model_checkpoints/hmm_regime.pkl` | HMM regime detector, fit once from 60 days of history (shared across asset classes — regime detection reads Binance BTC data regardless of `ASSET_TYPE`) | Regime detection falls back to refitting on next start (slow, not catastrophic) |
| `backtrack_logs/paper_ledger_<asset_type>.json` | Live paper-trading ledger for that asset class: balance, every closed trade, win rate, and `last_learned_count` (the GNN learning job's bookmark) | That asset class's entire win-rate track record, and its GNN learning job would treat all history as new |
| **Neo4j database itself** | All Token/Protocol/Macro_Event nodes and edges — the knowledge graph the GNN reads. Crypto and stock Token nodes coexist in the same database, distinguished by an `asset_type` property (see below) | Every relationship learned from news enrichment, LLM causal discovery, and the base seed graph, for both asset classes |

`backtest_ledger*.json` and `dashboard/status.json` are **not** in this list —
they're regenerated from scratch each run and aren't part of what the system
has "learned."

## Crypto vs stock: separate models, one graph

Switching `.env`'s `ASSET_TYPE` between `crypto` and `stock` does NOT share
learning between them — each gets its own GNN checkpoint and paper ledger
file (the `_<asset_type>` suffix above), and `gnn/dataset.build_hetero_data()`
filters Neo4j `Token` nodes by `asset_type` before building the graph the GNN
trains/predicts on. This is deliberate: a model trained on BTC/ETH price
action has no business predicting AAPL, and vice versa — sharing one
checkpoint between them would load without error but produce meaningless
predictions, silently.

Both asset classes' Token nodes live in the *same* Neo4j database — `graph/schema.py`
seeds both crypto and stock tokens unconditionally, tagged with
`asset_type: "crypto"` or `asset_type: "stock"`. Edges between two crypto
tokens or two stock tokens work as normal; there's no meaningful edge type
connecting a crypto token to a stock token yet, so the two subgraphs are
disjoint in practice even though they share a database.

To train the stock model for the first time: `ASSET_TYPE=stock` in `.env`,
then `python train_gnn.py` (it already knows to fetch `STOCK_WATCHLIST`
tickers instead of crypto ones). `python backtest.py` and `python main.py`
follow `ASSET_TYPE` the same way.

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

## Neo4j: currently local Neo4j Desktop, not Docker (2026-08-28)

`NEO4J_URI` points at `neo4j://127.0.0.1:7687`, the user's native Neo4j
Desktop instance — switched off a Docker container (`neo4j-trader`) after
Docker Desktop's own reliability (crashes, "manually paused" states,
especially after laptop sleep) was the dominant operational failure mode
for weeks. The switch was a straightforward MERGE-keyed migration (no
`neo4j-admin dump/load` needed — that risks Neo4j edition/version
mismatches between a Docker image and a Desktop install; a plain Cypher
read-from-source/MERGE-into-target script is more portable and was used
instead) and lost nothing — Docker's own data turned out to be a
strict-ish subset of what local already had from an earlier period
before something switched the live system onto Docker.

Practical consequence for the recovery scripts: `start_system.sh`/
`stop_system.sh` no longer have a Docker fallback for Neo4j at all — if
the local Desktop DBMS isn't running, there's no CLI way to start it (it's
a GUI app), so a monitoring loop can only flag "Neo4j unreachable, start
it manually" rather than self-heal the way it could restart Docker
Desktop's process.

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
