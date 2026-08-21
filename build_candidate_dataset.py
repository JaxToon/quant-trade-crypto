"""
Stage 1 (cont'd), entry-selection research line: offline dataset builder.
Joins the candidate-trade log (dashboard/candidate_trades_{asset_type}.json
-- features at decision time, see main.py's _record_candidate()) against
the paper ledger's actual outcomes (backtrack_logs/paper_ledger_{asset_type}.json)
to produce one training-ready table: candidate features -> realized label.

Deliberately built and testable BEFORE the candidate log has accumulated
real data -- this is pure code, the join logic doesn't need live data to be
correct, only to be useful. Run for real once Stage 1 has been live long
enough to matter.

Join key: (symbol, market, timeframe) + nearest opened_at within
JOIN_TOLERANCE_SECONDS -- not an exact timestamp match. main.py captures
the candidate log's opened_at a few microseconds before maybe_open()'s own
internal timestamp (no I/O in between), so they're expected to differ by
well under a second in practice; the tolerance window exists to be robust
to that, not because the two are expected to diverge meaningfully. False
matches are unlikely given the 60s decision-loop cadence -- two trades on
the exact same (symbol, market, timeframe) within a few seconds of each
other essentially never happens.

Label design (matches the reviewed playbook's spec):
  "accepted" candidate, later in closed_trades  -> "win" | "loss" (from
    the ledger's own outcome field), pnl_dollars carried through
  "accepted" candidate, still in open_positions -> "pending" (not yet
    resolved -- excluded from any training use until it resolves)
  "accepted" candidate, found in neither        -> "unmatched" (should be
    rare; flagged for debugging, not silently dropped)
  "rejected" candidate                          -> "rejected" (never got
    capital -- a distinct category from a resolved loss, not folded into it)

Usage: python build_candidate_dataset.py [--asset-type crypto|stock]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

JOIN_TOLERANCE_SECONDS = 10.0


def _load_json(path: str) -> dict | list:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "").split("+")[0])
    except ValueError:
        return None


def _build_outcome_index(closed_trades: list[dict]) -> dict[tuple, list[dict]]:
    """(symbol, market, timeframe) -> [trade dicts], for nearest-timestamp lookup."""
    index: dict[tuple, list[dict]] = {}
    for t in closed_trades:
        key = (t.get("symbol"), t.get("market"), t.get("timeframe"))
        index.setdefault(key, []).append(t)
    return index


def _build_open_index(open_positions: list[dict]) -> dict[tuple, list[dict]]:
    index: dict[tuple, list[dict]] = {}
    for p in open_positions:
        key = (p.get("symbol"), p.get("market"), p.get("timeframe"))
        index.setdefault(key, []).append(p)
    return index


def _find_nearest(candidates: list[dict], target_ts: datetime, tolerance_s: float) -> dict | None:
    best, best_delta = None, None
    for c in candidates:
        ts = _parse_ts(c.get("opened_at"))
        if ts is None:
            continue
        delta = abs((ts - target_ts).total_seconds())
        if delta <= tolerance_s and (best_delta is None or delta < best_delta):
            best, best_delta = c, delta
    return best


def build_dataset(candidate_log: list[dict], closed_trades: list[dict], open_positions: list[dict]) -> list[dict]:
    outcome_index = _build_outcome_index(closed_trades)
    open_index = _build_open_index(open_positions)
    rows: list[dict] = []

    for cand in candidate_log:
        key = (cand.get("symbol"), cand.get("market"), cand.get("timeframe"))
        row = dict(cand)  # keep all original fields (features, entry/stop/target, etc.)

        if cand.get("decision") == "rejected":
            row["label"] = "rejected"
            row["pnl_dollars"] = None
            rows.append(row)
            continue

        target_ts = _parse_ts(cand.get("opened_at"))
        if target_ts is None:
            row["label"] = "unmatched"
            row["pnl_dollars"] = None
            rows.append(row)
            continue

        matched_closed = _find_nearest(outcome_index.get(key, []), target_ts, JOIN_TOLERANCE_SECONDS)
        if matched_closed is not None:
            row["label"] = matched_closed.get("outcome")  # "win" | "loss"
            row["pnl_dollars"] = matched_closed.get("pnl_dollars")
            row["closed_at"] = matched_closed.get("closed_at")
            rows.append(row)
            continue

        matched_open = _find_nearest(open_index.get(key, []), target_ts, JOIN_TOLERANCE_SECONDS)
        if matched_open is not None:
            row["label"] = "pending"
            row["pnl_dollars"] = None
            rows.append(row)
            continue

        row["label"] = "unmatched"
        row["pnl_dollars"] = None
        rows.append(row)

    return rows


def summarize(rows: list[dict]) -> None:
    from collections import Counter

    label_counts = Counter(r["label"] for r in rows)
    print(f"Total candidates: {len(rows)}")
    for label, n in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<10} {n}")

    resolved = [r for r in rows if r["label"] in ("win", "loss")]
    if resolved:
        wins = sum(1 for r in resolved if r["label"] == "win")
        pnl = sum(r["pnl_dollars"] for r in resolved if r["pnl_dollars"] is not None)
        print(f"\nResolved (win/loss): n={len(resolved)}  win_rate={wins / len(resolved) * 100:.1f}%  total_pnl={pnl:+.2f}")

    if label_counts.get("unmatched", 0) > 0:
        print(f"\nWARNING: {label_counts['unmatched']} unmatched accepted candidates -- "
              "either JOIN_TOLERANCE_SECONDS is too tight, or something is genuinely wrong. Investigate before trusting this dataset.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-type", default=None, help="crypto, stock, or omit for both")
    args = parser.parse_args()

    asset_types = [args.asset_type] if args.asset_type else ["crypto", "stock"]

    for asset_type in asset_types:
        candidate_path = os.path.join("dashboard", f"candidate_trades_{asset_type}.json")
        ledger_path = os.path.join("backtrack_logs", f"paper_ledger_{asset_type}.json")

        candidate_log = _load_json(candidate_path)
        ledger = _load_json(ledger_path)
        if not candidate_log:
            print(f"=== {asset_type}: no candidate log yet at {candidate_path} -- nothing to build. ===\n")
            continue

        closed_trades = ledger.get("closed_trades", [])
        open_positions = ledger.get("open_positions", [])

        rows = build_dataset(candidate_log, closed_trades, open_positions)

        print(f"=== {asset_type} ===")
        summarize(rows)

        out_path = os.path.join("dashboard", f"candidate_dataset_{asset_type}.json")
        with open(out_path, "w") as f:
            json.dump(rows, f)
        print(f"Saved {len(rows)} rows -> {out_path}\n")


if __name__ == "__main__":
    main()
