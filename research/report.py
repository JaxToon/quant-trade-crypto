"""Render research/results/*.json as markdown tables, so numbers reach the docs
without being copied by hand.

Usage: python -m research.report dev|holdout|yearly
"""
from __future__ import annotations

import json
import math
import os
import sys

OUT = os.path.join("research", "results")


def _pct(v, d=1):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:+.{d}f}%"


def _num(v, d=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


def _plain(v, d=0):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:.{d}f}%"


def table(results: dict, with_null: bool) -> str:
    head = ("| Strategi | CAGR | Sharpe | CI 95% Sharpe | MaxDD | Calmar | Win rate | Trade/thn | "
            "Ekspektasi/trade | Eksposur rata-rata |" + (" Uji acak p95 |" if with_null else ""))
    sep = "|---" * (10 + (1 if with_null else 0)) + "|"
    rows = [head, sep]
    for name, s in results.items():
        if not s:
            rows.append(f"| {name} | (tidak ada data) |")
            continue
        bench = name.startswith("BH-")
        ci = s.get("sharpe_ci")
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        wr = "—" if bench else _plain(s.get("win_rate"))
        tpy = "—" if bench else _num(s.get("trades_per_year"), 1)
        exp = "—" if bench else _pct(s.get("expectancy_pct"), 2)
        row = (f"| {name} | {_pct(s['cagr'])} | {_num(s['sharpe'])} | {ci_s} | {_plain(s['maxdd'], 1)} | "
               f"{_num(s['calmar'])} | {wr} | {tpy} | {exp} | {_plain(s.get('avg_exposure'))} |")
        if with_null:
            row += f" {_num(s.get('null_p95'))} |" if "null_p95" in s else " n/a |"
        rows.append(row)
    return "\n".join(rows)


def yearly_table(results: dict, names: list[str]) -> str:
    years = sorted({int(y) for n in names if results.get(n) for y in results[n]["yearly"]})
    head = "| Tahun | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    rows = [head, sep]
    for y in years:
        cells = [_pct(results[n]["yearly"].get(str(y), results[n]["yearly"].get(y)), 0)
                 if results.get(n) else "—" for n in names]
        rows.append(f"| {y} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "dev":
        sel = json.load(open(os.path.join(OUT, "selection.json")))
        print(table(sel["dev_results"], with_null=True))
    elif what == "holdout":
        h = json.load(open(os.path.join(OUT, "holdout.json")))
        print(table(h["results"], with_null=False))
    elif what == "yearly":
        src = sys.argv[2]
        names = sys.argv[3].split(",")
        d = json.load(open(os.path.join(OUT, src)))
        res = d.get("dev_results") or d.get("results")
        print(yearly_table(res, names))
