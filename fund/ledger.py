"""The fund's book: cash, quantities, signal memory and a full run history.

One JSON file per mode, written atomically. The SMC paper ledgers
(paper_ledger_*.json) are a different file and are never touched here.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field


@dataclass
class Book:
    mode: str
    start_capital: float
    cash: float
    qty: dict[str, float] = field(default_factory=dict)
    last_sig: dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    runs: list[dict] = field(default_factory=list)      # one per trading day
    trades: list[dict] = field(default_factory=list)    # one per fill

    @property
    def last_run_date(self) -> str | None:
        return self.runs[-1]["date"] if self.runs else None


def load(path: str, mode: str, start_capital: float) -> Book:
    if not os.path.exists(path):
        return Book(mode=mode, start_capital=start_capital, cash=start_capital,
                    peak_equity=start_capital)
    with open(path, encoding="utf-8") as f:
        return Book(**json.load(f))


def save(book: Book, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(book), f, indent=1)
    os.replace(tmp, path)
