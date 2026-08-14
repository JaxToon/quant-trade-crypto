"""
TraderStuff monitoring dashboard API.
Serves the HTML dashboard and provides /api/status JSON.

Run: python run_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from graph.neo4j_client import Neo4jClient

app = FastAPI(title="TraderStuff Dashboard")

DASHBOARD_DIR = Path(__file__).parent
# Asset-type-suffixed to match main.py's write path — lets a crypto process
# and a stock process run side by side (different DASHBOARD_PORT each)
# without one silently overwriting the other's status file.
STATUS_FILE = DASHBOARD_DIR / f"status_{settings.ASSET_TYPE}.json"
LOG_FILE = Path(__file__).parent.parent / "trader.log"


def _read_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "timestamp": None,
        "regime": "UNKNOWN",
        "bear_veto": None,
        "gnn_signals": {},
    }


def _read_log_tail(n: int = 60) -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except Exception:
        return []


def _query_neo4j() -> dict:
    try:
        db = Neo4jClient.get()
        nodes = db.run_read(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
        )
        edges = db.run_read(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
        )
        trades = db.run_read(
            "MATCH (t:Trade) RETURN t ORDER BY coalesce(t.timestamp,0) DESC LIMIT 10"
        )
        return {
            "nodes": {r["label"]: r["cnt"] for r in nodes},
            "edges": {r["rel"]: r["cnt"] for r in edges},
            "total_nodes": sum(r["cnt"] for r in nodes),
            "total_edges": sum(r["cnt"] for r in edges),
            "recent_trades": [dict(r["t"]) for r in trades],
        }
    except Exception as e:
        return {
            "nodes": {},
            "edges": {},
            "total_nodes": 0,
            "total_edges": 0,
            "recent_trades": [],
            "error": str(e),
        }


@app.get("/")
def index():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/api/status")
def status():
    live = _read_status()
    graph = _query_neo4j()
    logs = _read_log_tail(60)

    # Parse regime from log if status file missing
    if live["regime"] == "UNKNOWN" and logs:
        for line in reversed(logs):
            if "HIGH_STRESS_TURBULENT" in line:
                live["regime"] = "HIGH_STRESS_TURBULENT"
                live["bear_veto"] = True
                break
            elif "NORMAL" in line and "REGIME" in line:
                live["regime"] = "NORMAL"
                live["bear_veto"] = False
                break

    return JSONResponse({
        **live,
        "graph": graph,
        "recent_logs": logs,
        "server_time": datetime.utcnow().isoformat(),
    })
