"""
Regression test for main.py's _run_paper_trading price wiring.

Live bug (2026-08-19): the first version of the already-invalidated-setup
guard (see tests/test_paper_trading.py) used insight["price"] as
current_price — but that's the signal's OWN timeframe's last candle price
(e.g. a 1h layer only updates once an hour), which can already be stale
relative to the live price check_and_close() checks moments later in the
same cycle. Comparing the stale price against itself always passed,
silently defeating the guard: NVDA kept re-opening and instantly
stopping out even after the guard was deployed. Fix: use prices[symbol]
(the same cross-timeframe price check_and_close() uses), not
insight["price"].
"""
from __future__ import annotations

from unittest.mock import MagicMock

import main


def _insight(symbol="NVDA", timeframe="1h", stale_price=219.084999):
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": stale_price,  # stale: this 1h layer's last candle price
        "confidence": 78.0,
        "spot": {
            "actionable": True,
            "direction": "long",
            "entry_zone": [stale_price, stale_price],
            "stop_loss": 217.524002,
            "target": 221.314995,
        },
        "derivative": {"actionable": False, "direction": "long"},
    }


def test_maybe_open_receives_live_price_not_stale_insight_price(monkeypatch):
    """The whole point of the guard: if the real live price (in `prices`)
    has already crossed the stop, maybe_open must be told that — not the
    signal's own stale per-timeframe price, which never crosses its own
    stop by construction."""
    mock_ledger = MagicMock()
    mock_ledger.open_positions = {}
    mock_ledger.maybe_open.return_value = None
    monkeypatch.setattr(main, "_paper_ledger", mock_ledger)
    monkeypatch.setattr(main, "_record_rejection", lambda *a, **kw: None)
    monkeypatch.setattr(main, "_record_candidate", lambda *a, **kw: None)  # not under test here

    insights = {"NVDA_1h": _insight()}
    live_prices = {"NVDA": 217.524002}  # already at/through the stop

    main._run_paper_trading(insights, live_prices)

    for call in mock_ledger.maybe_open.call_args_list:
        assert call.kwargs["current_price"] == 217.524002
        assert call.kwargs["current_price"] != 219.084999  # not the stale insight price
