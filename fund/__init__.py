"""Daily trend fund: the C4 DONCH-VT strategy run once a day, paper first.

Signals come from research/strategies.py unchanged, and the trading rule in
fund/core.py is a one-step copy of research/engine.py's loop. A test
(tests/test_fund.py) replays history through the fund's code path and checks
it reproduces the research backtest to floating-point precision, so what runs
live is what was measured.
"""
