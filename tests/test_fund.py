"""The fund must trade exactly the strategy that was researched, and its guards must hold."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from fund import broker as fbroker
from fund import config, core, ledger, market
from fund import run_daily
from research import engine
from research.data import CACHE_DIR, SYMBOLS

HAVE_DATA = all(os.path.exists(os.path.join(CACHE_DIR, f"{s}_1d.parquet")) for s in SYMBOLS)


def _panel():
    frames = {s: pd.read_parquet(os.path.join(CACHE_DIR, f"{s}_1d.parquet")) for s in SYMBOLS}
    return {f: pd.DataFrame({s: d[f] for s, d in frames.items()}).sort_index() for f in ("open", "close")}


def _simulate_fund(p, min_order=0.0):
    """Drive the fund's own decision code day by day over history."""
    close, open_ = p["close"], p["open"]
    sig, tgt = core.targets(close)
    syms = list(close.columns)
    O = open_.fillna(1.0)  # pre-listing coins hold nothing; any price works
    qty, cash, last = {}, 1.0, np.zeros(len(syms))
    eq = []
    for i in range(1, len(close) - 1):
        d_prev, d = close.index[i - 1], close.index[i]
        prices = O.loc[d].to_dict()
        _, qty, cash, last, _, _ = core.plan_orders(
            syms, qty, cash, prices, tgt.loc[d_prev].to_numpy(), sig.loc[d_prev].to_numpy(),
            last, d_prev.dayofweek == 0, min_order=min_order)
        nxt = O.iloc[i + 1]
        eq.append((close.index[i], cash + sum(qty[k] * nxt[k] for k in syms)))
    return pd.Series(dict(eq))


@pytest.mark.skipif(not HAVE_DATA, reason="research kline cache not present")
def test_fund_reproduces_research_backtest():
    p = _panel()
    sig, tgt = core.targets(p["close"])
    res = engine.run(p["open"], tgt, sig, config.COST, engine.mondays(p["close"].index), config.BAND,
                     track_trades=False)
    research_eq = (1 + res.returns).cumprod()
    fund_eq = _simulate_fund(p)
    research_eq = research_eq.reindex(fund_eq.index)
    years = (fund_eq.index[-1] - fund_eq.index[0]).days / 365.25
    cagr = lambda s: s.iloc[-1] ** (1 / years) - 1
    # Fund pays fees from cash; the engine spreads them over holdings. A few bp/yr at most.
    assert abs(cagr(fund_eq) - cagr(research_eq)) < 0.001
    assert (fund_eq / research_eq - 1).abs().max() < 0.005


@pytest.mark.skipif(not HAVE_DATA, reason="research kline cache not present")
def test_min_order_costs_little():
    """A $1,000 book that skips sub-$5 orders still tracks the strategy."""
    p = _panel()
    p = {k: v[v.index >= "2023-01-01"] for k, v in p.items()}
    full = _simulate_fund(p)
    small = _simulate_fund({k: v for k, v in p.items()}, min_order=5.0 / 1000.0)
    assert abs(small.iloc[-1] / full.iloc[-1] - 1) < 0.02


@pytest.mark.skipif(not HAVE_DATA, reason="research kline cache not present")
def test_risk_multiple_is_a_pure_scaling_of_research_c4():
    from research import strategies
    close = _panel()["close"]
    sig = strategies.sig_donchian(close).where(strategies.eligible(close), 0.0)
    base = strategies.w_voltarget(sig, close)
    _, t1 = core.targets(close, risk=1)
    _, t2 = core.targets(close, risk=2)
    pd.testing.assert_frame_equal(t1, base)
    pd.testing.assert_frame_equal(t2, strategies._cap(base * 2))
    assert t2.sum(axis=1).max() <= 1 + 1e-9


def test_plan_orders_never_leaves_negative_cash():
    syms = ["A", "B"]
    orders, q, cash, *_ = core.plan_orders(
        syms, {"A": 1.0}, 0.0, {"A": 100.0, "B": 100.0},
        t=np.array([0.5, 0.5]), s=np.array([1.0, 1.0]), last_sig=np.array([1.0, 0.0]),
        rebalance=True)
    assert cash >= -1e-9
    assert sum(o.notional for o in orders if o.side == "BUY") <= 100.0


# ---------------------------------------------------------------- run_daily guards

@pytest.fixture
def fake_market(monkeypatch, tmp_path):
    idx = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, len(idx))))
                          for s in config.UNIVERSE}, index=idx)
    today = idx[-1] + pd.Timedelta(days=1)
    monkeypatch.setattr(market, "history", lambda syms, t: {"open": close, "close": close})
    monkeypatch.setattr(market, "todays_open", lambda syms, t: close.iloc[-1].to_dict())
    monkeypatch.setattr(config, "PAPER_LEDGER", str(tmp_path / "paper.json"))
    monkeypatch.setattr(config, "KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.setattr(config, "LOCK_FILE", str(tmp_path / "fund.lock"))
    monkeypatch.setattr(run_daily.notify, "send", lambda text: True)
    return today, tmp_path


def test_paper_run_is_idempotent(fake_market):
    today, tmp = fake_market
    first = run_daily.run("paper", today=today)
    assert first is not None
    assert run_daily.run("paper", today=today) is None
    book = ledger.load(config.PAPER_LEDGER, "paper", 1000)
    assert len(book.runs) == 1
    assert book.cash >= 0


def test_kill_switch_blocks_buys(fake_market):
    today, tmp = fake_market
    (tmp / "KILL").write_text("stop")
    rec = run_daily.run("paper", today=today)
    assert all(o["side"] == "SELL" for o in rec["orders"])
    assert ledger.load(config.PAPER_LEDGER, "paper", 1000).halted


def test_stale_data_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PAPER_LEDGER", str(tmp_path / "paper.json"))
    monkeypatch.setattr(run_daily.notify, "send", lambda text: True)

    def stale(syms, t):
        raise market.StaleData("old")
    monkeypatch.setattr(market, "history", stale)
    with pytest.raises(market.StaleData):
        run_daily.run("paper", today=pd.Timestamp("2026-11-06", tz="UTC"))
    assert not os.path.exists(config.PAPER_LEDGER)


# ---------------------------------------------------------------- exchange broker

class FakeClient:
    API_URL = "https://www.tokocrypto.site/api"

    def __init__(self):
        self.orders = []
        self.usdt = 1000.0

    def get_account(self):
        return {"balances": [{"asset": "USDT", "free": str(self.usdt), "locked": "0"}]}

    def get_symbol_info(self, symbol):
        return {"filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "5"}]}

    def create_order(self, **kw):
        self.orders.append(kw)
        return {"executedQty": kw["quantity"], "cummulativeQuoteQty": str(float(kw["quantity"]) * 100),
                "fills": [], "orderId": len(self.orders), "status": "FILLED"}


def test_broker_refuses_real_money_without_confirmation():
    with pytest.raises(fbroker.RefuseToArm):
        fbroker.ExchangeBroker(client=FakeClient(), venue="tokocrypto", enabled=True,
                               confirm_real_money=False)
    with pytest.raises(fbroker.RefuseToArm):
        fbroker.ExchangeBroker(client=FakeClient(), venue="tokocrypto", enabled=False,
                               confirm_real_money=True)


def test_broker_sells_first_and_skips_dust():
    c = FakeClient()
    b = fbroker.ExchangeBroker(client=c, venue="tokocrypto", enabled=True, confirm_real_money=True)
    fills = b.execute([core.Order("BTCUSDT", "BUY", 1.0, 100.0),
                       core.Order("ETHUSDT", "SELL", 0.5, 100.0),
                       core.Order("SOLUSDT", "BUY", 0.01, 100.0)],  # $1: below minimum
                      cash_available=1000.0)
    assert [o["side"] for o in c.orders] == ["SELL", "BUY"]
    assert all(o["symbol"] != "SOLUSDT" for o in c.orders)
    assert c.orders[0]["quantity"] == "0.500"
    assert len(fills) == 2


def test_broker_never_spends_money_outside_the_fund():
    """The account holds $1,000 USDT, but the fund only owns $20 of it."""
    c = FakeClient()
    b = fbroker.ExchangeBroker(client=c, venue="tokocrypto", enabled=True, confirm_real_money=True)
    b.execute([core.Order("BTCUSDT", "BUY", 5.0, 100.0)], cash_available=20.0)
    assert float(c.orders[0]["quantity"]) * 100 <= 20.0


def test_live_refuses_without_explicit_capital(fake_market, monkeypatch):
    today, tmp = fake_market
    monkeypatch.setattr(config, "LIVE_LEDGER", str(tmp / "live.json"))
    monkeypatch.setattr(config, "LIVE_CAPITAL", 0.0)
    c = FakeClient()
    c.get_symbol_ticker = lambda symbol: {"price": "100"}
    b = fbroker.ExchangeBroker(client=c, venue="tokocrypto", enabled=True, confirm_real_money=True)
    with pytest.raises(RuntimeError, match="FUND_LIVE_CAPITAL"):
        run_daily.run("live", broker=b, today=today)
    assert c.orders == []


class BalanceClient(FakeClient):
    """Fake exchange whose balances move with fills; can fail on the n-th order."""

    def __init__(self, fail_on: int | None = None):
        super().__init__()
        self.bal = {"USDT": 1000.0}
        self.fail_on = fail_on

    def get_account(self):
        return {"balances": [{"asset": a, "free": str(v), "locked": "0"} for a, v in self.bal.items()]}

    def get_symbol_ticker(self, symbol):
        return {"price": "100"}

    def create_order(self, **kw):
        if self.fail_on is not None and len(self.orders) + 1 == self.fail_on:
            raise RuntimeError("exchange rejected the order")
        q = float(kw["quantity"])
        base = kw["symbol"].removesuffix("USDT")
        sign = 1 if kw["side"] == "BUY" else -1
        self.bal[base] = self.bal.get(base, 0.0) + sign * q
        self.bal["USDT"] -= sign * q * 100
        return super().create_order(**kw)


def test_live_order_failure_mid_rebalance_books_what_happened(fake_market, monkeypatch):
    today, tmp = fake_market
    monkeypatch.setattr(config, "LIVE_LEDGER", str(tmp / "live.json"))
    monkeypatch.setattr(config, "LIVE_MARKER", str(tmp / "live_marker"))
    monkeypatch.setattr(config, "LOCK_FILE", str(tmp / "fund.lock"))
    monkeypatch.setattr(config, "LIVE_CAPITAL", 1000.0)
    c = BalanceClient(fail_on=2)
    b = fbroker.ExchangeBroker(client=c, venue="tokocrypto", enabled=True, confirm_real_money=True)
    with pytest.raises(RuntimeError, match="mid-rebalance"):
        run_daily.run("live", broker=b, today=today)
    book = ledger.load(str(tmp / "live.json"), "live", 1000)
    assert book.halted and len(book.runs) == 1
    filled = c.orders[0]
    base_sym = filled["symbol"]
    assert book.qty[base_sym] == pytest.approx(float(filled["quantity"]))   # the fill that happened is booked
    assert book.cash == pytest.approx(c.bal["USDT"])                       # and the book matches the exchange
    assert not os.path.exists(config.LOCK_FILE)


def test_live_refuses_to_restart_from_scratch_after_ledger_loss(fake_market, monkeypatch):
    today, tmp = fake_market
    monkeypatch.setattr(config, "LIVE_LEDGER", str(tmp / "live.json"))
    monkeypatch.setattr(config, "LIVE_MARKER", str(tmp / "live_marker"))
    monkeypatch.setattr(config, "LOCK_FILE", str(tmp / "fund.lock"))
    (tmp / "live_marker").write_text("")
    with pytest.raises(RuntimeError, match="missing"):
        run_daily.run("live", broker=object(), today=today)


def test_run_lock_blocks_a_second_run_and_expires(fake_market, monkeypatch):
    today, tmp = fake_market
    lock = tmp / "fund.lock"
    monkeypatch.setattr(config, "LOCK_FILE", str(lock))
    lock.write_text("123")
    with pytest.raises(run_daily.RunLocked):
        run_daily.run("paper", today=today)
    old = os.path.getmtime(lock) - config.LOCK_STALE_SECONDS - 10
    os.utime(lock, (old, old))
    assert run_daily.run("paper", today=today) is not None   # a crashed run's lock is taken over
    assert not lock.exists()


def test_stale_coin_is_held_while_others_trade(fake_market, monkeypatch):
    today, tmp = fake_market
    monkeypatch.setattr(config, "LOCK_FILE", str(tmp / "fund.lock"))
    real_history = market.history
    h = market.history(config.UNIVERSE, today)   # the fixture's frames
    close = h["close"].copy()
    close.loc[close.index[-1], "DOTUSDT"] = np.nan
    monkeypatch.setattr(market, "history", lambda syms, t: {"open": close, "close": close, "stale": ["DOTUSDT"]})
    opens = close.iloc[-2].to_dict()
    opens.pop("DOTUSDT")
    monkeypatch.setattr(market, "todays_open", lambda syms, t: dict(opens))
    rec = run_daily.run("paper", today=today)
    assert rec["stale"] == ["DOTUSDT"]
    assert all(o["symbol"] != "DOTUSDT" for o in rec["orders"])


def test_missed_monday_still_rebalances():
    book = ledger.Book(mode="paper", start_capital=1000, cash=1000,
                       runs=[{"date": "2026-11-08", "decision_day": "2026-11-07"}])  # Saturday
    assert run_daily._rebalance_due(book, pd.Timestamp("2026-11-10", tz="UTC"))       # Tuesday, Monday missed
    assert not run_daily._rebalance_due(book, pd.Timestamp("2026-11-08", tz="UTC"))   # Sunday


def test_env_cannot_raise_risk_or_disable_the_halt(monkeypatch):
    import importlib
    monkeypatch.setenv("FUND_RISK_MULTIPLIER", "5")
    monkeypatch.setenv("FUND_MAX_DRAWDOWN_HALT", "1")
    try:
        importlib.reload(config)
        assert config.RISK_MULTIPLIER == 2.0
        assert config.MAX_DRAWDOWN_HALT == 0.30
    finally:
        monkeypatch.delenv("FUND_RISK_MULTIPLIER")
        monkeypatch.delenv("FUND_MAX_DRAWDOWN_HALT")
        importlib.reload(config)


def test_report_paper_matches_backtest_when_every_day_runs(fake_market, monkeypatch):
    from fund import report
    today, tmp = fake_market
    close = market.history(config.UNIVERSE, today)["close"]
    monkeypatch.setattr(market, "todays_open", lambda syms, t: close.loc[t].to_dict())
    days = close.index[-12:-2]
    for d in days:
        run_daily.run("paper", today=d)
    c = report.compare("paper", panel={"open": close, "close": close})
    assert c["runs"] == len(days)
    assert c["gap"] == pytest.approx(0.0, abs=1e-12)        # same code, same prices, every day
    assert "Syarat 1" in report.text(c)


def test_report_flags_missed_days(fake_market, monkeypatch):
    from fund import report
    today, tmp = fake_market
    close = market.history(config.UNIVERSE, today)["close"]
    monkeypatch.setattr(market, "todays_open", lambda syms, t: close.loc[t].to_dict())
    days = [d for i, d in enumerate(close.index[-30:-2]) if i % 3 != 1]   # laptop off every third day
    for d in days:
        run_daily.run("paper", today=d)
    c = report.compare("paper", panel={"open": close, "close": close})
    assert c["runs"] == len(days) and c["days"] > c["runs"]
    assert c["backtest_return"] is not None


def test_telegram_stop_and_resume_need_confirmation(fake_market):
    from fund import telegram_bot as tb
    today, tmp = fake_market
    run_daily.run("paper", today=today)
    pending = {}
    assert "Ekuitas" in tb.handle("/status", pending)
    assert "Yakin" in tb.handle("/stop", pending)
    assert not os.path.exists(config.KILL_FILE)          # nothing happens without "ya"
    assert "Stop pembelian aktif" in tb.handle("ya", pending)
    assert os.path.exists(config.KILL_FILE)
    assert "Yakin" in tb.handle("/lanjut", pending)
    tb.handle("something else", pending)                  # any other message cancels
    assert "/status" in tb.handle("ya", pending)          # stale "ya" does nothing
    assert os.path.exists(config.KILL_FILE)
    tb.handle("/lanjut", pending)
    tb.handle("ya", pending)
    assert not os.path.exists(config.KILL_FILE)


def test_live_halts_when_exchange_holds_less_than_book(fake_market, monkeypatch):
    today, tmp = fake_market
    path = tmp / "live.json"
    monkeypatch.setattr(config, "LIVE_LEDGER", str(path))
    book = ledger.Book(mode="live", start_capital=1000, cash=500, qty={"BTCUSDT": 5.0},
                       peak_equity=1000, runs=[{"date": "2026-01-01"}])
    ledger.save(book, str(path))
    c = FakeClient()  # holds no BTC at all
    c.get_symbol_ticker = lambda symbol: {"price": "100"}
    b = fbroker.ExchangeBroker(client=c, venue="tokocrypto", enabled=True, confirm_real_money=True)
    with pytest.raises(RuntimeError, match="less than the book"):
        run_daily.run("live", broker=b, today=today)
    assert c.orders == []
    assert ledger.load(str(path), "live", 1000).halted
