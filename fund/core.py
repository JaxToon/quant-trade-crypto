"""The fund's decision: target weights, then one step of the research trading rule.

Pure functions only (no I/O), so the parity test can drive them over history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import strategies
from fund import config


def targets(close: pd.DataFrame, strategy: str = config.STRATEGY,
            risk: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(signal, target weight) frames for every date, from the research code as-is,
    scaled by the risk multiple and capped at 100% gross (no borrowing)."""
    risk = config.RISK_MULTIPLIER if risk is None else risk
    sig_fn, w_fn = strategies.CANDIDATES[strategy]
    sig = sig_fn(close).where(strategies.eligible(close), 0.0)   # as run_comparison._run
    w = w_fn(sig, close)
    return sig, (w if risk == 1 else strategies._cap(w * risk))


def decide(t: np.ndarray, s: np.ndarray, h: np.ndarray, last_sig: np.ndarray,
           rebalance: bool, band: float = config.BAND):
    """One step of research/engine.py's loop, verbatim.

    t, s: target weight and signal decided at the previous close.
    h: weights held right now (after drift). Returns (new weights, trade mask,
    updated last_sig).
    """
    exit_ = (t <= 0) & (h > 0)
    enter = (t > 0) & (h <= 0)
    sig_chg = (t > 0) & (h > 0) & (s != last_sig)
    drift = (t > 0) & (h > 0) & rebalance & (np.abs(t - h) > band * t)
    trade = exit_ | enter | sig_chg | drift
    new = np.where(trade, np.where(t > 0, t, 0.0), h)
    if new.sum() > 1.0 + 1e-9:
        trade = (t > 0) | (h > 0)
        new = np.where(t > 0, t, 0.0)
    last_sig = np.where(trade, s, last_sig)
    return new, trade, last_sig


@dataclass
class Order:
    symbol: str
    side: str          # "BUY" / "SELL"
    qty: float
    price: float       # reference price (the day's open)

    @property
    def notional(self) -> float:
        return self.qty * self.price


def plan_orders(symbols: list[str], qty: dict[str, float], cash: float, prices: dict[str, float],
                t: np.ndarray, s: np.ndarray, last_sig: np.ndarray, rebalance: bool,
                cost: float = config.COST, min_order: float = 0.0):
    """Turn a decision into orders for a quantity-based book.

    Real-world accounting: fees come out of cash and untraded positions keep
    their quantity. The research engine instead spreads the fee across every
    holding, so the two differ by a few basis points a year; the parity test
    bounds that gap. Cash never goes negative: buys are trimmed to fit.
    Orders below min_order are dropped (the position keeps its old size and
    its signal is not marked as traded, so the trade is retried next day).
    """
    px = np.array([prices[k] for k in symbols])
    q = np.array([qty.get(k, 0.0) for k in symbols])
    equity = cash + float((q * px).sum())
    h = q * px / equity if equity > 0 else np.zeros(len(symbols))
    new, trade, new_last = decide(t, s, h, last_sig, rebalance)

    dw = new - h
    small = trade & (np.abs(dw) * equity < min_order) & (dw != 0)
    if small.any():
        new = np.where(small, h, new)
        new_last = np.where(small, last_sig, new_last)
        dw = new - h
    fee = cost * float(np.abs(dw).sum()) * equity
    eq_after = equity - fee
    new_q = np.where(dw != 0, new * eq_after / px, q)
    # Untraded positions keep their quantity; their value already sits in equity.
    new_cash = eq_after - float((new_q * px).sum())
    if new_cash < 0:
        buys = (new_q - q) * px
        buy_total = float(buys.clip(min=0).sum())
        if buy_total > 0:
            shrink = min(1.0, -new_cash / buy_total)
            new_q = np.where(buys > 0, new_q - shrink * buys / px, new_q)
        fee = cost * float(np.abs((new_q - q) * px).sum())
        new_cash = equity - fee - float((new_q * px).sum())
    orders = [Order(k, "BUY" if nq > oq else "SELL", abs(nq - oq), p)
              for k, oq, nq, p in zip(symbols, q, new_q, px) if abs(nq - oq) > 0]
    return orders, dict(zip(symbols, new_q)), new_cash, new_last, equity, fee
