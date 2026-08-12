"""
Bootstrap GNN training from historical market data via yfinance.

Builds many (feature, label) snapshots per symbol by replaying 5m candles
through the same SmartMoneyAnalyzer live main.py uses, instead of a single
"now" snapshot. The old approach trained 150 epochs of gradient descent on
one static point per symbol, which is memorization, not learning a feature
-> forward-return mapping. Graph topology (edges, Protocol/Macro_Event
features) is taken once from the live Neo4j graph and held fixed; only each
symbol's feature vector and its forward-return label vary per sample.

The 9 SMC feature slots are now REAL SmartMoneyAnalyzer output from
replaying the historical candles — not zeroed placeholders. Earlier versions
of this script zeroed them out (no cheap way to replay historical order-
block/FVG state was in place yet), which meant the model never saw the
feature distribution it actually gets fed at live inference time. That gap
is closed now that backtest.py's candle-replay pattern exists.

Usage: python train_gnn.py
"""
from __future__ import annotations

import math
import random
from datetime import datetime

import yfinance as yf

from gnn.dataset import TOKEN_FEATURES
from gnn.trainer import GNNTrainer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# Binance symbol -> yfinance ticker (try multiple for each)
SYMBOL_TICKERS = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    "SOLUSDT": ["SOL-USD", "SOL1-USD"],
    "BNBUSDT": ["BNB-USD", "BNB-USDT"],
    "XRPUSDT": ["XRP-USD"],
    "ADAUSDT": ["ADA-USD"],
    "AVAXUSDT": ["AVAX-USD"],
    "DOTUSDT": ["DOT-USD"],
}

YF_MAX_DAYS_5M = 60      # Yahoo Finance's hard limit for 5-minute-interval history
STRIDE_CANDLES = 30      # spacing between training snapshots (~2.5h at 5m)
LABEL_HORIZON = 24       # candles forward for the return label (~2h at 5m)
MIN_LOOKBACK = 100       # candles needed before SMC market-structure is meaningful
MAX_SAMPLES_PER_SYMBOL = 200


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    window = closes[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(window)):
        delta = window[i] - window[i - 1]
        (gains if delta > 0 else losses).append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(period: int, data: list[float]) -> list[float]:
    k = 2 / (period + 1)
    emas = [data[0]]
    for p in data[1:]:
        emas.append(p * k + emas[-1] * (1 - k))
    return emas


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    fast_ema = _ema(fast, closes)
    slow_ema = _ema(slow, closes)
    macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
    signal_line = _ema(signal, macd_line)
    return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]


def _bollinger(closes: list[float], period: int = 20, n_std: float = 2.0) -> tuple[float, float, float]:
    window = closes[-period:]
    mean = sum(window) / period
    std = math.sqrt(sum((c - mean) ** 2 for c in window) / period)
    return mean + n_std * std, mean, mean - n_std * std


def _label_at(closes: list[float], t: int, horizon: int) -> float:
    """Forward return from t to t+horizon, clamped to [-1, +1] at a +-15% move."""
    entry = closes[t]
    future = closes[t + horizon]
    if entry == 0:
        return 0.0
    momentum = (future - entry) / entry
    momentum_signal = max(-1.0, min(1.0, momentum / 0.15))

    rsi = _rsi(closes[: t + 1])
    if rsi < 30:
        rsi_signal = 0.5
    elif rsi > 70:
        rsi_signal = -0.5
    else:
        rsi_signal = (50 - rsi) / 100

    return max(-1.0, min(1.0, 0.7 * momentum_signal + 0.3 * rsi_signal))


def _encode_market_structure(ms: str) -> float:
    if ms == "bullish":
        return 1.0
    elif ms == "bearish":
        return -1.0
    return 0.0


def fetch_candles(symbol: str, tickers: list[str]) -> list[Candle]:
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{YF_MAX_DAYS_5M}d", interval="5m", progress=False, auto_adjust=True)
            if df.empty or len(df) < MIN_LOOKBACK + LABEL_HORIZON + STRIDE_CANDLES:
                continue

            opens = df["Open"].values.flatten().tolist()
            highs = df["High"].values.flatten().tolist()
            lows = df["Low"].values.flatten().tolist()
            closes = df["Close"].values.flatten().tolist()
            volumes = df["Volume"].values.flatten().tolist() if "Volume" in df.columns else [0.0] * len(closes)
            timestamps = [ts.to_pydatetime().replace(tzinfo=None) for ts in df.index]

            candles = [
                Candle(timestamp=timestamps[i], open=opens[i], high=highs[i], low=lows[i],
                       close=closes[i], volume=volumes[i])
                for i in range(len(closes))
            ]
            print(f"  {symbol} ({ticker}): {len(candles)} x 5m candles")
            return candles
        except Exception as e:
            print(f"  {symbol} ({ticker}): failed — {e}")
            continue
    return []


def build_symbol_series(symbol: str, tickers: list[str]) -> list[tuple[list[float], float]]:
    """
    Replays real candles through SmartMoneyAnalyzer and samples (feature,
    label) snapshots every STRIDE_CANDLES — feature vectors carry the SMC
    state that was ACTUALLY detected at that point in history, matching what
    live inference feeds the model.
    """
    candles = fetch_candles(symbol, tickers)
    if not candles:
        return []

    smc = SmartMoneyAnalyzer(lookback=MIN_LOOKBACK)
    closes = [c.close for c in candles]

    series = []
    last_t = len(candles) - LABEL_HORIZON - 1
    for t, candle in enumerate(candles):
        smc_signals = smc.update(candle)

        if t < MIN_LOOKBACK or t >= last_t or t % STRIDE_CANDLES != 0:
            continue

        closes_so_far = closes[: t + 1]
        rsi = _rsi(closes_so_far)
        macd, macd_signal, macd_hist = _macd(closes_so_far)
        bb_upper, bb_mid, bb_lower = _bollinger(closes_so_far)

        tech_features = [candle.close, candle.volume, rsi, macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower]
        smc_features = [
            smc_signals.get("smc_signal", 0.0),
            _encode_market_structure(smc_signals.get("market_structure", "unknown")),
            float(smc_signals.get("active_order_blocks", 0)),
            float(smc_signals.get("active_fvgs", 0)),
            float(smc_signals.get("liquidity_zones", 0)),
            1.0 if smc_signals.get("in_bullish_ob") else 0.0,
            1.0 if smc_signals.get("in_bearish_ob") else 0.0,
            1.0 if smc_signals.get("in_bullish_fvg") else 0.0,
            1.0 if smc_signals.get("in_bearish_fvg") else 0.0,
        ]
        features = tech_features + smc_features
        assert len(features) == len(TOKEN_FEATURES)

        label = _label_at(closes, t, LABEL_HORIZON)
        series.append((features, label))

    if len(series) > MAX_SAMPLES_PER_SYMBOL:
        series = random.sample(series, MAX_SAMPLES_PER_SYMBOL)

    print(f"  {symbol}: {len(series)} training snapshots (real SMC features)")
    return series


def main():
    print(f"Building historical training snapshots ({YF_MAX_DAYS_5M}d of 5m candles, real SMC replay)...\n")
    per_symbol: dict[str, list[tuple[list[float], float]]] = {}
    for symbol, tickers in SYMBOL_TICKERS.items():
        series = build_symbol_series(symbol, tickers)
        if series:
            per_symbol[symbol] = series

    if not per_symbol:
        print("\nERROR: No data fetched for any symbol. Check internet connection.")
        return

    # Align by relative index position across symbols (not true calendar time —
    # each symbol keeps its own indicator history, we just pair up the i-th
    # snapshot of each symbol into one joint graph sample).
    min_len = min(len(s) for s in per_symbol.values())
    samples = []
    for i in range(min_len):
        samples.append({
            "features": {sym: series[i][0] for sym, series in per_symbol.items()},
            "label": {sym: series[i][1] for sym, series in per_symbol.items()},
        })
    print(f"\nBuilt {len(samples)} joint training samples across {len(per_symbol)} symbols.")

    print("\nBuilding GNN model...")
    trainer = GNNTrainer()
    trainer.build_model()

    print("Training from scratch on historical snapshots (fresh weights avoids stale-checkpoint mismatches)...")
    epochs = min(100, len(samples) * 2)
    print(f"\nTraining ({epochs} epochs over {len(samples)} samples each)...")
    loss = trainer.train_on_history(samples, epochs=epochs)

    trainer.save()
    print(f"\nDone. Final avg loss: {loss:.6f}")

    if loss > 0.3:
        print("Loss still high — graph is sparse. Let main.py run a few hours to")
        print("populate edges from news, then retrain.")
    elif loss > 0.1:
        print("Decent. Retrain again after a few hours of news ingestion.")
    else:
        print("Good convergence. Restart main.py to start live trading.")


if __name__ == "__main__":
    main()
