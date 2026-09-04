"""
Does sourcing the STOCK GNN's training set from real 1-minute candles
(data/stock_ohlcv_1m/, fetched by experiments/fetch_hf_stock_ohlcv.py)
actually beat the current production path (yfinance 1h/730d)?

Pre-registered question, answered BEFORE touching train_gnn.py -- the
1h/730d path was itself a deliberate fix for a documented stock collapse
("trained stock model returned near-identical output ~-0.0017 across
wildly different inputs"), so replacing it needs evidence, not just "more
data is better" intuition.

Four arms, all sliced to the SAME calendar window so the only thing that
varies is candle resolution / source:

  YF_1H          current production path: yfinance 1h
  LOCAL_1H       local 1m parquet resampled to 1h -- control that isolates
                 "different source" from "different resolution"
  LOCAL_1M       local 1m native, stride 240min -- same wall-clock sample
                 spacing as the 1h arms, so sample COUNT matches and only
                 the SMC/FVG feature richness differs
  LOCAL_1M_DENSE local 1m native, stride 30min -- spends the extra data on
                 more samples instead of richer features

Protocol per arm (identical): build (feature, label) snapshots by replaying
real candles through SmartMoneyAnalyzer exactly as train_gnn.py does ->
chronological 80/20 train/test split (never shuffled across the boundary)
-> z-score stats fit on TRAIN ONLY -> fresh GNNTrainer -> evaluate on the
held-out tail.

Metrics: held-out directional accuracy (the metric this project already
used to validate feature normalization, 55.0% -> 76.9%), information
coefficient (corr(pred, label)), prediction spread (std -- directly tests
the documented collapse-to-constant failure), and MSE.

Training compute is equalized across arms (samples x epochs ~ constant) so
a denser arm isn't just buying accuracy with more gradient steps.

NOTHING is persisted: no checkpoint save, no feature-norm file. Writing
either would clobber live stock state -- train_gnn.py's own comment
documents an incident where a stray gnn_feature_norm_*.json silently
desynced a RUNNING process's inference from its in-memory model.

Usage: ASSET_TYPE=stock python -m experiments.stock_training_data_ab [--months 12]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics

import pandas as pd

from config import settings
from gnn.dataset import TOKEN_FEATURES
from gnn.feature_norm import compute_feature_stats, normalize_feature_rows
from strategy.smart_money import Candle, SmartMoneyAnalyzer
from train_gnn import _bollinger, _encode_market_structure, _label_at, _macd, _rsi

LOCAL_DIR = "data/stock_ohlcv_1m"
MIN_LOOKBACK = 100          # candles before SMC market-structure is meaningful
HORIZON_MINUTES = 240       # 4h forward-return label, identical wall-clock for every arm
TEST_FRACTION = 0.2

# Regular NASDAQ session only (14:30-21:00 UTC). The local dataset also carries
# pre/post-market minutes, but those are thin, wide-spread, and nothing the live
# system trades on -- including them would train SMC structure on microstructure
# the model never sees at inference time.
SESSION_START_UTC = pd.Timestamp("2000-01-01 14:30:00+00:00").time()
SESSION_END_UTC = pd.Timestamp("2000-01-01 21:00:00+00:00").time()


def load_local_1m(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = os.path.join(LOCAL_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    times = df["timestamp"].dt.time
    df = df[(times >= SESSION_START_UTC) & (times < SESSION_END_UTC)]
    return df.sort_values("timestamp").reset_index(drop=True)


def df_to_candles(df: pd.DataFrame) -> list[Candle]:
    return [
        Candle(
            timestamp=row.timestamp.to_pydatetime().replace(tzinfo=None),
            open=float(row.open), high=float(row.high), low=float(row.low),
            close=float(row.close), volume=float(row.volume),
        )
        for row in df.itertuples()
    ]


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = (
        df.set_index("timestamp")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out


def fetch_yf_1h(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], utc=True),
        "open": df["Open"].astype(float), "high": df["High"].astype(float),
        "low": df["Low"].astype(float), "close": df["Close"].astype(float),
        "volume": df["Volume"].astype(float) if "Volume" in df.columns else 0.0,
    })
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]
    return out.sort_values("timestamp").reset_index(drop=True)


def build_series(candles: list[Candle], stride: int, horizon: int) -> list[tuple[list[float], float]]:
    """
    Same replay-and-snapshot logic as train_gnn.build_symbol_series, but takes
    candles directly (so every arm can share it) and keeps chronological order
    instead of random-sampling -- the split below needs time order preserved.
    """
    if len(candles) < MIN_LOOKBACK + horizon + stride:
        return []

    smc = SmartMoneyAnalyzer(lookback=MIN_LOOKBACK)
    closes = [c.close for c in candles]
    last_t = len(candles) - horizon - 1

    series = []
    for t, candle in enumerate(candles):
        smc_signals = smc.update(candle)
        if t < MIN_LOOKBACK or t >= last_t or t % stride != 0:
            continue

        closes_so_far = closes[: t + 1]
        rsi = _rsi(closes_so_far)
        macd, macd_signal, macd_hist = _macd(closes_so_far)
        bb_upper, bb_mid, bb_lower = _bollinger(closes_so_far)

        features = [
            candle.close, candle.volume, rsi, macd, macd_signal, macd_hist,
            bb_upper, bb_mid, bb_lower,
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
        assert len(features) == len(TOKEN_FEATURES)
        series.append((features, _label_at(closes, t, horizon)))

    return series


def to_joint_samples(per_symbol: dict[str, list[tuple[list[float], float]]]) -> list[dict]:
    """Pair the i-th snapshot of each symbol into one joint graph sample
    (same index-alignment train_gnn.py uses)."""
    if not per_symbol:
        return []
    min_len = min(len(s) for s in per_symbol.values())
    return [
        {
            "features": {sym: series[i][0] for sym, series in per_symbol.items()},
            "label": {sym: series[i][1] for sym, series in per_symbol.items()},
        }
        for i in range(min_len)
    ]


def evaluate(trainer, samples: list[dict]) -> dict:
    """
    Held-out forward pass. Runs the model directly rather than calling
    predict_from_features per sample -- that rebuilds the whole hetero graph
    (and hits Neo4j) on every call, which would dominate runtime here.
    """
    import torch

    from gnn.dataset import build_hetero_data

    base_data, _ = build_hetero_data()
    edge_index_dict = trainer._filter_edges(base_data)
    token_symbols = base_data["Token"].symbol if hasattr(base_data["Token"], "symbol") else []
    static_x = {nt: base_data[nt].x.to(trainer.device) for nt in base_data.node_types if nt != "Token"}

    preds, labels = [], []
    trainer.model.eval()
    with torch.no_grad():
        for sample in samples:
            feats, lbls = sample["features"], sample["label"]
            token_x = torch.tensor(
                [feats.get(sym, [0.0] * len(TOKEN_FEATURES)) for sym in token_symbols],
                dtype=torch.float, device=trainer.device,
            )
            out = trainer.model({**static_x, "Token": token_x}, edge_index_dict)
            for sym in feats:
                if sym in token_symbols:
                    preds.append(float(out[token_symbols.index(sym), 0].cpu()))
                    labels.append(float(lbls.get(sym, 0.0)))

    n = len(preds)
    directional = sum(
        1 for p, l in zip(preds, labels) if (p > 0) == (l > 0) and l != 0
    )
    nonzero = sum(1 for l in labels if l != 0)
    mse = sum((p - l) ** 2 for p, l in zip(preds, labels)) / n if n else float("nan")
    pred_std = statistics.pstdev(preds) if n > 1 else 0.0

    # Information coefficient: corr(prediction, realized label)
    if n > 1 and pred_std > 0:
        mp, ml = sum(preds) / n, sum(labels) / n
        cov = sum((p - mp) * (l - ml) for p, l in zip(preds, labels)) / n
        sl = statistics.pstdev(labels)
        ic = cov / (pred_std * sl) if sl > 0 else 0.0
    else:
        ic = 0.0

    return {
        "n": n,
        "dir_acc": 100.0 * directional / nonzero if nonzero else float("nan"),
        "ic": ic,
        "pred_std": pred_std,
        "mse": mse,
    }


def run_arm(name: str, per_symbol: dict[str, list[tuple[list[float], float]]], budget: int) -> dict | None:
    from gnn.trainer import GNNTrainer

    samples = to_joint_samples(per_symbol)
    if len(samples) < 20:
        print(f"  {name}: only {len(samples)} joint samples — too few, skipping")
        return None

    split = int(len(samples) * (1 - TEST_FRACTION))
    train_samples, test_samples = samples[:split], samples[split:]

    # Fit normalization on TRAIN ONLY (never saved to disk — see module docstring)
    means, stds = compute_feature_stats(train_samples, len(TOKEN_FEATURES))
    for bucket in (train_samples, test_samples):
        for sample in bucket:
            sample["features"] = {
                sym: normalize_feature_rows([f], means, stds)[0]
                for sym, f in sample["features"].items()
            }

    epochs = max(5, min(100, budget // max(1, len(train_samples))))
    print(f"  {name}: {len(train_samples)} train / {len(test_samples)} test samples, {epochs} epochs", flush=True)

    trainer = GNNTrainer()
    trainer.build_model()
    loss = trainer.train_on_history(train_samples, epochs=epochs)

    metrics = evaluate(trainer, test_samples)
    metrics.update({"arm": name, "train_loss": loss, "n_train": len(train_samples), "epochs": epochs})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12, help="calendar window, ending at the local dataset's last month")
    parser.add_argument("--budget", type=int, default=40_000,
                        help="training steps (samples x epochs) per arm, held constant so a denser "
                             "arm can't buy accuracy with extra gradient steps. Lower it for a smoke test.")
    parser.add_argument("--skip-yf", action="store_true", help="skip the yfinance arm (useful offline / for faster iteration)")
    args = parser.parse_args()

    if settings.ASSET_TYPE != "stock":
        print(f"WARNING: ASSET_TYPE={settings.ASSET_TYPE}, expected 'stock'. "
              "Re-run with ASSET_TYPE=stock or the graph/model will be built for the wrong asset class.")

    end = pd.Timestamp("2026-03-31 23:59:00+00:00")
    start = end - pd.DateOffset(months=args.months)
    tickers = list(settings.STOCK_WATCHLIST)
    print(f"Window: {start.date()} .. {end.date()} ({args.months} months), tickers: {tickers}\n")

    arms: dict[str, dict[str, list]] = {k: {} for k in ("YF_1H", "LOCAL_1H", "LOCAL_1M", "LOCAL_1M_DENSE")}

    for ticker in tickers:
        df_1m = load_local_1m(ticker, start, end)
        if df_1m.empty:
            print(f"  {ticker}: no local 1m data in window, skipping")
            continue

        candles_1m = df_to_candles(df_1m)
        candles_local_1h = df_to_candles(resample_1h(df_1m))
        if args.skip_yf:
            candles_yf_1h = []
        else:
            df_yf = fetch_yf_1h(ticker, start, end)
            candles_yf_1h = df_to_candles(df_yf) if not df_yf.empty else []

        print(f"  {ticker}: 1m={len(candles_1m)}  local_1h={len(candles_local_1h)}  yf_1h={len(candles_yf_1h)}", flush=True)

        if candles_yf_1h:
            arms["YF_1H"][ticker] = build_series(candles_yf_1h, stride=4, horizon=4)
        arms["LOCAL_1H"][ticker] = build_series(candles_local_1h, stride=4, horizon=4)

        # One 1m replay, two arms: sample at the finest stride, then take every
        # 8th of those for the 240min arm (30 * 8 = 240) instead of replaying
        # the same 100k candles a second time.
        dense = build_series(candles_1m, stride=30, horizon=HORIZON_MINUTES)
        arms["LOCAL_1M_DENSE"][ticker] = dense
        arms["LOCAL_1M"][ticker] = dense[::8]

    print(f"\n{'=' * 92}")
    print("STOCK GNN TRAINING-DATA A/B — held-out evaluation (chronological 80/20 split)")
    print(f"{'=' * 92}")

    results = []
    for name, per_symbol in arms.items():
        per_symbol = {k: v for k, v in per_symbol.items() if v}
        if not per_symbol:
            print(f"  {name}: no data, skipping")
            continue
        res = run_arm(name, per_symbol, args.budget)
        if res:
            results.append(res)

    print(f"\n{'=' * 92}")
    print(f"{'Arm':<16}{'Train n':<10}{'Epochs':<9}{'TrainLoss':<12}{'DirAcc':<10}{'IC':<10}{'PredStd':<11}{'MSE':<10}")
    print("-" * 92)
    for r in results:
        print(f"{r['arm']:<16}{r['n_train']:<10}{r['epochs']:<9}{r['train_loss']:<12.6f}"
              f"{r['dir_acc']:<10.1f}{r['ic']:<10.3f}{r['pred_std']:<11.5f}{r['mse']:<10.5f}")
    print("-" * 92)
    print("DirAcc  = held-out directional accuracy %, the metric feature-normalization was validated on")
    print("IC      = corr(prediction, realized label) on held-out data — the real predictive-quality measure")
    print("PredStd = spread of predictions; near-zero means the model collapsed to a constant")
    print(f"{'=' * 92}")


if __name__ == "__main__":
    main()
