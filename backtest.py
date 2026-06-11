"""
Backtest harness for the Kronos signal pipeline.

Replays Kronos predictions over historical intraday windows and measures:
  1. Forecast accuracy   — directional hit rate + close MAPE of the raw model
  2. Signal performance  — WIN/LOSS/EXPIRED, win rate, expectancy (reuses the
                           exact tracker.py outcome logic, so backtest == live)
  3. Config sweep        — re-scores the cached predictions across a grid of
                           target-quantile / directional-agreement settings to
                           tune them WITHOUT re-running the (expensive) model.

The model is run once per anchor; sweeping configs is then almost free.

Usage:
    python backtest.py --symbols RELIANCE TCS INFY [--anchors 30] [--samples 20]
                       [--variant small] [--interval 1h] [--days 1]
                       [--sweep] [--no-trend]
"""

import argparse
import sys
import numpy as np
import pandas as pd

from pipeline.data_fetcher import fetch_ohlcv
from pipeline.predictor import load_model, predict_next_day, CONTEXT_LEN
from pipeline.trend_analyzer import fetch_daily, analyze as analyze_trend
from pipeline import signal_generator as sg
from pipeline.signal_generator import generate_signal
from pipeline.data_fetcher import CANDLES_PER_DAY
from tracker import _determine_outcome


def _collect_records(symbols, interval, pred_len, context_len, sample_count,
                     n_anchors, step, use_trend):
    """
    Runs the model once per historical anchor and returns a list of records:
    {symbol, entry, pred_df, fut, trend}. Anchors are non-overlapping windows
    stepping back from the most recent fully-realized window.
    """
    records = []
    for sym in symbols:
        df = fetch_ohlcv(sym, interval=interval)
        if df is None or len(df) < context_len + pred_len + 5:
            print(f"  [skip] {sym}: not enough history")
            continue

        daily = fetch_daily(sym, days=400) if use_trend else None

        # Most recent anchor leaves a full realized window after it; walk back.
        last_anchor = len(df) - pred_len
        anchors = list(range(last_anchor, context_len, -step))[:n_anchors]
        print(f"  {sym}: {len(anchors)} anchors")

        for i in anchors:
            ctx = df.iloc[i - context_len:i]
            fut = df.iloc[i:i + pred_len]
            if len(fut) < pred_len:
                continue

            x_ts = ctx.index.to_series().reset_index(drop=True)
            y_ts = fut.index.to_series().reset_index(drop=True)

            pred_df = predict_next_day(sym, ctx, x_ts, y_ts,
                                       sample_count=sample_count)
            if pred_df is None:
                continue

            trend = None
            if use_trend and daily is not None:
                asof = fut.index[0]
                asof_naive = asof.tz_localize(None) if asof.tzinfo else asof
                d_slice = daily.loc[:asof_naive.normalize()]
                if len(d_slice) >= 30:
                    trend = analyze_trend(sym, df=d_slice)

            records.append({
                "symbol": sym,
                "entry":  float(fut.iloc[0]["open"]),
                "pred_df": pred_df,
                "fut":     fut,
                "trend":   trend,
            })
    return records


def _forecast_metrics(records):
    """Config-independent raw-model accuracy: directional hit rate + close MAPE."""
    hits = total = 0
    mapes = []
    for r in records:
        entry = r["entry"]
        pred_close = float(r["pred_df"]["close"].iloc[-1])
        real_close = float(r["fut"]["close"].iloc[-1])
        pred_dir = np.sign(pred_close - entry)
        real_dir = np.sign(real_close - entry)
        if pred_dir != 0:
            total += 1
            hits += int(pred_dir == real_dir)
        mapes.append(abs(pred_close - real_close) / real_close * 100)
    return {
        "n": len(records),
        "dir_acc": (hits / total * 100) if total else 0.0,
        "dir_n": total,
        "close_mape": float(np.mean(mapes)) if mapes else 0.0,
    }


def _evaluate_config(records, target_quantile, min_dir_agreement,
                     min_move=None, min_rr=None, sl_cap=None, sl_floor=None):
    """Re-scores cached predictions under one config. Returns signal stats."""
    sg.TARGET_QUANTILE   = target_quantile
    sg.MIN_DIR_AGREEMENT = min_dir_agreement
    if min_move is not None:
        sg.MIN_MOVE_PCT = min_move
    if min_rr is not None:
        sg.MIN_RR_RATIO = min_rr
    if sl_cap is not None:
        sg.SL_CAP_PCT = sl_cap
    if sl_floor is not None:
        sg.SL_FLOOR_PCT = sl_floor

    wins = losses = expired = fired = 0
    pnls = []
    for r in records:
        sig = generate_signal(r["symbol"], r["pred_df"],
                              current_price=r["entry"], trend=r["trend"])
        if sig.direction not in ("LONG", "SHORT"):
            continue
        fired += 1
        entry   = r["entry"]
        tgt_pct = abs((sig.target - entry) / entry * 100)
        sl_pct  = abs((sig.stop_loss - entry) / entry * 100)
        outcome, pnl, *_ = _determine_outcome(
            sig.direction, entry, tgt_pct, sl_pct, r["fut"])
        pnls.append(pnl)
        if   outcome == "WIN":  wins += 1
        elif outcome == "LOSS": losses += 1
        else:                   expired += 1

    decisive = wins + losses
    win_rate = (wins / decisive * 100) if decisive else 0.0
    return {
        "fired": fired, "wins": wins, "losses": losses, "expired": expired,
        "win_rate": win_rate,
        "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,        # per fired trade
        "pnl_per_window": float(np.sum(pnls) / len(records)) if records else 0.0,  # per opportunity
    }


def _print_config(label, m):
    print(f"  {label:<30} fired={m['fired']:>3} | "
          f"W/L/E={m['wins']}/{m['losses']}/{m['expired']} | "
          f"win%={m['win_rate']:>5.1f} | avgP&L/trade={m['avg_pnl']:+.2f}% | "
          f"P&L/window={m['pnl_per_window']:+.3f}%")


def main():
    p = argparse.ArgumentParser(description="Kronos signal pipeline backtest")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--variant", default="small", choices=["mini", "small", "base"])
    p.add_argument("--interval", default="1h", choices=["1h", "15m", "5m", "1m"])
    p.add_argument("--days", type=int, default=1, help="Prediction horizon in trading days")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--anchors", type=int, default=30, help="Historical windows per symbol")
    p.add_argument("--step", type=int, default=0,
                   help="Candles between anchors (default: one prediction window)")
    p.add_argument("--no-trend", action="store_true",
                   help="Skip trend confluence (pure forecast/target test)")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep target-quantile / agreement grid to tune defaults")
    args = p.parse_args()

    context_len = CONTEXT_LEN[args.variant]
    pred_len    = args.days * CANDLES_PER_DAY[args.interval]
    step        = args.step or pred_len

    print("=" * 68)
    print("  KRONOS PIPELINE BACKTEST")
    print(f"  variant={args.variant} interval={args.interval} horizon={args.days}d "
          f"({pred_len} candles) samples={args.samples}")
    print("=" * 68)

    try:
        load_model(variant=args.variant)
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nCollecting predictions ({len(args.symbols)} symbols)...")
    records = _collect_records(
        args.symbols, args.interval, pred_len, context_len,
        args.samples, args.anchors, step, use_trend=not args.no_trend)

    if not records:
        print("[ERROR] No backtestable windows collected.")
        sys.exit(1)

    fm = _forecast_metrics(records)
    print("\n--- Forecast accuracy (raw model, config-independent) ---")
    print(f"  Windows evaluated   : {fm['n']}")
    print(f"  Directional hit rate: {fm['dir_acc']:.1f}%  (n={fm['dir_n']}, 50% = coin flip)")
    print(f"  Final-close MAPE    : {fm['close_mape']:.2f}%")

    print("\n--- Signal performance @ current defaults "
          f"(q={sg.TARGET_QUANTILE}, agree={sg.MIN_DIR_AGREEMENT}) ---")
    base = _evaluate_config(records, sg.TARGET_QUANTILE, sg.MIN_DIR_AGREEMENT)
    _print_config(f"q={sg.TARGET_QUANTILE} agree={sg.MIN_DIR_AGREEMENT}", base)

    if args.sweep:
        print("\n--- Config sweep (target_quantile x min_dir_agreement) ---")
        best = None
        for q in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
            for a in (0.5, 0.55, 0.6, 0.7):
                m = _evaluate_config(records, q, a)
                _print_config(f"q={q} agree={a}", m)
                # Rank by expectancy, require a minimum sample of fired trades
                if m["fired"] >= max(5, len(records) // 10):
                    score = m["avg_pnl"]
                    if best is None or score > best[0]:
                        best = (score, q, a, m)
        if best:
            _, q, a, m = best
            print(f"\n  >> Best by avg P&L: target_quantile={q}, "
                  f"min_dir_agreement={a}  (avgP&L={m['avg_pnl']:+.2f}%, "
                  f"win%={m['win_rate']:.1f}, fired={m['fired']})")
    print()


if __name__ == "__main__":
    main()
