"""
NSE Intraday Signal Pipeline using Kronos Foundation Model
----------------------------------------------------------
1. Scans NSE top gainers & losers by market-cap tier (Large / Mid / Small)
2. Fetches historical hourly OHLCV for each stock
3. Analyses weekly + monthly trend (RSI, SMA, ADX, momentum)
4. Runs Kronos to predict tomorrow's intraday candles
5. Generates trade signals — target = Kronos predicted range, stop capped at 2.5%

Usage:
    python main.py [--variant small|mini|base] [--top N] [--samples N]
                   [--cap large|mid|small|all] [--save] [--symbols ...]
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime

from pipeline.market_scanner import (
    get_top_gainers_losers, display_scanner_results, all_symbols_from_results
)
from pipeline.data_fetcher import (fetch_ohlcv, prepare_context_and_forecast_timestamps,
                                   get_current_price, CANDLES_PER_DAY)
from pipeline.predictor import load_model, predict_batch
from pipeline.signal_generator import generate_signal, display_signals, signals_to_dataframe
from pipeline.trend_analyzer import analyze as analyze_trend, display_trend
from pipeline.sentiment_analyzer import analyze_batch as analyze_sentiments
from pipeline.symbol_resolver import resolve_symbols

CONTEXT_LEN = {"mini": 2048, "small": 512, "base": 512}


def parse_args():
    p = argparse.ArgumentParser(description="NSE Intraday Signal Pipeline (Kronos)")
    p.add_argument("--variant", default="small", choices=["mini", "small", "base"])
    p.add_argument("--top",     type=int, default=10,
                   help="Top N gainers/losers per cap tier (default: 10)")
    p.add_argument("--samples", type=int, default=20,
                   help="Kronos ensemble samples (default: 20)")
    p.add_argument("--cap",     default="all",
                   choices=["large", "mid", "small", "all"],
                   help="Cap tier to scan (default: all)")
    p.add_argument("--interval", default="1h", choices=["1h", "15m", "5m", "1m"],
                   help="Candle interval (default: 1h). Use 15m/5m for detailed intraday.")
    p.add_argument("--days",    type=int, default=3,
                   help="Number of trading days to predict (default: 3)")
    p.add_argument("--save",    action="store_true", help="Save results to CSV")
    p.add_argument("--track",   action="store_true", help="Log signals to tracker DB for outcome evaluation")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Skip scanner — predict specific symbols instead")
    p.add_argument("--no-sentiment", action="store_true",
                   help="Skip sentiment analysis (faster runs)")
    return p.parse_args()


def get_symbols_from_scanner(top_n: int, cap: str) -> tuple:
    """Returns (flat_symbol_list, tiered_results_dict)."""
    tiers = ["large", "mid", "small"] if cap == "all" else [cap]
    print(f"\n[1/6] Scanning NSE — top {top_n} gainers & losers per tier: {', '.join(t.upper() for t in tiers)}")

    try:
        results = get_top_gainers_losers(top_n=top_n, tiers=tiers)
        display_scanner_results(results)
        symbols = all_symbols_from_results(results)
        return symbols, results
    except Exception as e:
        print(f"[ERROR] Scanner failed: {e}")
        fallback = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                    "HINDUNILVR", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]
        print(f"       Falling back to: {', '.join(fallback)}")
        return fallback, {}


def fetch_sentiments(symbols: list) -> dict:
    print(f"\n[4/6] Analysing news sentiment for {len(symbols)} stocks...")
    return analyze_sentiments(symbols)


def fetch_all_data(symbols: list, context_len: int, pred_len: int, interval: str) -> list:
    print(f"\n[2/6] Fetching {interval} candles for {len(symbols)} stocks...")
    stocks = []
    for sym in symbols:
        df = fetch_ohlcv(sym, interval=interval)
        if df is None:
            continue
        x_df, x_ts, y_ts = prepare_context_and_forecast_timestamps(
            df, pred_len=pred_len, context_len=context_len, interval=interval
        )
        stocks.append((sym, x_df, x_ts, y_ts))
        print(f"  {sym}: {len(x_df)} context candles | predicting {pred_len} candles")
    return stocks


def fetch_trends(symbols: list) -> dict:
    print(f"\n[3/6] Analysing weekly & monthly trends for {len(symbols)} stocks...")
    trends = {}
    for sym in symbols:
        t = analyze_trend(sym)
        trends[sym] = t
        if t:
            display_trend(t)
        else:
            print(f"  {sym}: trend data unavailable")
    return trends


def run_predictions(stocks: list, sample_count: int) -> dict:
    print(f"\n[5/6] Running Kronos predictions (samples={sample_count})...")
    return predict_batch(stocks, sample_count=sample_count)


def build_signals(stocks: list, predictions: dict, trends: dict,
                  scan_results: dict, sentiments: dict = None) -> list:
    print("\n[6/6] Generating trade signals with trend confluence...")

    # Build symbol → cap_tier lookup
    tier_map = {}
    for tier, (gainers, losers) in scan_results.items():
        for sym in pd.concat([gainers, losers])["symbol"].tolist():
            tier_map.setdefault(sym, tier)

    signals = []
    for symbol, *_ in stocks:
        pred_df = predictions.get(symbol)
        if pred_df is None:
            continue
        current_price = get_current_price(symbol)
        trend  = trends.get(symbol)
        signal = generate_signal(symbol, pred_df, current_price, trend)
        signal.cap_tier = tier_map.get(symbol, "unknown")
        if sentiments and symbol in sentiments:
            s = sentiments[symbol]
            signal.sentiment       = s.label
            signal.sentiment_score = s.score
            signal.sentiment_count = s.count
        signals.append(signal)
    return signals


def display_tiered_signals(signals: list) -> None:
    """Groups actionable signals by cap tier for cleaner output."""
    tier_order  = ["large", "mid", "small", "unknown"]
    tier_labels = {
        "large":   "LARGE CAP",
        "mid":     "MID CAP",
        "small":   "SMALL CAP",
        "unknown": "OTHER",
    }

    actionable = [s for s in signals if s.direction != "NO TRADE"]
    skipped    = [s for s in signals if s.direction == "NO TRADE"]

    print("\n" + "="*72)
    print("  ACTIONABLE INTRADAY TRADE SIGNALS (Tomorrow)")
    print("="*72)

    if not actionable:
        print("  No high-conviction trades found.")
    else:
        for tier in tier_order:
            tier_signals = [s for s in actionable if getattr(s, "cap_tier", "unknown") == tier]
            if not tier_signals:
                continue
            print(f"\n  --- {tier_labels[tier]} ---")
            for s in tier_signals:
                tag    = "BUY  ^" if s.direction == "LONG" else "SELL v"
                stars  = {"STRONG": "***", "MODERATE": "**", "WEAK": "*"}.get(s.confluence, "")
                pct    = abs((s.target - s.entry) / s.entry * 100)
                sl_pct = abs((s.stop_loss - s.entry) / s.entry * 100)
                print(f"\n    [{tag}] {s.symbol}  [{s.confidence} confidence]  "
                      f"Confluence: {s.confluence} {stars}")
                print(f"      Entry:     {s.entry}")
                print(f"      Target:    {s.target}  "
                      f"({'+' if s.direction=='LONG' else '-'}{pct:.1f}%)")
                print(f"      Stop Loss: {s.stop_loss}  (-{sl_pct:.1f}%)")
                print(f"      R:R Ratio: {s.rr_ratio}:1")
                print(f"      Trend:     {s.trend_bias}  (score {s.trend_score:+d})")
                if s.sentiment_count > 0:
                    print(f"      Sentiment: {s.sentiment} ({s.sentiment_score:.2f}, {s.sentiment_count} headlines)")
                for r in s.reasons:
                    print(f"      * {r}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)} stocks — no edge / low R:R / against trend):")
        for s in skipped:
            tier = getattr(s, "cap_tier", "")
            print(f"    [{tier.upper():<5}] {s.symbol:<14} {' | '.join(s.reasons)}")
    print()


def save_results(signals: list, variant: str) -> None:
    signals = [s for s in signals if s.direction != "NO TRADE"]
    if not signals:
        print("\n  Nothing to save — no actionable signals.")
        return
    rows = []
    for s in signals:
        rows.append({
            "Cap Tier":   getattr(s, "cap_tier", ""),
            "Symbol":     s.symbol,
            "Direction":  s.direction,
            "Entry":      s.entry,
            "Target":     s.target,
            "Stop Loss":  s.stop_loss,
            "R:R":        s.rr_ratio,
            "Confidence": s.confidence,
            "Confluence": s.confluence,
            "Trend":           s.trend_bias,
            "Trend Score":     s.trend_score,
            "Sentiment":       s.sentiment,
            "Sentiment Score": s.sentiment_score,
            "Sentiment Count": s.sentiment_count,
            "Reason":          " | ".join(s.reasons),
        })
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"signals_{ts}_{variant}.csv")
    df.to_csv(path, index=False)
    print(f"\n  Results saved to: {path}")


def main():
    args = parse_args()
    context_len = CONTEXT_LEN[args.variant]
    pred_len    = args.days * CANDLES_PER_DAY[args.interval]

    print("=" * 65)
    print("  NSE INTRADAY SIGNAL PIPELINE — Powered by Kronos")
    print(f"  Model: Kronos-{args.variant} | Interval: {args.interval} | Predicting {args.days}d ({pred_len} candles) | {datetime.now().strftime('%d %b %Y %H:%M')}")
    print("=" * 65)

    try:
        load_model(variant=args.variant)
    except ImportError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Step 1: Get symbols
    if args.symbols:
        symbols = resolve_symbols(args.symbols)
        scan_results = {}
        print(f"\n[1/6] Using provided symbols: {', '.join(symbols)}")
    else:
        symbols, scan_results = get_symbols_from_scanner(args.top, args.cap)

    if not symbols:
        print("[ERROR] No symbols to process. Exiting.")
        sys.exit(1)

    # Steps 2-5
    stocks      = fetch_all_data(symbols, context_len, pred_len, args.interval)
    if not stocks:
        print("[ERROR] No valid data fetched. Exiting.")
        sys.exit(1)

    trends      = fetch_trends([s[0] for s in stocks])
    sentiments  = fetch_sentiments([s[0] for s in stocks]) if not args.no_sentiment else {}
    predictions = run_predictions(stocks, args.samples)
    signals     = build_signals(stocks, predictions, trends, scan_results, sentiments)

    display_tiered_signals(signals)

    # Summary table
    df_sig = signals_to_dataframe(signals)
    actionable = df_sig[df_sig["Direction"] != "NO TRADE"]
    if not actionable.empty:
        print("  SUMMARY:")
        print(actionable[["Symbol", "Direction", "Entry", "Target",
                           "Stop Loss", "R:R", "Confidence", "Confluence"]].to_string(index=False))

    if args.save:
        save_results(signals, args.variant)

    if args.track:
        from tracker import log_signals
        log_signals(signals, pred_days=args.days, interval=args.interval, trends=trends)

    print("\nDone. Always place a hard stop-loss order with your broker.")


if __name__ == "__main__":
    main()
