"""
Fetches historical intraday OHLCV data for NSE stocks via yfinance.
Supports multiple candle intervals: 1h, 15m, 5m, 1m.
"""

import yfinance as yf
import pandas as pd
from typing import Optional


# yfinance max lookback per interval
LOOKBACK_DAYS = {
    "1h":  365,   # 730 days available, 365 is enough for 512-candle context
    "15m": 55,    # max 60 days
    "5m":  55,    # max 60 days
    "1m":  6,     # max 7 days
}

# NSE session = 375 minutes (9:15 AM to 3:30 PM)
CANDLES_PER_DAY = {
    "1h":  7,
    "15m": 25,
    "5m":  75,
    "1m":  375,
}

# Interval in minutes (for timestamp generation)
INTERVAL_MINUTES = {
    "1h":  60,
    "15m": 15,
    "5m":  5,
    "1m":  1,
}

KRONOS_CONTEXT = 512   # default; overridden per model variant

# NSE market hours (IST)
_MARKET_OPEN  = pd.Timedelta(hours=9,  minutes=15)
_MARKET_CLOSE = pd.Timedelta(hours=15, minutes=30)


def _to_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        symbol += ".NS"
    return symbol


def fetch_ohlcv(symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
    """
    Downloads OHLCV for the given NSE symbol at the specified interval.
    Returns a DataFrame with lowercase columns: open, high, low, close, volume.
    Returns None on failure.
    """
    if interval not in LOOKBACK_DAYS:
        raise ValueError(f"Unsupported interval '{interval}'. Choose: {list(LOOKBACK_DAYS)}")

    ticker = _to_nse_ticker(symbol)
    days   = LOOKBACK_DAYS[interval]

    try:
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] yfinance error for {symbol}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] No data returned for {symbol}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low":  "low",  "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    # For sub-hour intervals, keep only market-hours candles
    if interval != "1h":
        df = _filter_market_hours(df)

    if len(df) < 30:
        print(f"[WARN] Too few candles for {symbol} ({len(df)}). Skipping.")
        return None

    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only candles that fall within NSE market hours (9:15–15:30 IST)."""
    # Normalise timezone for comparison
    if df.index.tzinfo is not None:
        idx = df.index.tz_convert("Asia/Kolkata")
    else:
        idx = df.index

    time_of_day = idx.hour * 60 + idx.minute   # minutes since midnight
    market_open  = 9 * 60 + 15   # 555
    market_close = 15 * 60 + 30  # 930

    mask = (time_of_day >= market_open) & (time_of_day <= market_close)
    return df[mask]


def _next_market_timestamps(last_ts: pd.Timestamp, n: int, interval: str) -> list:
    """
    Generates n future candle timestamps strictly within NSE market hours,
    skipping weekends and non-market-hours gaps.
    """
    step_min  = INTERVAL_MINUTES[interval]
    step      = pd.Timedelta(minutes=step_min)
    future    = []
    ts        = last_ts

    while len(future) < n:
        ts = ts + step

        # Skip weekends
        if ts.weekday() >= 5:
            continue

        # Get time of day as timedelta
        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
        midnight  = ts_naive.normalize()
        tod       = ts_naive - midnight

        # Skip before market open or after market close
        if tod < _MARKET_OPEN or tod > _MARKET_CLOSE:
            # Jump to next day's open if we've passed close
            if tod > _MARKET_CLOSE:
                next_day = (midnight + pd.Timedelta(days=1))
                # Skip weekends
                while next_day.weekday() >= 5:
                    next_day += pd.Timedelta(days=1)
                # Reconstruct with original timezone
                if ts.tzinfo:
                    ts = pd.Timestamp(next_day + _MARKET_OPEN, tz=ts.tzinfo)
                else:
                    ts = next_day + _MARKET_OPEN
                ts -= step  # will be re-added at top of loop
            continue

        future.append(ts)

    return future


def prepare_context_and_forecast_timestamps(
    df: pd.DataFrame,
    pred_len: int = 21,
    context_len: int = KRONOS_CONTEXT,
    interval: str = "1h",
) -> tuple:
    """
    Returns (x_df, x_timestamp, y_timestamp) ready for Kronos.
    x_df         : last context_len candles
    x_timestamp  : timestamps for context candles
    y_timestamp  : pred_len future market-hours timestamps
    """
    x_df        = df.iloc[-context_len:].copy()
    x_timestamp = x_df.index.to_series().reset_index(drop=True)
    last_ts     = x_df.index[-1]

    future      = _next_market_timestamps(last_ts, pred_len, interval)
    y_timestamp = pd.Series(future)

    return x_df, x_timestamp, y_timestamp


def get_current_price(symbol: str) -> Optional[float]:
    ticker = _to_nse_ticker(symbol)
    try:
        return float(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        return None


if __name__ == "__main__":
    for iv in ["1h", "15m", "5m"]:
        df = fetch_ohlcv("RELIANCE", interval=iv)
        if df is not None:
            x_df, x_ts, y_ts = prepare_context_and_forecast_timestamps(
                df, pred_len=CANDLES_PER_DAY[iv], interval=iv
            )
            print(f"[{iv}] {len(df)} candles fetched | "
                  f"context={len(x_df)} | forecast={len(y_ts)} | "
                  f"first future: {y_ts.iloc[0]}")
