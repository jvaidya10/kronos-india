"""
Computes weekly and monthly trend for a stock using daily OHLCV data.
Uses price momentum, moving averages, RSI, ADX, RVOL, and OBV.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrendSnapshot:
    symbol:          str
    monthly_bias:    str    # "BULLISH", "BEARISH", "NEUTRAL"
    weekly_bias:     str
    monthly_chg_pct: float
    weekly_chg_pct:  float
    above_sma20:     bool
    above_sma50:     bool
    rsi14:           float
    adx14:           float
    rvol:            float  # today's volume / 20-day avg volume
    volume_spike:    bool   # rvol > 1.5
    obv_trend:       str    # "RISING", "FALLING", "FLAT"
    overall_bias:    str
    score:           int    # -8 to +8


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta).clip(lower=0).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)


def _rvol(volume: pd.Series, lookback: int = 20) -> tuple:
    avg   = float(volume.iloc[-lookback-1:-1].mean())
    today = float(volume.iloc[-1])
    rvol  = round(today / avg, 2) if avg > 0 else 1.0
    return rvol, rvol > 1.5


def _obv_trend(close: pd.Series, volume: pd.Series, period: int = 10) -> str:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv       = (direction * volume).cumsum()
    slope     = float(obv.iloc[-1]) - float(obv.iloc[-period])
    if slope > 0:  return "RISING"
    if slope < 0:  return "FALLING"
    return "FLAT"


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    dm_p = (high - high.shift()).clip(lower=0)
    dm_m = (low.shift() - low).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0)
    dm_m = dm_m.where(dm_m > dm_p, 0)
    atr  = tr.rolling(period).mean()
    di_p = 100 * dm_p.rolling(period).mean() / atr
    di_m = 100 * dm_m.rolling(period).mean() / atr
    dx   = (100 * (di_p - di_m).abs() / (di_p + di_m)).replace([np.inf, -np.inf], 0)
    return round(float(dx.rolling(period).mean().iloc[-1]), 2)


def fetch_daily(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    ticker = symbol.strip().upper()
    if not ticker.endswith((".NS", ".BO")):
        ticker += ".NS"
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low",
                                 "Close":"close","Volume":"volume"})
        df = df[["open","high","low","close","volume"]].dropna()
        return df if len(df) >= 30 else None
    except Exception:
        return None


def _bias(chg_pct: float, threshold: float = 1.5) -> str:
    if chg_pct >  threshold: return "BULLISH"
    if chg_pct < -threshold: return "BEARISH"
    return "NEUTRAL"


def analyze(symbol: str) -> Optional[TrendSnapshot]:
    df = fetch_daily(symbol)
    if df is None:
        return None

    close   = df["close"]
    current = float(close.iloc[-1])

    weekly_chg  = ((current - float(close.iloc[-6]))  / float(close.iloc[-6]))  * 100
    monthly_chg = ((current - float(close.iloc[-23])) / float(close.iloc[-23])) * 100

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20
    above_sma20 = current > sma20
    above_sma50 = current > sma50

    rsi              = _rsi(close)
    adx              = _adx(df)
    rvol, vol_spike  = _rvol(df["volume"])
    obv              = _obv_trend(close, df["volume"])

    weekly_bias  = _bias(weekly_chg,  threshold=1.0)
    monthly_bias = _bias(monthly_chg, threshold=2.0)

    score = 0
    score += 1 if monthly_bias == "BULLISH" else (-1 if monthly_bias == "BEARISH" else 0)
    score += 1 if weekly_bias  == "BULLISH" else (-1 if weekly_bias  == "BEARISH" else 0)
    score += 1 if above_sma20 else -1
    score += 1 if above_sma50 else -1
    score += 1 if rsi > 55 else (-1 if rsi < 45 else 0)
    # Volume spike confirms direction of the move
    if vol_spike:
        score += 1 if monthly_bias != "BEARISH" else -1
    score += 1 if obv == "RISING" else (-1 if obv == "FALLING" else 0)
    if adx > 25:
        score += 1 if score > 0 else (-1 if score < 0 else 0)

    score = max(-8, min(8, score))

    if score >= 5:    overall = "STRONGLY BULLISH"
    elif score >= 3:  overall = "BULLISH"
    elif score >= 1:  overall = "MILD BULLISH"
    elif score == 0:  overall = "NEUTRAL"
    elif score >= -2: overall = "MILD BEARISH"
    elif score >= -4: overall = "BEARISH"
    else:             overall = "STRONGLY BEARISH"

    return TrendSnapshot(
        symbol=symbol,
        monthly_bias=monthly_bias, weekly_bias=weekly_bias,
        monthly_chg_pct=round(monthly_chg, 2), weekly_chg_pct=round(weekly_chg, 2),
        above_sma20=above_sma20, above_sma50=above_sma50,
        rsi14=rsi, adx14=adx,
        rvol=rvol, volume_spike=vol_spike, obv_trend=obv,
        overall_bias=overall, score=score,
    )


def display_trend(t: TrendSnapshot) -> None:
    sma = f"SMA20={'above' if t.above_sma20 else 'below'} SMA50={'above' if t.above_sma50 else 'below'}"
    vol = f"RVOL={t.rvol}x{'[SPIKE]' if t.volume_spike else ''} OBV={t.obv_trend}"
    print(
        f"  {t.symbol:<14} "
        f"Monthly:{t.monthly_bias:<10}({t.monthly_chg_pct:+.1f}%)  "
        f"Weekly:{t.weekly_bias:<10}({t.weekly_chg_pct:+.1f}%)  "
        f"RSI:{t.rsi14:>6}  ADX:{t.adx14:>6}  "
        f"{sma}  {vol:<22}  => {t.overall_bias}  [score {t.score:+d}]"
    )
