"""
Converts Kronos price predictions into actionable trade signals.
Target and stop-loss are Kronos-native: target = the model's predicted extreme
(pred_high for LONG, pred_low for SHORT), stop = the opposite extreme clamped to
a risk ceiling. No hardcoded profit magnitude.
Incorporates weekly + monthly trend confluence from trend_analyzer.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from trend_analyzer import TrendSnapshot


MIN_MOVE_PCT = 1.5   # noise filter — ignore predicted moves below this
SL_CAP_PCT   = 2.5   # max risk per trade (stop-loss ceiling)
SL_FLOOR_PCT = 1.0   # whipsaw protection (stop-loss floor)
MIN_RR_RATIO = 2.0   # reject setups below this reward:risk


@dataclass
class TradeSignal:
    symbol:      str
    direction:   str    # "LONG", "SHORT", "NO TRADE"
    entry:       float
    target:      float
    stop_loss:   float
    rr_ratio:    float
    pred_high:   float
    pred_low:    float
    confidence:  str    # "HIGH", "MEDIUM", "LOW"
    trend_bias:  str
    trend_score: int
    confluence:      str    # "STRONG", "MODERATE", "WEAK", "AGAINST TREND"
    reasons:         list = field(default_factory=list)
    sentiment:       str  = "NEUTRAL"   # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float = 0.0
    sentiment_count: int   = 0


def _confidence_from_iqr(pred_df: pd.DataFrame, entry: float) -> str:
    if "close_q25" not in pred_df.columns:
        return "MEDIUM"
    iqr     = (pred_df["close_q75"] - pred_df["close_q25"]).mean()
    iqr_pct = (iqr / entry) * 100 if entry > 0 else 99
    if iqr_pct < 1.5:  return "HIGH"
    if iqr_pct < 3.5:  return "MEDIUM"
    return "LOW"


def _confluence(direction: str, trend: "TrendSnapshot") -> str:
    score = trend.score  # -8 to +8
    if direction == "LONG":
        if score >= 3:   return "STRONG"       # ≥ +3
        if score >= 1:   return "MODERATE"     # +1 or +2
        if score == 0:   return "WEAK"         #  0
        return "AGAINST TREND"                 # < 0
    if direction == "SHORT":
        if score <= -3:  return "STRONG"       # ≤ -3
        if score <= -1:  return "MODERATE"     # -1 or -2
        if score == 0:   return "WEAK"         #  0
        return "AGAINST TREND"                 # > 0
    return "NEUTRAL"


def generate_signal(
    symbol: str,
    pred_df: pd.DataFrame,
    current_price: Optional[float] = None,
    trend: Optional["TrendSnapshot"] = None,
) -> TradeSignal:
    pred_open    = float(pred_df["open"].iloc[0])
    entry        = current_price if current_price is not None else pred_open
    pred_high    = float(pred_df["high"].max())
    pred_low     = float(pred_df["low"].min())
    upside_pct   = ((pred_high - entry) / entry) * 100
    downside_pct = ((entry - pred_low)  / entry) * 100
    confidence   = _confidence_from_iqr(pred_df, entry)
    trend_bias   = trend.overall_bias if trend else "UNKNOWN"
    trend_score  = trend.score        if trend else 0

    def _make(direction, move_pct, opposite_pct):
        # opposite_pct = Kronos's predicted move toward the stop side.
        # Clamp it to a risk band: never riskier than SL_CAP, never tighter than SL_FLOOR.
        sl_pct = min(max(opposite_pct, SL_FLOOR_PCT), SL_CAP_PCT)

        if direction == "LONG":
            target    = round(pred_high, 2)
            stop_loss = round(entry * (1 - sl_pct / 100), 2)
            risk, reward = entry - stop_loss, target - entry
        else:
            target    = round(pred_low, 2)
            stop_loss = round(entry * (1 + sl_pct / 100), 2)
            risk, reward = stop_loss - entry, entry - target

        rr         = round(reward / risk, 2) if risk > 0 else 0
        confluence = _confluence(direction, trend) if trend else "UNKNOWN"

        if confluence == "AGAINST TREND":
            return TradeSignal(
                symbol=symbol, direction="NO TRADE",
                entry=round(entry,2), target=0, stop_loss=0,
                rr_ratio=rr, pred_high=round(pred_high,2), pred_low=round(pred_low,2),
                confidence=confidence, trend_bias=trend_bias, trend_score=trend_score,
                confluence=confluence,
                reasons=[f"Kronos says {direction} but trend is {trend_bias} (score {trend_score:+d}) — blocked"],
            )

        if rr < MIN_RR_RATIO:
            return TradeSignal(
                symbol=symbol, direction="NO TRADE",
                entry=round(entry,2), target=0, stop_loss=0,
                rr_ratio=rr, pred_high=round(pred_high,2), pred_low=round(pred_low,2),
                confidence=confidence, trend_bias=trend_bias, trend_score=trend_score,
                confluence=confluence,
                reasons=[f"R:R {rr} below minimum {MIN_RR_RATIO}"],
            )

        reasons = [f"Kronos predicts {'upside' if direction=='LONG' else 'downside'} {move_pct:.1f}% "
                   f"(target {move_pct:.1f}% / stop {sl_pct:.1f}%)"]
        if trend:
            reasons.append(f"Monthly: {trend.monthly_bias} ({trend.monthly_chg_pct:+.1f}%) | Weekly: {trend.weekly_bias} ({trend.weekly_chg_pct:+.1f}%)")
            reasons.append(f"RSI {trend.rsi14} | ADX {trend.adx14} | Score {trend_score:+d}/8")
            vol_str = f"RVOL {trend.rvol}x" + (" [VOLUME SPIKE — confirms move]" if trend.volume_spike else " [normal volume]")
            reasons.append(f"{vol_str} | OBV {trend.obv_trend}")

        return TradeSignal(
            symbol=symbol, direction=direction,
            entry=round(entry,2), target=target, stop_loss=stop_loss,
            rr_ratio=rr, pred_high=round(pred_high,2), pred_low=round(pred_low,2),
            confidence=confidence, trend_bias=trend_bias, trend_score=trend_score,
            confluence=confluence, reasons=reasons,
        )

    if upside_pct >= MIN_MOVE_PCT and upside_pct > downside_pct:
        return _make("LONG", upside_pct, downside_pct)

    if downside_pct >= MIN_MOVE_PCT and downside_pct > upside_pct:
        return _make("SHORT", downside_pct, upside_pct)

    return TradeSignal(
        symbol=symbol, direction="NO TRADE",
        entry=round(entry,2), target=0, stop_loss=0, rr_ratio=0,
        pred_high=round(pred_high,2), pred_low=round(pred_low,2),
        confidence=confidence, trend_bias=trend_bias, trend_score=trend_score,
        confluence="NEUTRAL",
        reasons=[f"No clear edge — upside {upside_pct:.1f}% | downside {downside_pct:.1f}%"],
    )


def signals_to_dataframe(signals: list) -> pd.DataFrame:
    rows = []
    for s in signals:
        rows.append({
            "Symbol":     s.symbol,
            "Direction":  s.direction,
            "Entry":      s.entry,
            "Target":     s.target,
            "Stop Loss":  s.stop_loss,
            "R:R":        s.rr_ratio,
            "Pred High":  s.pred_high,
            "Pred Low":   s.pred_low,
            "Confidence": s.confidence,
            "Trend":      s.trend_bias,
            "Confluence": s.confluence,
            "Reason":     " | ".join(s.reasons),
            "Sentiment":  s.sentiment,
        })
    return pd.DataFrame(rows)


def display_signals(signals: list) -> None:
    actionable = [s for s in signals if s.direction != "NO TRADE"]
    skipped    = [s for s in signals if s.direction == "NO TRADE"]

    print("\n" + "="*70)
    print("  ACTIONABLE TRADE SIGNALS")
    print(f"  Target: Kronos range (min {MIN_MOVE_PCT}%)  |  Max SL {SL_CAP_PCT}%  |  Min R:R {MIN_RR_RATIO}:1")
    print("="*70)

    if not actionable:
        print("  No high-conviction trades found.")
    else:
        for s in actionable:
            tag    = "BUY  ^" if s.direction == "LONG" else "SELL v"
            stars  = {"STRONG": "***", "MODERATE": "**", "WEAK": "*"}.get(s.confluence, "")
            tgt_pct = abs((s.target - s.entry) / s.entry * 100)
            sl_pct  = abs((s.stop_loss - s.entry) / s.entry * 100)
            print(f"\n  [{tag}] {s.symbol}  [{s.confidence} confidence]  Confluence: {s.confluence} {stars}")
            print(f"    Entry:     {s.entry}")
            print(f"    Target:    {s.target}  ({'+' if s.direction=='LONG' else '-'}{tgt_pct:.1f}%)")
            print(f"    Stop Loss: {s.stop_loss}  (-{sl_pct:.1f}%)")
            print(f"    R:R Ratio: {s.rr_ratio}:1")
            print(f"    Trend:     {s.trend_bias}  (score {s.trend_score:+d}/8)")
            if s.sentiment_count > 0:
                print(f"    Sentiment: {s.sentiment} ({s.sentiment_score:.2f}, {s.sentiment_count} headlines)")
            for r in s.reasons:
                print(f"    * {r}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)} stocks):")
        for s in skipped:
            print(f"    {s.symbol:<14} {s.reasons[0] if s.reasons else ''}")
    print()
