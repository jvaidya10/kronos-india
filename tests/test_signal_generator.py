"""
Unit tests for pipeline/signal_generator.py — the Kronos-native variable
target/stop logic: target = predicted extreme, stop = opposite extreme clamped
to the [SL_FLOOR_PCT, SL_CAP_PCT] risk band, with noise and R:R filters.

Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from pipeline.signal_generator import (
    generate_signal, MIN_MOVE_PCT, SL_CAP_PCT, SL_FLOOR_PCT, MIN_RR_RATIO,
    TARGET_QUANTILE, MIN_DIR_AGREEMENT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeTrend:
    """Minimal stand-in for TrendSnapshot — only the fields the generator reads."""
    def __init__(self, score):
        self.score           = score
        self.overall_bias    = "BULLISH" if score >= 0 else "BEARISH"
        self.monthly_bias    = self.overall_bias
        self.monthly_chg_pct = 0.0
        self.weekly_bias     = self.overall_bias
        self.weekly_chg_pct  = 0.0
        self.rsi14           = 50.0
        self.adx14           = 20.0
        self.rvol            = 1.0
        self.volume_spike    = False
        self.obv_trend       = "FLAT"


def _pred(open_, high, low):
    """One-row predicted OHLCV with an IQR band, as the predictor emits."""
    close = (high + low) / 2
    return pd.DataFrame({
        "open": [open_], "high": [high], "low": [low], "close": [close],
        "close_q25": [low], "close_q75": [high],
    })


# ── Variable target ───────────────────────────────────────────────────────────

class TestVariableTarget:

    def test_small_move_fires(self):
        # Up 3% / down 1% — old 5% floor would have rejected this.
        s = generate_signal("X", _pred(100, 103, 99), current_price=100.0,
                            trend=FakeTrend(5))
        assert s.direction == "LONG"
        assert s.target == 103.0                    # target = pred_high, not capped
        assert abs(s.stop_loss - 99.0) < 0.01       # stop = pred_low (1% down)
        assert s.rr_ratio == 3.0

    def test_large_move_not_capped(self):
        # TCS-style 10% fall — old 7% cap would have clipped the target.
        s = generate_signal("X", _pred(100, 100.5, 90), current_price=100.0,
                            trend=FakeTrend(-5))
        assert s.direction == "SHORT"
        assert s.target == 90.0                      # full -10% captured
        assert abs(s.stop_loss - 101.0) < 0.01       # opposite extreme = +0.5% -> floored to 1%
        assert s.rr_ratio == 10.0


# ── Stop-loss clamping ────────────────────────────────────────────────────────

class TestStopClamping:

    def test_stop_capped_at_max(self):
        # Predicted downside 5% on a long -> stop held at SL_CAP_PCT (2.5%).
        # Up 6% keeps direction LONG and R:R above the 2:1 minimum.
        s = generate_signal("X", _pred(100, 106, 95), current_price=100.0,
                            trend=FakeTrend(5))
        assert s.direction == "LONG"
        assert abs(s.stop_loss - (100 * (1 - SL_CAP_PCT / 100))) < 0.01

    def test_stop_floored_at_min(self):
        # Predicted downside 0.5% -> stop widened to SL_FLOOR_PCT (1%).
        s = generate_signal("X", _pred(100, 106, 99.5), current_price=100.0,
                            trend=FakeTrend(6))
        assert s.direction == "LONG"
        assert abs(s.stop_loss - (100 * (1 - SL_FLOOR_PCT / 100))) < 0.01


# ── Filters ───────────────────────────────────────────────────────────────────

class TestFilters:

    def test_noise_floor_blocks_tiny_move(self):
        # Up 1.2% is below MIN_MOVE_PCT (1.5%) -> NO TRADE.
        s = generate_signal("X", _pred(100, 101.2, 99.8), current_price=100.0,
                            trend=FakeTrend(2))
        assert s.direction == "NO TRADE"

    def test_rr_filter_rejects_weak_setup(self):
        # Up 2.8% target, down 2.5% risk -> stop capped to 2.5% -> R:R 1.12 < 1.5.
        s = generate_signal("X", _pred(100, 102.8, 97.5), current_price=100.0,
                            trend=FakeTrend(5))
        assert s.direction == "NO TRADE"
        assert s.rr_ratio < MIN_RR_RATIO

    def test_against_trend_blocks(self):
        # Kronos says LONG (up dominates) but trend is strongly bearish.
        s = generate_signal("X", _pred(100, 105, 99), current_price=100.0,
                            trend=FakeTrend(-5))
        assert s.direction == "NO TRADE"
        assert s.confluence == "AGAINST TREND"


# ── Entry fallback ────────────────────────────────────────────────────────────

def test_entry_falls_back_to_pred_open_when_no_price():
    # current_price=None -> entry uses predicted open (first candle).
    s = generate_signal("X", _pred(100, 103, 99), current_price=None,
                        trend=FakeTrend(5))
    assert s.entry == 100.0


# ── Ensemble distribution path (per-sample stats in pred_df.attrs) ─────────────

def _pred_dist(open_, high, low, highs, lows, up_frac):
    """Predicted candle with the per-sample distribution metadata the model emits."""
    df = _pred(open_, high, low)
    df.attrs["n_samples"]     = len(highs)
    df.attrs["high_samples"]  = np.array(highs, dtype=float)
    df.attrs["low_samples"]   = np.array(lows, dtype=float)
    df.attrs["up_fraction"]   = up_frac
    df.attrs["down_fraction"] = 1.0 - up_frac
    return df


class TestDistributionPath:

    def test_target_is_quantile_of_sample_highs(self):
        # TARGET_QUANTILE=0.5 -> target = median of per-sample peak highs (104), not the max (106).
        s = generate_signal("X", _pred_dist(100, 106, 99,
                            highs=[102, 104, 106], lows=[99, 99, 99], up_frac=0.9),
                            current_price=100.0, trend=FakeTrend(5))
        assert s.direction == "LONG"
        assert s.target == round(float(np.quantile([102, 104, 106], TARGET_QUANTILE)), 2)
        assert s.target == 104.0

    def test_conviction_picks_short_despite_larger_upside_excursion(self):
        # Big high spike (excursion would pick LONG) but 90% of samples close DOWN.
        # Conviction-first must trade SHORT, the side the ensemble agrees on.
        s = generate_signal("X", _pred_dist(100, 108, 94,
                            highs=[101, 102, 108], lows=[94, 95, 96], up_frac=0.1),
                            current_price=100.0, trend=FakeTrend(-5))
        assert s.direction == "SHORT"
        assert s.dir_agreement >= MIN_DIR_AGREEMENT

    def test_low_agreement_blocks_trade(self):
        # 50/50 split ensemble -> below MIN_DIR_AGREEMENT -> NO TRADE on low conviction.
        s = generate_signal("X", _pred_dist(100, 106, 99,
                            highs=[104, 105, 106], lows=[98, 99, 99], up_frac=0.5),
                            current_price=100.0, trend=FakeTrend(5))
        assert s.direction == "NO TRADE"
        assert "conviction" in " ".join(s.reasons).lower()
