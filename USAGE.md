# NSE Intraday Signal Pipeline — Usage Guide

Run from the `kronos-india` folder:

```bash
python main.py [arguments]
```

---

## Arguments

### `--variant`

Which Kronos model to use for prediction.

| Value   | Params | Context Window                     | Speed (RTX 5070) | Best For          |
|---------|--------|------------------------------------|------------------|-------------------|
| `small` | 24.7M  | 512 candles (~3 months of 1h data) | ~2-5 sec/stock   | Default, balanced |
| `mini`  | 4.1M   | 2048 candles (~1 year of 1h data)  | ~1-2 sec/stock   | Longer history    |
| `base`  | 102.3M | 512 candles                        | ~8-12 sec/stock  | Best accuracy     |

```bash
python main.py --variant small
python main.py --variant base
python main.py --variant mini
```

---

### `--cap`

Which market-cap tier to scan for gainers and losers.

| Value   | Universe                | Stocks Scanned |
|---------|-------------------------|----------------|
| `all`   | Large + Mid + Small cap | ~500 stocks    |
| `large` | Nifty 100               | ~100 stocks    |
| `mid`   | Nifty Midcap 150        | ~150 stocks    |
| `small` | Nifty Smallcap 250      | ~250 stocks    |

```bash
python main.py --cap all
python main.py --cap large
python main.py --cap mid
python main.py --cap small
```

---

### `--universe`

Chooses the stock universe. This is the most important lever for signal quality.

- **`scanner`** *(default)* — ranks each cap tier's biggest gainers & losers by
  today's % move, i.e. the day's most *volatile* movers.
- **`nifty50`** — runs a **fixed** list of Nifty 50 constituents, no ranking.
- **`nifty100`** — runs a **fixed** list of Nifty 100 constituents.

**Why this matters:** the `scanner` selects stocks *because* they moved the most
today — by construction the volatile, news-driven names. Backtests and live
tracking both show Kronos has a (small) directional edge on **calm, liquid
large-caps** and essentially none on volatile movers. The `nifty50` / `nifty100`
universes point the model at the stocks it predicts best, instead of the day's
chaos. The constituent lists are fetched live from NSE (with a hardcoded
fallback). `--cap` and `--top` are ignored when a fixed universe is selected.

```bash
python main.py --universe nifty50              # fixed liquid large-caps
python main.py --universe nifty100 --interval 1h --track
```

---

### `--top`

How many top gainers and top losers to pick **per cap tier**.

Default is `10` (10 gainers + 10 losers per tier).

```bash
python main.py --top 5    # top 5 gainers + 5 losers per tier
python main.py --top 15   # top 15 gainers + 15 losers per tier
```

---

### `--interval`

Candle size used for fetching data and predicting. Smaller intervals give more
candles per day — better resolution for intraday trades.

| Value | Candles/day | Context (512 candles) | History available | Best For |
| --- | --- | --- | --- | --- |
| `1h` | 7 | ~73 trading days | 730 days | Swing / multi-day |
| `15m` | 25 | ~20 trading days | 60 days | Intraday (recommended) |
| `5m` | 75 | ~7 trading days | 60 days | Detailed intraday |
| `1m` | 375 | ~1.5 trading days | 7 days only | Scalping (limited) |

```bash
python main.py --interval 1h    # default
python main.py --interval 15m --days 1   # 25 candles for tomorrow
python main.py --interval 5m  --days 1   # 75 candles for tomorrow
```

> Tip: For `--days 1` (intraday), use `--interval 15m` or `--interval 5m`.
> For `--days 3` or more, stick with `--interval 1h` to have enough history.

---

### `--days`

How many trading days ahead to predict. Each day = 7 hourly candles (NSE session).

Default is `3` (21 candles = next 3 trading days).

| Value | Candles | Use When                      |
|-------|---------|-------------------------------|
| `1`   | 7       | Pure intraday — tomorrow only |
| `3`   | 21      | Short swing trade (default)   |
| `5`   | 35      | Weekly outlook                |

> Kronos-small and Kronos-base have a 512-candle context window. Predicting more days extends the forecast horizon without reducing context.

```bash
python main.py --days 1   # tomorrow only
python main.py --days 3   # next 3 days (default)
python main.py --days 5   # full week
```

---

### `--samples`

Number of times Kronos runs prediction per stock. Results are aggregated using median.
More samples = more stable signal + better confidence estimate, but slower.

| Value | Speed          | Use When                 |
|-------|----------------|--------------------------|
| `5`   | Very fast      | Quick test / exploration |
| `20`  | Fast (default) | Daily use                |
| `50`  | Moderate       | Higher conviction needed |
| `100` | Slow           | Maximum confidence       |

```bash
python main.py --samples 5
python main.py --samples 50
```

---

### `--symbols`

Skip the NSE scanner entirely and predict specific stocks directly.
Accepts one or more NSE ticker symbols **or company names** — the pipeline resolves
names to symbols automatically using a fuzzy index of 2,456 NSE equities.

```bash
# By exact symbol
python main.py --symbols RELIANCE
python main.py --symbols RELIANCE TCS INFY HDFCBANK

# By company name (partial names work too)
python main.py --symbols "reliance industries" "hdfc bank" "bajaj finance"
python main.py --symbols "infosys" "zomato" --samples 30

# Mix of symbols and names
python main.py --symbols RELIANCE "hdfc bank" INFY
```

> If a name is ambiguous or the company uses an abbreviated name in the exchange
> listing (e.g. TCS is listed as "TCS", SBI as "SBI"), a warning is printed and
> you can fall back to the exact symbol instead.

---

### `--save`

Save actionable signals (LONG/SHORT only, NO TRADE excluded) to a timestamped CSV
inside the `outputs/` folder.

```bash
python main.py --save
```

Output path: `outputs/signals_YYYYMMDD_HHMM_<variant>.csv`

---

### `--no-sentiment`

Skip the news sentiment analysis step (step 4/6). Useful for faster runs or when offline.

By default, sentiment is **on** — FinBERT classifies recent Google News headlines per stock into BULLISH / BEARISH / NEUTRAL and displays the result alongside each signal. No API key required.

```bash
python main.py --no-sentiment                    # skip sentiment, faster run
python main.py --symbols RELIANCE --no-sentiment # quick single-stock test
```

---

### `--workers`

Number of concurrent network workers used when fetching data (OHLCV download,
trend analysis, news sentiment, and current-price lookups). These steps are
I/O-bound, so running them in parallel substantially speeds up large scans.

Default is `8`. Use `--workers 1` for fully serial fetching (e.g. to debug, or
if you hit news-feed rate limits).

```bash
python main.py --cap all --top 10          # parallel fetch (default 8 workers)
python main.py --workers 1                  # serial (slowest, most conservative)
```

---

### `--track`

Log all actionable signals to the prediction tracker database (`outputs/tracker.db`).
The candle interval used for the run is stored alongside each signal and replayed during
evaluation — a 15m signal is evaluated on 15m candles, a 1h signal on 1h candles.
The `eval_by` date is calculated using NSE holiday-aware business days.
After `--days` trading days have passed, run `tracker.py evaluate` to check outcomes.
Use this daily to build a performance record and measure model accuracy over time.

```bash
python main.py --track
python main.py --cap large --days 3 --track
python main.py --interval 15m --days 1 --track   # stores interval=15m for evaluation
```

---

## Examples

```bash
# Quick test on specific stocks
python main.py --symbols RELIANCE TCS INFY --samples 5

# Full daily scan — all cap tiers, log signals for tracking
python main.py --track

# Large cap only, save CSV + track
python main.py --cap large --save --track

# Mid cap, high-accuracy model, 50 samples
python main.py --cap mid --variant base --samples 50

# Small cap top 5 per tier, tomorrow only
python main.py --cap small --top 5 --days 1

# Best accuracy run
python main.py --variant base --samples 100 --cap large --save --track
```

---

## Prediction Tracker

The tracker logs every LONG/SHORT signal and evaluates actual outcomes after the
prediction window closes. Use it to measure win rate, P&L, and which setups work best.

### Commands

```bash
# Check outcomes for all matured signals (run after --days trading days)
python tracker.py evaluate

# View full performance report
python tracker.py report

# List all logged signals (last 50)
python tracker.py show

# Force-evaluate signals whose eval_by has passed but were skipped due to data gaps
# (signals with eval_by still in the future are skipped with a clear message)
python tracker.py evaluate --force

# Import a saved signals CSV into the tracker (for runs done without --track)
python tracker.py import --csv outputs/signals_20260604_0410_base.csv
python tracker.py import --csv outputs/signals_...csv --days 1 --interval 15m
```

### Importing CSVs

If you ran the pipeline with `--save` but forgot `--track`, the signals only
exist in the CSV. The `import` command loads them into `tracker.db` so they can
be evaluated and reported like any tracked signal:

- `logged_at` is parsed from the filename timestamp (`signals_YYYYMMDD_HHMM_*.csv`)
- `eval_by` is computed from `--days` (default 3) using NSE holiday-aware business days
- `--interval` sets which candle size to replay during evaluation (default `1h`)
- Duplicate rows (same symbol + direction + logged_at) are skipped, so re-importing is safe

In the Streamlit dashboard, the same feature appears as an **"Import CSV into
Tracker"** expander in the Tracker section whenever saved CSVs exist.

### Report output

```text
Total signals evaluated : 24
Wins                    : 15  (62.5%)
Losses                  : 7
Expired (no hit)        : 2
Avg win P&L             : +5.8%
Avg loss P&L            : -2.4%
Expectancy              : +2.7% per trade

--- By Confluence Level ---
STRONG     :  8 trades | Win rate 75% | Avg P&L +3.8%
MODERATE   : 12 trades | Win rate 58% | Avg P&L +2.1%

--- By Confidence Level ---
HIGH       : 10 trades | Win rate 80% | Avg P&L +4.1%
MEDIUM     : 11 trades | Win rate 55% | Avg P&L +1.8%
LOW        :  3 trades | Win rate 33% | Avg P&L -0.6%

--- By Sentiment ---
BULLISH  :  9 trades | Win rate 78% | Avg P&L +3.9%
NEUTRAL  : 11 trades | Win rate 55% | Avg P&L +1.4%
BEARISH  :  4 trades | Win rate 25% | Avg P&L -1.2%

--- By Cap Tier ---
LARGE cap  : 10 trades | Win rate 70% | Avg P&L +3.2%
SMALL cap  :  8 trades | Win rate 50% | Avg P&L +0.8%

--- By Direction ---
LONG   : 18 trades | Win rate 67% | Avg P&L +2.9%
SHORT  :  6 trades | Win rate 50% | Avg P&L +1.1%

--- By Interval ---
1h   : 14 trades | Win rate 64% | Avg P&L +2.4%
15m  : 10 trades | Win rate 60% | Avg P&L +3.1%

--- Streaks ---
Current streak  : 3 × WIN
Best win streak : 5 in a row
Worst loss run  : 3 in a row
```

#### Evaluation accuracy notes

- P&L is calculated from the **actual next-day open price**, not the signal's logged close price
- Targets and stop-losses are recalculated from that actual open using the original percentages
- EXPIRED trades are closed at the last candle of the `eval_by` date
- `eval_by` dates skip weekends and official NSE holidays

---

## Backtesting

`backtest.py` replays Kronos over historical intraday windows so you can measure
the model and tune the signal logic *without waiting weeks for live outcomes*. It
reuses the exact tracker outcome logic, so backtest results and live `--track`
results are directly comparable.

It reports three things:

1. **Forecast accuracy** — the raw model's directional hit rate (vs a 50% coin
   flip) and final-close MAPE, independent of any signal thresholds.
2. **Signal performance** — WIN / LOSS / EXPIRED, win rate and average P&L at the
   current settings.
3. **Config sweep** (`--sweep`) — re-scores the cached predictions across a grid
   of `target_quantile` × `min_dir_agreement` so you can pick the best defaults.
   The model runs once per window; the sweep itself is almost free.

```bash
# Forecast accuracy + signal performance on a basket
python backtest.py --symbols RELIANCE TCS INFY HDFCBANK --anchors 25

# Tune the target-quantile / agreement defaults
python backtest.py --symbols TATASTEEL VEDL SAIL ADANIENT --days 3 --sweep

# Isolate pure forecast/target quality (skip trend confluence)
python backtest.py --symbols RELIANCE INFY --no-trend --sweep
```

Key options: `--anchors N` (historical windows per symbol), `--samples N`
(ensemble size), `--variant`, `--interval`, `--days`, `--no-trend`, `--sweep`.

**Two findings worth knowing:** the forecast directional edge is strongest on
liquid large-caps (calm names, small moves) and weakest on volatile movers; and
a *reachable* target (`target_quantile` ≈ 0.5, the default) produces a higher win
rate and far fewer expiries than targeting the single most optimistic predicted
price. See `signal_generator.py` (`TARGET_QUANTILE`, `MIN_DIR_AGREEMENT`).

---

## Signal Output Explained

```text
[BUY  ^] INFY  [HIGH confidence]  Confluence: STRONG ***
  Entry:     1202.50
  Target:    1237.40  (+2.9%)
  Stop Loss: 1190.50  (-1.0%)
  R:R Ratio: 2.90:1
  Trend:     BULLISH  (score +5/8)
  Sentiment: BULLISH (0.81, 5 headlines)
  * Kronos predicts upside 2.9% (target 2.9% / stop 1.0%)
  * Monthly: BULLISH (+3.6%) | Weekly: BULLISH (+2.9%)
  * RSI 63.7 | ADX 19.0 | Score +5/8
  * RVOL 2.1x [VOLUME SPIKE - confirms move] | OBV RISING
```

| Field            | Meaning                                                         |
|------------------|-----------------------------------------------------------------|
| `BUY ^ / SELL v` | Long (buy) or Short (sell) signal                               |
| `Confidence`     | How closely Kronos's samples agreed (HIGH / MEDIUM / LOW)       |
| `Confluence`     | Alignment between Kronos signal and weekly + monthly trend      |
| `Entry`          | Current price — your trade entry                                |
| `Target`         | TARGET_QUANTILE (median) of Kronos's per-sample predicted range |
| `Stop Loss`      | Kronos's opposite extreme, clamped to 1.0–2.5% risk band        |
| `R:R Ratio`      | Reward-to-risk ratio — minimum 1.5:1 to take the trade          |
| `Sentiment`      | FinBERT news sentiment — BULLISH / BEARISH / NEUTRAL            |
| `Trend score`    | -8 (strongly bearish) to +8 (strongly bullish)                  |
| `RVOL`           | Today's volume vs 20-day average — spike confirms the move      |
| `OBV`            | On Balance Volume trend — RISING means money flowing in         |

### Confluence levels

| Level | Meaning |
| --- | --- |
| `STRONG ***` | Trend score >= +3 (LONG) or <= -3 (SHORT) — strongly agrees with Kronos |
| `MODERATE **` | Trend score +1 or +2 (LONG) or -1 or -2 (SHORT) — leans same way |
| `WEAK *` | Trend score 0 — neutral, take smaller position |
| `AGAINST TREND` | Trend opposes signal direction — trade blocked, shown as NO TRADE |

---

## How Targets and Stops Are Set (and why)

Kronos forecasts a full OHLCV range for the prediction window — both a predicted
high and a predicted low. The signal generator uses **both**, instead of imposing
fixed profit/loss percentages.

| | Target | Stop-loss |
| --- | --- | --- |
| LONG | `TARGET_QUANTILE` of per-sample peak highs | opposite extreme, clamped to the 1.0–2.5% risk band |
| SHORT | `TARGET_QUANTILE` of per-sample trough lows | opposite extreme, clamped to the 1.0–2.5% risk band |

Direction is chosen by the ensemble's **net-direction vote** (the side most of
Kronos's samples close toward), not the larger excursion — a volatile up-spike
can occur even when most samples close lower. A signal only fires if at least
`MIN_DIR_AGREEMENT` (55%) of samples agree on the direction, the predicted move
toward the target is at least **1.5%**, and the reward:risk is at least **1.5:1**
(lowered from 2:1 after backtesting — see the Backtesting section).

**Why this replaced the old fixed 5–7% target / 2.5% stop:**

- **The old 5% minimum missed real moves.** Large- and mid-cap stocks (TECHM,
  INFY, HCLTECH) routinely move 2–4% intraday but rarely 5%+. Those valid,
  correctly-predicted moves were discarded as `NO TRADE`. In one tracked run, 13
  of 14 signals expired with the stock moving the *right* direction (e.g. THERMAX
  +2.7%) but never reaching the 7% target — a 0% win rate despite correct
  direction calls.
- **The old 7% cap clipped the big winners.** When Kronos predicts a large move
  (e.g. an 8–10% fall, as has happened on large caps), capping the target at 7%
  left profit on the table and understated the trade's true reward:risk.
- **A stop derived from the prediction is meaningful.** The stop now sits at the
  level where Kronos's own forecast is invalidated (its predicted opposite
  extreme), not at an arbitrary fixed percentage. Because it varies per stock,
  the R:R filter now does real work — it rejects setups where the predicted risk
  is large relative to the predicted reward, instead of every trade passing at a
  constant 2.0.

**The two clamps are safety rails, not the main logic:**

- `SL_CAP_PCT = 2.5` — caps maximum loss per trade. If Kronos predicts a downside
  larger than 2.5% on a long, the stop is held at 2.5% (a transient predicted dip
  could stop you out early — the accepted cost of a hard risk ceiling).
- `SL_FLOOR_PCT = 1.0` — prevents a too-tight stop on a narrow predicted range,
  which would whipsaw out on normal intraday noise.

These four constants live at the top of `pipeline/signal_generator.py`
(`MIN_MOVE_PCT`, `SL_CAP_PCT`, `SL_FLOOR_PCT`, `MIN_RR_RATIO`) and can be tuned
once enough tracked signals reveal the best thresholds.

---

## Notes

- Predictions cover the next `--days` trading days (default 3 days = 21 hourly candles)
- NSE trading hours: 9:15 AM to 3:30 PM IST
- Always place a **hard stop-loss order** with your broker — do not rely on manual exits
- Kronos predictions are probabilistic, not guaranteed
- Run the pipeline **after market close** (after 3:30 PM IST) for next-day signals
- Use `--track` daily to build a performance record — aim for 20-30 signals before drawing conclusions
