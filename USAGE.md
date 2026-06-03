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
Accepts one or more NSE ticker symbols (without `.NS`).

```bash
python main.py --symbols RELIANCE
python main.py --symbols RELIANCE TCS INFY HDFCBANK
python main.py --symbols ZOMATO NAUKRI --samples 30
```

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
```

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

## Signal Output Explained

```text
[BUY  ^] INFY  [HIGH confidence]  Confluence: STRONG ***
  Entry:     1202.50
  Target:    1262.60  (+5.0%)
  Stop Loss: 1172.40  (-2.5%)
  R:R Ratio: 2.00:1
  Trend:     BULLISH  (score +5/8)
  Sentiment: BULLISH (0.81, 5 headlines)
  * Kronos predicts upside 5.8%
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
| `Target`         | Price to book profit (5-7% from entry)                          |
| `Stop Loss`      | Price to exit if trade goes wrong (2.5% from entry)             |
| `R:R Ratio`      | Reward-to-risk ratio — minimum 2:1 to take the trade            |
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

## Notes

- Predictions cover the next `--days` trading days (default 3 days = 21 hourly candles)
- NSE trading hours: 9:15 AM to 3:30 PM IST
- Always place a **hard stop-loss order** with your broker — do not rely on manual exits
- Kronos predictions are probabilistic, not guaranteed
- Run the pipeline **after market close** (after 3:30 PM IST) for next-day signals
- Use `--track` daily to build a performance record — aim for 20-30 signals before drawing conclusions
