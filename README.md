# Kronos India — NSE Intraday Signal Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12%20CUDA-orange?logo=pytorch)
![Exchange](https://img.shields.io/badge/Exchange-NSE%20%7C%20BSE-green)
![Model](https://img.shields.io/badge/Model-Kronos%20Foundation-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> AI-powered intraday trade signal generator for Indian stock markets, built on the [Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model — accepted at **AAAI 2026**.

---

> ⚠️ **Disclaimer:** This tool is for **research and educational purposes only**. It does not constitute financial advice. Past model performance does not guarantee future results. Always use a hard stop-loss order with your broker. Trade only what you can afford to lose.

---
## Dashboard Preview

![Dashboard](images/dashboard.png)

## Trend Analysis

![Trend Analysis](images/trend_analysis.png)

## Experimental Results

⚠️ Results are currently being collected.

Current experiment:

- Model: Kronos-base
- Market Universe: NSE Large Cap + Mid Cap + Small Cap
- Interval: 1 minute
- Prediction Horizon: 1 trading day
- Signal Tracking: Enabled

The built-in tracker is collecting live signal outcomes. Performance statistics will be published once a statistically meaningful sample size has been accumulated.
## What is Kronos?

[Kronos](https://github.com/shiyu-coder/Kronos) is a decoder-only foundation model pre-trained specifically on financial candlestick (K-line) data — OHLCV sequences — from over 45 global exchanges, covering more than 12 billion K-line records. Unlike general-purpose time series models, Kronos is designed from the ground up for the unique noise characteristics of financial markets. It uses a specialized tokenizer that discretizes continuous price and volume data into discrete tokens, then applies autoregressive pre-training to learn temporal and cross-asset patterns.

In zero-shot benchmarks, Kronos outperforms the leading time series foundation model by 93% on price forecasting RankIC and achieves 9% lower MAE on volatility forecasting.

- 📄 [arXiv Paper](https://arxiv.org/abs/2508.02739)
- 🤗 [HuggingFace](https://huggingface.co/papers/2508.02739)
- 💻 [Official Repository](https://github.com/shiyu-coder/Kronos)

---

## What This Pipeline Does

This project wraps Kronos into a complete signal generation and tracking system for the Indian stock market (NSE/BSE). Every evening after market close, it:

1. **Scans NSE** for today's top gainers and losers across Large Cap (Nifty 100), Mid Cap (Nifty Midcap 150), and Small Cap (Nifty Smallcap 250)
2. **Analyses trends** using RSI, ADX, SMA20/50, weekly/monthly momentum, RVOL, and OBV
3. **Analyses sentiment** using FinBERT on recent Google News headlines — BULLISH / BEARISH / NEUTRAL per stock
4. **Predicts price** using Kronos — fed the last 512 candles of OHLCV history at your chosen interval
5. **Generates signals** with entry, a Kronos-native target (the model's own predicted high/low — no fixed cap), a stop-loss clamped to a 1.0–2.5% risk band, and R:R ratio — only when Kronos and the trend agree
6. **Tracks outcomes** over time to measure real-world accuracy against actual market data

---

## Architecture

```text
NSE Archive CSVs          →  Large / Mid / Small cap universe
yfinance OHLCV            →  Historical candles (1h / 15m / 5m / 1m)
Trend Analyzer            →  RSI · ADX · SMA · RVOL · OBV · Momentum
Google News RSS + FinBERT →  Sentiment per stock (BULLISH / BEARISH / NEUTRAL)
        ↓
Kronos Foundation Model   →  Predicts next N trading days (OHLCV)
        ↓
Signal Generator          →  LONG / SHORT / NO TRADE
                              Entry · Target · Stop-Loss · R:R · Confluence · Sentiment
        ↓
Prediction Tracker        →  SQLite log · WIN/LOSS evaluation · Performance report
```

---

## Prerequisites

| Requirement | Version |
| --- | --- |
| Python | 3.10+ |
| CUDA (recommended) | 12.8+ |
| GPU VRAM (recommended) | 4 GB+ |

> **GPU strongly recommended:** CPU inference is possible but very slow — Kronos-base may take several minutes per stock on CPU.

> **Note on `--interval 1m`:** Kronos has a 512-candle context window. One full NSE session is 375 one-minute candles, so `--interval 1m` gives Kronos only approximately 1.5 trading days of history as context — very limited. For broader market context use `--interval 15m` instead, which fits approximately 20 trading days within the same 512-candle window.

---

## Installation

### 1. Clone both repositories

```bash
git clone https://github.com/shiyu-coder/Kronos.git
git clone https://github.com/jvaidya10/kronos-india.git
cd kronos-india
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install einops==0.8.1 matplotlib==3.9.3
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 13.2 (RTX 40/50 series)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# For CUDA 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. Set the Kronos path

```bash
# Windows
setx KRONOS_PATH "C:\path\to\Kronos"

# Linux / Mac
export KRONOS_PATH="/path/to/Kronos"
```

### 5. Verify setup

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Quick Start

### Web Dashboard (recommended)

```bash
pip install streamlit
streamlit run app.py
```

Opens at `http://localhost:8501` — all arguments as dropdowns, per-step progress bars, styled scanner tables (green/red), trend analysis table, colour-coded signals table (green for LONG, red for SHORT), and tracker controls built in.

### Command Line

```bash
# Daily scan — all cap tiers, predict next 3 days, log signals
python main.py --track

# Intraday only — 15m candles, predict tomorrow, large cap
python main.py --interval 15m --days 1 --cap large --track

# Test on specific stocks — by symbol or company name
python main.py --symbols RELIANCE TCS INFY --samples 5
python main.py --symbols "reliance industries" "hdfc bank" "bajaj finance"
```

---

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--variant` | `small` | Kronos model: `mini` (4.1M) · `small` (24.7M) · `base` (102.3M) |
| `--cap` | `all` | Cap tier: `large` · `mid` · `small` · `all` |
| `--top` | `10` | Top N gainers + losers per tier |
| `--interval` | `1h` | Candle size: `1h` · `15m` · `5m` · `1m` |
| `--days` | `3` | Trading days to predict (1 day = 7/25/75 candles by interval) |
| `--samples` | `20` | Kronos ensemble samples — more = stable but slower |
| `--symbols` | — | Skip scanner, predict specific symbols or company names |
| `--save` | off | Save actionable signals to `outputs/` CSV |
| `--track` | off | Log signals to tracker DB for outcome evaluation |

> See [USAGE.md](USAGE.md) for full documentation with examples.

---

## Candle Intervals

| Interval | Candles/day | Context window | History | Best for |
| --- | --- | --- | --- | --- |
| `1h` | 7 | ~73 trading days | 730 days | Swing / multi-day |
| `15m` | 25 | ~20 trading days | 60 days | Intraday (recommended) |
| `5m` | 75 | ~7 trading days | 60 days | Detailed intraday |
| `1m` | 375 | ~1.5 trading days | 7 days | Scalping (limited) |

---

## Sample Output

```text
================================================================
  NSE INTRADAY SIGNAL PIPELINE — Powered by Kronos
  Model: Kronos-small | Interval: 15m | Predicting 1d (25 candles)
================================================================

[1/5] Scanning NSE — top 10 gainers & losers: LARGE, MID, SMALL
  TOP GAINERS :: Large Cap (Nifty 100)
      symbol      ltp  change_pct  change
       TECHM  1543.20        4.00   59.30
        INFY  1202.50        3.58   41.60

[3/6] Analysing weekly & monthly trends...
  TECHM    Monthly:BULLISH (+6.6%)  Weekly:BULLISH (+7.7%)
           RSI:77.28  ADX:14.30  SMA20=above SMA50=above
           RVOL=1.77x[SPIKE] OBV=RISING  => STRONGLY BULLISH [score +7]

[4/6] Analysing news sentiment...
  TECHM    Sentiment: BULLISH (0.79, 6 headlines)

[5/6] Running Kronos predictions (samples=20)...
  Predicting: TECHM...

======================================================================
  ACTIONABLE TRADE SIGNALS (Next 1 Trading Day)
  Target: Kronos range (min 1.5%)  |  Max SL 2.5%  |  Min R:R 2.0:1
======================================================================

  [BUY  ^] TECHM  [HIGH confidence]  Confluence: STRONG ***
    Entry:     1543.20
    Target:    1638.88  (+6.2%)
    Stop Loss: 1504.62  (-2.5%)
    R:R Ratio: 2.48:1
    Trend:     STRONGLY BULLISH  (score +7/8)
    Sentiment: BULLISH (0.79, 6 headlines)
    * Kronos predicts upside 6.2% (target 6.2% / stop 2.5%)
    * Monthly: BULLISH (+6.6%) | Weekly: BULLISH (+7.7%)
    * RSI 77.28 | ADX 14.3 | Score +7/8
    * RVOL 1.77x [VOLUME SPIKE — confirms move] | OBV RISING
```

> **Why the target isn't a fixed percentage:** Kronos forecasts a full price
> range, so the target is its predicted high (long) or low (short) and the stop
> is its predicted opposite extreme, clamped to a 1.0–2.5% risk band. A fixed
> 5–7% target both missed real 2–4% large-cap moves (filtered out as NO TRADE)
> and clipped the occasional 8–10% move. Letting Kronos's own range drive the
> levels also makes the 2:1 reward:risk filter meaningful. See
> [USAGE.md](USAGE.md#how-targets-and-stops-are-set-and-why) for the full rationale.

---

## Trend Scoring

Each stock gets a score from **-8 to +8** based on:

| Factor | Bullish | Bearish |
| --- | --- | --- |
| Monthly momentum (>2%) | +1 | -1 |
| Weekly momentum (>1%) | +1 | -1 |
| Price above SMA 20 | +1 | -1 |
| Price above SMA 50 | +1 | -1 |
| RSI > 55 / < 45 | +1 | -1 |
| Volume spike in trend direction | +1 | -1 |
| OBV rising / falling | +1 | -1 |
| ADX > 25 (trend strength bonus) | +1 | -1 |

### Confluence levels

| Level | Score | Meaning |
| --- | --- | --- |
| `STRONG ***` | ≥ +3 / ≤ -3 | Trend strongly agrees with Kronos |
| `MODERATE **` | +1 or +2 / -1 or -2 | Trend leans same way |
| `WEAK *` | 0 | Trend is neutral — take smaller position |
| `AGAINST TREND` | Opposite sign to signal | Trade blocked |

---

## Prediction Tracker

Every signal logged with `--track` is evaluated after the prediction window closes. The tracker walks through candles in order to determine which level — target or stop-loss — was hit first, making evaluation more realistic than simply checking the end-of-period close.

- The candle **interval** used for a run is stored alongside each signal and used when fetching actual price data during evaluation — a 15m signal is evaluated on 15m candles, not 1h
- **eval_by** date is calculated using NSE holiday-aware business days — weekends and official NSE holidays are both skipped
- Evaluation uses the **actual next-day open price** as the realistic entry point, not the signal's logged close price — targets and stop-losses are recalculated from there
- The EXPIRED exit price is the last close of the eval_by date specifically, not any earlier candle

```bash
# Log signals after each run
python main.py --track

# Evaluate outcomes after prediction window closes
python tracker.py evaluate

# View full performance report
python tracker.py report

# List all logged signals
python tracker.py show

# Force-evaluate overdue signals skipped due to data gaps
python tracker.py evaluate --force

# Import a saved CSV into the tracker (for runs done without --track)
python tracker.py import --csv outputs/signals_YYYYMMDD_HHMM_<variant>.csv
```

### Report example

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

---

## Project Structure

```text
kronos-india/
├── main.py              # CLI entry point — orchestrates the full pipeline
├── app.py               # Streamlit web dashboard
├── tracker.py           # Signal logger + outcome evaluator + report
├── requirements.txt     # Python dependencies
├── USAGE.md             # Full argument reference
├── pipeline/            # Internal pipeline modules
│   ├── market_scanner.py    # NSE gainers/losers by cap tier
│   ├── data_fetcher.py      # Historical OHLCV via yfinance (multi-interval)
│   ├── trend_analyzer.py    # RSI, ADX, SMA, RVOL, OBV, momentum scoring
│   ├── sentiment_analyzer.py# FinBERT news sentiment (Google News RSS)
│   ├── predictor.py         # Kronos model wrapper (GPU-accelerated)
│   ├── signal_generator.py  # LONG/SHORT signal with entry/target/SL
│   ├── symbol_resolver.py   # Company name -> NSE symbol lookup (fuzzy search)
│   └── symbol_names.csv     # 2,456 NSE equities: symbol + company name
├── tests/
│   └── test_tracker.py  # Unit tests for outcome evaluation and business day logic
├── outputs/
│   ├── signals_*.csv    # Saved signal CSVs (--save)
│   └── tracker.db       # SQLite prediction log (--track)
└── ../Kronos/           # Kronos repo (cloned separately)
```

---

## Kronos Model Variants

| Model | Parameters | Context | Speed (RTX 5070) |
| --- | --- | --- | --- |
| `Kronos-mini` | 4.1M | 2048 candles | ~1-2 sec/stock |
| `Kronos-small` | 24.7M | 512 candles | ~2-5 sec/stock |
| `Kronos-base` | 102.3M | 512 candles | ~8-12 sec/stock |

Models are downloaded automatically from HuggingFace Hub on first run (~500MB for small) and cached locally. All subsequent runs are fully offline.

---

## Testing

Unit tests cover the tracker's outcome evaluation logic and NSE holiday-aware business day calculation — the two functions that directly affect whether performance claims can be trusted.

```bash
python -m pytest tests/
```

---

## Roadmap

- [x] NSE gainers/losers by market cap tier (Large/Mid/Small)
- [x] Multi-interval candles (1h / 15m / 5m / 1m)
- [x] Trend confluence scoring (RSI, ADX, SMA, RVOL, OBV)
- [x] GPU-accelerated Kronos inference (CUDA)
- [x] Prediction tracker with WIN/LOSS evaluation
- [x] Tracker: interval-aware evaluation, NSE holiday scheduling, actual-entry P&L, streak tracking
- [x] Web dashboard (Streamlit) with live progress, styled tables, tracker controls
- [x] FinBERT news sentiment analysis (Google News RSS, no API key required)
- [ ] Fine-tuning Kronos on Indian market data
- [ ] Live candle loop (re-predict every candle during market hours)
- [ ] Telegram / email alerts for actionable signals

---

## Acknowledgements

Built on [Kronos](https://github.com/shiyu-coder/Kronos) by Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang, and Jian Li — accepted at AAAI 2026.

```bibtex
@misc{shi2025kronos,
  title={Kronos: A Foundation Model for the Language of Financial Markets},
  author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
  year={2025},
  eprint={2508.02739},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST},
  url={https://arxiv.org/abs/2508.02739}
}
```