"""
Kronos India — Streamlit Dashboard
Run with: streamlit run app.py
"""

import os
import sys
import subprocess
from datetime import datetime

os.environ["TQDM_DISABLE"] = "1"          # suppress Kronos tqdm before any imports

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import pandas as pd
import streamlit as st

from pipeline.data_fetcher     import fetch_ohlcv, prepare_context_and_forecast_timestamps, get_current_price, CANDLES_PER_DAY
from pipeline.trend_analyzer   import analyze as analyze_trend
from pipeline.signal_generator import generate_signal
from pipeline.predictor        import predict_next_day
from pipeline.sentiment_analyzer import analyze_batch as analyze_sentiments

CONTEXT_LEN = {"mini": 2048, "small": 512, "base": 512}


@st.cache_data
def _symbol_options() -> list[str]:
    """Loads all NSE equities as 'SYMBOL — Company Name' strings for multiselect."""
    import csv
    path = os.path.join(APP_DIR, "pipeline", "symbol_names.csv")
    opts = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            opts.append(f"{row['symbol'].strip()} — {row['name'].strip()}")
    return sorted(opts)

st.set_page_config(page_title="Kronos India", page_icon="📈", layout="wide")
st.title("📈 Kronos India — NSE Intraday Signal Pipeline")
st.caption("Powered by Kronos Foundation Model · NSE/BSE")
st.divider()


# ── Prediction state machine — persists across reruns ────────────────────────
# Processing one stock per Streamlit run lets the Stop button be responsive.
_PRED_DEFAULTS = {
    "pred_mode":         False,
    "pred_all":          [],   # full stocks list — never mutated during run
    "pred_idx":          0,    # index of next stock to predict
    "pred_done":         {},   # {sym: pred_df} results so far
    "pred_stopped":      False,
    "pred_trends":       {},
    "pred_trend_rows":   [],
    "pred_scan_results":  {},
    "pred_interval":      "1h",
    "pred_days":          3,
    "pred_samples":       20,
    "pred_save":          False,
    "pred_track":         False,
    "pred_variant":       "small",
    "pred_sentiments":    {},
    "pred_no_sentiment":  False,
}
for _k, _v in _PRED_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

run_clicked = False   # defined here so the output panel can always reference it


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(variant: str) -> None:
    from pipeline.predictor import load_model
    load_model(variant)


# ── Table stylers ─────────────────────────────────────────────────────────────

def _style_scanner(df: pd.DataFrame, gainers: bool):
    color   = "#198754" if gainers else "#dc3545"
    display = df[["symbol", "ltp", "change_pct", "change"]].copy()
    display.columns = ["Symbol", "LTP", "Change %", "Change ₹"]
    def _col(s):
        return [f"color: {color}; font-weight: bold"] * len(s)
    return (display.style
                   .apply(_col, subset=["Change %", "Change ₹"])
                   .format({"LTP": "{:.2f}", "Change %": "{:.2f}", "Change ₹": "{:.2f}"}))


def _style_trend(df: pd.DataFrame):
    def _score(s):
        return s.map(lambda v:
            "color: #198754; font-weight: bold" if v > 0 else
            ("color: #dc3545; font-weight: bold" if v < 0 else "color: gray"))
    def _bias(s):
        return s.map(lambda v:
            "color: #198754" if "BULL" in str(v) else
            ("color: #dc3545" if "BEAR" in str(v) else "color: gray"))
    return (df.style
              .apply(_score, subset=["Score"])
              .apply(_bias,  subset=["Trend", "Monthly %", "Weekly %"])
              .format({"RSI": "{:.2f}", "ADX": "{:.2f}"}))


def _style_signals(df: pd.DataFrame):
    def _row(row):
        if row["Direction"] == "LONG":
            return ["color: #198754; font-weight: bold"] * len(row)
        if row["Direction"] == "SHORT":
            return ["color: #dc3545; font-weight: bold"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row, axis=1).format({"Entry ₹": "{:.2f}"})


# ── Signal display helper ─────────────────────────────────────────────────────

def _show_signals(predictions, trends, scan_results, stocks, save, track, days, interval, variant, sentiments=None):
    tier_map = {}
    for tier, (g, l) in scan_results.items():
        for sym in pd.concat([g, l])["symbol"].tolist():
            tier_map.setdefault(sym, tier)

    signals = []
    for sym, *_ in stocks:
        pred_df = predictions.get(sym)
        if pred_df is None:
            continue
        sig          = generate_signal(sym, pred_df, get_current_price(sym), trends.get(sym))
        sig.cap_tier = tier_map.get(sym, "unknown")
        if sentiments and sym in sentiments:
            s = sentiments[sym]
            sig.sentiment       = s.label
            sig.sentiment_score = s.score
            sig.sentiment_count = s.count
        signals.append(sig)

    actionable = [s for s in signals if s.direction != "NO TRADE"]

    if actionable:
        rows = []
        for s in actionable:
            tgt_pct = abs((s.target    - s.entry) / s.entry * 100)
            sl_pct  = abs((s.stop_loss - s.entry) / s.entry * 100)
            sign    = "+" if s.direction == "LONG" else "-"
            rows.append({
                "Symbol":     s.symbol,
                "Cap Tier":   getattr(s, "cap_tier", ""),
                "Direction":  s.direction,
                "Entry ₹":    s.entry,
                "Target":     f"{s.target}  ({sign}{tgt_pct:.1f}%)",
                "Stop Loss":  f"{s.stop_loss}  (-{sl_pct:.1f}%)",
                "R:R":        f"{s.rr_ratio}:1",
                "Confidence": s.confidence,
                "Confluence": s.confluence,
                "Trend":      s.trend_bias,
                "Score /8":   s.trend_score,
                "Sentiment":  f"{s.sentiment} ({s.sentiment_score:.2f}, {s.sentiment_count}h)" if s.sentiment_count > 0 else s.sentiment,
            })
        with st.expander(f"🎯 Trade Signals — {len(actionable)} actionable", expanded=True):
            st.dataframe(_style_signals(pd.DataFrame(rows)),
                         use_container_width=True, hide_index=True)
    else:
        st.info("No high-conviction trades found in this scan.")

    if save and actionable:
        df_save = pd.DataFrame([{
            "Cap Tier": getattr(s, "cap_tier", ""), "Symbol": s.symbol,
            "Direction": s.direction, "Entry": s.entry, "Target": s.target,
            "Stop Loss": s.stop_loss, "R:R": s.rr_ratio,
            "Confidence": s.confidence, "Confluence": s.confluence,
            "Trend": s.trend_bias, "Trend Score": s.trend_score,
        } for s in actionable])
        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        path = os.path.join(APP_DIR, "outputs", f"signals_{ts}_{variant}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_save.to_csv(path, index=False)
        st.success(f"Saved → {path}")

    if track and signals:
        from tracker import log_signals
        n = log_signals(signals, pred_days=int(days), interval=interval, trends=trends)
        st.success(f"Logged {n} signal(s) to tracker.db")


# ── Settings panel ────────────────────────────────────────────────────────────

left, right = st.columns([1, 2])

with left:
    st.subheader("Settings")
    variant  = st.selectbox("Model Variant",   ["small", "mini", "base"],        index=0,
                            help="small = balanced · mini = fastest · base = most accurate")
    cap      = st.selectbox("Cap Tier",        ["all", "large", "mid", "small"], index=0,
                            help="Nifty universe to scan")
    interval = st.selectbox("Candle Interval", ["1h", "15m", "5m", "1m"],        index=0,
                            help="1h swing · 15m intraday · 5m detailed")
    top     = st.number_input("Top N per Tier",   min_value=1, max_value=50,  value=10, step=5)
    days    = st.number_input("Predict Days",     min_value=1, max_value=10,  value=3,  step=1)
    samples = st.number_input("Ensemble Samples", min_value=5, max_value=200, value=20, step=5)
    st.markdown("---")
    symbols_selected = st.multiselect(
        "Symbols Override (optional)",
        options=_symbol_options(),
        placeholder="Type symbol or company name to search...",
        help="Leave empty to run the full NSE scanner. Select one or more stocks to override.",
    )
    c1, c2, c3 = st.columns(3)
    with c1: save         = st.checkbox("Save CSV")
    with c2: track        = st.checkbox("Track")
    with c3: no_sentiment = st.checkbox("Skip Sentiment")
    st.markdown("")

    # During predictions show Stop; otherwise show Run
    if st.session_state.pred_mode:
        if st.button("⏹  Stop Pipeline", type="secondary", use_container_width=True):
            st.session_state.pred_stopped = True
            st.rerun()
    else:
        run_clicked = st.button("▶  Run Pipeline", type="primary", use_container_width=True)


# ── Pipeline output ───────────────────────────────────────────────────────────

with right:
    st.subheader("Output")
    main_prog = st.progress(0, text="Ready — configure settings and press Run")

    # ── Prediction mode ───────────────────────────────────────────────────────
    # Each rerun processes exactly one stock then calls st.rerun().
    # This makes the Stop button responsive between stocks.
    if st.session_state.pred_mode:
        ss        = st.session_state
        total     = len(ss.pred_all)
        completed = ss.pred_idx

        # Re-render scanner tables on every prediction rerun so they stay visible
        if ss.pred_scan_results:
            for tier, (gainers_df, losers_df) in ss.pred_scan_results.items():
                with st.expander(f"📊 {tier.upper()} CAP", expanded=True):
                    gc, lc = st.columns(2)
                    with gc:
                        st.markdown("**Gainers**")
                        st.dataframe(_style_scanner(gainers_df, gainers=True),
                                     use_container_width=True, hide_index=True)
                    with lc:
                        st.markdown("**Losers**")
                        st.dataframe(_style_scanner(losers_df, gainers=False),
                                     use_container_width=True, hide_index=True)
        elif ss.pred_all:
            symbols_list = [sym for sym, *_ in ss.pred_all]
            st.info(f"Symbols override: {', '.join(symbols_list)}")

        if ss.pred_trend_rows:
            with st.expander("📈 Trend Analysis", expanded=False):
                st.dataframe(_style_trend(pd.DataFrame(ss.pred_trend_rows)),
                             use_container_width=True, hide_index=True)

        main_prog.progress(
            0.60 + 0.25 * (completed / max(total, 1)),
            text=f"[5/6] Kronos predictions  ({completed} / {total})"
        )
        pred_prog = st.progress(
            completed / max(total, 1),
            text=f"Predicting {completed + 1} / {total}" if completed < total else "✓  Complete"
        )

        all_done = completed >= total
        stopped  = ss.pred_stopped

        if all_done or stopped:
            label = "⏹ Stopped" if stopped else "✅ Pipeline complete"
            main_prog.progress(1.0 if all_done else (0.60 + 0.25 * completed / max(total, 1)),
                               text=label)

            if ss.pred_done:
                main_prog.progress(0.90, text="[6/6] Generating signals...")
                _show_signals(
                    predictions  = ss.pred_done,
                    trends       = ss.pred_trends,
                    scan_results = ss.pred_scan_results,
                    stocks       = ss.pred_all,
                    save         = ss.pred_save,
                    track        = ss.pred_track,
                    days         = ss.pred_days,
                    interval     = ss.pred_interval,
                    variant      = ss.pred_variant,
                    sentiments   = ss.pred_sentiments,
                )
                main_prog.progress(1.0, text=label)
            else:
                st.warning("No predictions completed — nothing to show.")

            # Reset state (takes effect on next user interaction)
            for _k, _v in _PRED_DEFAULTS.items():
                st.session_state[_k] = _v

        else:
            # Process the next stock, then rerun
            sym, x_df, x_ts, y_ts = ss.pred_all[ss.pred_idx]
            pred_prog.progress(
                completed / max(total, 1),
                text=f"Predicting {sym}  ({completed + 1} / {total})"
            )
            with st.spinner(f"Running Kronos on {sym}  ({ss.pred_samples} samples)..."):
                result = predict_next_day(sym, x_df, x_ts, y_ts,
                                          sample_count=ss.pred_samples)
            st.session_state.pred_done[sym] = result
            st.session_state.pred_idx      += 1
            st.rerun()

    # ── Initial run — steps 1–3, then hand off to prediction mode ─────────────
    elif run_clicked:
        context_len = CONTEXT_LEN[variant]
        pred_len    = int(days) * CANDLES_PER_DAY[interval]
        tiers       = ["large", "mid", "small"] if cap == "all" else [cap]

        # Load model
        main_prog.progress(0.05, text="Loading Kronos model...")
        with st.spinner(f"Loading Kronos-{variant} model..."):
            _load_model(variant)

        # Step 1: Scanner
        main_prog.progress(0.10, text="[1/6] Scanning NSE...")
        scan_results, symbols = {}, []
        if symbols_selected:
            symbols = [s.split(" — ")[0] for s in symbols_selected]
            st.info(f"Symbols override: {', '.join(symbols)}")
        else:
            from pipeline.market_scanner import get_top_gainers_losers
            scan_results = get_top_gainers_losers(top_n=int(top), tiers=tiers)
            for tier, (gainers_df, losers_df) in scan_results.items():
                with st.expander(f"📊 {tier.upper()} CAP", expanded=True):
                    gc, lc = st.columns(2)
                    with gc:
                        st.markdown("**Gainers**")
                        st.dataframe(_style_scanner(gainers_df, gainers=True),
                                     use_container_width=True, hide_index=True)
                    with lc:
                        st.markdown("**Losers**")
                        st.dataframe(_style_scanner(losers_df, gainers=False),
                                     use_container_width=True, hide_index=True)
                symbols += gainers_df["symbol"].tolist() + losers_df["symbol"].tolist()
            symbols = list(dict.fromkeys(symbols))

        if not symbols:
            st.error("No symbols found.")
            st.stop()

        # Step 2: Fetch
        main_prog.progress(0.22, text="[2/6] Fetching OHLCV data...")
        fetch_prog = st.progress(0.0, text=f"Fetching  0 / {len(symbols)}")
        stocks = []
        for i, sym in enumerate(symbols):
            fetch_prog.progress((i + 1) / len(symbols),
                                text=f"Fetching {sym}  ({i + 1} / {len(symbols)})")
            df = fetch_ohlcv(sym, interval=interval)
            if df is None:
                continue
            x_df, x_ts, y_ts = prepare_context_and_forecast_timestamps(
                df, pred_len=pred_len, context_len=context_len, interval=interval
            )
            stocks.append((sym, x_df, x_ts, y_ts))
        fetch_prog.progress(1.0, text=f"✓  {len(stocks)} / {len(symbols)} stocks fetched")
        if not stocks:
            st.error("No valid OHLCV data fetched.")
            st.stop()

        # Step 3: Trends
        main_prog.progress(0.38, text="[3/6] Analysing trends...")
        trends, trend_rows = {}, []
        for sym, *_ in stocks:
            t = analyze_trend(sym)
            trends[sym] = t
            if t:
                trend_rows.append({
                    "Symbol":    t.symbol,
                    "Trend":     t.overall_bias,
                    "Score":     t.score,
                    "Monthly %": f"{t.monthly_chg_pct:+.1f}%",
                    "Weekly %":  f"{t.weekly_chg_pct:+.1f}%",
                    "RSI":       t.rsi14,
                    "ADX":       t.adx14,
                    "RVOL":      f"{t.rvol}x{'↑' if t.volume_spike else ''}",
                    "OBV":       t.obv_trend,
                })
        if trend_rows:
            with st.expander("📈 Trend Analysis", expanded=False):
                st.dataframe(_style_trend(pd.DataFrame(trend_rows)),
                             use_container_width=True, hide_index=True)

        # Step 4: Sentiment
        sentiments = {}
        if not no_sentiment:
            main_prog.progress(0.52, text="[4/6] Analysing news sentiment...")
            sent_prog = st.progress(0.0, text=f"Sentiment  0 / {len(stocks)}")
            sym_list  = [s[0] for s in stocks]
            for i, sym in enumerate(sym_list):
                sent_prog.progress((i + 1) / len(sym_list),
                                   text=f"Sentiment {sym}  ({i + 1} / {len(sym_list)})")
                try:
                    from pipeline.sentiment_analyzer import analyze as _analyze_sent
                    import time as _time
                    sentiments[sym] = _analyze_sent(sym)
                    _time.sleep(0.3)
                except Exception:
                    pass
            sent_prog.progress(1.0, text=f"✓  Sentiment done ({len(sentiments)} stocks)")

        # Hand off to prediction mode — one stock per rerun from here
        main_prog.progress(0.60, text="[5/6] Starting Kronos predictions...")
        st.session_state.pred_mode         = True
        st.session_state.pred_all          = stocks
        st.session_state.pred_idx          = 0
        st.session_state.pred_done         = {}
        st.session_state.pred_stopped      = False
        st.session_state.pred_trends       = trends
        st.session_state.pred_trend_rows   = trend_rows
        st.session_state.pred_scan_results = scan_results
        st.session_state.pred_interval     = interval
        st.session_state.pred_days         = int(days)
        st.session_state.pred_samples      = int(samples)
        st.session_state.pred_save         = save
        st.session_state.pred_track        = track
        st.session_state.pred_variant      = variant
        st.session_state.pred_sentiments   = sentiments
        st.session_state.pred_no_sentiment = no_sentiment
        st.rerun()


# ── Prediction Tracker ────────────────────────────────────────────────────────

st.divider()
st.subheader("Prediction Tracker")

import sqlite3

_DB_PATH = os.path.join(APP_DIR, "outputs", "tracker.db")

st.session_state.setdefault("tracker_view",  None)
st.session_state.setdefault("tracker_eval",  [])   # [(kind, text), ...]


def _tracker_load() -> pd.DataFrame | None:
    if not os.path.exists(_DB_PATH):
        return None
    try:
        with sqlite3.connect(_DB_PATH) as con:
            return pd.read_sql("SELECT * FROM signals ORDER BY id DESC", con)
    except Exception:
        return None


def _run_evaluate(force: bool = False):
    cmd = [sys.executable, os.path.join(APP_DIR, "tracker.py"), "evaluate"]
    if force:
        cmd.append("--force")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=APP_DIR)
    lines = []
    for line in (result.stdout + (result.stderr or "")).splitlines():
        t = line.strip()
        if not t:
            continue
        if "[WIN" in t:
            lines.append(("success", t))
        elif "[LOSS" in t:
            lines.append(("error", t))
        elif "[EXP" in t:
            lines.append(("warning", t))
        elif "complete" in t.lower() or "logged" in t.lower():
            lines.append(("success", t))
        elif "skipping" in t.lower() or "no signals" in t.lower():
            lines.append(("info", t))
        else:
            lines.append(("caption", t))
    st.session_state.tracker_eval = lines
    st.session_state.tracker_view = "evaluate"


def _render_evaluate():
    for kind, text in st.session_state.tracker_eval:
        if   kind == "success": st.success(text)
        elif kind == "error":   st.error(text)
        elif kind == "warning": st.warning(text)
        elif kind == "info":    st.info(text)
        else:                   st.caption(text)


def _render_report():
    df = _tracker_load()
    if df is None:
        st.info("No tracker database found — run the pipeline with **Track** enabled.")
        return
    if df.empty:
        st.info("No signals logged yet.")
        return

    evaluated = df[df["outcome"].isin(["WIN", "LOSS", "EXPIRED"])].copy()
    if evaluated.empty:
        pending = df[df["outcome"].isna()]
        st.info(f"No evaluated signals yet — {len(pending)} pending. "
                "Click **Evaluate** after the prediction window closes.")
        return

    wins    = evaluated[evaluated["outcome"] == "WIN"]
    losses  = evaluated[evaluated["outcome"] == "LOSS"]
    expired = evaluated[evaluated["outcome"] == "EXPIRED"]
    total   = len(evaluated)
    wr      = len(wins) / total * 100
    avg_win  = wins["pnl_pct"].mean()   if not wins.empty   else 0.0
    avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0
    avg_pnl  = evaluated["pnl_pct"].mean()
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)

    # Summary metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total",      total)
    c2.metric("Win Rate",   f"{wr:.1f}%")
    c3.metric("Wins",       len(wins))
    c4.metric("Losses",     len(losses))
    c5.metric("Expired",    len(expired))
    c6.metric("Expectancy", f"{expectancy:+.2f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("Avg Win P&L",  f"{avg_win:+.2f}%")
    c8.metric("Avg Loss P&L", f"{avg_loss:+.2f}%")
    c9.metric("Avg All P&L",  f"{avg_pnl:+.2f}%")

    st.markdown("---")

    def _breakdown(col, values):
        if col not in evaluated.columns:
            return None
        rows = []
        for v in values:
            sub = evaluated[evaluated[col] == v]
            if sub.empty:
                continue
            w = len(sub[sub["outcome"] == "WIN"])
            rows.append({"": v, "Trades": len(sub),
                         "Win %": f"{w / len(sub) * 100:.0f}%",
                         "Avg P&L": f"{sub['pnl_pct'].mean():+.2f}%"})
        return pd.DataFrame(rows) if rows else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**By Confluence**")
        d = _breakdown("confluence", ["STRONG", "MODERATE", "WEAK"])
        if d is not None: st.dataframe(d, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**By Confidence**")
        d = _breakdown("confidence", ["HIGH", "MEDIUM", "LOW"])
        if d is not None: st.dataframe(d, hide_index=True, use_container_width=True)
    with col3:
        st.markdown("**By Sentiment**")
        d = _breakdown("sentiment", ["BULLISH", "NEUTRAL", "BEARISH"])
        if d is not None:
            st.dataframe(d, hide_index=True, use_container_width=True)
        else:
            st.caption("No sentiment data yet")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("**By Cap Tier**")
        d = _breakdown("cap_tier", ["large", "mid", "small"])
        if d is not None: st.dataframe(d, hide_index=True, use_container_width=True)
    with col5:
        st.markdown("**By Direction**")
        d = _breakdown("direction", ["LONG", "SHORT"])
        if d is not None: st.dataframe(d, hide_index=True, use_container_width=True)
    with col6:
        st.markdown("**By Interval**")
        d = _breakdown("interval", ["1h", "15m", "5m", "1m"])
        if d is not None: st.dataframe(d, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Best / worst trades
    vcols = [c for c in ["symbol", "direction", "entry", "pnl_pct",
                          "confluence", "logged_at"] if c in evaluated.columns]
    cb, cw = st.columns(2)
    with cb:
        st.markdown("**Best Trades**")
        st.dataframe(evaluated.nlargest(3, "pnl_pct")[vcols],
                     hide_index=True, use_container_width=True)
    with cw:
        st.markdown("**Worst Trades**")
        st.dataframe(evaluated.nsmallest(3, "pnl_pct")[vcols],
                     hide_index=True, use_container_width=True)

    # Streaks
    decisive = evaluated[evaluated["outcome"].isin(["WIN", "LOSS"])].sort_values("logged_at")
    if not decisive.empty:
        outcomes = decisive["outcome"].tolist()
        cur_val, cur_count = outcomes[-1], 0
        for o in reversed(outcomes):
            if o == cur_val: cur_count += 1
            else: break

        best_win = worst_loss = run = 1
        for i in range(1, len(outcomes)):
            run = run + 1 if outcomes[i] == outcomes[i - 1] else 1
            if outcomes[i] == "WIN":  best_win   = max(best_win,   run)
            if outcomes[i] == "LOSS": worst_loss = max(worst_loss, run)

        st.markdown("---")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Current Streak",  f"{cur_count} × {cur_val}")
        sc2.metric("Best Win Streak", f"{best_win} in a row")
        sc3.metric("Worst Loss Run",  f"{worst_loss} in a row")


def _render_show():
    df = _tracker_load()
    if df is None or df.empty:
        st.info("No signals logged yet.")
        return

    pending   = df[df["outcome"].isna()].copy()
    evaluated = df[df["outcome"].notna()].copy()

    tab_p, tab_e = st.tabs(
        [f"Pending  ({len(pending)})", f"Evaluated  ({len(evaluated)})"]
    )

    with tab_p:
        if pending.empty:
            st.info("No pending signals.")
        else:
            pcols = [c for c in ["logged_at", "symbol", "cap_tier", "direction",
                                  "entry", "target", "rr_ratio", "confidence",
                                  "confluence", "eval_by"] if c in pending.columns]
            st.dataframe(pending[pcols], hide_index=True, use_container_width=True)

    with tab_e:
        if evaluated.empty:
            st.info("No evaluated signals yet.")
        else:
            ecols = [c for c in ["logged_at", "symbol", "cap_tier", "direction",
                                  "entry", "pnl_pct", "outcome",
                                  "confidence", "confluence", "eval_by"]
                     if c in evaluated.columns]

            def _color_row(row):
                o = row.get("outcome", "")
                if o == "WIN":     return ["background-color: #1a3a1a"] * len(row)
                if o == "LOSS":    return ["background-color: #3a1a1a"] * len(row)
                if o == "EXPIRED": return ["background-color: #2a2a1a"] * len(row)
                return [""] * len(row)

            st.dataframe(
                evaluated[ecols].style.apply(_color_row, axis=1),
                hide_index=True, use_container_width=True,
            )


# Buttons
b1, b2, b3, b4, _ = st.columns([1, 1.5, 1, 1.3, 3])
with b1:
    if st.button("Evaluate", use_container_width=True):
        _run_evaluate(force=False)
with b2:
    if st.button("Evaluate --force", use_container_width=True):
        _run_evaluate(force=True)
with b3:
    if st.button("Report", use_container_width=True):
        st.session_state.tracker_view = "report"
with b4:
    if st.button("Show Signals", use_container_width=True):
        st.session_state.tracker_view = "show"

# Render active view
_view = st.session_state.tracker_view
if   _view == "evaluate": _render_evaluate()
elif _view == "report":   _render_report()
elif _view == "show":     _render_show()
