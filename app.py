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

from data_fetcher     import fetch_ohlcv, prepare_context_and_forecast_timestamps, get_current_price, CANDLES_PER_DAY
from trend_analyzer   import analyze as analyze_trend
from signal_generator import generate_signal
from predictor        import predict_next_day

CONTEXT_LEN = {"mini": 2048, "small": 512, "base": 512}

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
    "pred_scan_results": {},
    "pred_interval":     "1h",
    "pred_days":         3,
    "pred_samples":      20,
    "pred_save":         False,
    "pred_track":        False,
    "pred_variant":      "small",
}
for _k, _v in _PRED_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

run_clicked = False   # defined here so the output panel can always reference it


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(variant: str) -> None:
    from predictor import load_model
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

def _show_signals(predictions, trends, scan_results, stocks, save, track, days, interval, variant):
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
        for sig in signals:
            sig._trend = trends.get(sig.symbol)
        n = log_signals(signals, pred_days=int(days), interval=interval)
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
    symbols_input = st.text_input("Symbols Override (optional)", placeholder="RELIANCE TCS INFY")
    c1, c2 = st.columns(2)
    with c1: save  = st.checkbox("Save CSV")
    with c2: track = st.checkbox("Track")
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

        main_prog.progress(
            0.60 + 0.25 * (completed / max(total, 1)),
            text=f"[4/5] Kronos predictions  ({completed} / {total})"
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

            if ss.pred_trend_rows:
                with st.expander("📈 Trend Analysis", expanded=False):
                    st.dataframe(_style_trend(pd.DataFrame(ss.pred_trend_rows)),
                                 use_container_width=True, hide_index=True)

            if ss.pred_done:
                main_prog.progress(0.90, text="[5/5] Generating signals...")
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
        main_prog.progress(0.10, text="[1/5] Scanning NSE...")
        scan_results, symbols = {}, []
        if symbols_input.strip():
            symbols = [s.upper() for s in symbols_input.strip().split()]
            st.info(f"Symbols override: {', '.join(symbols)}")
        else:
            from market_scanner import get_top_gainers_losers
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
        main_prog.progress(0.25, text="[2/5] Fetching OHLCV data...")
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
        main_prog.progress(0.45, text="[3/5] Analysing trends...")
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

        # Hand off to prediction mode — one stock per rerun from here
        main_prog.progress(0.60, text="[4/5] Starting Kronos predictions...")
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
        st.rerun()


# ── Prediction Tracker ────────────────────────────────────────────────────────

st.divider()
st.subheader("Prediction Tracker")

b1, b2, b3, b4, _ = st.columns([1, 1.5, 1, 1.3, 3])
tracker_out = st.empty()


def _run_tracker(command: str, extra: list = None):
    cmd = [sys.executable, os.path.join(APP_DIR, "tracker.py"), command]
    if extra:
        cmd += extra
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=APP_DIR)
    tracker_out.code(result.stdout + (result.stderr or "") or "(no output)", language=None)


with b1:
    if st.button("Evaluate", use_container_width=True):
        _run_tracker("evaluate")
with b2:
    if st.button("Evaluate --force", use_container_width=True):
        _run_tracker("evaluate", ["--force"])
with b3:
    if st.button("Report", use_container_width=True):
        _run_tracker("report")
with b4:
    if st.button("Show Signals", use_container_width=True):
        _run_tracker("show")
