"""
Prediction Tracker — logs every signal and evaluates actual outcomes.

Commands:
    python tracker.py evaluate  -- check outcomes for matured signals
    python tracker.py report    -- show performance statistics
    python tracker.py show      -- list all logged signals

Storage: SQLite at outputs/tracker.db (no extra dependencies needed)
"""

import os
import sys
import sqlite3
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "outputs", "tracker.db")

# NSE equity-segment trading holidays for 2025–2026.
# Source: https://www.nseindia.com/resources/exchange-communication-holidays
# Update this set each year by checking the official NSE circular.
NSE_HOLIDAYS = {
    # ── 2025 (14 holidays, all confirmed from NSE circulars) ──────────────────
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr (Eid)
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti / Dussehra
    "2025-10-21",  # Diwali — Laxmi Puja
    "2025-10-22",  # Diwali — Balipratipada
    "2025-11-05",  # Gurunanak Jayanti
    "2025-12-25",  # Christmas
    # ── 2026 (confirmed from NSE/broker sources where available) ─────────────
    "2026-01-26",  # Republic Day
    "2026-02-17",  # Mahashivratri
    "2026-03-04",  # Holi
    "2026-03-20",  # Id-Ul-Fitr (Eid) — verify against moon sighting
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id (Eid-ul-Adha)
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-11-24",  # Gurunanak Jayanti
    "2026-12-25",  # Christmas
}


# ─── Database setup ───────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at     TEXT NOT NULL,
            eval_by       TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            cap_tier      TEXT,
            direction     TEXT NOT NULL,
            entry         REAL,
            target        REAL,
            stop_loss     REAL,
            target_pct    REAL,
            stoploss_pct  REAL,
            rr_ratio      REAL,
            pred_days     INTEGER,
            interval      TEXT,
            confidence    TEXT,
            confluence    TEXT,
            trend_bias    TEXT,
            trend_score   INTEGER,
            rvol          REAL,
            volume_spike  INTEGER,
            obv_trend     TEXT,
            outcome       TEXT,
            actual_entry  REAL,
            actual_high   REAL,
            actual_low    REAL,
            actual_close  REAL,
            pnl_pct       REAL,
            evaluated_at  TEXT
        )
        """)
        # Migrate existing databases — silently ignored if column already exists
        try:
            con.execute("ALTER TABLE signals ADD COLUMN interval TEXT DEFAULT '1h'")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE signals ADD COLUMN actual_entry REAL")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE signals ADD COLUMN sentiment TEXT DEFAULT 'NEUTRAL'")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE signals ADD COLUMN sentiment_score REAL DEFAULT 0.0")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE signals ADD COLUMN sentiment_count INTEGER DEFAULT 0")
        except Exception:
            pass


def _is_trading_day(d: datetime) -> bool:
    """True if d is a weekday and not an NSE holiday."""
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS


def _next_business_days(from_date: datetime, n: int) -> datetime:
    """Returns the date that is n NSE trading days after from_date."""
    d, count = from_date, 0
    while count < n:
        d += timedelta(days=1)
        if _is_trading_day(d):
            count += 1
    return d


_MARKET_CLOSE_HOUR   = 15
_MARKET_CLOSE_MINUTE = 30


def _nth_trading_day_from(from_date: datetime, n: int) -> datetime:
    """Returns the nth trading day ON OR AFTER from_date (1-indexed).

    Today counts as day 1 only when the market session is still open or not yet
    started (i.e. from_date is before 15:30).  Signals generated after market
    close belong to the next session, so today is skipped in that case.
    """
    market_close = from_date.replace(
        hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE,
        second=0, microsecond=0,
    )
    d = from_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # If generated after market close, today's session is over — skip to tomorrow
    if from_date >= market_close:
        d += timedelta(days=1)

    # Snap to first trading day on or after d
    while not _is_trading_day(d):
        d += timedelta(days=1)

    # Advance n-1 more trading days
    count = 1
    while count < n:
        d += timedelta(days=1)
        if _is_trading_day(d):
            count += 1
    return d


# ─── Logging ──────────────────────────────────────────────────────────────────

def log_signals(signals: list, pred_days: int = 3, interval: str = "1h",
                trends: dict = None) -> int:
    """
    Saves actionable signals to the tracker DB.
    trends: optional dict {symbol: TrendSnapshot} — used to store RVOL/OBV alongside each signal.
    Returns number of signals logged.
    """
    init_db()
    now     = datetime.now()
    eval_by = _nth_trading_day_from(now, pred_days)
    logged  = 0

    with _conn() as con:
        for s in signals:
            if s.direction == "NO TRADE":
                continue

            tgt_pct = abs((s.target - s.entry) / s.entry * 100) if s.entry else 0
            sl_pct  = abs((s.stop_loss - s.entry) / s.entry * 100) if s.entry else 0
            trend   = (trends or {}).get(s.symbol)

            con.execute("""
                INSERT INTO signals
                (logged_at, eval_by, symbol, cap_tier, direction,
                 entry, target, stop_loss, target_pct, stoploss_pct,
                 rr_ratio, pred_days, interval, confidence, confluence,
                 trend_bias, trend_score, rvol, volume_spike, obv_trend,
                 sentiment, sentiment_score, sentiment_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now.strftime("%Y-%m-%d %H:%M"),
                eval_by.strftime("%Y-%m-%d"),
                s.symbol,
                getattr(s, "cap_tier", ""),
                s.direction,
                s.entry, s.target, s.stop_loss,
                round(tgt_pct, 2), round(sl_pct, 2),
                s.rr_ratio, pred_days, interval,
                s.confidence, s.confluence,
                s.trend_bias, s.trend_score,
                getattr(trend, "rvol", None),
                int(getattr(trend, "volume_spike", False)),
                getattr(trend, "obv_trend", None),
                getattr(s, "sentiment", "NEUTRAL"),
                getattr(s, "sentiment_score", 0.0),
                getattr(s, "sentiment_count", 0),
            ))
            logged += 1

    print(f"  [Tracker] Logged {logged} signal(s) — evaluate after {eval_by.strftime('%d %b %Y')}")
    return logged


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _next_market_open(logged_at: str) -> str:
    """Returns the NSE market open (09:15) to start the evaluation window from.

    If the signal was logged before today's market open on a trading day,
    return today's 09:15 — this enables same-day evaluation for pre-market runs
    (e.g. generated at 07:00, evaluated after 15:30 the same day).
    Otherwise return the next trading day's 09:15.
    """
    ts       = pd.Timestamp(logged_at)
    today    = ts.normalize()
    mkt_open = today + pd.Timedelta(hours=9, minutes=15)

    if ts < mkt_open and _is_trading_day(today.to_pydatetime()):
        return today.strftime("%Y-%m-%d") + " 09:15:00"

    next_day = today + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        next_day += pd.Timedelta(days=1)
    return next_day.strftime("%Y-%m-%d") + " 09:15:00"


def _fetch_actual(symbol: str, from_dt: str, to_dt: str,
                  interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetches OHLCV at the signal's own interval between from_dt and to_dt."""
    from pipeline.data_fetcher import fetch_ohlcv
    df = fetch_ohlcv(symbol, interval=interval)
    if df is None:
        return None

    def _to_ist(s: str) -> pd.Timestamp:
        ts = pd.Timestamp(s)
        return ts.tz_localize("Asia/Kolkata") if ts.tzinfo is None else ts.tz_convert("Asia/Kolkata")

    start = _to_ist(from_dt)
    end   = _to_ist(to_dt)
    mask  = (df.index >= start) & (df.index <= end)
    return df[mask] if mask.any() else None


def _determine_outcome(direction: str, actual_entry: float,
                        target_pct: float, stoploss_pct: float,
                        candles: pd.DataFrame) -> tuple:
    """
    Recalculates target and stop-loss from actual next-day open (actual_entry),
    walks candles in order, and returns which level was hit first.
    Returns (outcome, pnl_pct, actual_high, actual_low, actual_close)
    """
    if direction == "LONG":
        adj_target   = actual_entry * (1 + target_pct   / 100)
        adj_stoploss = actual_entry * (1 - stoploss_pct / 100)
    else:  # SHORT
        adj_target   = actual_entry * (1 - target_pct   / 100)
        adj_stoploss = actual_entry * (1 + stoploss_pct / 100)

    actual_high = float(candles["high"].max())
    actual_low  = float(candles["low"].min())

    # EXPIRED exit: last close of the eval_by date specifically
    last_date = candles.index[-1].date()
    final_day = candles[pd.Series(candles.index).apply(lambda t: t.date() == last_date).values]
    actual_close = (float(final_day["close"].iloc[-1])
                    if not final_day.empty
                    else float(candles["close"].iloc[-1]))

    for _, row in candles.iterrows():
        if direction == "LONG":
            if row["high"] >= adj_target:
                pnl = ((adj_target - actual_entry) / actual_entry) * 100
                return "WIN", round(pnl, 2), actual_high, actual_low, actual_close
            if row["low"] <= adj_stoploss:
                pnl = ((adj_stoploss - actual_entry) / actual_entry) * 100
                return "LOSS", round(pnl, 2), actual_high, actual_low, actual_close
        else:
            if row["low"] <= adj_target:
                pnl = ((actual_entry - adj_target) / actual_entry) * 100
                return "WIN", round(pnl, 2), actual_high, actual_low, actual_close
            if row["high"] >= adj_stoploss:
                pnl = ((actual_entry - adj_stoploss) / actual_entry) * 100
                return "LOSS", round(pnl, 2), actual_high, actual_low, actual_close

    # Neither target nor stop hit — expired
    if direction == "LONG":
        pnl = ((actual_close - actual_entry) / actual_entry) * 100
    else:
        pnl = ((actual_entry - actual_close) / actual_entry) * 100
    return "EXPIRED", round(pnl, 2), actual_high, actual_low, actual_close


def evaluate(force: bool = False) -> None:
    """
    Checks all pending signals whose eval_by date has passed.
    Updates outcome in DB using actual next-day open as entry.
    """
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")

    with _conn() as con:
        query = "SELECT * FROM signals WHERE outcome IS NULL"
        if not force:
            query += f" AND eval_by <= '{today}'"
        cur  = con.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    if not rows:
        print("  [Tracker] No signals ready for evaluation.")
        return

    print(f"  [Tracker] Evaluating {len(rows)} signal(s)...")

    with _conn() as con:
        for row in rows:
            r = dict(zip(cols, row))

            if r["eval_by"] > today:
                print(f"    {r['symbol']:<14} eval_by {r['eval_by']} not reached yet — skipping")
                continue

            interval = r.get("interval") or "1h"
            from_dt  = _next_market_open(r["logged_at"])
            candles  = _fetch_actual(
                r["symbol"],
                from_dt,
                r["eval_by"] + " 15:30:00",
                interval=interval,
            )

            if candles is None or candles.empty:
                print(f"    {r['symbol']:<14} no price data in window "
                      f"({from_dt[:10]} -> {r['eval_by']}) - skipping")
                continue

            actual_entry = float(candles.iloc[0]["open"])
            outcome, pnl, hi, lo, cl = _determine_outcome(
                r["direction"],
                actual_entry,
                r["target_pct"],
                r["stoploss_pct"],
                candles,
            )

            con.execute("""
                UPDATE signals
                SET outcome=?, actual_entry=?, actual_high=?, actual_low=?,
                    actual_close=?, pnl_pct=?, evaluated_at=?
                WHERE id=?
            """, (outcome, actual_entry, hi, lo, cl, pnl, today, r["id"]))

            tag = "WIN " if outcome == "WIN" else ("LOSS" if outcome == "LOSS" else "EXP ")
            print(f"    [{tag}] {r['symbol']:<14} {r['direction']:<5} "
                  f"ActualEntry:{actual_entry:.2f}  P&L:{pnl:+.2f}%  ({outcome})")

    print("  [Tracker] Evaluation complete.")


# ─── Report ───────────────────────────────────────────────────────────────────

def report() -> None:
    init_db()
    with _conn() as con:
        df = pd.read_sql("SELECT * FROM signals WHERE outcome IS NOT NULL", con)

    if df.empty:
        print("  [Tracker] No evaluated signals yet. Run: python tracker.py evaluate")
        return

    evaluated  = df[df["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]
    wins       = evaluated[evaluated["outcome"] == "WIN"]
    losses     = evaluated[evaluated["outcome"] == "LOSS"]
    expired    = evaluated[evaluated["outcome"] == "EXPIRED"]

    total      = len(evaluated)
    decisive   = len(wins) + len(losses)
    hit_rate   = len(wins) / total * 100 if total else 0          # wins / ALL trades
    win_rate   = len(wins) / decisive * 100 if decisive else 0    # wins / (wins + losses)
    avg_win    = wins["pnl_pct"].mean()    if not wins.empty    else 0
    avg_loss   = losses["pnl_pct"].mean()  if not losses.empty  else 0
    avg_expired= expired["pnl_pct"].mean() if not expired.empty else 0
    avg_pnl    = evaluated["pnl_pct"].mean()

    # Expectancy = expected P&L per trade taken, weighting EACH outcome class by
    # its own realised average P&L. Expired trades exit at the eval-date close, so
    # they are counted at their real (often non-zero) P&L instead of being lumped
    # in with losses. This equals the mean realised P&L across all evaluated trades.
    p_win = len(wins)    / total if total else 0
    p_los = len(losses)  / total if total else 0
    p_exp = len(expired) / total if total else 0
    expectancy = p_win * avg_win + p_los * avg_loss + p_exp * avg_expired

    print("\n" + "=" * 60)
    print("  KRONOS PREDICTION TRACKER — PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Total signals evaluated : {total}")
    print(f"  Wins                    : {len(wins)}  ({win_rate:.1f}% of decided, {hit_rate:.1f}% of all)")
    print(f"  Losses                  : {len(losses)}")
    print(f"  Expired (no hit)        : {len(expired)}")
    print(f"  Avg win P&L             : {avg_win:+.2f}%")
    print(f"  Avg loss P&L            : {avg_loss:+.2f}%")
    print(f"  Avg expired P&L         : {avg_expired:+.2f}%")
    print(f"  Avg P&L all trades      : {avg_pnl:+.2f}%")
    print(f"  Expectancy              : {expectancy:+.2f}% per trade  "
          f"[win {p_win*100:.0f}%×{avg_win:+.1f} + loss {p_los*100:.0f}%×{avg_loss:+.1f} "
          f"+ exp {p_exp*100:.0f}%×{avg_expired:+.1f}]")

    # By confluence
    print("\n  --- By Confluence Level ---")
    for conf in ["STRONG", "MODERATE", "WEAK"]:
        sub = evaluated[evaluated["confluence"] == conf]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {conf:<10} : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # By confidence
    print("\n  --- By Confidence Level ---")
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        sub = evaluated[evaluated["confidence"] == conf]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {conf:<8} : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # By sentiment
    print("\n  --- By Sentiment ---")
    for sent in ["BULLISH", "NEUTRAL", "BEARISH"]:
        sub = evaluated[evaluated["sentiment"] == sent]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {sent:<8} : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # By cap tier
    print("\n  --- By Cap Tier ---")
    for tier in ["large", "mid", "small"]:
        sub = evaluated[evaluated["cap_tier"] == tier]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {tier.upper():<8} cap : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # By direction
    print("\n  --- By Direction ---")
    for d in ["LONG", "SHORT"]:
        sub = evaluated[evaluated["direction"] == d]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {d:<6} : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # By interval
    print("\n  --- By Interval ---")
    has_interval = "interval" in evaluated.columns
    for iv in ["1h", "15m", "5m", "1m"]:
        if not has_interval:
            break
        sub = evaluated[evaluated["interval"] == iv]
        if sub.empty:
            continue
        wr = len(sub[sub["outcome"] == "WIN"]) / len(sub) * 100
        print(f"  {iv:<4} : {len(sub):>3} trades | Win rate {wr:.0f}% | "
              f"Avg P&L {sub['pnl_pct'].mean():+.2f}%")

    # Best and worst trades
    print("\n  --- Best Trades ---")
    best = evaluated.nlargest(3, "pnl_pct")[
        ["symbol", "direction", "entry", "pnl_pct", "confluence", "logged_at"]
    ]
    print(best.to_string(index=False))

    print("\n  --- Worst Trades ---")
    worst = evaluated.nsmallest(3, "pnl_pct")[
        ["symbol", "direction", "entry", "pnl_pct", "confluence", "logged_at"]
    ]
    print(worst.to_string(index=False))

    # Streaks — WIN/LOSS only, ignoring EXPIRED
    print("\n  --- Streaks ---")
    decisive = (evaluated[evaluated["outcome"].isin(["WIN", "LOSS"])]
                .sort_values("logged_at"))
    if decisive.empty:
        print("  Not enough WIN/LOSS outcomes yet.")
    else:
        outcomes = decisive["outcome"].tolist()

        # Current streak: consecutive identical outcomes at the tail
        cur_val   = outcomes[-1]
        cur_count = 0
        for o in reversed(outcomes):
            if o == cur_val:
                cur_count += 1
            else:
                break

        # Best win streak and worst loss streak across full history
        best_win   = 1 if outcomes[0] == "WIN"  else 0
        worst_loss = 1 if outcomes[0] == "LOSS" else 0
        run_count  = 1
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i - 1]:
                run_count += 1
            else:
                run_count = 1
            if outcomes[i] == "WIN"  and run_count > best_win:   best_win  = run_count
            if outcomes[i] == "LOSS" and run_count > worst_loss: worst_loss = run_count

        print(f"  Current streak  : {cur_count} × {cur_val}")
        print(f"  Best win streak : {best_win} in a row")
        print(f"  Worst loss run  : {worst_loss} in a row")
    print()


def show() -> None:
    """Lists all logged signals (evaluated and pending)."""
    init_db()
    with _conn() as con:
        df = pd.read_sql("""
            SELECT logged_at, symbol, cap_tier, direction, entry, target,
                   stop_loss, rr_ratio, confidence, confluence,
                   eval_by, outcome, pnl_pct
            FROM signals ORDER BY id DESC LIMIT 50
        """, con)

    if df.empty:
        print("  [Tracker] No signals logged yet.")
        return

    print("\n" + "=" * 60)
    print("  LOGGED SIGNALS (last 50)")
    print("=" * 60)
    pending   = df[df["outcome"].isna()]
    evaluated = df[df["outcome"].notna()]
    if not evaluated.empty:
        print("\n  Evaluated:")
        print(evaluated.to_string(index=False))
    if not pending.empty:
        print("\n  Pending evaluation:")
        print(pending[["logged_at", "symbol", "direction", "entry",
                        "target", "eval_by"]].to_string(index=False))
    print()


# ─── CSV Import ───────────────────────────────────────────────────────────────

def import_csv(csv_path: str, pred_days: int = 3, interval: str = "1h") -> None:
    """
    Imports signals from a saved CSV into the tracker DB.
    logged_at is parsed from the filename (signals_YYYYMMDD_HHMM_*.csv);
    falls back to current time if the filename doesn't match that pattern.
    Skips rows that are already in the DB (same symbol + direction + logged_at).
    """
    import re

    if not os.path.isfile(csv_path):
        print(f"  [Tracker] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"Symbol", "Direction", "Entry", "Target", "Stop Loss"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [Tracker] CSV is missing columns: {missing}")
        return

    # Parse logged_at from filename: signals_YYYYMMDD_HHMM_*.csv
    fname = os.path.basename(csv_path)
    m = re.search(r"signals_(\d{8})_(\d{4})", fname)
    if m:
        logged_at = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    else:
        logged_at = datetime.now()
        print(f"  [Tracker] Could not parse timestamp from filename — using now as logged_at")

    eval_by = _nth_trading_day_from(logged_at, pred_days)

    init_db()
    inserted = skipped = 0

    with _conn() as con:
        for _, row in df.iterrows():
            if str(row.get("Direction", "")).strip() in ("NO TRADE", "", "nan"):
                continue

            symbol    = str(row["Symbol"]).strip()
            direction = str(row["Direction"]).strip()
            entry     = float(row["Entry"])
            target    = float(row["Target"])
            stop_loss = float(row["Stop Loss"])

            # Duplicate check
            exists = con.execute(
                "SELECT 1 FROM signals WHERE symbol=? AND direction=? AND logged_at=?",
                (symbol, direction, logged_at.strftime("%Y-%m-%d %H:%M")),
            ).fetchone()
            if exists:
                skipped += 1
                continue

            tgt_pct = abs((target - entry) / entry * 100) if entry else 0
            sl_pct  = abs((stop_loss - entry) / entry * 100) if entry else 0
            rr      = round(tgt_pct / sl_pct, 2) if sl_pct else 0

            con.execute("""
                INSERT INTO signals
                (logged_at, eval_by, symbol, cap_tier, direction,
                 entry, target, stop_loss, target_pct, stoploss_pct,
                 rr_ratio, pred_days, interval, confidence, confluence,
                 trend_bias, trend_score, sentiment, sentiment_score, sentiment_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                logged_at.strftime("%Y-%m-%d %H:%M"),
                eval_by.strftime("%Y-%m-%d"),
                symbol,
                str(row.get("Cap Tier", "")).strip(),
                direction,
                entry, target, stop_loss,
                round(tgt_pct, 2), round(sl_pct, 2),
                rr, pred_days, interval,
                str(row.get("Confidence", "")).strip() or None,
                str(row.get("Confluence", "")).strip() or None,
                str(row.get("Trend", "")).strip() or None,
                int(row["Trend Score"]) if "Trend Score" in df.columns and pd.notna(row.get("Trend Score")) else None,
                str(row.get("Sentiment", "NEUTRAL")).strip() or "NEUTRAL",
                float(row["Sentiment Score"]) if "Sentiment Score" in df.columns and pd.notna(row.get("Sentiment Score")) else 0.0,
                int(row["Sentiment Count"]) if "Sentiment Count" in df.columns and pd.notna(row.get("Sentiment Count")) else 0,
            ))
            inserted += 1

    print(f"  [Tracker] Imported {inserted} signal(s) from {fname}"
          + (f" | {skipped} duplicate(s) skipped" if skipped else "")
          + f" — evaluate after {eval_by.strftime('%d %b %Y')}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kronos Prediction Tracker")
    p.add_argument("command", choices=["evaluate", "report", "show", "import"],
                   help="evaluate: check outcomes | report: stats | show: list signals | import: load from CSV")
    p.add_argument("--force", action="store_true",
                   help="Evaluate all pending signals regardless of eval_by date")
    p.add_argument("--csv",      default=None, help="Path to CSV file (for import command)")
    p.add_argument("--days",     type=int, default=3, help="Prediction horizon in trading days (for import, default 3)")
    p.add_argument("--interval", default="1h", choices=["1h", "15m", "5m", "1m"],
                   help="Candle interval (for import, default 1h)")
    args = p.parse_args()

    if args.command == "evaluate":
        evaluate(force=args.force)
    elif args.command == "report":
        report()
    elif args.command == "show":
        show()
    elif args.command == "import":
        if not args.csv:
            p.error("import requires --csv <path>")
        import_csv(args.csv, pred_days=args.days, interval=args.interval)
