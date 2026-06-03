"""
Fetches top gainers and losers segmented by market-cap tier:
  Large Cap  — Nifty 100      (~100 stocks)
  Mid Cap    — Nifty Midcap 150  (~150 stocks)
  Small Cap  — Nifty Smallcap 250 (~250 stocks)

Constituent lists are downloaded from NSE archives (public CSV).
Prices and % change are computed via yfinance.
"""

import requests
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple

# NSE public CSVs — no auth required
NSE_INDEX_CSVS = {
    "large": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "mid":   "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "small": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

# Fallbacks if NSE archive is unreachable
FALLBACK = {
    "large": [
        "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFY","SBIN",
        "HINDUNILVR","ITC","LT","KOTAKBANK","HCLTECH","AXISBANK","MARUTI",
        "SUNPHARMA","TITAN","ULTRACEMCO","NESTLEIND","WIPRO","POWERGRID",
        "NTPC","TECHM","BAJFINANCE","ONGC","COALINDIA","ADANIENT","ADANIPORTS",
        "BAJAJFINSV","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HDFCLIFE",
        "HEROMOTOCO","HINDALCO","INDUSINDBK","JSWSTEEL","M&M","SBILIFE",
        "SHRIRAMFIN","TATACONSUM","TATAPOWER","TATASTEEL","CIPLA","BPCL",
        "BRITANNIA","APOLLOHOSP","BAJAJ-AUTO","LICI","VEDL",
    ],
    "mid": [
        "ABCAPITAL","ABFRL","ALKEM","ASHOKLEY","AUROPHARMA","BANDHANBNK",
        "BANKBARODA","BHEL","BIOCON","BOSCHLTD","CANBK","CHOLAFIN","COLPAL",
        "CONCOR","CUMMINSIND","DABUR","DALBHARAT","DEEPAKNTR","DLF","ESCORTS",
        "FEDERALBNK","GAIL","GODREJCP","GODREJPROP","HAVELLS","ICICIGI",
        "ICICIPRULI","IDFCFIRSTB","INDHOTEL","INDUSTOWER","IRCTC","JINDALSTEL",
        "JUBLFOOD","LTF","LUPIN","MANKIND","MARICO","MAXHEALTH","MFSL",
        "MPHASIS","MRF","MUTHOOTFIN","NAUKRI","OBEROIRLTY","OFSS","PAGEIND",
        "PERSISTENT","PIIND","PNB","POLICYBZR","POONAWALLA","PVRINOX","SBICARD",
        "TRENT","VGUARD","VOLTAS","ZOMATO","ZYDUSLIFE",
    ],
    "small": [
        "AFFLE","ANGELONE","ANURAS","APARINDS","APTUS","ASTRAL","ATGL",
        "ATUL","BALRAMCHIN","BSOFT","CAMPUS","CANFINHOME","CDSL","CESC",
        "CLEAN","CSBBANK","DATAPATTNS","DCMSHRIRAM","DELHIVERY","ELGIEQUIP",
        "EQUITASBNK","FINPIPE","FIVESTAR","GLAND","GLAXO","GNFC","GRAPHITE",
        "GRINDWELL","HAPPSTMNDS","HFCL","IEX","INDIAMART","INGERRAND",
        "JBCHEPHARM","JKCEMENT","JKLAKSHMI","JKPAPER","JYOTHYLAB","KFINTECH",
        "KPIL","KRBL","LAXMIMACH","LTTS","LUXIND","MAHSCOOTER","MEDANTA",
        "METROBRAND","MINDAIND","MMTC","MTAR","NATCOPHARM","NAVNETEDUL",
        "NBCC","NCC","NIACL","NLCINDIA","NMDC","NOCIL","NUVOCO",
    ],
}


def _fetch_nse_constituents(tier: str) -> list:
    """Downloads NSE index CSV and returns list of ticker symbols."""
    url = NSE_INDEX_CSVS[tier]
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if col:
            return df[col].str.strip().dropna().tolist()
    except Exception:
        pass
    print(f"  [WARN] Could not fetch {tier}-cap list from NSE — using fallback list.")
    return FALLBACK[tier]


def _get_price_changes(symbols: list) -> pd.DataFrame:
    """Downloads 5-day daily close for all symbols, returns % change df."""
    tickers = [s + ".NS" for s in symbols]
    raw = yf.download(
        tickers, period="5d", interval="1d",
        auto_adjust=True, progress=False, group_by="ticker",
    )
    rows = []
    for sym, ticker in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw[ticker]["Close"].dropna()
            else:
                close = raw["Close"].dropna()
            if len(close) < 2:
                continue
            prev, today = float(close.iloc[-2]), float(close.iloc[-1])
            rows.append({
                "symbol":     sym,
                "ltp":        round(today, 2),
                "prev_close": round(prev, 2),
                "change_pct": round(((today - prev) / prev) * 100, 2),
                "change":     round(today - prev, 2),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def get_top_gainers_losers(
    top_n: int = 10,
    tiers: list = None,           # None = all three
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Returns {tier: (gainers_df, losers_df)} for each requested cap tier.
    tier keys: "large", "mid", "small"
    """
    if tiers is None:
        tiers = ["large", "mid", "small"]

    results = {}
    for tier in tiers:
        label = {"large": "Large Cap (Nifty 100)",
                 "mid":   "Mid Cap (Nifty Midcap 150)",
                 "small": "Small Cap (Nifty Smallcap 250)"}[tier]
        print(f"  Fetching {label}...")
        symbols = _fetch_nse_constituents(tier)
        df = _get_price_changes(symbols)
        if df.empty:
            results[tier] = (pd.DataFrame(), pd.DataFrame())
            continue
        gainers = df.nlargest(top_n,  "change_pct").reset_index(drop=True)
        losers  = df.nsmallest(top_n, "change_pct").reset_index(drop=True)
        # Tag with tier for use downstream
        gainers["cap_tier"] = tier
        losers["cap_tier"]  = tier
        results[tier] = (gainers, losers)

    return results


def display_scanner_results(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
) -> None:
    tier_labels = {
        "large": "LARGE CAP  (Nifty 100)",
        "mid":   "MID CAP    (Nifty Midcap 150)",
        "small": "SMALL CAP  (Nifty Smallcap 250)",
    }
    cols = ["symbol", "ltp", "change_pct", "change"]

    for tier, (gainers, losers) in results.items():
        label = tier_labels.get(tier, tier.upper())
        width = 58

        print("\n" + "="*width)
        print(f"  TOP GAINERS :: {label}")
        print("="*width)
        print(gainers[cols].to_string(index=False) if not gainers.empty else "  No data.")

        print("\n" + "-"*width)
        print(f"  TOP LOSERS  :: {label}")
        print("-"*width)
        print(losers[cols].to_string(index=False) if not losers.empty else "  No data.")
    print()


def all_symbols_from_results(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
) -> list:
    """Returns a deduplicated flat list of all scanned symbols (gainers + losers)."""
    seen, out = set(), []
    for gainers, losers in results.values():
        for sym in pd.concat([gainers, losers])["symbol"].tolist():
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


if __name__ == "__main__":
    results = get_top_gainers_losers(top_n=10)
    display_scanner_results(results)
