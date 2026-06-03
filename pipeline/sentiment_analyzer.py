"""
News sentiment analysis using FinBERT (ProsusAI/finbert).
Data source: Google News RSS — free, no API key required.
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import feedparser


@dataclass
class SentimentResult:
    label: str    # "BULLISH", "BEARISH", "NEUTRAL"
    score: float  # dominant class probability 0.0–1.0
    count: int    # headlines analysed


_model     = None
_tokenizer = None


def _load_finbert():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("  Loading FinBERT (first run downloads ~440 MB)...")
    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
    print("  FinBERT ready.")
    return _model, _tokenizer


def _fetch_headlines(symbol: str, max_count: int = 8) -> list:
    """Pulls recent Google News RSS headlines for an NSE symbol."""
    query  = f"{symbol}+NSE+stock+India"
    url    = (f"https://news.google.com/rss/search"
              f"?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        headlines = []
        for entry in feed.entries[:25]:
            title = re.sub(r"<[^>]+>", "", entry.get("title", "")).strip()
            if not title:
                continue
            pub = entry.get("published_parsed")
            if not pub:
                continue
            pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
            if pub_dt < cutoff:
                continue
            headlines.append(title)
            if len(headlines) >= max_count:
                break
        return headlines
    except Exception:
        return []


def _classify(headlines: list) -> SentimentResult:
    """Runs FinBERT on a batch of headlines and returns aggregated sentiment."""
    import torch

    model, tokenizer = _load_finbert()
    device = next(model.parameters()).device

    inputs = tokenizer(
        headlines, padding=True, truncation=True,
        max_length=64, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    probs       = torch.softmax(logits, dim=-1).mean(dim=0).cpu()
    pos, neg, neu = probs[0].item(), probs[1].item(), probs[2].item()

    if pos >= neg and pos >= neu and pos > 0.38:
        return SentimentResult("BULLISH", round(pos, 2), len(headlines))
    if neg > pos and neg >= neu and neg > 0.38:
        return SentimentResult("BEARISH", round(neg, 2), len(headlines))
    return SentimentResult("NEUTRAL", round(max(pos, neg, neu), 2), len(headlines))


def analyze(symbol: str) -> SentimentResult:
    """Returns sentiment for a single NSE symbol based on recent news."""
    headlines = _fetch_headlines(symbol)
    if not headlines:
        return SentimentResult("NEUTRAL", 0.0, 0)
    return _classify(headlines)


def analyze_batch(symbols: list, delay: float = 0.3) -> dict:
    """
    Returns {symbol: SentimentResult} for a list of NSE symbols.
    Small delay between RSS requests avoids rate-limiting.
    """
    results = {}
    for sym in symbols:
        try:
            results[sym] = analyze(sym)
            time.sleep(delay)
        except Exception as e:
            print(f"  [WARN] Sentiment error for {sym}: {e}")
            results[sym] = SentimentResult("NEUTRAL", 0.0, 0)
    return results
