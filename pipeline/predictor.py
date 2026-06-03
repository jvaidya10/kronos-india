"""
Wraps the Kronos model for intraday prediction.
Assumes the Kronos repo is cloned at ../Kronos or at KRONOS_PATH env variable.
"""

import os
import sys
import pandas as pd
from typing import Optional

def _hf_cache_exists() -> bool:
    """Returns True if any Kronos model weights are already in the HuggingFace cache."""
    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    if not os.path.isdir(cache_dir):
        return False
    return any(
        entry.startswith("models--NeoQuasar--Kronos")
        for entry in os.listdir(cache_dir)
    )

if _hf_cache_exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # skip version-check if cached

# Allow overriding Kronos path via env variable
KRONOS_PATH = os.environ.get("KRONOS_PATH", os.path.join(os.path.dirname(__file__), "..", "Kronos"))
if KRONOS_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(KRONOS_PATH))

# Lazily loaded — imported only when load_model() is called
_predictor = None
_model_name = None


def load_model(variant: str = "small") -> None:
    """
    Loads the Kronos model into module-level cache.
    variant: 'mini' (4.1M, ctx=2048), 'small' (24.7M, ctx=512), 'base' (102.3M, ctx=512)
    """
    global _predictor, _model_name

    if _predictor is not None and _model_name == variant:
        return  # already loaded

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        raise ImportError(
            f"Cannot import Kronos model. Make sure the Kronos repo is cloned at:\n"
            f"  {KRONOS_PATH}\n"
            f"Or set KRONOS_PATH env variable to its location.\n"
            f"Original error: {e}"
        )

    tokenizer_map = {
        "mini":  "NeoQuasar/Kronos-Tokenizer-2k",
        "small": "NeoQuasar/Kronos-Tokenizer-base",
        "base":  "NeoQuasar/Kronos-Tokenizer-base",
    }
    model_map = {
        "mini":  "NeoQuasar/Kronos-mini",
        "small": "NeoQuasar/Kronos-small",
        "base":  "NeoQuasar/Kronos-base",
    }
    context_map = {
        "mini":  2048,
        "small": 512,
        "base":  512,
    }

    if variant not in model_map:
        raise ValueError(f"Invalid variant '{variant}'. Choose from: mini, small, base")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"[Kronos] GPU detected: {gpu} ({vram} GB VRAM) — using CUDA")
    else:
        print("[Kronos] No GPU found — running on CPU (slower)")

    source = "local cache" if os.environ.get("HF_HUB_OFFLINE") == "1" else "HuggingFace Hub"
    print(f"[Kronos] Loading {variant} model from {source}...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_map[variant])
    model     = Kronos.from_pretrained(model_map[variant])
    model     = model.to(device)
    _predictor = KronosPredictor(model, tokenizer, max_context=context_map[variant])
    _model_name = variant
    print(f"[Kronos] Model loaded: Kronos-{variant} on {device.upper()}")


def predict_next_day(
    symbol: str,
    x_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    sample_count: int = 20,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Optional[pd.DataFrame]:
    """
    Runs Kronos prediction for a single stock.
    Returns a DataFrame with predicted OHLCV for tomorrow's intraday candles,
    plus an 'ensemble_close' column averaging across samples.
    Returns None on failure.
    """
    if _predictor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    pred_len = len(y_timestamp)

    try:
        # Run multiple samples for uncertainty estimation
        all_preds = []
        for _ in range(sample_count):
            pred = _predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
            )
            all_preds.append(pred)

        # Aggregate: median across samples for robustness
        close_stack = pd.concat([p["close"] for p in all_preds], axis=1)
        high_stack  = pd.concat([p["high"]  for p in all_preds], axis=1)
        low_stack   = pd.concat([p["low"]   for p in all_preds], axis=1)
        open_stack  = pd.concat([p["open"]  for p in all_preds], axis=1)

        result = pd.DataFrame(index=all_preds[0].index)
        result["open"]  = open_stack.median(axis=1)
        result["high"]  = high_stack.median(axis=1)
        result["low"]   = low_stack.median(axis=1)
        result["close"] = close_stack.median(axis=1)

        # Confidence band: IQR of close predictions
        result["close_q25"] = close_stack.quantile(0.25, axis=1)
        result["close_q75"] = close_stack.quantile(0.75, axis=1)

        return result

    except Exception as e:
        print(f"[WARN] Kronos prediction failed for {symbol}: {e}")
        return None


def predict_batch(
    stocks: list,           # list of (symbol, x_df, x_timestamp, y_timestamp)
    sample_count: int = 20,
) -> dict:
    """
    Runs predictions for multiple stocks. Returns {symbol: pred_df or None}.
    """
    results = {}
    for symbol, x_df, x_ts, y_ts in stocks:
        print(f"  Predicting: {symbol}...")
        results[symbol] = predict_next_day(symbol, x_df, x_ts, y_ts, sample_count=sample_count)
    return results
