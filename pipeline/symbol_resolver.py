"""
Symbol resolver — maps company name queries to NSE ticker symbols.

Priority order:
  1. Exact symbol match (case-insensitive)
  2. Exact full name match (case-insensitive)
  3. Substring match (query appears anywhere in name)
  4. Fuzzy match via difflib (catches typos / partial words)

If a query already looks like a valid symbol it passes through unchanged.
"""

import csv
import difflib
import os

_NAMES_CSV = os.path.join(os.path.dirname(__file__), "symbol_names.csv")
_cache: dict[str, str] | None = None   # {SYMBOL: "Company Name"}


def _load() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        with open(_NAMES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _cache[row["symbol"].strip().upper()] = row["name"].strip()
    return _cache


def resolve(query: str) -> tuple[str, str] | None:
    """
    Returns (symbol, name) for a symbol or company name query.
    Returns None if no confident match found.
    """
    names = _load()
    q = query.strip()

    # 1. Exact symbol match
    if q.upper() in names:
        sym = q.upper()
        return sym, names[sym]

    # 2. Exact full name match
    q_lower = q.lower()
    for sym, name in names.items():
        if name.lower() == q_lower:
            return sym, name

    # 3. Substring match — query words all present in name
    words = q_lower.split()
    matches = [
        (sym, name) for sym, name in names.items()
        if all(w in name.lower() for w in words)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer shorter names (more specific match) then alphabetical
        matches.sort(key=lambda x: (len(x[1]), x[0]))
        return matches[0]

    # 4. Fuzzy match on full names — require at least one query word to appear
    #    in the matched name so we don't pick completely unrelated companies
    all_names = list(names.values())
    close = difflib.get_close_matches(q, all_names, n=1, cutoff=0.65)
    if close:
        matched_name = close[0]
        words_in_match = any(w in matched_name.lower() for w in words)
        if words_in_match:
            for sym, name in names.items():
                if name == matched_name:
                    return sym, name

    return None


def resolve_symbols(queries: list[str]) -> list[str]:
    """
    Resolves a list of symbol or name queries to NSE symbols.
    Logs what was resolved so the user can see the mapping.
    Unresolvable queries are passed through with a warning.
    """
    resolved = []
    for q in queries:
        result = resolve(q)
        if result is None:
            print(f"  [WARN] Could not resolve '{q}' — using as-is")
            resolved.append(q.upper())
        elif result[0] != q.upper():
            print(f"  Resolved  '{q}'  ->  {result[0]}  ({result[1]})")
            resolved.append(result[0])
        else:
            resolved.append(result[0])
    return resolved
