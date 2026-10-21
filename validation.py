"""Small, strict validators for model-generated metadata."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def bounded_float(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return default
    return min(max(number, low), high) if math.isfinite(number) else default


def evidence_ids(raw: Any, valid: set[str], limit: int = 8) -> list[str]:
    if not isinstance(raw, list):
        return []
    return list(
        dict.fromkeys(item for item in raw if isinstance(item, str) and item in valid)
    )[:limit]


def contains_term(term: str, text: str) -> bool:
    """Match Chinese phrases and whole ASCII words, avoiding e.g. ai in rain."""
    term = str(term or "").strip().casefold()
    text = str(text or "").casefold()
    if not term:
        return False
    left = r"(?<![a-z0-9_])" if term[0].isascii() and term[0].isalnum() else ""
    right = r"(?![a-z0-9_])" if term[-1].isascii() and term[-1].isalnum() else ""
    return bool(re.search(left + re.escape(term) + right, text))
