"""Value normalization for comparing predicted vs gold fields.

Receipts use many spellings of the same value ("$12.34" vs "12.34" vs "12.3"
rounded). Each metric normalizes through these helpers so e.g. case
differences and currency symbols don't count as mismatches.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Optional
import re


_PUNCT = re.compile(r"[\s\.,;:!?]+")


def normalize_string(value: str | None) -> str:
    """Casefold, strip whitespace and trailing punctuation. Empty if None."""
    if value is None:
        return ""
    s = value.strip().casefold()
    s = _PUNCT.sub(" ", s).strip()
    return s


def normalize_decimal(value: Decimal | str | float | None) -> Optional[Decimal]:
    """Coerce to Decimal at 2dp; None if unparseable or empty."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"))
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("$", "").replace("€", "").replace("£", "").replace(",", "")
    try:
        return Decimal(s).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def normalize_date(value: str | None) -> str:
    """Expects YYYY-MM-DD; returns "" for empty or invalid."""
    if not value:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return value.strip()
    return ""


def normalize_currency(value: str | None) -> str:
    """3-letter ISO codes, uppercased. Empty if invalid."""
    if not value:
        return ""
    s = value.strip().upper()
    return s if re.fullmatch(r"[A-Z]{3}", s) else ""
