"""Character Error Rate on the raw OCR text.

CER = (substitutions + deletions + insertions) / total_gold_chars.
Lower is better. Useful as a sanity check independent of the parser —
if Vision recognized the text well, CER stays low even when the
structured parser fails.

Uses `jiwer` if available. Falls back to a simple Levenshtein
implementation so the bench still runs without it.
"""

from __future__ import annotations

from typing import Optional

try:
    from jiwer import cer as jiwer_cer
    _HAS_JIWER = True
except ImportError:  # pragma: no cover
    _HAS_JIWER = False


def cer(gold_text: Optional[str], predicted_text: Optional[str]) -> Optional[float]:
    """Character error rate. Returns None if either text is unavailable."""
    if gold_text is None or predicted_text is None:
        return None
    gold = gold_text or ""
    pred = predicted_text or ""
    if not gold and not pred:
        return 0.0
    if _HAS_JIWER:
        try:
            return float(jiwer_cer(gold, pred))
        except Exception:  # pragma: no cover
            return _levenshtein_cer(gold, pred)
    return _levenshtein_cer(gold, pred)


def _levenshtein_cer(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if pred else 0.0
    dist = _levenshtein(gold, pred)
    return dist / len(gold)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]
