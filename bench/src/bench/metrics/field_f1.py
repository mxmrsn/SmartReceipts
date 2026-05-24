"""Per-field precision / recall / F1 across all evaluated samples.

For scalar fields (merchant, date, currency, total, subtotal):
    correct          — predicted matches gold (after normalization)
    present_in_pred  — pipeline produced any non-empty value
    present_in_gold  — gold label has any non-empty value

For line items: per-sample F1 by matching items on (normalized description,
total price within tolerance). Then averaged across samples.

F1 = 2 PR / (P + R). Falls back to 0 when divisor is 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from bench.schema import Receipt, LineItem
from bench.metrics.normalize import (
    normalize_string,
    normalize_decimal,
    normalize_date,
    normalize_currency,
)


@dataclass
class FieldScore:
    correct: int = 0
    present_in_pred: int = 0
    present_in_gold: int = 0
    sample_count: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.present_in_pred if self.present_in_pred else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.present_in_gold if self.present_in_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.sample_count if self.sample_count else 0.0


@dataclass
class LineItemsScore:
    """Per-sample F1 averaged across all samples that had any line items."""

    sample_count: int = 0
    samples_with_items: int = 0
    micro_correct: int = 0     # true positives across all samples
    micro_pred_count: int = 0  # all predicted items
    micro_gold_count: int = 0  # all gold items
    per_sample_f1: list[float] = field(default_factory=list)

    @property
    def micro_precision(self) -> float:
        return self.micro_correct / self.micro_pred_count if self.micro_pred_count else 0.0

    @property
    def micro_recall(self) -> float:
        return self.micro_correct / self.micro_gold_count if self.micro_gold_count else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def macro_f1(self) -> float:
        return (sum(self.per_sample_f1) / len(self.per_sample_f1)) if self.per_sample_f1 else 0.0


def score_sample(gold: Receipt, predicted: Optional[Receipt]) -> dict[str, dict]:
    """Score one (gold, predicted) pair. Returns per-field match info that
    aggregate functions later sum up."""
    out: dict[str, dict] = {}

    def scalar(field_name: str, gold_val, pred_val, equal) -> None:
        present_in_gold = bool(gold_val)
        present_in_pred = bool(pred_val) if predicted is not None else False
        is_correct = predicted is not None and equal(gold_val, pred_val)
        out[field_name] = {
            "correct": int(is_correct),
            "present_in_pred": int(present_in_pred),
            "present_in_gold": int(present_in_gold),
            "gold_value": gold_val,
            "pred_value": pred_val,
        }

    # Merchant
    g_merchant = normalize_string(gold.header.merchant.name)
    p_merchant = normalize_string(predicted.header.merchant.name) if predicted else ""
    scalar("merchant.name", g_merchant, p_merchant, lambda a, b: a and a == b)

    # Date
    g_date = normalize_date(gold.header.date.value)
    p_date = normalize_date(predicted.header.date.value) if predicted else ""
    # Treat the sentinel "1970-01-01" as absent.
    if g_date == "1970-01-01":
        g_date = ""
    if p_date == "1970-01-01":
        p_date = ""
    scalar("date.value", g_date, p_date, lambda a, b: a and a == b)

    # Currency
    g_curr = normalize_currency(gold.header.currency)
    p_curr = normalize_currency(predicted.header.currency) if predicted else ""
    scalar("header.currency", g_curr, p_curr, lambda a, b: a and a == b)

    # Total / Subtotal
    for key, gv, pv in [
        ("totals.total", gold.totals.total, predicted.totals.total if predicted else None),
        ("totals.subtotal", gold.totals.subtotal, predicted.totals.subtotal if predicted else None),
    ]:
        gn = normalize_decimal(gv)
        pn = normalize_decimal(pv)
        scalar(key, gn, pn, lambda a, b: a is not None and a == b)

    # Line items — set-match by (normalized description, total price within ±2%)
    out["lineItems"] = score_line_items(gold.lineItems, predicted.lineItems if predicted else [])

    return out


def score_line_items(gold_items: list[LineItem], pred_items: list[LineItem]) -> dict:
    """Greedy bipartite matching on (normalized description, price ±2%)."""

    def key_of(item: LineItem) -> tuple[str, Optional[Decimal]]:
        return (normalize_string(item.description), normalize_decimal(item.totalPrice))

    gold_keys = [key_of(i) for i in gold_items]
    pred_keys = [key_of(i) for i in pred_items]

    matched_pred: set[int] = set()
    matches = 0
    for g in gold_keys:
        gd, gp = g
        for i, p in enumerate(pred_keys):
            if i in matched_pred:
                continue
            pd, pp = p
            if not (gd and pd):
                continue
            if gp is None or pp is None:
                desc_match = gd == pd or (gd in pd) or (pd in gd)
                if desc_match:
                    matched_pred.add(i)
                    matches += 1
                    break
                continue
            price_close = abs(gp - pp) <= max(gp, pp) * Decimal("0.02")
            desc_match = gd == pd or (gd in pd) or (pd in gd)
            if desc_match and price_close:
                matched_pred.add(i)
                matches += 1
                break

    precision = matches / len(pred_items) if pred_items else 0.0
    recall = matches / len(gold_items) if gold_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "correct": matches,
        "gold_count": len(gold_items),
        "pred_count": len(pred_items),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def aggregate_field_scores(per_sample_results: list[dict]) -> dict[str, FieldScore]:
    """Aggregate per-sample field scores into FieldScore totals."""
    if not per_sample_results:
        return {}
    field_keys = [k for k in per_sample_results[0].keys() if k != "lineItems"]
    out: dict[str, FieldScore] = {k: FieldScore() for k in field_keys}
    for sample in per_sample_results:
        for k in field_keys:
            entry = sample.get(k)
            if not entry:
                continue
            s = out[k]
            s.correct += entry["correct"]
            s.present_in_pred += entry["present_in_pred"]
            s.present_in_gold += entry["present_in_gold"]
            s.sample_count += 1
    return out


def aggregate_line_items(per_sample_results: list[dict]) -> LineItemsScore:
    out = LineItemsScore()
    for sample in per_sample_results:
        li = sample.get("lineItems")
        if not li:
            continue
        out.sample_count += 1
        if li["gold_count"] or li["pred_count"]:
            out.samples_with_items += 1
        out.micro_correct += li["correct"]
        out.micro_pred_count += li["pred_count"]
        out.micro_gold_count += li["gold_count"]
        out.per_sample_f1.append(li["f1"])
    return out
