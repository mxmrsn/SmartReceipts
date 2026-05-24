"""Lint passes over `dataset/labels/*.json` to surface obvious data-quality
issues — the kind that quietly poison a benchmark if you don't catch them.

Runs without changing the labels on disk. Output is a punch list of
(image_id, rule_id, severity, message). Severities are 'error' (label is
unusable / clearly wrong) or 'warning' (probably wrong, but worth a human
glance).

Run with:
    bench lint            # all labels
    bench lint --status verified
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

LABELS_DIR = Path(__file__).resolve().parents[3] / "dataset" / "labels"


# Description prefixes that mean "this is a payment / totals footer row,
# not a real purchased line item". The FM pipeline has the same denylist
# at extraction time, but stale drafts produced before it was wired in can
# still carry these rows, so lint should also catch them after the fact.
_LINE_ITEM_FOOTER_PREFIXES = (
    "balance", "balance due",
    "credit", "credit card", "debit", "card", "visa", "mastercard", "mc ",
    "amex", "american express", "discover",
    "change", "cash", "tender", "tendered",
    "subtotal", "sub total", "sub-total",
    "tax", "sales tax", "vat", "gst", "hst",
    "tip", "gratuity",
    "total", "grand total", "amount due", "amount paid", "amount payable",
    "auth", "approval", "approved",
    "payment", "paid",
    "discount", "savings", "you saved", "coupon",
    "cashback", "cash back",
    "rounding",
    "loyalty", "rewards", "points",
    "invoice", "receipt", "transaction",
)

_DATE_PATTERN = re.compile(r"^(19|20)\d{2}-\d{2}-\d{2}$")
_BOGUS_DATE = "1970-01-01"   # FM's sentinel for "no date extracted"


@dataclass
class LintIssue:
    image_id: str
    rule: str
    severity: Literal["error", "warning"]
    message: str


def lint_label_file(path: Path) -> list[LintIssue]:
    """Run all lint rules over a single label JSON file."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [LintIssue(
            image_id=path.stem, rule="invalid-json", severity="error",
            message=f"label file is not valid JSON: {e}"
        )]

    image_id = data.get("imageId", path.stem)
    issues: list[LintIssue] = []

    # ---- date ----
    date_value = (
        data.get("header", {}).get("date", {}).get("value", "")
    )
    if not _DATE_PATTERN.fullmatch(date_value):
        issues.append(LintIssue(
            image_id=image_id, rule="date-format", severity="error",
            message=f"date.value '{date_value}' is not YYYY-MM-DD with year in 19xx/20xx",
        ))
    elif date_value == _BOGUS_DATE:
        issues.append(LintIssue(
            image_id=image_id, rule="date-sentinel", severity="warning",
            message=f"date.value is the 'no date' sentinel ({_BOGUS_DATE}); fill it in",
        ))

    # ---- merchant ----
    merchant = data.get("header", {}).get("merchant", {}).get("name", "").strip()
    if not merchant or merchant.lower() in {"unknown", "..."}:
        issues.append(LintIssue(
            image_id=image_id, rule="merchant-empty", severity="warning",
            message=f"merchant.name is empty or placeholder ('{merchant}')",
        ))

    # ---- totals ----
    total = data.get("totals", {}).get("total")
    if total in (None, 0, 0.0):
        issues.append(LintIssue(
            image_id=image_id, rule="total-zero", severity="error",
            message="totals.total is missing or zero — a receipt always has a total",
        ))

    # ---- line items: footer rows ----
    for idx, item in enumerate(data.get("lineItems", []) or []):
        desc = str(item.get("description", "")).strip().lower()
        if not desc:
            continue
        for prefix in _LINE_ITEM_FOOTER_PREFIXES:
            if desc == prefix or desc.startswith(prefix + " ") \
               or desc.startswith(prefix + ":") or desc.startswith(prefix + "\t"):
                issues.append(LintIssue(
                    image_id=image_id, rule="footer-as-line-item", severity="error",
                    message=f"lineItems[{idx}] '{item.get('description')}' is a payment/totals footer row",
                ))
                break

    # ---- line items: zero total ----
    for idx, item in enumerate(data.get("lineItems", []) or []):
        tp = item.get("totalPrice")
        if tp in (None, 0, 0.0):
            issues.append(LintIssue(
                image_id=image_id, rule="line-item-zero-price", severity="warning",
                message=f"lineItems[{idx}] '{item.get('description')}' has totalPrice = 0",
            ))

    return issues


def lint_all(status: str | None = None) -> list[LintIssue]:
    """Lint every label file. Filter by `_label.status` if given."""
    issues: list[LintIssue] = []
    for path in sorted(LABELS_DIR.glob("*.json")):
        if status:
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("_label", {}).get("status") != status:
                continue
        issues.extend(lint_label_file(path))
    return issues


def summarize(issues: Iterable[LintIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.rule] = counts.get(issue.rule, 0) + 1
    return counts
