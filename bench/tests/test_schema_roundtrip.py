"""M1 verification: a canonical sample JSON round-trips through both the Swift
and Python schema mirrors and produces byte-identical canonical output.

The Swift round-trip is delegated to `tools/ocr-cli` via a future
`--roundtrip-stdin` subcommand; for M1 we verify Python parity against a
hand-checked fixture JSON.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from bench.schema import (
    BBox,
    Category,
    Header,
    LineItem,
    Merchant,
    Payment,
    PaymentMethod,
    Provenance,
    Receipt,
    ReceiptDate,
    TaxLine,
    Totals,
)


def _fixture() -> Receipt:
    return Receipt(
        imageId=UUID("11111111-2222-3333-4444-555555555555"),
        header=Header(
            merchant=Merchant(name="Joe's Coffee", address="123 Main St", phone="555-1234"),
            date=ReceiptDate(value="2026-05-24", time="09:14"),
            transactionId="TX-001",
            currency="USD",
        ),
        lineItems=[
            LineItem(description="Latte", quantity=Decimal("1"), unitPrice=Decimal("4.50"), totalPrice=Decimal("4.50"), category=Category.food),
            LineItem(description="Croissant", quantity=Decimal("2"), unitPrice=Decimal("3.25"), totalPrice=Decimal("6.50"), category=Category.food),
        ],
        totals=Totals(
            subtotal=Decimal("11.00"),
            tax=[TaxLine(label="Sales Tax", rate=Decimal("0.0875"), amount=Decimal("0.96"))],
            tip=Decimal("2.00"),
            total=Decimal("13.96"),
        ),
        payment=Payment(method=PaymentMethod.card, cardLast4="1234"),
        provenance=Provenance(
            pipelineId="vision-regex",
            modelVersion="vision-accurate.1",
            confidence=0.42,
            fieldConfidence={"merchant.name": 0.5, "totals.total": 0.6},
            bboxes={"merchant.name": BBox(x=0.1, y=0.05, width=0.8, height=0.05)},
        ),
    )


def test_receipt_roundtrip_python():
    r = _fixture()
    data = r.model_dump(mode="json")
    r2 = Receipt.model_validate(data)
    assert r2 == r


def test_canonical_json_stable():
    r = _fixture()
    a = json.dumps(r.model_dump(mode="json"), sort_keys=True)
    b = json.dumps(Receipt.model_validate(json.loads(a)).model_dump(mode="json"), sort_keys=True)
    assert a == b


def test_schema_file_exists():
    # tests/ lives at bench/tests/ — parents[2] is repo root.
    repo_root = Path(__file__).resolve().parents[2]
    schema_file = repo_root / "shared" / "schema" / "receipt.schema.json"
    assert schema_file.exists(), "Canonical JSON schema missing"
    raw = json.loads(schema_file.read_text())
    assert raw["title"] == "Receipt"
    assert "imageId" in raw["properties"]
