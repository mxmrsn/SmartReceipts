"""Python mirror of `shared/schema/receipt.schema.json`.

Hand-maintained against the JSON Schema. The round-trip test in
`bench/tests/test_schema_roundtrip.py` catches drift. When this module is
regenerated via datamodel-code-generator later, the JSON file becomes the
unambiguous source of truth.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    populate_by_name=True,
    use_enum_values=True,
)


class Category(str, Enum):
    food = "Food"
    fuel = "Fuel"
    groceries = "Groceries"
    office = "Office"
    transport = "Transport"
    lodging = "Lodging"
    entertainment = "Entertainment"
    health = "Health"
    other = "Other"


class PaymentMethod(str, Enum):
    cash = "cash"
    card = "card"
    check = "check"
    other = "other"


class Merchant(BaseModel):
    model_config = _MODEL_CONFIG

    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    taxId: Optional[str] = None


class ReceiptDate(BaseModel):
    model_config = _MODEL_CONFIG

    value: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: Optional[str] = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class Header(BaseModel):
    model_config = _MODEL_CONFIG

    merchant: Merchant
    date: ReceiptDate
    transactionId: Optional[str] = None
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


class LineItem(BaseModel):
    model_config = _MODEL_CONFIG

    description: str
    quantity: Optional[Decimal] = None
    unitPrice: Optional[Decimal] = None
    totalPrice: Decimal
    category: Optional[Category] = None


class TaxLine(BaseModel):
    model_config = _MODEL_CONFIG

    label: str
    rate: Optional[Decimal] = None
    amount: Decimal


class Totals(BaseModel):
    model_config = _MODEL_CONFIG

    subtotal: Optional[Decimal] = None
    tax: list[TaxLine] = Field(default_factory=list)
    discount: Optional[Decimal] = None
    tip: Optional[Decimal] = None
    serviceCharge: Optional[Decimal] = None
    total: Decimal


class Payment(BaseModel):
    model_config = _MODEL_CONFIG

    method: Optional[PaymentMethod] = None
    cardLast4: Optional[str] = Field(default=None, pattern=r"^\d{4}$")


class BBox(BaseModel):
    model_config = _MODEL_CONFIG

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class Provenance(BaseModel):
    model_config = _MODEL_CONFIG

    pipelineId: str
    modelVersion: str
    confidence: float = Field(ge=0, le=1)
    fieldConfidence: dict[str, float] = Field(default_factory=dict)
    bboxes: dict[str, BBox] = Field(default_factory=dict)


class Receipt(BaseModel):
    model_config = _MODEL_CONFIG

    imageId: UUID
    header: Header
    lineItems: list[LineItem] = Field(default_factory=list)
    totals: Totals
    payment: Optional[Payment] = None
    provenance: Provenance


class ExtractionResult(BaseModel):
    """Output of any pipeline run via `tools/ocr-cli`."""

    model_config = _MODEL_CONFIG

    receipt: Receipt
    latencyMs: int
    peakMemoryMB: Optional[int] = None
    rawText: Optional[str] = None
