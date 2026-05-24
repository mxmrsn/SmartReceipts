import Foundation
import SwiftData

/// Persisted receipt. Mirrors the columnar projection of `OCRKit.Receipt` for
/// fast queries (Library list, Dashboard) and keeps the full canonical JSON
/// payload alongside so the model can evolve without losing detail.
@Model
final class Receipt {
    @Attribute(.unique) var id: UUID
    var capturedAt: Date
    /// Path relative to the app's Documents/ directory, e.g. "receipts/{uuid}.jpg".
    var imageRelativePath: String
    var receiptDate: Date?
    var merchantName: String?
    var total: Decimal?
    var currency: String
    /// Full canonical Receipt JSON (matches shared/schema/receipt.schema.json).
    var parsedPayloadJSON: Data
    var pipelineId: String
    var overallConfidence: Double
    @Relationship(deleteRule: .cascade, inverse: \ReceiptLineItem.receipt)
    var lineItems: [ReceiptLineItem]

    init(
        id: UUID,
        capturedAt: Date,
        imageRelativePath: String,
        receiptDate: Date?,
        merchantName: String?,
        total: Decimal?,
        currency: String,
        parsedPayloadJSON: Data,
        pipelineId: String,
        overallConfidence: Double,
        lineItems: [ReceiptLineItem] = []
    ) {
        self.id = id
        self.capturedAt = capturedAt
        self.imageRelativePath = imageRelativePath
        self.receiptDate = receiptDate
        self.merchantName = merchantName
        self.total = total
        self.currency = currency
        self.parsedPayloadJSON = parsedPayloadJSON
        self.pipelineId = pipelineId
        self.overallConfidence = overallConfidence
        self.lineItems = lineItems
    }
}

@Model
final class ReceiptLineItem {
    var itemDescription: String
    var quantity: Decimal?
    var unitPrice: Decimal?
    var totalPrice: Decimal
    /// Stored as raw String of the canonical category enum (e.g. "Food").
    var category: String?
    var receipt: Receipt?

    init(
        itemDescription: String,
        quantity: Decimal? = nil,
        unitPrice: Decimal? = nil,
        totalPrice: Decimal,
        category: String? = nil
    ) {
        self.itemDescription = itemDescription
        self.quantity = quantity
        self.unitPrice = unitPrice
        self.totalPrice = totalPrice
        self.category = category
    }
}
