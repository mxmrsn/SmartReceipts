import Foundation
import OCRKit
import Observation

/// Editable in-memory representation of an extracted receipt. Pre-filled from
/// `OCRKit.Receipt`, mutated by `ReviewView`, then folded back into the
/// canonical schema on save via `applying(to:newID:)`.
@Observable
final class ReceiptDraft {
    var merchantName: String
    var receiptDate: Date
    var total: Decimal
    var currency: String
    var lineItems: [LineItemDraft]
    let fieldConfidence: [String: Double]

    init(from canonical: OCRKit.Receipt) {
        self.merchantName = canonical.header.merchant.name
        self.receiptDate = ReceiptMapping.parseISODate(canonical.header.date.value) ?? Date()
        self.total = canonical.totals.total
        self.currency = canonical.header.currency
        self.lineItems = canonical.lineItems.map { LineItemDraft(from: $0) }
        self.fieldConfidence = canonical.provenance.fieldConfidence
    }

    var isSavable: Bool {
        !merchantName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Produce a new canonical Receipt by overlaying the edited values onto
    /// `base`. Provenance, bboxes, and any field we did not surface for edit
    /// are preserved from `base`.
    func applying(to base: OCRKit.Receipt, newID: UUID) -> OCRKit.Receipt {
        var copy = base
        copy.imageId = newID
        copy.header.merchant.name = merchantName
        copy.header.date = OCRKit.Receipt.ReceiptDate(value: ReceiptMapping.formatISODate(receiptDate))
        copy.header.currency = currency
        copy.totals.total = total
        copy.lineItems = lineItems.map { $0.asCanonical }
        return copy
    }
}

@Observable
final class LineItemDraft: Identifiable {
    let id = UUID()
    var itemDescription: String
    var quantity: Decimal?
    var unitPrice: Decimal?
    var totalPrice: Decimal
    var category: OCRKit.Receipt.Category?

    init(from item: OCRKit.Receipt.LineItem) {
        self.itemDescription = item.description
        self.quantity = item.quantity
        self.unitPrice = item.unitPrice
        self.totalPrice = item.totalPrice
        self.category = item.category
    }

    var asCanonical: OCRKit.Receipt.LineItem {
        OCRKit.Receipt.LineItem(
            description: itemDescription,
            quantity: quantity,
            unitPrice: unitPrice,
            totalPrice: totalPrice,
            category: category
        )
    }
}
