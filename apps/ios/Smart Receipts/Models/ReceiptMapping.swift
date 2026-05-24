import Foundation
import OCRKit

/// Bridges between the canonical `OCRKit.Receipt` schema and the SwiftData `Receipt`
/// model. Keep this file as the single conversion point so schema drift is
/// localized.
enum ReceiptMapping {

    static func makePersistedReceipt(
        from canonical: OCRKit.Receipt,
        capturedAt: Date,
        imageRelativePath: String
    ) throws -> Receipt {
        let payload = try canonical.canonicalJSON()

        let receipt = Receipt(
            id: canonical.imageId,
            capturedAt: capturedAt,
            imageRelativePath: imageRelativePath,
            receiptDate: Self.parseISODate(canonical.header.date.value),
            merchantName: canonical.header.merchant.name,
            total: canonical.totals.total,
            currency: canonical.header.currency,
            parsedPayloadJSON: payload,
            pipelineId: canonical.provenance.pipelineId,
            overallConfidence: canonical.provenance.confidence
        )

        receipt.lineItems = canonical.lineItems.map { item in
            ReceiptLineItem(
                itemDescription: item.description,
                quantity: item.quantity,
                unitPrice: item.unitPrice,
                totalPrice: item.totalPrice,
                category: item.category?.rawValue
            )
        }
        return receipt
    }

    /// Re-decode the stored canonical JSON into an `OCRKit.Receipt`. Used by
    /// the detail view to render provenance/bbox info.
    static func canonical(from receipt: Receipt) -> OCRKit.Receipt? {
        try? OCRKit.Receipt.from(json: receipt.parsedPayloadJSON)
    }

    static func parseISODate(_ s: String) -> Date? {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        return f.date(from: s)
    }

    static func formatISODate(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: d)
    }
}
