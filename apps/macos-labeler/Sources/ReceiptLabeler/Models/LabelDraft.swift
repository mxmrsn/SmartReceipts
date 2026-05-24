import Foundation
import OCRKit
import Observation

/// Editable in-memory representation of a label being authored in the
/// LabelingView. Pre-populated from either an existing label document or a
/// fresh OCRKit draft, then folded back into a canonical `OCRKit.Receipt` on
/// save.
@Observable
@MainActor
final class LabelDraft {

    var merchantName: String
    var receiptDate: Date
    var total: Decimal
    var currency: String
    var lineItems: [LineItemDraft]
    var status: LabelStatus
    var notes: String

    /// Carried through so a Save preserves provenance/bboxes from the source.
    private let basisReceipt: OCRKit.Receipt
    private let labelExisting: LabelDocument.LabelMetadata?

    init(from document: LabelDocument) {
        let r = document.receipt
        self.basisReceipt = r
        self.labelExisting = document.label
        self.merchantName = r.header.merchant.name
        self.receiptDate = LabelDraft.parseISODate(r.header.date.value) ?? Date()
        self.total = r.totals.total
        self.currency = r.header.currency
        self.lineItems = r.lineItems.map { LineItemDraft(from: $0) }
        self.status = document.label.status
        self.notes = document.label.notes ?? ""
    }

    /// Create a fresh draft (status = draft) from a pipeline output.
    convenience init(fromPreLabel receipt: OCRKit.Receipt, sourceFilename: String, pipelineId: String) {
        let meta = LabelDocument.LabelMetadata(
            status: .draft,
            sourceFilename: sourceFilename,
            labeler: nil,
            verifiedAt: nil,
            sourcePipeline: pipelineId,
            notes: nil
        )
        self.init(from: LabelDocument(receipt: receipt, label: meta))
    }

    var isSavable: Bool {
        !merchantName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    func snapshot(asStatus newStatus: LabelStatus, labeler: String?) -> LabelDocument {
        var receipt = basisReceipt
        receipt.header.merchant.name = merchantName
        receipt.header.date = OCRKit.Receipt.ReceiptDate(value: LabelDraft.formatISODate(receiptDate))
        receipt.header.currency = currency
        receipt.totals.total = total
        receipt.lineItems = lineItems.map { $0.asCanonical }

        let metadata = LabelDocument.LabelMetadata(
            status: newStatus,
            sourceFilename: labelExisting?.sourceFilename,
            labeler: labeler ?? labelExisting?.labeler,
            verifiedAt: newStatus == .verified ? Date() : labelExisting?.verifiedAt,
            sourcePipeline: labelExisting?.sourcePipeline ?? receipt.provenance.pipelineId,
            notes: notes.isEmpty ? nil : notes
        )
        return LabelDocument(receipt: receipt, label: metadata)
    }

    private static func parseISODate(_ s: String) -> Date? {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        return f.date(from: s)
    }

    private static func formatISODate(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: d)
    }
}

@Observable
@MainActor
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

    init(blank: ()) {
        self.itemDescription = ""
        self.quantity = nil
        self.unitPrice = nil
        self.totalPrice = 0
        self.category = nil
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
