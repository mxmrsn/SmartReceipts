import Foundation
import OCRKit
import Observation

/// Editable in-memory representation of a label being authored.
///
/// Two birth paths:
/// 1. `.existingLabel`  — we loaded a verified/draft label off disk
/// 2. `.preLabel(rawText:)` — we just ran a pipeline; the form should advertise
///    every field as "Auto" until the user touches it, with the raw OCR text
///    available alongside for cross-checking.
@Observable
@MainActor
final class LabelDraft {

    enum Source: Sendable {
        case existingLabel
        case preLabel(rawText: String?)
    }

    // Editable
    var merchantName: String
    var receiptDate: Date
    var total: Decimal
    var currency: String
    var lineItems: [LineItemDraft]
    var status: LabelStatus
    var notes: String

    // Immutable context
    let source: Source
    let basis: OCRKit.Receipt
    /// Where the date came from. "ocr" = found in receipt text; "exif"/"file"
    /// = pulled from image metadata when OCR found none; nil = unknown.
    let dateSource: String?
    private let labelExisting: LabelDocument.LabelMetadata?

    // MARK: - Init

    init(from document: LabelDocument) {
        let r = document.receipt
        self.basis = r
        self.labelExisting = document.label
        self.source = .existingLabel
        self.dateSource = nil
        self.merchantName = r.header.merchant.name
        self.receiptDate = LabelDraft.parseISODate(r.header.date.value) ?? Date()
        self.total = r.totals.total
        self.currency = r.header.currency
        self.lineItems = r.lineItems.map { LineItemDraft(from: $0) }
        self.status = document.label.status
        self.notes = document.label.notes ?? ""
    }

    init(
        fromPreLabel receipt: OCRKit.Receipt,
        sourceFilename: String,
        pipelineId: String,
        rawText: String?,
        dateSource: String? = nil
    ) {
        let metadata = LabelDocument.LabelMetadata(
            status: .draft,
            sourceFilename: sourceFilename,
            labeler: nil,
            verifiedAt: nil,
            sourcePipeline: pipelineId,
            notes: nil
        )
        self.basis = receipt
        self.labelExisting = metadata
        self.source = .preLabel(rawText: rawText)
        self.dateSource = dateSource
        self.merchantName = receipt.header.merchant.name
        self.receiptDate = LabelDraft.parseISODate(receipt.header.date.value) ?? Date()
        self.total = receipt.totals.total
        self.currency = receipt.header.currency
        self.lineItems = receipt.lineItems.map { LineItemDraft(from: $0) }
        self.status = .draft
        self.notes = ""
    }

    // MARK: - Derived

    var rawText: String? {
        if case .preLabel(let text) = source { return text }
        return nil
    }

    var isPreLabel: Bool {
        if case .preLabel = source { return true }
        return false
    }

    var pipelineId: String { basis.provenance.pipelineId }

    var isSavable: Bool {
        !merchantName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    // Per-field "did the user touch this field" checks. Only meaningful when
    // source is .preLabel — for an existing label everything is technically
    // editable but there's no "auto" baseline to compare against.
    var merchantWasEdited: Bool { merchantName != basis.header.merchant.name }
    var dateWasEdited: Bool { LabelDraft.formatISODate(receiptDate) != basis.header.date.value }
    var totalWasEdited: Bool { total != basis.totals.total }
    var currencyWasEdited: Bool { currency != basis.header.currency }
    var lineItemsWereEdited: Bool {
        if lineItems.count != basis.lineItems.count { return true }
        for (draft, original) in zip(lineItems, basis.lineItems) {
            if draft.itemDescription != original.description
                || draft.quantity != original.quantity
                || draft.unitPrice != original.unitPrice
                || draft.totalPrice != original.totalPrice {
                return true
            }
        }
        return false
    }

    // MARK: - Save

    func snapshot(asStatus newStatus: LabelStatus, labeler: String?) -> LabelDocument {
        var receipt = basis
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

    // MARK: - Date helpers

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
