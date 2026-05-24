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
    /// All individual OCR observations from the pipeline. Used by the
    /// BoundingBoxOverlay to render every detection and let the user
    /// click-to-(re)assign lines to fields.
    let ocrLines: [OCRKit.OCRLine]
    /// True when our preferred pipeline failed and we used the fallback.
    /// Used by the banner to show a warning.
    let preferredPipelineFailed: Bool
    let preferredPipelineError: String?

    /// User overrides to the bboxes coming from the pipeline. When the user
    /// click-assigns an OCR line to a field, we record the line's bbox here.
    /// Merged into `effectiveBBoxes` for the overlay + into provenance on save.
    var bboxOverrides: [String: OCRKit.Receipt.BBox] = [:]

    /// Currently-selected bbox key (e.g. "merchant.name", "totals.total").
    /// Driven from both sides — tapping a form row sets it, tapping a bbox
    /// sets it — and powers the selection ring + drag-to-move affordance
    /// on the overlay.
    var selectedBBoxKey: String? = nil

    private let labelExisting: LabelDocument.LabelMetadata?

    // MARK: - Init

    init(from document: LabelDocument) {
        let r = document.receipt
        self.basis = r
        self.labelExisting = document.label
        self.source = .existingLabel
        self.dateSource = nil
        self.ocrLines = []
        self.preferredPipelineFailed = false
        self.preferredPipelineError = nil
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
        dateSource: String? = nil,
        ocrLines: [OCRKit.OCRLine] = [],
        preferredPipelineFailed: Bool = false,
        preferredPipelineError: String? = nil
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
        self.ocrLines = ocrLines
        self.preferredPipelineFailed = preferredPipelineFailed
        self.preferredPipelineError = preferredPipelineError
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

    // MARK: - BBox overrides + click-assignment

    /// Pipeline bboxes merged with user overrides. Overlay reads from here so
    /// the moment the user click-assigns a line to a field, the overlay
    /// reflects it.
    var effectiveBBoxes: [String: OCRKit.Receipt.BBox] {
        basis.provenance.bboxes.merging(bboxOverrides) { _, override in override }
    }

    enum FieldTarget: String, Sendable, CaseIterable {
        case merchant
        case date
        case total
        case subtotal
        case lineItem

        /// The bbox key this target writes to (for non-lineItem cases).
        var bboxKey: String? {
            switch self {
            case .merchant: return "merchant.name"
            case .date:     return "date.value"
            case .total:    return "totals.total"
            case .subtotal: return "totals.subtotal"
            case .lineItem: return nil  // dynamic index, computed at assignment time
            }
        }
    }

    /// If the field has no bbox yet (commonly because the value came from
    /// EXIF metadata rather than the receipt text), create a small default
    /// box at the image center. The user can then drag/resize it into the
    /// right place using the overlay handles.
    func ensureBBox(for key: String) {
        guard effectiveBBoxes[key] == nil else { return }
        // 30% wide × 5% tall, centered. Reasonable for a single text line.
        bboxOverrides[key] = OCRKit.Receipt.BBox(
            x: 0.35, y: 0.475, width: 0.30, height: 0.05
        )
    }

    /// Move a bbox by a normalized delta. Used by drag gestures on the
    /// selected bbox; clamps so the box stays inside [0,1].
    func translateBBox(key: String, by delta: CGSize) {
        guard var box = effectiveBBoxes[key] else { return }
        let newX = max(0, min(1 - box.width, box.x + Double(delta.width)))
        let newY = max(0, min(1 - box.height, box.y + Double(delta.height)))
        box.x = newX
        box.y = newY
        bboxOverrides[key] = box
    }

    /// Resize a bbox by dragging one of its corners. `corner` is 0=TL,
    /// 1=TR, 2=BL, 3=BR. Delta is in normalized image coordinates.
    func resizeBBox(key: String, corner: Int, by delta: CGSize) {
        guard var box = effectiveBBoxes[key] else { return }
        let dx = Double(delta.width)
        let dy = Double(delta.height)
        switch corner {
        case 0: // top-left: position grows, size shrinks
            let newX = box.x + dx
            let newY = box.y + dy
            let newW = box.width - dx
            let newH = box.height - dy
            if newW > 0.01, newH > 0.01, newX >= 0, newY >= 0 {
                box.x = newX; box.y = newY; box.width = newW; box.height = newH
            }
        case 1: // top-right: y grows, height shrinks, width grows
            let newY = box.y + dy
            let newW = box.width + dx
            let newH = box.height - dy
            if newW > 0.01, newH > 0.01, newY >= 0, box.x + newW <= 1 {
                box.y = newY; box.width = newW; box.height = newH
            }
        case 2: // bottom-left: x grows, width shrinks, height grows
            let newX = box.x + dx
            let newW = box.width - dx
            let newH = box.height + dy
            if newW > 0.01, newH > 0.01, newX >= 0, box.y + newH <= 1 {
                box.x = newX; box.width = newW; box.height = newH
            }
        case 3: // bottom-right: width + height grow
            let newW = box.width + dx
            let newH = box.height + dy
            if newW > 0.01, newH > 0.01, box.x + newW <= 1, box.y + newH <= 1 {
                box.width = newW; box.height = newH
            }
        default: return
        }
        bboxOverrides[key] = box
    }

    /// Assign an OCR line's text + bbox to a field. The form value updates
    /// immediately and the bbox override is recorded so the overlay highlight
    /// jumps to the clicked line.
    func assign(line: OCRKit.OCRLine, to field: FieldTarget) {
        switch field {
        case .merchant:
            merchantName = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            bboxOverrides["merchant.name"] = line.box
        case .date:
            if let parsed = LabelDraft.parseLooseDate(line.text) {
                receiptDate = parsed
            }
            bboxOverrides["date.value"] = line.box
        case .total:
            if let amount = LabelDraft.parseAmount(line.text) {
                total = amount
            }
            bboxOverrides["totals.total"] = line.box
        case .subtotal:
            bboxOverrides["totals.subtotal"] = line.box
        case .lineItem:
            // Best-effort: extract description + price from one line, append
            // as a new line item.
            let (desc, price) = LabelDraft.splitDescriptionAndPrice(line.text)
            let item = LineItemDraft(from: OCRKit.Receipt.LineItem(
                description: desc,
                quantity: nil,
                unitPrice: nil,
                totalPrice: price ?? 0,
                category: nil
            ))
            lineItems.append(item)
            // Index it under lineItem.NNN
            let idx = lineItems.count - 1
            bboxOverrides["lineItem.\(String(format: "%03d", idx))"] = line.box
        }
    }

    // MARK: - Save

    func snapshot(asStatus newStatus: LabelStatus, labeler: String?) -> LabelDocument {
        var receipt = basis
        receipt.header.merchant.name = merchantName
        receipt.header.date = OCRKit.Receipt.ReceiptDate(value: LabelDraft.formatISODate(receiptDate))
        receipt.header.currency = currency
        receipt.totals.total = total
        receipt.lineItems = lineItems.map { $0.asCanonical }
        receipt.provenance.bboxes = effectiveBBoxes

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

    // MARK: - Parsing helpers (used by click-assign)

    private static func parseLooseDate(_ raw: String) -> Date? {
        let formats = [
            "yyyy-MM-dd", "yyyy/MM/dd",
            "MM/dd/yyyy", "M/d/yyyy", "M/d/yy", "MM/dd/yy",
            "MM-dd-yyyy", "M-d-yyyy", "M-d-yy",
            "dd/MM/yyyy", "dd-MM-yyyy",
            "MMM d, yyyy", "MMM dd, yyyy",
            "MMMM d, yyyy",
            "d MMM yyyy", "dd MMM yyyy"
        ]
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // First, try the whole text. Then try matching a substring containing a date.
        for fmt in formats {
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = fmt
            df.timeZone = TimeZone(identifier: "UTC")
            if let d = df.date(from: trimmed) { return d }
        }
        // Extract any date-shaped substring and retry
        if let r = trimmed.range(of: #"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}"#, options: .regularExpression) {
            return parseLooseDate(String(trimmed[r]))
        }
        return nil
    }

    private static func parseAmount(_ raw: String) -> Decimal? {
        guard let r = raw.range(of: #"-?\$?\s*\d{1,5}(?:,\d{3})*\.\d{2}"#, options: .regularExpression) else { return nil }
        let cleaned = String(raw[r])
            .replacingOccurrences(of: "$", with: "")
            .replacingOccurrences(of: ",", with: "")
            .trimmingCharacters(in: .whitespaces)
        return Decimal(string: cleaned, locale: Locale(identifier: "en_US_POSIX"))
    }

    private static func splitDescriptionAndPrice(_ raw: String) -> (description: String, price: Decimal?) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let priceRange = trimmed.range(of: #"-?\$?\s*\d{1,5}(?:,\d{3})*\.\d{2}\s*$"#, options: .regularExpression) else {
            return (trimmed, nil)
        }
        let price = parseAmount(String(trimmed[priceRange]))
        let desc = String(trimmed[trimmed.startIndex..<priceRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (desc.isEmpty ? trimmed : desc, price)
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
