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

        // Migrate any per-line-item bboxes off provenance and onto the
        // LineItemDraft itself, so deletes/reorders keep them in sync.
        // Both keys are read: `lineItem.NNN` (description / whole row) and
        // `lineItem.NNN.price` (price column).
        for (idx, item) in lineItems.enumerated() {
            let key = String(format: "lineItem.%03d", idx)
            if let box = r.provenance.bboxes[key] {
                item.bbox = box
            }
            if let box = r.provenance.bboxes["\(key).price"] {
                item.priceBBox = box
            }
        }
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

        // Same migration as the existing-label init: per-line-item bboxes
        // ride on the LineItemDraft so deleting the item drops the box.
        for (idx, item) in lineItems.enumerated() {
            let key = String(format: "lineItem.%03d", idx)
            if let box = receipt.provenance.bboxes[key] {
                item.bbox = box
            }
            if let box = receipt.provenance.bboxes["\(key).price"] {
                item.priceBBox = box
            }
        }
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
    ///
    /// Line-item bboxes are NOT sourced from `basis.provenance.bboxes` —
    /// they live on each `LineItemDraft`. We rebuild both the
    /// `lineItem.NNN` (description) and `lineItem.NNN.price` (price column)
    /// keys from the live array so that deleting an item removes both
    /// bboxes (and surviving items reindex to match their new positions).
    var effectiveBBoxes: [String: OCRKit.Receipt.BBox] {
        var result: [String: OCRKit.Receipt.BBox] = [:]
        for (key, box) in basis.provenance.bboxes where !key.hasPrefix("lineItem.") {
            result[key] = box
        }
        for (key, box) in bboxOverrides {
            result[key] = box
        }
        for (idx, item) in lineItems.enumerated() {
            let key = String(format: "lineItem.%03d", idx)
            if let box = item.bbox {
                result[key] = box
            }
            if let box = item.priceBBox {
                result["\(key).price"] = box
            }
        }
        return result
    }

    /// Public helper: the bbox key the form should select when the user
    /// taps a line item's description / price zone.
    static func lineItemBBoxKey(index: Int, isPrice: Bool = false) -> String {
        let base = String(format: "lineItem.%03d", index)
        return isPrice ? "\(base).price" : base
    }

    /// Parse `"lineItem.NNN"` or `"lineItem.NNN.price"` → (index, isPrice).
    /// Returns nil for non-line-item keys.
    private static func parseLineItemKey(_ key: String) -> (index: Int, isPrice: Bool)? {
        let prefix = "lineItem."
        guard key.hasPrefix(prefix) else { return nil }
        let rest = String(key.dropFirst(prefix.count))
        if rest.hasSuffix(".price") {
            let idxPart = rest.dropLast(".price".count)
            if let i = Int(idxPart) { return (i, true) }
            return nil
        }
        if let i = Int(rest) { return (i, false) }
        return nil
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
    ///
    /// For `lineItem.NNN` / `lineItem.NNN.price` we write to the
    /// LineItemDraft itself so the box stays in sync with deletes/reorders.
    /// Description and price get distinct default positions so they don't
    /// overlap on creation (left-of-center for desc, right-of-center for $).
    func ensureBBox(for key: String) {
        guard effectiveBBoxes[key] == nil else { return }
        if let parsed = LabelDraft.parseLineItemKey(key),
           parsed.index >= 0, parsed.index < lineItems.count {
            let item = lineItems[parsed.index]
            if parsed.isPrice {
                // Narrow box on the right side — price column.
                item.priceBBox = OCRKit.Receipt.BBox(
                    x: 0.70, y: 0.475, width: 0.20, height: 0.04
                )
            } else {
                // Wider box on the left — description column.
                item.bbox = OCRKit.Receipt.BBox(
                    x: 0.10, y: 0.475, width: 0.55, height: 0.04
                )
            }
            return
        }
        // 30% wide × 5% tall, centered. Reasonable for a single text line.
        bboxOverrides[key] = OCRKit.Receipt.BBox(
            x: 0.35, y: 0.475, width: 0.30, height: 0.05
        )
    }

    /// Move a bbox by a normalized delta. Used by drag gestures on the
    /// selected bbox; clamps so the box stays inside [0,1].
    func translateBBox(key: String, by delta: CGSize) {
        if let parsed = LabelDraft.parseLineItemKey(key),
           parsed.index >= 0, parsed.index < lineItems.count {
            let item = lineItems[parsed.index]
            guard var box = parsed.isPrice ? item.priceBBox : item.bbox else { return }
            box.x = max(0, min(1 - box.width, box.x + Double(delta.width)))
            box.y = max(0, min(1 - box.height, box.y + Double(delta.height)))
            if parsed.isPrice {
                item.priceBBox = box
            } else {
                item.bbox = box
            }
            return
        }
        guard var box = effectiveBBoxes[key] else { return }
        box.x = max(0, min(1 - box.width, box.x + Double(delta.width)))
        box.y = max(0, min(1 - box.height, box.y + Double(delta.height)))
        bboxOverrides[key] = box
    }

    /// Resize a bbox by dragging one of its corners. `corner` is 0=TL,
    /// 1=TR, 2=BL, 3=BR. Delta is in normalized image coordinates.
    func resizeBBox(key: String, corner: Int, by delta: CGSize) {
        if let parsed = LabelDraft.parseLineItemKey(key),
           parsed.index >= 0, parsed.index < lineItems.count {
            let item = lineItems[parsed.index]
            guard let current = parsed.isPrice ? item.priceBBox : item.bbox,
                  let next = LabelDraft.applyCornerResize(box: current, corner: corner, delta: delta) else {
                return
            }
            if parsed.isPrice {
                item.priceBBox = next
            } else {
                item.bbox = next
            }
            return
        }
        guard let current = effectiveBBoxes[key],
              let next = LabelDraft.applyCornerResize(box: current, corner: corner, delta: delta) else {
            return
        }
        bboxOverrides[key] = next
    }

    /// Pure helper: given a bbox + a corner index + a normalized drag delta,
    /// return the resized bbox, or nil if the move would produce an invalid
    /// (too small / out-of-bounds) result.
    private static func applyCornerResize(
        box: OCRKit.Receipt.BBox,
        corner: Int,
        delta: CGSize
    ) -> OCRKit.Receipt.BBox? {
        var box = box
        let dx = Double(delta.width)
        let dy = Double(delta.height)
        switch corner {
        case 0: // top-left
            let newX = box.x + dx
            let newY = box.y + dy
            let newW = box.width - dx
            let newH = box.height - dy
            guard newW > 0.01, newH > 0.01, newX >= 0, newY >= 0 else { return nil }
            box.x = newX; box.y = newY; box.width = newW; box.height = newH
        case 1: // top-right
            let newY = box.y + dy
            let newW = box.width + dx
            let newH = box.height - dy
            guard newW > 0.01, newH > 0.01, newY >= 0, box.x + newW <= 1 else { return nil }
            box.y = newY; box.width = newW; box.height = newH
        case 2: // bottom-left
            let newX = box.x + dx
            let newW = box.width - dx
            let newH = box.height + dy
            guard newW > 0.01, newH > 0.01, newX >= 0, box.y + newH <= 1 else { return nil }
            box.x = newX; box.width = newW; box.height = newH
        case 3: // bottom-right
            let newW = box.width + dx
            let newH = box.height + dy
            guard newW > 0.01, newH > 0.01, box.x + newW <= 1, box.y + newH <= 1 else { return nil }
            box.width = newW; box.height = newH
        default: return nil
        }
        return box
    }

    /// Assign an OCR line to a specific line-item bbox slot identified by
    /// its dotted key. Used when the user has selected a `lineItem.NNN`
    /// (description) or `lineItem.NNN.price` slot and then clicks an OCR
    /// line. We record the bbox AND, as a convenience, fill the matching
    /// text/price field if it's currently empty.
    func assignLine(_ line: OCRKit.OCRLine, toLineItemKey key: String) {
        guard let parsed = LabelDraft.parseLineItemKey(key),
              parsed.index >= 0, parsed.index < lineItems.count else { return }
        let item = lineItems[parsed.index]
        if parsed.isPrice {
            item.priceBBox = line.box
            // Convenience: parse a price from the line text. Only overwrite
            // if the existing total is 0 — don't clobber user edits.
            if let amount = LabelDraft.parseAmount(line.text), item.totalPrice == 0 {
                item.totalPrice = amount
            }
        } else {
            item.bbox = line.box
            // Convenience: try splitting `desc … $price` and fill in both
            // fields if the description is currently empty.
            if item.itemDescription.trimmingCharacters(in: .whitespaces).isEmpty {
                let (desc, price) = LabelDraft.splitDescriptionAndPrice(line.text)
                item.itemDescription = desc
                if let price, item.totalPrice == 0 {
                    item.totalPrice = price
                }
            }
        }
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
            // as a new line item with its bbox riding on the draft itself.
            let (desc, price) = LabelDraft.splitDescriptionAndPrice(line.text)
            let item = LineItemDraft(from: OCRKit.Receipt.LineItem(
                description: desc,
                quantity: nil,
                unitPrice: nil,
                totalPrice: price ?? 0,
                category: nil
            ))
            item.bbox = line.box
            lineItems.append(item)
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
    /// Description / whole-row bbox attached to this specific line item.
    /// Lives with the LineItemDraft (rather than in a separate
    /// `lineItem.NNN`-keyed dictionary) so that deleting the item also
    /// drops its bbox, and reordering re-keys automatically.
    var bbox: OCRKit.Receipt.BBox?
    /// Separate bbox for the price column. Optional and independent — a
    /// line item can have a description bbox but no price bbox (or
    /// vice-versa) until the user finishes annotating.
    var priceBBox: OCRKit.Receipt.BBox?

    init(from item: OCRKit.Receipt.LineItem) {
        self.itemDescription = item.description
        self.quantity = item.quantity
        self.unitPrice = item.unitPrice
        self.totalPrice = item.totalPrice
        self.category = item.category
        self.bbox = nil
        self.priceBBox = nil
    }

    init(blank: ()) {
        self.itemDescription = ""
        self.quantity = nil
        self.unitPrice = nil
        self.totalPrice = 0
        self.category = nil
        self.bbox = nil
        self.priceBBox = nil
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
