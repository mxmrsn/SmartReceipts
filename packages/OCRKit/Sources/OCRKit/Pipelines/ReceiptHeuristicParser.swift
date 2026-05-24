import Foundation

/// Heuristic parser used by `VisionOnlyPipeline`.
///
/// This is the baseline pipeline's structured-extraction step — the OCRKit
/// "floor" against which smarter pipelines (Foundation Models, MLX VLMs) get
/// benchmarked. We spend a bit more effort here than a pure dumb regex so
/// the manual-labeling workflow has reasonable pre-labels to start from.
///
/// Strategy:
/// 1. Sort lines top-to-bottom by Y (input may be in observation order).
/// 2. Locate spatial anchors — date line, subtotal/total lines, tax lines.
/// 3. Merchant = best-scoring candidate in the top ~8 lines.
/// 4. Line items = lines between the date and the first subtotal/total that
///    end in a money amount, minus known footer keywords.
enum ReceiptHeuristicParser {

    struct Output {
        var header: Receipt.Header
        var lineItems: [Receipt.LineItem]
        var totals: Receipt.Totals
        var payment: Receipt.Payment?
        var overallConfidence: Double
        var fieldConfidence: [String: Double]
        var bboxes: [String: Receipt.BBox]
    }

    typealias Line = (text: String, box: Receipt.BBox)

    // MARK: - Entry

    static func parse(lines unsorted: [Line]) -> Output {
        let lines = unsorted.sorted { $0.box.y < $1.box.y }

        var bboxes: [String: Receipt.BBox] = [:]
        var fieldConfidence: [String: Double] = [:]

        // Anchors first — drive line-item boundaries and totals.
        let subtotalAnchor = findKeywordAmount(
            lines: lines,
            anyOf: ["subtotal", "sub total", "sub-total"],
            excluding: []
        )
        let totalAnchor = findKeywordAmount(
            lines: lines,
            anyOf: ["grand total", "total due", "balance due", "amount due",
                    "amount payable", "total amount", "total", "balance", "amount"],
            excluding: ["sub", "subtotal", "ex tax", "ex. tax", "before tax"]
        )
        let taxAnchors = findTaxLines(lines: lines)
        let tipAnchor = findKeywordAmount(lines: lines, anyOf: ["tip", "gratuity"], excluding: [])
        let discountAnchor = findKeywordAmount(lines: lines, anyOf: ["discount", "savings"], excluding: ["you saved"])

        // Currency
        let currency = detectCurrency(lines: lines)

        // Merchant
        let merchant = findMerchant(lines: lines)
        if let m = merchant {
            bboxes["merchant.name"] = m.box
            fieldConfidence["merchant.name"] = m.confidence
        }

        // Date
        let dateHit = findDate(lines: lines)
        if let d = dateHit {
            bboxes["date.value"] = d.box
            fieldConfidence["date.value"] = d.confidence
        }

        // Total + subtotal bboxes/confidence
        if let t = totalAnchor {
            bboxes["totals.total"] = t.box
            fieldConfidence["totals.total"] = t.confidence
        }
        if let s = subtotalAnchor {
            bboxes["totals.subtotal"] = s.box
            fieldConfidence["totals.subtotal"] = s.confidence
        }

        // Line items live between the header (date, fall back to top) and the
        // first footer anchor (subtotal/tax/total, whichever comes first).
        let itemStart = (dateHit?.lineIndex ?? -1) + 1
        let footerIndices: [Int] = [
            subtotalAnchor?.lineIndex,
            taxAnchors.first?.lineIndex,
            totalAnchor?.lineIndex
        ].compactMap { $0 }
        let itemEnd = footerIndices.min() ?? lines.count
        let items = extractLineItems(lines: lines, from: itemStart, upTo: itemEnd)

        // Build canonical output
        let header = Receipt.Header(
            merchant: Receipt.Merchant(name: merchant?.text ?? "Unknown"),
            date: Receipt.ReceiptDate(value: dateHit?.text ?? "1970-01-01"),
            currency: currency
        )

        let totals = Receipt.Totals(
            subtotal: subtotalAnchor?.value,
            tax: taxAnchors.map {
                Receipt.TaxLine(label: $0.label, rate: $0.rate, amount: $0.value)
            },
            discount: discountAnchor?.value,
            tip: tipAnchor?.value,
            serviceCharge: nil,
            total: totalAnchor?.value ?? 0
        )

        // Aggregate overall confidence. Bonus for finding line items.
        let valueParts = fieldConfidence.values
        var overall: Double = valueParts.isEmpty
            ? 0.1
            : valueParts.reduce(0, +) / Double(valueParts.count)
        if !items.isEmpty {
            overall = min(1.0, overall + 0.05)
        }

        return Output(
            header: header,
            lineItems: items,
            totals: totals,
            payment: nil,
            overallConfidence: overall,
            fieldConfidence: fieldConfidence,
            bboxes: bboxes
        )
    }

    // MARK: - Merchant

    private static let trivialHeaderTokens: [String] = [
        "receipt", "customer copy", "merchant copy", "duplicate", "duplicate copy",
        "store copy", "guest copy", "thank you", "welcome", "transaction",
        "order #", "ticket #", "invoice"
    ]

    private static let addressTokens: [String] = [
        " street", " st.", " st,", " avenue", " ave.", " ave,", " road", " rd.", " rd,",
        " blvd", " drive", " dr.", " way", " suite ", " ste ", " hwy", " highway",
        " circle ", " ct.", " court ", " lane ", " ln.", " pl.", " place "
    ]

    private static func findMerchant(lines: [Line]) -> (text: String, box: Receipt.BBox, confidence: Double)? {
        let topSlice = Array(lines.prefix(min(8, lines.count)))
        var best: (text: String, box: Receipt.BBox, score: Double)?

        for (i, line) in topSlice.enumerated() {
            let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.count >= 3 else { continue }

            let lc = text.lowercased()
            if trivialHeaderTokens.contains(where: { lc.contains($0) }) { continue }

            // Skip address lines
            if addressTokens.contains(where: { lc.contains($0) }) { continue }

            // Skip phone numbers / pure digits / pure punctuation
            if text.range(of: #"^\s*\+?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}([-.\s]?\d{2,4})?\s*$"#, options: .regularExpression) != nil { continue }

            // Skip URLs / emails
            if text.contains("@") || lc.contains("www.") || lc.contains("http") { continue }

            // Letters vs digits balance
            let letters = text.filter(\.isLetter).count
            let digits = text.filter(\.isNumber).count
            guard letters > 0 else { continue }
            if digits > letters { continue }

            // Skip date-only lines
            if text.range(of: dateRegex, options: [.regularExpression, .caseInsensitive]) != nil { continue }

            // Score: letters ratio, length sweet-spot, top-of-page bias
            let letterRatio = Double(letters) / Double(text.count)
            let lenBias = 1.0 - min(1.0, abs(Double(text.count) - 16.0) / 24.0)
            let topBias = 1.0 - Double(i) / Double(max(topSlice.count, 1))
            let score = letterRatio * 0.4 + lenBias * 0.3 + topBias * 0.3

            if best == nil || score > best!.score {
                best = (text, line.box, score)
            }
        }

        guard let chosen = best else { return nil }
        // Map score (0..1) → confidence (0.4..0.85) to avoid extreme reads.
        let confidence = 0.4 + min(chosen.score, 1.0) * 0.45
        return (text: chosen.text, box: chosen.box, confidence: confidence)
    }

    // MARK: - Date

    private static let dateFormats: [String] = [
        "yyyy-MM-dd", "yyyy/MM/dd",
        "MM/dd/yyyy", "M/d/yyyy", "M/d/yy", "MM/dd/yy",
        "MM-dd-yyyy", "M-d-yyyy", "M-d-yy",
        "dd/MM/yyyy", "dd-MM-yyyy",
        "MMM d, yyyy", "MMM dd, yyyy",
        "MMMM d, yyyy", "MMMM dd, yyyy",
        "d MMM yyyy", "dd MMM yyyy",
        "d-MMM-yyyy", "dd-MMM-yyyy",
        "yyyyMMdd"
    ]

    private static let dateRegex: String = [
        #"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"#,
        #"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"#,
        #"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b"#,
        #"\b\d{1,2}[-/ ](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ ]\d{2,4}\b"#
    ].joined(separator: "|")

    private static func findDate(lines: [Line]) -> (text: String, box: Receipt.BBox, confidence: Double, lineIndex: Int)? {
        for (idx, line) in lines.enumerated() {
            guard let r = line.text.range(of: dateRegex, options: [.regularExpression, .caseInsensitive]) else { continue }
            let raw = String(line.text[r])
            if let normalized = normalizeDate(raw) {
                return (text: normalized, box: line.box, confidence: 0.75, lineIndex: idx)
            }
        }
        return nil
    }

    private static func normalizeDate(_ raw: String) -> String? {
        for fmt in dateFormats {
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = fmt
            df.timeZone = TimeZone(identifier: "UTC")
            guard let parsed = df.date(from: raw) else { continue }

            // 2-digit-year fix: anything < 50 → 20xx, else 19xx
            let cal = Calendar(identifier: .gregorian)
            var comps = cal.dateComponents([.year, .month, .day], from: parsed)
            if let y = comps.year, y < 100 {
                comps.year = (y < 50 ? 2000 : 1900) + y
            }
            guard let final = cal.date(from: comps) else { continue }

            let out = DateFormatter()
            out.locale = Locale(identifier: "en_US_POSIX")
            out.dateFormat = "yyyy-MM-dd"
            out.timeZone = TimeZone(identifier: "UTC")
            return out.string(from: final)
        }
        return nil
    }

    // MARK: - Currency

    private static func detectCurrency(lines: [Line]) -> String {
        let blob = lines.map(\.text).joined(separator: " ")
        if blob.contains("€") || blob.range(of: #"\bEUR\b"#, options: .regularExpression) != nil { return "EUR" }
        if blob.contains("£") || blob.range(of: #"\bGBP\b"#, options: .regularExpression) != nil { return "GBP" }
        if blob.contains("¥") {
            // Could be JPY or CNY; default to JPY (more common on receipts)
            return "JPY"
        }
        if blob.contains("₹") || blob.range(of: #"\bINR\b"#, options: .regularExpression) != nil { return "INR" }
        if blob.range(of: #"\bCAD\b"#, options: .regularExpression) != nil { return "CAD" }
        if blob.range(of: #"\bAUD\b"#, options: .regularExpression) != nil { return "AUD" }
        return "USD"
    }

    // MARK: - Keyword amount (subtotal, total, tip, discount)

    private static let strongTotalKeywords = ["grand total", "total due", "balance due", "amount due", "amount payable"]

    private static func findKeywordAmount(
        lines: [Line],
        anyOf keywords: [String],
        excluding: [String]
    ) -> (value: Decimal, box: Receipt.BBox, confidence: Double, lineIndex: Int)? {
        // Iterate bottom-to-top: the latest "total" is typically the final one.
        for (idx, line) in lines.enumerated().reversed() {
            let lc = line.text.lowercased()
            guard keywords.contains(where: { lc.contains($0) }) else { continue }
            if excluding.contains(where: { lc.contains($0) }) { continue }

            // Direct amount on the keyword line
            if let amt = firstAmount(in: line.text) {
                let conf = strongTotalKeywords.contains(where: { lc.contains($0) }) ? 0.85 : 0.70
                return (amt, line.box, conf, idx)
            }
            // Amount on the next line (right column on a different OCR line)
            let lookahead = min(idx + 3, lines.count)
            for j in (idx + 1)..<lookahead {
                if let amt = firstAmount(in: lines[j].text) {
                    let conf = strongTotalKeywords.contains(where: { lc.contains($0) }) ? 0.75 : 0.60
                    return (amt, line.box, conf, idx)
                }
            }
        }
        return nil
    }

    // MARK: - Tax (possibly multiple lines)

    private struct TaxRow {
        var label: String
        var value: Decimal
        var rate: Decimal?
        var box: Receipt.BBox
        var lineIndex: Int
    }

    private static func findTaxLines(lines: [Line]) -> [TaxRow] {
        let triggers = ["sales tax", "state tax", "county tax", "city tax", "local tax", "vat", "gst", "hst", "pst", "tax"]
        var out: [TaxRow] = []
        for (idx, line) in lines.enumerated() {
            let lc = line.text.lowercased()
            guard triggers.contains(where: { lc.contains($0) }) else { continue }
            // Exclude false friends
            if lc.contains("exempt") || lc.contains("tax id") || lc.contains("tax#") || lc.contains("tax #")
                || lc.contains("ex tax") || lc.contains("ex. tax") || lc.contains("non-tax") || lc.contains("non taxable") {
                continue
            }
            guard let amt = firstAmount(in: line.text) else { continue }

            // Optional rate (e.g. "Tax (8.5%) $1.23")
            var rate: Decimal?
            if let r = line.text.range(of: #"(\d{1,2}(?:\.\d{1,3})?)\s*%"#, options: .regularExpression) {
                let s = line.text[r].replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespaces)
                if let dec = Decimal(string: s, locale: Locale(identifier: "en_US_POSIX")) {
                    rate = dec / 100
                }
            }

            // Skip duplicate amounts (some OCRs double-recognize the same line)
            if out.contains(where: { $0.value == amt }) { continue }

            let label: String = {
                if lc.contains("sales tax") { return "Sales Tax" }
                if lc.contains("state tax")  { return "State Tax" }
                if lc.contains("county tax") { return "County Tax" }
                if lc.contains("city tax")   { return "City Tax" }
                if lc.contains("local tax")  { return "Local Tax" }
                if lc.contains("vat")        { return "VAT" }
                if lc.contains("gst")        { return "GST" }
                if lc.contains("hst")        { return "HST" }
                if lc.contains("pst")        { return "PST" }
                return "Tax"
            }()
            out.append(TaxRow(label: label, value: amt, rate: rate, box: line.box, lineIndex: idx))
        }
        return out
    }

    // MARK: - Line items

    private static let itemSkipKeywords: [String] = [
        "subtotal", "sub total", "sub-total", "tax", "tip", "gratuity",
        "total", "discount", "savings", "you saved", "rewards", "loyalty",
        "change", "cash", "credit", "debit", "card", "visa", "mastercard",
        "amex", "discover", "balance", "payment", "tender", "due",
        "thank you", "approval", "auth", "ref ", "ref:", "ref#",
        "merchant", "terminal", "store #", "order #", "ticket #",
        "items sold", "items:"
    ]

    private static let itemTrailingAmount = #"-?\$?\s*\d{1,5}(?:,\d{3})*\.\d{2}\s*$"#

    private static func extractLineItems(lines: [Line], from start: Int, upTo end: Int) -> [Receipt.LineItem] {
        var items: [Receipt.LineItem] = []
        guard start < end, start < lines.count else { return items }
        let upper = min(end, lines.count)

        for i in start..<upper {
            let raw = lines[i].text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard raw.count >= 4 else { continue }

            let lc = raw.lowercased()
            if itemSkipKeywords.contains(where: { lc.contains($0) }) { continue }
            // Skip address-y or contact-y lines that snuck past the header
            if raw.contains("@") || lc.contains("www.") { continue }
            if raw.range(of: #"\b\d{3}-\d{4}\b|\(\d{3}\)"#, options: .regularExpression) != nil { continue }

            // Require a trailing money amount
            guard let priceRange = raw.range(of: itemTrailingAmount, options: .regularExpression),
                  let price = parseDecimal(String(raw[priceRange]))
            else { continue }

            var description = raw[raw.startIndex..<priceRange.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Trim a trailing "@ unit_price" if present (kept for unitPrice later, ignored for v1)
            if let atRange = description.range(of: #"\s+@\s*\$?\d+(?:\.\d{1,2})?\s*$"#, options: .regularExpression) {
                description = String(description[description.startIndex..<atRange.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
            }

            // Quantity prefix detection: "2 ITEM", "2x ITEM", "2 @ ITEM"
            var quantity: Decimal?
            if let qRange = description.range(of: #"^(\d+(?:\.\d+)?)\s*(?:x|X|@)?\s+"#, options: .regularExpression) {
                let qText = description[qRange]
                    .replacingOccurrences(of: "x", with: "")
                    .replacingOccurrences(of: "X", with: "")
                    .replacingOccurrences(of: "@", with: "")
                    .trimmingCharacters(in: .whitespaces)
                if let q = Decimal(string: qText, locale: Locale(identifier: "en_US_POSIX")), q > 0, q < 1000 {
                    quantity = q
                    description = String(description[qRange.upperBound...])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }

            // Filter clearly junk descriptions
            let letterCount = description.filter(\.isLetter).count
            if letterCount < 2 { continue }
            if description.count > 80 { continue }

            items.append(Receipt.LineItem(
                description: description,
                quantity: quantity,
                unitPrice: nil,
                totalPrice: price,
                category: nil
            ))
        }
        return items
    }

    // MARK: - Amount helpers

    private static let amountAnywhere = #"-?\$?\s*\d{1,5}(?:,\d{3})*(?:\.\d{2})?"#

    private static func firstAmount(in text: String) -> Decimal? {
        guard let r = text.range(of: amountAnywhere, options: .regularExpression) else { return nil }
        return parseDecimal(String(text[r]))
    }

    private static func parseDecimal(_ raw: String) -> Decimal? {
        let cleaned = raw
            .replacingOccurrences(of: "$", with: "")
            .replacingOccurrences(of: ",", with: "")
            .trimmingCharacters(in: .whitespaces)
        return Decimal(string: cleaned, locale: Locale(identifier: "en_US_POSIX"))
    }
}
