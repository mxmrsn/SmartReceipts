import Foundation

/// Lightweight regex/heuristic parser used by `VisionOnlyPipeline`.
///
/// This is the M1 baseline. It is deliberately small — extend cautiously, and
/// only when the benchmark shows that a richer heuristic would close a gap that
/// the LLM pipeline cannot. The expected long-term role of this parser is the
/// floor measurement, not the production code path.
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

    static func parse(lines: [(text: String, box: Receipt.BBox)]) -> Output {
        let texts = lines.map(\.text)
        var bboxes: [String: Receipt.BBox] = [:]
        var fieldConfidence: [String: Double] = [:]

        let merchant = pickMerchant(lines: lines)
        if let m = merchant {
            bboxes["merchant.name"] = m.box
            fieldConfidence["merchant.name"] = 0.5
        }

        let date = pickDate(lines: lines)
        if let d = date {
            bboxes["date.value"] = d.box
            fieldConfidence["date.value"] = 0.5
        }

        let total = pickTotal(lines: lines)
        if let t = total {
            bboxes["totals.total"] = t.box
            fieldConfidence["totals.total"] = 0.6
        }

        let subtotal = pickAmount(lines: lines, anyOf: ["subtotal", "sub total", "sub-total"])
        let tax = pickAmount(lines: lines, anyOf: ["tax", "vat", "gst", "hst", "sales tax"])

        let header = Receipt.Header(
            merchant: Receipt.Merchant(name: merchant?.text ?? "Unknown"),
            date: Receipt.ReceiptDate(value: date?.text ?? "1970-01-01"),
            currency: "USD"
        )

        let totals = Receipt.Totals(
            subtotal: subtotal?.value,
            tax: tax.map { [Receipt.TaxLine(label: "Tax", amount: $0.value)] } ?? [],
            total: total?.value ?? 0
        )

        let overall: Double = {
            let parts = fieldConfidence.values
            guard !parts.isEmpty else { return 0.1 }
            return parts.reduce(0, +) / Double(parts.count)
        }()

        _ = texts // silence unused for future use
        return Output(
            header: header,
            lineItems: [],
            totals: totals,
            payment: nil,
            overallConfidence: overall,
            fieldConfidence: fieldConfidence,
            bboxes: bboxes
        )
    }

    // MARK: - Field heuristics

    private static func pickMerchant(lines: [(text: String, box: Receipt.BBox)]) -> (text: String, box: Receipt.BBox)? {
        // First non-trivial line near the top of the image is usually the merchant.
        let candidates = lines
            .filter { $0.text.count >= 3 }
            .sorted { $0.box.y < $1.box.y }
        guard let first = candidates.first else { return nil }
        return (text: first.text.trimmingCharacters(in: .whitespacesAndNewlines), box: first.box)
    }

    private static func pickDate(lines: [(text: String, box: Receipt.BBox)]) -> (text: String, box: Receipt.BBox)? {
        let patterns: [String] = [
            #"\b\d{4}-\d{2}-\d{2}\b"#,
            #"\b\d{2}/\d{2}/\d{4}\b"#,
            #"\b\d{1,2}/\d{1,2}/\d{2,4}\b"#,
            #"\b\d{1,2}-\d{1,2}-\d{2,4}\b"#
        ]
        for line in lines {
            for p in patterns {
                if let match = line.text.range(of: p, options: .regularExpression) {
                    let raw = String(line.text[match])
                    return (text: normalizeDate(raw), box: line.box)
                }
            }
        }
        return nil
    }

    private static func normalizeDate(_ raw: String) -> String {
        // Best-effort; the bench measures how often we get this right.
        let formats = ["yyyy-MM-dd", "MM/dd/yyyy", "M/d/yyyy", "M/d/yy", "MM-dd-yyyy"]
        for f in formats {
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = f
            if let d = df.date(from: raw) {
                let out = DateFormatter()
                out.locale = Locale(identifier: "en_US_POSIX")
                out.dateFormat = "yyyy-MM-dd"
                return out.string(from: d)
            }
        }
        return raw
    }

    private static func pickTotal(lines: [(text: String, box: Receipt.BBox)]) -> (value: Decimal, box: Receipt.BBox)? {
        // Prefer a line that contains the literal "total" but not "subtotal".
        for line in lines {
            let lc = line.text.lowercased()
            guard lc.contains("total"), !lc.contains("sub") else { continue }
            if let amt = firstAmount(in: line.text) {
                return (value: amt, box: line.box)
            }
        }
        // Fallback: largest amount on the receipt.
        let amounts = lines.compactMap { line -> (value: Decimal, box: Receipt.BBox)? in
            guard let v = firstAmount(in: line.text) else { return nil }
            return (value: v, box: line.box)
        }
        return amounts.max(by: { $0.value < $1.value })
    }

    private static func pickAmount(
        lines: [(text: String, box: Receipt.BBox)],
        anyOf keywords: [String]
    ) -> (value: Decimal, box: Receipt.BBox)? {
        for line in lines {
            let lc = line.text.lowercased()
            guard keywords.contains(where: { lc.contains($0) }) else { continue }
            if let v = firstAmount(in: line.text) {
                return (value: v, box: line.box)
            }
        }
        return nil
    }

    private static func firstAmount(in text: String) -> Decimal? {
        let pattern = #"-?\$?\s*\d{1,5}(?:,\d{3})*(?:\.\d{2})?"#
        guard let range = text.range(of: pattern, options: .regularExpression) else { return nil }
        let raw = String(text[range])
            .replacingOccurrences(of: "$", with: "")
            .replacingOccurrences(of: ",", with: "")
            .trimmingCharacters(in: .whitespaces)
        return Decimal(string: raw, locale: Locale(identifier: "en_US_POSIX"))
    }
}
