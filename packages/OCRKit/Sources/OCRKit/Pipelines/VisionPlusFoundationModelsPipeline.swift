import CoreGraphics
import Foundation
import ImageIO
import Vision

#if canImport(FoundationModels)
import FoundationModels

/// Production-target pipeline: Apple Vision for text recognition, then Apple's
/// on-device Foundation Models for *structured* extraction into the canonical
/// `Receipt` schema.
///
/// Vision is a near-perfect text reader but does no reasoning — so the M1
/// regex parser fell over on anything non-trivial. Here, the OCR text is
/// streamed into a `LanguageModelSession` and we ask the model for a strict
/// JSON object that mirrors the canonical schema. The on-device 3B-parameter
/// model handles layout reasoning natively: line items in any format, dates
/// in any locale, totals that span two lines, merchant names buried below a
/// logo, etc.
///
/// We *don't* use the `@Generable` macro because SwiftPM-built packages
/// can't currently resolve Apple's `FoundationModelsMacros` macro plugin.
/// The JSON-prompt fallback works on both SwiftPM and Xcode toolchains.
///
/// Availability:
/// - macOS 26+, iOS 26+
/// - Apple Intelligence must be enabled on-device
/// The pipeline throws `OCRError.modelNotAvailable` if Apple Intelligence
/// isn't ready, so the labeler can fall back to `vision-regex`.
@available(macOS 26.0, iOS 26.0, *)
public struct VisionPlusFoundationModelsPipeline: OCRPipeline {
    public static let id = "vision-fm"
    public static let displayName = "Apple Vision + Foundation Models"
    public static let modelVersion = "vision-fm.1"

    public init() {}

    public func extract(image: CGImage, orientation: CGImagePropertyOrientation) async throws -> ExtractionResult {
        let startNs = DispatchTime.now().uptimeNanoseconds

        // ---- 1) Vision OCR ----
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["en-US"]

        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw OCRError.visionRequestFailed(error.localizedDescription)
        }

        let observations: [VNRecognizedTextObservation] = request.results ?? []
        let lines: [(text: String, box: Receipt.BBox)] = observations
            .compactMap { obs -> (text: String, box: Receipt.BBox)? in
                guard let top = obs.topCandidates(1).first else { return nil }
                return (text: top.string, box: Self.bbox(from: obs.boundingBox))
            }
            .sorted { $0.box.y < $1.box.y }
        let rawText = lines.map(\.text).joined(separator: "\n")

        guard !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OCRError.parseFailed("Vision returned no recognized text.")
        }

        // ---- 2) Foundation Models structured extraction ----
        guard case .available = SystemLanguageModel.default.availability else {
            throw OCRError.modelNotAvailable(
                "Apple Foundation Models unavailable. Enable Apple Intelligence in System Settings."
            )
        }

        // Feed the model the OCR lines WITH their Y coordinates so it can
        // group descriptions and prices that landed on adjacent rows.
        let spatialOCR = lines.enumerated().map { idx, line in
            let yPct = Int((line.box.y * 100).rounded())
            return String(format: "[L%02d y=%02d] %@", idx, yPct, line.text)
        }.joined(separator: "\n")

        let session = LanguageModelSession(instructions: Self.instructions)
        let prompt = "Receipt OCR (each line prefixed with [L## y=YY] where YY is vertical position 0=top..99=bottom). " +
                     "When you see an item description followed by a price-only line at a similar y (within ~3), treat them as ONE line item.\n\n\(spatialOCR)\n\nReturn ONLY the JSON object."

        let rawResponse: String
        do {
            let response = try await session.respond(to: prompt)
            rawResponse = response.content
        } catch {
            throw OCRError.parseFailed(
                "Foundation Models generation failed: \(error.localizedDescription)"
            )
        }

        let extracted: ReceiptExtraction
        do {
            extracted = try Self.parseJSON(rawResponse)
        } catch {
            throw OCRError.parseFailed(
                "Could not parse model output as JSON: \(error.localizedDescription). Raw output: \(rawResponse.prefix(400))"
            )
        }

        let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds &- startNs) / 1_000_000)

        // ---- 3) Build canonical Receipt ----
        var receipt = Self.buildReceipt(from: extracted)
        // Backfill bboxes by matching extracted field values back to OCR lines.
        // The bbox overlay in the labeler uses these to highlight detections.
        receipt = Self.attachBBoxes(receipt: receipt, lines: lines)

        return ExtractionResult(
            receipt: receipt,
            latencyMs: elapsedMs,
            peakMemoryMB: nil,
            rawText: rawText,
            ocrLines: lines.map { OCRLine(text: $0.text, box: $0.box) }
        )
    }

    // MARK: - Instructions

    private static let instructions: String = """
        You parse OCR text from receipts. The user supplies text that was recognized from a photo of a paper receipt, with each line prefixed by its line number and rough vertical position. \
        Output ONLY a valid JSON object — no markdown, no commentary, no code fence.

        Strict shape (every key required, use empty string "" for fields not visible on the receipt):
        {
          "merchantName": "",
          "date": "YYYY-MM-DD",
          "currency": "USD",
          "total": "",
          "subtotal": "",
          "tax": "",
          "tip": "",
          "discount": "",
          "lineItems": [
            { "description": "", "quantity": "", "unitPrice": "", "totalPrice": "" }
          ]
        }

        Rules:
        - All money fields are plain decimal strings like "12.34". No currency symbols, no thousands separators.
        - Currency is a 3-letter ISO 4217 code. Default to "USD" if unclear.
        - Date is strictly YYYY-MM-DD. If the receipt only shows a 2-digit year, assume 20YY.
        - lineItems contains only purchased products in receipt order. Skip subtotal/tax/total/tip/discount/payment/change/cashback/loyalty/rewards lines.
        - When a description on one line is followed by a price on the next line at a similar vertical position (Δy ≤ 3), combine them into ONE line item.
        - When a quantity prefix appears ("2 ITEM", "2x ITEM", "QTY 2  ITEM"), set the quantity field separately and put just the item name in description.
        - Use "" (empty string) for any field not visible. Do not guess merchant from filenames.
        - Do NOT wrap the JSON in markdown. Start your reply with { and end with }.
        """

    // MARK: - Mapping

    private static func buildReceipt(from x: ReceiptExtraction) -> Receipt {
        let total: Decimal = parseDecimal(x.total) ?? 0
        let subtotal: Decimal? = parseDecimal(x.subtotal)
        let taxAmount: Decimal? = parseDecimal(x.tax)
        let tip: Decimal? = parseDecimal(x.tip)
        let discount: Decimal? = parseDecimal(x.discount)

        let currencyCode: String = {
            let c = x.currency.trimmingCharacters(in: .whitespaces).uppercased()
            return c.isEmpty ? "USD" : c
        }()

        let dateValue: String = {
            let d = x.date.trimmingCharacters(in: .whitespaces)
            return d.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil
                ? d
                : "1970-01-01"
        }()

        var fieldConf: [String: Double] = [:]
        let merchant = x.merchantName.trimmingCharacters(in: .whitespaces)
        if !merchant.isEmpty            { fieldConf["merchant.name"] = 0.90 }
        if dateValue != "1970-01-01"    { fieldConf["date.value"]    = 0.90 }
        if total > 0                    { fieldConf["totals.total"]  = 0.90 }
        if subtotal != nil              { fieldConf["totals.subtotal"] = 0.85 }

        let items: [Receipt.LineItem] = x.lineItems.compactMap { row in
            let desc = row.description.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !desc.isEmpty else { return nil }
            guard let price = parseDecimal(row.totalPrice) else { return nil }
            return Receipt.LineItem(
                description: desc,
                quantity: parseDecimal(row.quantity),
                unitPrice: parseDecimal(row.unitPrice),
                totalPrice: price,
                category: nil
            )
        }

        let taxLines: [Receipt.TaxLine] = taxAmount
            .map { [Receipt.TaxLine(label: "Tax", amount: $0)] } ?? []

        let overall = fieldConf.values.isEmpty
            ? 0.5
            : fieldConf.values.reduce(0, +) / Double(fieldConf.values.count)

        return Receipt(
            imageId: UUID(),
            header: Receipt.Header(
                merchant: Receipt.Merchant(name: merchant.isEmpty ? "Unknown" : merchant),
                date: Receipt.ReceiptDate(value: dateValue),
                currency: currencyCode
            ),
            lineItems: items,
            totals: Receipt.Totals(
                subtotal: subtotal,
                tax: taxLines,
                discount: discount,
                tip: tip,
                serviceCharge: nil,
                total: total
            ),
            payment: nil,
            provenance: Receipt.Provenance(
                pipelineId: Self.id,
                modelVersion: Self.modelVersion,
                confidence: overall,
                fieldConfidence: fieldConf,
                bboxes: [:]
            )
        )
    }

    // MARK: - JSON cleanup + parsing

    private static func parseJSON(_ output: String) throws -> ReceiptExtraction {
        let cleaned = cleanJSONFence(output)
        guard let data = cleaned.data(using: .utf8) else {
            throw NSError(domain: "FMPipeline", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Empty model output"
            ])
        }
        return try JSONDecoder().decode(ReceiptExtraction.self, from: data)
    }

    /// Strip Markdown code fences and stray prose around the JSON object.
    private static func cleanJSONFence(_ s: String) -> String {
        var x = s.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip ```json ... ``` or ``` ... ``` fences.
        if x.hasPrefix("```") {
            x = x.replacingOccurrences(of: #"^```(?:json|JSON)?\s*"#, with: "", options: .regularExpression)
            x = x.replacingOccurrences(of: #"\s*```\s*$"#, with: "", options: .regularExpression)
        }
        // Keep only from the first '{' to the last '}'.
        if let firstBrace = x.firstIndex(of: "{"), let lastBrace = x.lastIndex(of: "}") {
            x = String(x[firstBrace...lastBrace])
        }
        return x.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - BBox matching

    /// Match extracted field values back to OCR lines so the labeler can show
    /// the spatial highlight where each field came from. We pick the line
    /// with the closest substring match.
    private static func attachBBoxes(receipt input: Receipt, lines: [(text: String, box: Receipt.BBox)]) -> Receipt {
        var receipt = input
        var bboxes = receipt.provenance.bboxes

        if let box = locateBBox(value: receipt.header.merchant.name, in: lines) {
            bboxes["merchant.name"] = box
        }
        // Date appears in many formats; try the canonical YYYY-MM-DD first, then any line containing digits
        if receipt.header.date.value != "1970-01-01",
           let box = locateBBox(value: receipt.header.date.value, in: lines, fallbackKeywords: ["date", "time"]) {
            bboxes["date.value"] = box
        }
        if receipt.totals.total > 0,
           let box = locateBBox(amount: receipt.totals.total, in: lines, requireKeyword: "total") {
            bboxes["totals.total"] = box
        }
        if let subtotal = receipt.totals.subtotal,
           let box = locateBBox(amount: subtotal, in: lines, requireKeyword: "subtotal") {
            bboxes["totals.subtotal"] = box
        }
        // Line items: walk in order, find first match per row that we haven't claimed yet.
        var claimed = Set<Int>()
        for (idx, item) in receipt.lineItems.enumerated() {
            if let (lineIdx, box) = locateItemBBox(description: item.description, totalPrice: item.totalPrice, in: lines, excluding: claimed) {
                bboxes["lineItem.\(String(format: "%03d", idx))"] = box
                claimed.insert(lineIdx)
            }
        }

        receipt.provenance.bboxes = bboxes
        return receipt
    }

    private static func locateBBox(
        value: String,
        in lines: [(text: String, box: Receipt.BBox)],
        fallbackKeywords: [String] = []
    ) -> Receipt.BBox? {
        let needle = value.trimmingCharacters(in: .whitespaces).lowercased()
        guard needle.count >= 2 else { return nil }
        if let hit = lines.first(where: { $0.text.lowercased().contains(needle) }) {
            return hit.box
        }
        for kw in fallbackKeywords {
            if let hit = lines.first(where: { $0.text.lowercased().contains(kw) }) {
                return hit.box
            }
        }
        return nil
    }

    private static func locateBBox(
        amount: Decimal,
        in lines: [(text: String, box: Receipt.BBox)],
        requireKeyword: String
    ) -> Receipt.BBox? {
        let amountStr = NSDecimalNumber(decimal: amount)
            .stringValue
        // Pick a line that mentions the keyword AND contains the amount text.
        // Search bottom-up since totals are at the bottom.
        for line in lines.reversed() {
            let lc = line.text.lowercased()
            if lc.contains(requireKeyword), line.text.contains(amountStr) {
                return line.box
            }
        }
        // Fallback: any line with the keyword
        return lines.reversed().first(where: { $0.text.lowercased().contains(requireKeyword) })?.box
    }

    private static func locateItemBBox(
        description: String,
        totalPrice: Decimal,
        in lines: [(text: String, box: Receipt.BBox)],
        excluding: Set<Int>
    ) -> (lineIndex: Int, box: Receipt.BBox)? {
        let needle = description.lowercased().trimmingCharacters(in: .whitespaces)
        let firstWord = needle.split(separator: " ").first.map(String.init) ?? needle
        for (idx, line) in lines.enumerated() {
            if excluding.contains(idx) { continue }
            let lc = line.text.lowercased()
            if lc.contains(needle) || (firstWord.count >= 3 && lc.contains(firstWord)) {
                return (idx, line.box)
            }
        }
        return nil
    }

    private static func parseDecimal(_ s: String) -> Decimal? {
        let cleaned = s
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "$", with: "")
            .replacingOccurrences(of: "€", with: "")
            .replacingOccurrences(of: "£", with: "")
            .replacingOccurrences(of: ",", with: "")
        guard !cleaned.isEmpty else { return nil }
        return Decimal(string: cleaned, locale: Locale(identifier: "en_US_POSIX"))
    }

    private static func bbox(from vn: CGRect) -> Receipt.BBox {
        Receipt.BBox(
            x: Double(vn.minX),
            y: Double(1.0 - vn.maxY),
            width: Double(vn.width),
            height: Double(vn.height)
        )
    }
}

// MARK: - Plain Codable shape (no macros)

private struct ReceiptExtraction: Codable {
    var merchantName: String
    var date: String
    var currency: String
    var total: String
    var subtotal: String
    var tax: String
    var tip: String
    var discount: String
    var lineItems: [ReceiptLineItemExtraction]
}

private struct ReceiptLineItemExtraction: Codable {
    var description: String
    var quantity: String
    var unitPrice: String
    var totalPrice: String
}

#endif
