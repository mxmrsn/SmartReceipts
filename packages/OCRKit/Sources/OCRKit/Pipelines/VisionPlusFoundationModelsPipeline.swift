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

        // Pre-cluster Vision observations into visual rows so a description
        // and its price on the same printed line arrive at the model as a
        // single row, separated by "  ". Massively reduces the
        // pair-across-lines errors we'd otherwise see.
        let rows = Self.clusterByRow(lines)
        let spatialOCR = rows.enumerated().map { idx, row in
            let yPct = Int((row.meanY * 100).rounded())
            return String(format: "[R%02d y=%02d] %@", idx, yPct, row.joined)
        }.joined(separator: "\n")

        let session = LanguageModelSession(instructions: Self.instructions)
        let prompt = "Receipt OCR, pre-grouped into visual rows. Each row is prefixed with [R## y=YY] where YY is vertical position (0=top..99=bottom). Within a row, segments are separated by two spaces and are listed left-to-right; on a line-item row the rightmost segment is almost always the price.\n\n\(spatialOCR)\n\nReturn ONLY the JSON object."

        // Receipts with 20+ line items routinely overflowed the default budget
        // and the model truncated mid-array, breaking JSON parsing. 4096 tokens
        // comfortably fits a 40-item receipt as compact JSON.
        let options = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 4096
        )

        let rawResponse: String
        do {
            let response = try await session.respond(to: prompt, options: options)
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
        // Snap merchant to a canonical brand name when we can identify one
        // in the OCR header. Catches FM picking a city ("Colma" for
        // "Target Colma"), an OCR misread ("how doers" for "Home Depot"),
        // or an inconsistent specificity ("sprouts" vs full chain name).
        // The OCR lines are already sorted top-to-bottom by Y, so passing
        // them as-is gives the header band the right priority.
        let orderedLineTexts = lines.map(\.text)
        if let brand = MerchantBrands.canonicalBrand(inTopOf: orderedLineTexts) {
            // Override when FM either (a) returned a city / department /
            // store-number-style string, or (b) returned a known city
            // outright, or (c) returned a clearly looser variant. Otherwise
            // keep FM's pick (handles legit non-chain restaurants).
            let fmName = receipt.header.merchant.name.trimmingCharacters(in: .whitespacesAndNewlines)
            let shouldOverride =
                fmName.isEmpty
                || fmName.lowercased() == "unknown"
                || MerchantBrands.looksLikeCity(fmName)
                || !fmName.lowercased().contains(brand.lowercased().prefix(4))
            if shouldOverride {
                receipt.header.merchant.name = brand
                receipt.provenance.fieldConfidence["merchant.name"] = 0.95
            }
        }
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
        You parse OCR text from receipts. The user supplies text that was recognized from a photo of a paper receipt, pre-grouped into visual rows. Each row is prefixed with [R## y=YY] where YY is the row's vertical position (0=top..99=bottom). Within a row, segments are separated by two spaces and listed left-to-right. \
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
        - Date is strictly YYYY-MM-DD. If the receipt only shows a 2-digit year, assume 20YY. NEVER invent a date — if no date is visible, return "".
        - merchantName must be the STORE / CHAIN name (e.g. "Target", "Costco", "Philz Coffee"). It is almost always the largest text near the top. Do NOT use the city, the address line, the cashier name, the department, or a store number. "TARGET T-1234 COLMA" → "Target", not "Colma".
        - total is the GRAND TOTAL (the amount actually charged). It is usually labelled "TOTAL", "AMOUNT DUE", or "BALANCE DUE", appears near the bottom, and is ≥ subtotal. NEVER return subtotal in the total field — if you only see a subtotal, leave total "".
        - lineItems contains only purchased products in receipt order. Do NOT include subtotal / tax / total / tip / discount / payment / change / balance due / card / credit / debit / cashback / loyalty / rewards / auth rows — those go in their dedicated fields above (or are dropped entirely).
        - One row = at most one line item. On a line-item row, the rightmost segment is the price; segments before it form the description (and quantity if present).
        - When a quantity prefix appears ("2 ITEM", "2x ITEM", "QTY 2  ITEM"), set the quantity field separately and put just the item name in description.
        - If a description-only row has no price and the row immediately below it is price-only at a nearby Y, treat that pair as ONE line item — but this should be rare after row grouping.
        - Use "" (empty string) for any field not visible. Do not guess merchant from filenames.
        - Do NOT wrap the JSON in markdown. Start your reply with { and end with }.

        Example. Showing merchant normalization (TARGET COLMA → Target), date conversion, the total-vs-subtotal distinction, and the footer-row drop.

        INPUT:
        [R00 y=04] TARGET COLMA
        [R01 y=08] 04/02/23
        [R02 y=20] OREO COOKIE  3.99
        [R03 y=24] 2 BANANAS  1.58
        [R04 y=33] SUBTOTAL  5.57
        [R05 y=35] TAX  0.32
        [R06 y=38] TOTAL  5.89
        [R07 y=42] VISA  5.89

        OUTPUT:
        {"merchantName":"Target","date":"2023-04-02","currency":"USD","total":"5.89","subtotal":"5.57","tax":"0.32","tip":"","discount":"","lineItems":[{"description":"OREO COOKIE","quantity":"","unitPrice":"","totalPrice":"3.99"},{"description":"BANANAS","quantity":"2","unitPrice":"","totalPrice":"1.58"}]}
        """

    // MARK: - Mapping

    private static func buildReceipt(from x: ReceiptExtraction) -> Receipt {
        var total: Decimal = parseDecimal(x.total) ?? 0
        var subtotal: Decimal? = parseDecimal(x.subtotal)
        let taxAmount: Decimal? = parseDecimal(x.tax)
        let tip: Decimal? = parseDecimal(x.tip)
        let discount: Decimal? = parseDecimal(x.discount)

        // Sanity: the grand total can never be less than the subtotal.
        // FM occasionally flips them (we've seen `total = $28.79`,
        // `subtotal = $28.79`, but the actual total on the receipt was
        // $31.49 — FM grabbed the subtotal row twice). When they're
        // inverted, swap — the larger value is almost always the actual
        // grand total.
        if let sub = subtotal, total > 0, total < sub {
            let oldTotal = total
            total = sub
            subtotal = oldTotal
        }

        let currencyCode: String = {
            let c = x.currency.trimmingCharacters(in: .whitespaces).uppercased()
            return c.isEmpty ? "USD" : c
        }()

        let dateValue: String = {
            let d = x.date.trimmingCharacters(in: .whitespaces)
            // Year locked to 19xx/20xx so a 2-digit-year misread like
            // "0024" → "0024-02-04" doesn't sneak past the canonical schema.
            guard d.range(of: #"^(19|20)\d{2}-\d{2}-\d{2}$"#, options: .regularExpression) != nil else {
                return "1970-01-01"
            }
            // Plausibility: receipts are never from the future and almost
            // never more than ~10 years old in a labeling workflow. We've
            // seen FM hallucinate "2024-09-19" for a clearly-2023 receipt.
            // Returning the sentinel triggers the labeler's EXIF-date
            // fallback path, which is much more reliable than guessing.
            let year = Int(d.prefix(4)) ?? 0
            let currentYear = Calendar.current.component(.year, from: Date())
            if year < currentYear - 10 || year > currentYear + 1 {
                return "1970-01-01"
            }
            return d
        }()

        var fieldConf: [String: Double] = [:]
        let merchant = x.merchantName.trimmingCharacters(in: .whitespaces)
        if !merchant.isEmpty            { fieldConf["merchant.name"] = 0.90 }
        if dateValue != "1970-01-01"    { fieldConf["date.value"]    = 0.90 }
        if total > 0                    { fieldConf["totals.total"]  = 0.90 }
        if subtotal != nil              { fieldConf["totals.subtotal"] = 0.85 }

        // Three-stage line-item cleanup. FM is told to skip footer rows but
        // regresses often enough that we treat the LLM output as candidate
        // and re-filter here. Each stage drops a specific failure mode we
        // see repeatedly:
        //   (a) empty/footer rows
        //   (b) zero-price rows ("FREE GIFT" promos, etc. — the schema
        //       requires nonzero, and 0 makes line-item-sum metrics noisy)
        //   (c) prices ≥ grand-total (almost certainly the total line
        //       leaking through, since no single item on a multi-item
        //       receipt exceeds the receipt's own total)
        //   (d) adjacent duplicates (FM sometimes restates the same row)
        struct Candidate {
            let desc: String
            let qty: Decimal?
            let unit: Decimal?
            let price: Decimal
        }
        var candidates: [Candidate] = x.lineItems.compactMap { row in
            let desc = row.description.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !desc.isEmpty else { return nil }
            guard !isFooterRow(desc) else { return nil }
            // FM sometimes emits `unitPrice` but omits `totalPrice` even
            // when the receipt only prints one number per line. Treat
            // unitPrice as a fallback so we don't drop legit items just
            // because the model picked the "wrong" field name.
            let totalPriceDec = parseDecimal(row.totalPrice)
            let unitPriceDec = parseDecimal(row.unitPrice)
            guard let price = totalPriceDec ?? unitPriceDec, price > 0 else { return nil }
            return Candidate(
                desc: desc,
                qty: parseDecimal(row.quantity),
                unit: totalPriceDec != nil ? unitPriceDec : nil,
                price: price
            )
        }

        // Stage (c): drop items whose price meets or exceeds the receipt
        // total. Only safe when total is known and there are at least two
        // items — a single-item receipt legitimately has price == total.
        if total > 0, candidates.count > 1 {
            candidates.removeAll { $0.price >= total }
        }

        // Stage (d): collapse adjacent duplicates with the same desc + price.
        var deduped: [Candidate] = []
        for c in candidates {
            if let last = deduped.last, last.desc == c.desc, last.price == c.price {
                continue
            }
            deduped.append(c)
        }

        let items: [Receipt.LineItem] = deduped.map {
            Receipt.LineItem(
                description: $0.desc,
                quantity: $0.qty,
                unitPrice: $0.unit,
                totalPrice: $0.price,
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
        guard let data = (coerceNumericStrings(cleaned)).data(using: .utf8) else {
            throw NSError(domain: "FMPipeline", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Empty model output"
            ])
        }
        do {
            return try JSONDecoder().decode(ReceiptExtraction.self, from: data)
        } catch {
            // Best-effort salvage: when FM exceeds its response budget the
            // JSON is truncated mid-string, so the decoder fails. Drop the
            // trailing partial line item, close the array + object, and
            // try once more — better to keep 7 valid items than to drop
            // the whole receipt.
            if let salvaged = salvageTruncatedJSON(cleaned),
               let salvagedData = coerceNumericStrings(salvaged).data(using: .utf8),
               let receipt = try? JSONDecoder().decode(ReceiptExtraction.self, from: salvagedData) {
                return receipt
            }
            throw error
        }
    }

    /// FM occasionally emits `"quantity": 2` (numeric) when the schema
    /// asked for `"quantity": "2"` (string). Convert unquoted numbers to
    /// quoted strings for the four numeric-string fields so the Codable
    /// struct decodes either form. Other JSON shape stays intact.
    private static func coerceNumericStrings(_ json: String) -> String {
        let pattern = #""(quantity|unitPrice|totalPrice|total|subtotal|tax|tip|discount)"\s*:\s*(-?\d+(?:\.\d+)?)"#
        return json.replacingOccurrences(
            of: pattern,
            with: "\"$1\":\"$2\"",
            options: .regularExpression
        )
    }

    /// Attempt to repair JSON that was cut off mid-output (FM hit its
    /// response token budget or the model's context window). Strategy:
    ///   1. Find the last complete `},` inside the `lineItems` array.
    ///   2. Truncate everything after it.
    ///   3. Close the array with `]` and the outer object with `}`.
    /// Returns nil if the input doesn't look like our schema at all
    /// (e.g. no `"lineItems"` key), so the caller can re-raise the
    /// original parse error rather than papering over it.
    private static func salvageTruncatedJSON(_ json: String) -> String? {
        guard let itemsStart = json.range(of: "\"lineItems\"")?.upperBound else { return nil }
        // Find the last "}," in the lineItems region. That's the boundary
        // after the last complete line-item object.
        let afterItems = json[itemsStart...]
        guard let lastBoundary = afterItems.range(of: "},", options: .backwards) else {
            // No complete items in the array yet — close it empty.
            // First make sure the prefix up to the array opening is sane.
            guard let arrayOpen = afterItems.range(of: "[") else { return nil }
            let head = String(json[..<arrayOpen.upperBound])
            return head + "]}"
        }
        // Keep through the closing `}` of the last item (drop the trailing `,`)
        let kept = json[..<lastBoundary.lowerBound] + "}"
        return String(kept) + "]}"
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
        // Date appears in many formats. Try in order:
        // 1. literal match of the canonical YYYY-MM-DD value
        // 2. NSDataDetector on each OCR line — Apple's date detector handles
        //    "Jan 15, 2026" / "01/15/24" / "15-Jan-26" / etc.
        // 3. lines containing the "date" or "time" keyword (last resort).
        if receipt.header.date.value != "1970-01-01" {
            if let box = locateBBox(value: receipt.header.date.value, in: lines) {
                bboxes["date.value"] = box
            } else if let box = locateDateBBox(targetISO: receipt.header.date.value, in: lines) {
                bboxes["date.value"] = box
            } else if let box = locateBBox(value: "", in: lines, fallbackKeywords: ["date", "time"]) {
                bboxes["date.value"] = box
            }
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

    /// Use Apple's NSDataDetector to find any date-shaped text in the OCR
    /// lines, then pick the line whose detected date matches the target
    /// ISO date by Y/M/D components. Handles "Feb 4, 2024", "02/04/24",
    /// "4-Feb-24", etc. without case-by-case regex.
    ///
    /// Important: comparing components rather than going through
    /// startOfDay avoids the UTC-vs-local-time pitfall — FM's
    /// "2024-02-04" is just a date, and we don't want a midnight rollover
    /// to make us miss the match.
    private static func locateDateBBox(
        targetISO: String,
        in lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt.BBox? {
        // Parse target as Y/M/D directly so timezone doesn't shift it.
        let parts = targetISO.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        let targetYear = parts[0]
        let targetMonth = parts[1]
        let targetDay = parts[2]

        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue) else {
            return nil
        }
        let calendar = Calendar(identifier: .gregorian)

        for line in lines {
            let range = NSRange(line.text.startIndex..<line.text.endIndex, in: line.text)
            let matches = detector.matches(in: line.text, options: [], range: range)
            for match in matches {
                guard let d = match.date else { continue }
                let comps = calendar.dateComponents([.year, .month, .day], from: d)
                if comps.year == targetYear, comps.month == targetMonth, comps.day == targetDay {
                    return line.box
                }
            }
        }
        // Fallback: any line that contains a numeric date pattern.
        for line in lines {
            if line.text.range(of: #"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b"#, options: .regularExpression) != nil {
                return line.box
            }
        }
        return nil
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

    // MARK: - Line-item footer denylist

    /// Description prefixes that almost always indicate a payment/totals
    /// footer row rather than a real purchased item. FM is told to skip
    /// these in the prompt, but occasionally regresses — this is the safety
    /// net. Matched case-insensitive, against the start of the description.
    private static let lineItemFooterPrefixes: [String] = [
        "balance", "balance due",
        "credit", "credit card", "debit", "card", "visa", "mastercard", "mc ",
        "amex", "american express", "discover",
        "change", "cash", "tender", "tendered",
        "subtotal", "sub total", "sub-total",
        "tax", "sales tax", "vat", "gst", "hst",
        "tip", "gratuity",
        "total", "grand total", "amount due", "amount paid", "amount payable",
        "auth", "approval", "approved", "ref ", "ref#", "ref:",
        "payment", "paid",
        "discount", "savings", "you saved", "coupon",
        "cashback", "cash back",
        "rounding",
        "loyalty", "rewards", "points",
        "invoice", "receipt", "transaction",
    ]

    private static func isFooterRow(_ description: String) -> Bool {
        let lc = description.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lc.isEmpty else { return true }
        for prefix in lineItemFooterPrefixes {
            if lc == prefix { return true }
            // Require a word-boundary char after the prefix so "totalitarian"
            // wouldn't match "total", and "creditor" wouldn't match "credit".
            // (Hypothetical on receipts, but cheap insurance.)
            if lc.hasPrefix(prefix + " ") { return true }
            if lc.hasPrefix(prefix + ":") { return true }
            if lc.hasPrefix(prefix + "\t") { return true }
        }
        return false
    }

    // MARK: - OCR row clustering

    /// One visual row of the receipt, made up of one or more OCR
    /// observations that share roughly the same Y. Segments are kept
    /// left-to-right by X so prompt rows mirror printed receipt rows.
    fileprivate struct OCRRow {
        var segments: [(text: String, x: Double)]
        var yMin: Double
        var yMax: Double

        var meanY: Double { (yMin + yMax) / 2 }

        var joined: String {
            segments
                .sorted { $0.x < $1.x }
                .map(\.text)
                .joined(separator: "  ")
        }
    }

    /// Group OCR observations into visual rows. Two observations belong to
    /// the same row when they vertically overlap by at least ~40% of the
    /// shorter one's height. This collapses the very common case where the
    /// description and the price land on the same printed line but Vision
    /// returns them as two separate observations — the exact case FM was
    /// struggling to pair up using Y-percentage hints alone.
    fileprivate static func clusterByRow(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [OCRRow] {
        let sorted = lines.sorted {
            ($0.box.y + $0.box.height / 2) < ($1.box.y + $1.box.height / 2)
        }
        var rows: [OCRRow] = []
        for line in sorted {
            let yTop = line.box.y
            let yBot = line.box.y + line.box.height
            let height = line.box.height
            if var current = rows.last {
                let overlap = min(yBot, current.yMax) - max(yTop, current.yMin)
                let minH = min(height, current.yMax - current.yMin)
                if minH > 0, overlap >= minH * 0.4 {
                    current.segments.append((text: line.text, x: line.box.x))
                    current.yMin = min(current.yMin, yTop)
                    current.yMax = max(current.yMax, yBot)
                    rows[rows.count - 1] = current
                    continue
                }
            }
            rows.append(OCRRow(
                segments: [(text: line.text, x: line.box.x)],
                yMin: yTop,
                yMax: yBot
            ))
        }
        return rows
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

// Every field is decoded via `decodeIfPresent` so FM omitting one (a real
// failure mode we see when the response is truncated or the model swaps
// `unitPrice`/`totalPrice`) doesn't blow up the whole parse — we'd rather
// get a partial Receipt than nothing at all. Default-value-on-property
// alone does NOT help with the synthesized Codable; it still requires
// every key. Hence the manual init.
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

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        merchantName = try c.decodeIfPresent(String.self, forKey: .merchantName) ?? ""
        date         = try c.decodeIfPresent(String.self, forKey: .date)         ?? ""
        currency     = try c.decodeIfPresent(String.self, forKey: .currency)     ?? ""
        total        = try c.decodeIfPresent(String.self, forKey: .total)        ?? ""
        subtotal     = try c.decodeIfPresent(String.self, forKey: .subtotal)     ?? ""
        tax          = try c.decodeIfPresent(String.self, forKey: .tax)          ?? ""
        tip          = try c.decodeIfPresent(String.self, forKey: .tip)          ?? ""
        discount     = try c.decodeIfPresent(String.self, forKey: .discount)     ?? ""
        lineItems    = try c.decodeIfPresent([ReceiptLineItemExtraction].self, forKey: .lineItems) ?? []
    }
}

private struct ReceiptLineItemExtraction: Codable {
    var description: String
    var quantity: String
    var unitPrice: String
    var totalPrice: String

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        quantity    = try c.decodeIfPresent(String.self, forKey: .quantity)    ?? ""
        unitPrice   = try c.decodeIfPresent(String.self, forKey: .unitPrice)   ?? ""
        totalPrice  = try c.decodeIfPresent(String.self, forKey: .totalPrice)  ?? ""
    }
}

#endif
