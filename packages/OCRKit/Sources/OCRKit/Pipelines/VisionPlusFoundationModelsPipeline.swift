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
        //
        // Filter out OCR observations that are receipt metadata, not item
        // content. Safeway and similar chains print verbose annotations
        // around every line item ("Regular Price", "Member Savings",
        // "VOL+", "VOL-", "INFO", "WT") that interleave with the actual
        // item rows after row clustering, confusing the LLM. Stripping
        // them at the OCR level keeps the FM prompt focused on rows that
        // carry a description + price.
        let prunedLines = Self.dropMetadataObservations(lines)
        // Drop pure-numeric OCR observations that are clearly SKU / UPC
        // codes printed alongside each line item ("037129013" on Target,
        // "643231511367" on Home Depot, etc.). These add noise to the
        // prompt and make FM less likely to recognize the real item row;
        // they have no semantic value for receipt extraction.
        let withoutSKUs = Self.dropSKUObservations(prunedLines)
        // Normalize Vision OCR mistakes that look like decimal separators
        // but actually got read as hyphens (small-font receipts: "$11.75"
        // → "$11-75"). Done before clustering so the price-shape detection
        // downstream matches.
        let normalizedLines = Self.normalizePriceTokens(withoutSKUs)
        // Strip trailing tax / SKU category markers from price-shaped
        // segments so the LLM sees "8.49" instead of "8.49 S". Some chains
        // (Safeway, Lucky, Albertsons) suffix every line price with a
        // single uppercase letter indicating tax category, and FM
        // otherwise refuses to recognize the rightmost segment as a price
        // — it returned empty lineItems on the Safeway receipt until we
        // sanded these off.
        let cleanedLines = Self.stripTrailingTaxMarkers(normalizedLines)
        let rawRows = Self.clusterByRow(cleanedLines)
        // Some receipts (Happy Hound, etc.) wrap the price onto the row
        // BELOW the description when the description is too long. Anchor-
        // based clustering correctly leaves them as separate rows, but
        // FM then can't pair them. Merge a price-only row up into the
        // previous description-only row when they're vertically adjacent.
        let mergedRows = Self.mergeOrphanedPriceRows(rawRows)
        // After clustering + merge, drop rows whose entire content is a
        // money value (the regular-price and savings-amount lines whose
        // label sibling was pruned). Also strip trailing single-letter
        // tax-category segments that survive clustering on their own bbox.
        let rows = mergedRows
            .filter { !Self.isOrphanedNumericRow($0) }
            // Drop entire rows whose joined text contains a hard
            // discount/metadata marker. After clustering, the row "WT
            // 12.67 Regular Price" survives because none of its individual
            // observations is purely a "Regular Price" prefix, but the
            // row's `12.67` is a regular-price amount, not a real item
            // price. Feeding it to FM tempts the model to extract it as
            // a line item (see IMG_1317 picking $12.67 for APPLES when
            // the actual price was $7.77 on a different row).
            .filter { !Self.rowContainsHardMetadataMarker($0) }
            .map(Self.stripTrailingSingleLetterSegment)
        let spatialOCR = rows.enumerated().map { idx, row in
            let yPct = Int((row.meanY * 100).rounded())
            return String(format: "[R%02d y=%02d] %@", idx, yPct, row.joined)
        }.joined(separator: "\n")

        let session = LanguageModelSession(instructions: Self.instructions)
        let prompt = "Receipt OCR, pre-grouped into visual rows. Each row is prefixed with [R## y=YY] where YY is vertical position (0=top..99=bottom). Within a row, segments are separated by two spaces and are listed left-to-right; on a line-item row the rightmost segment is almost always the price.\n\n\(spatialOCR)\n\nReturn ONLY the JSON object."

        // Apple's on-device FM has a fixed total context (input + output).
        // Reserving too much for output starves the input — the long
        // Safeway-style receipts then trigger "Exceeded model context
        // window size" before generation even begins. The compact JSON
        // schema we ask for fits a 25-item receipt in well under 1500
        // tokens, so reserve that and leave plenty of room for the input
        // OCR rows.
        let options = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 1500
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
        // Use the price-normalized lines for all post-extraction work
        // (buildReceipt sanity checks, recovery, bbox attachment). The
        // raw `lines` array still has Vision's literal output like
        // "6. 49 S" with the spurious space — without normalization,
        // priceShape regexes reject the token and the item silently
        // vanishes from recovery (IMG_1326 B&J at $6.49).
        let postLines = Self.normalizePriceTokens(lines)
        var receipt = Self.buildReceipt(from: extracted, ocrLines: postLines)
        // PRIMARY line-item extraction: scan the price column directly.
        // The OCR data unambiguously tells us where the prices are. FM's
        // strength is reading text (merchant name, date, totals labels);
        // its weakness is structural decisions about WHICH price belongs
        // to WHICH description on cluttered Safeway / Sprouts receipts.
        // Column-anchored extraction sidesteps that — every price-shaped
        // observation in the column is a candidate, filtered against the
        // totals block, regular-price labels, and savings markers. FM's
        // items are still used as a HINT source for quantity/unit/category.
        let fmItems = receipt.lineItems
        let columnItems = Self.columnAnchoredLineItems(
            lines: postLines,
            fmItems: fmItems,
            totalsValues: Self.totalsValueSet(from: receipt),
            receiptTotal: receipt.totals.total
        )
        if !columnItems.isEmpty {
            receipt.lineItems = columnItems
        } else {
            // Fall back to FM-only items + the legacy post-processors
            // when column extraction finds nothing (single-item receipts,
            // very sparse OCR, etc.).
            receipt = Self.correctRegularPriceMistakes(receipt: receipt, lines: postLines)
            receipt = Self.recoverMissedLineItems(receipt: receipt, lines: postLines)
        }
        // Guardrail: FM sometimes picks a totally wrong value for the
        // grand total — a "Total Savings Value" from the SAVINGS block
        // (IMG_2942: $5.53 instead of $56.75) or a straight-up
        // hallucination not present in the OCR at all (IMG_7951:
        // $161.99 when the receipt prints $93.16). Cross-check against
        // (a) the value actually appearing in OCR near a totals label
        // and (b) the arithmetic sum of line items plus tax.
        receipt = Self.sanityCheckTotal(receipt: receipt, lines: postLines)
        // Same idea for subtotal / tax / tip — find the OCR label,
        // grab its adjacent value. FM sometimes cross-wires these
        // (Philz IMG: emits tip amount in the subtotal slot). If the
        // OCR is clear about which label goes with which value, we
        // should trust that over FM.
        receipt = Self.sanityCheckSubtotalTaxTip(receipt: receipt, lines: postLines)
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
        // Final non-item filter: drop any line items whose description is
        // clearly a department header ("LIQUOR"), a rewards/coupon line
        // ("BASKET Grocery Rewards"), or a savings amount ("3.50-").
        // Catches both FM-emitted and recovered items in one pass; FM
        // sometimes treats these as items even when the row text makes
        // it obvious they aren't.
        receipt.lineItems.removeAll { item in
            Self.looksLikeNonItemRecoveryCandidate(item.description)
        }
        // Backfill bboxes by matching extracted field values back to OCR lines.
        // The bbox overlay in the labeler uses these to highlight detections.
        // Uses the normalized lines so a price like "6. 49 S" still matches
        // its bbox slot — the bbox coordinates are identical to the
        // original observation either way.
        receipt = Self.attachBBoxes(receipt: receipt, lines: postLines)

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
        - Lines like "HEALTH ITEM TOTAL: X", "GROCERY TOTAL: X", "DEPT 25 TOTAL: X" are department/category summary totals shown in the footer recap — they are NOT purchased line items.
        - Item descriptions sometimes start with a long product code (SKU/UPC, 6+ digits) — drop the leading code, keep just the product name (e.g. "037129013 NTG Stubb Bd" → "NTG Stubb Bd"). Single-character trailing markers like "T", "+", "F", "N" indicate tax category and aren't part of the name — drop them too.
        - One row = at most one line item. On a line-item row, the rightmost segment is the price; segments before it form the description (and quantity if present).
        - Quantity rules — be conservative:
          * ONLY set `quantity` when the row's text EXPLICITLY contains a multiplier marker: a leading "2 ", "2x ", "QTY 2", "2 @", or similar. Without that marker, leave `quantity` as "" (it defaults to 1).
          * NEVER infer quantity from prices, savings, or totals math. If the row says "SIG PEANUT  2.99", quantity is "" — not 2.
          * The price printed on the row is the TOTAL for that line (already multiplied by quantity if applicable). Copy it verbatim into `totalPrice`. Leave `unitPrice` empty unless a separate per-unit price is printed on its own sub-row.
        - If a description-only row has no price and the row immediately below it is price-only at a nearby Y, treat that pair as ONE line item — but this should be rare after row grouping.
        - Use "" (empty string) for any field not visible. Do not guess merchant from filenames.
        - Do NOT wrap the JSON in markdown. Start your reply with { and end with }.

        Be aggressive about extracting items: every row that has a product-name-like segment AND a price-like number is a line item, even if it's mixed in among section headers ("GROCERY", "PRODUCE", "LIQUOR", etc.) or store-internal markers ("S", "T", "F"). When in doubt, INCLUDE the row — the postprocessor drops false positives. An empty lineItems array means the receipt has no purchased products, which is rare.
        """

    // MARK: - Mapping

    private static func buildReceipt(
        from x: ReceiptExtraction,
        ocrLines: [(text: String, box: Receipt.BBox)] = []
    ) -> Receipt {
        var total: Decimal = parseDecimal(x.total) ?? 0
        var subtotal: Decimal? = parseDecimal(x.subtotal)
        var taxAmount: Decimal? = parseDecimal(x.tax)
        // FM occasionally returns "0" / "0.00" for optional money fields
        // when the receipt didn't actually print them. Treat zero the
        // same as missing — saves the labeler from displaying a phantom
        // "$0.00 tip" the user has to clear manually.
        let tip: Decimal? = {
            let v = parseDecimal(x.tip)
            return (v ?? 0) == 0 ? nil : v
        }()
        let discount: Decimal? = {
            let v = parseDecimal(x.discount)
            return (v ?? 0) == 0 ? nil : v
        }()

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

        // Tax sanity: FM sometimes mis-extracts a tax RATE as the tax
        // amount when the receipt only prints "CA TAX 9.37500 on $12.89"
        // (no dollar amount on a dedicated line). Real US sales taxes
        // top out around 12%; any "tax" value > 30% of the subtotal is
        // almost certainly a rate. Drop it.
        if let tax = taxAmount, let sub = subtotal, sub > 0,
           tax > sub * Decimal(0.3)
        {
            taxAmount = nil
        }
        // Tax-slot-holds-total: on receipts where subtotal is missing
        // (Safeway-style: only prints "**** BALANCE"), FM sometimes
        // grabs the total value and puts it in the tax slot too. Any
        // tax that equals or exceeds the total, or is more than ~30%
        // of the total, isn't a real tax. Drop it. Covers IMG_2173
        // where FM emitted tax=$45.55 for a $45.55 total.
        if let tax = taxAmount, total > 0,
           (tax >= total || tax > total * Decimal(0.3))
        {
            taxAmount = nil
        }

        // If subtotal + total are present but tax is missing (either FM
        // didn't extract it, or we just dropped a rate-as-amount), derive
        // it from the receipt arithmetic:
        //     total = subtotal + tax + tip - discount
        //   ⇒ tax  = total - subtotal - tip + discount
        // We only accept the computed value when it's positive and not
        // larger than the subtotal — anything outside that window is more
        // likely a bug in our other extractions than a real tax.
        if taxAmount == nil, let sub = subtotal, total > sub {
            let computed = total - sub - (tip ?? 0) + (discount ?? 0)
            if computed > 0, computed <= sub {
                taxAmount = computed
            }
        }

        // Subtotal sanity: FM sometimes pulls "BALANCE 18.00" AND
        // "PAYMENT AMOUNT 18.00" off the receipt and emits BOTH as
        // total and subtotal — but that's the wrong answer when the
        // receipt has a real tax line. If subtotal == total and tax
        // is positive, the printed subtotal was almost certainly
        // missing and FM duplicated the total. Compute the real
        // subtotal from the arithmetic.
        if let sub = subtotal, sub == total, let tax = taxAmount, tax > 0 {
            let computed = total - tax - (tip ?? 0) + (discount ?? 0)
            if computed > 0, computed < total {
                subtotal = computed
            }
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
            // Clean up FM's raw description: strip leading SKU/UPC codes
            // and trailing tax-category markers (T, +, F, N, X). FM is
            // told to do this in the prompt but sometimes leaves the
            // prefix/suffix attached — this is the safety net.
            let desc = Self.cleanLineItemDescription(row.description)
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

        // OCR-anchored price correction. FM occasionally returns
        // `totalPrice = quantity × unitPrice` when the receipt actually
        // printed just the unit price (e.g. "SIG PEANUT 2.99" but FM
        // emits qty=2, unit=2.99, total=5.98). Trust the OCR over the
        // model's math: if the matching row's rightmost price token
        // equals the unitPrice (not the inflated total), reset total
        // to the printed value.
        if !ocrLines.isEmpty {
            for i in candidates.indices {
                let cand = candidates[i]
                guard let unit = cand.unit, unit > 0,
                      let qty = cand.qty, qty > 1
                else { continue }
                let computed = unit * qty
                // Only kick in when total == qty × unit (mathematical
                // consistency suggests FM did the multiplication).
                guard abs(cand.price - computed) < Decimal(0.01) else { continue }
                // Inspect every OCR row matching this description. If ANY
                // of them prints `unit` as the row price (not the inflated
                // total), FM did the multiplication on its own — reset.
                // Multi-row matching is the key for receipts like Sprouts
                // that print two separate "BONE-IN CHCKN THIGHS" lines at
                // different prices: looking at only the first row would
                // pick the wrong one half the time.
                let printedPrices = Self.rightmostPricesInOCRRows(
                    matching: cand.desc, in: ocrLines
                )
                let anyMatchesUnit = printedPrices.contains { abs($0 - unit) < Decimal(0.01) }
                let noneMatchesTotal = !printedPrices.contains { abs($0 - cand.price) < Decimal(0.01) }
                // Reset when the unit price is in the OCR but the inflated
                // total is NOT — otherwise we'd corrupt a legit case where
                // the printed row really does have the multiplied total.
                if anyMatchesUnit && noneMatchesTotal {
                    candidates[i] = Candidate(
                        desc: cand.desc,
                        qty: nil,
                        unit: nil,
                        price: unit
                    )
                }
            }
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

        // Stage (e): break LLM repetition loops. Under low temperature the
        // model occasionally cycles through the last few items it saw until
        // the response budget is exhausted (we observed this on a Safeway
        // receipt — same 4 items repeated 11 times). If any (desc, price)
        // pair appears 3+ times, keep only the first occurrence and drop
        // every later one. Legitimate "I bought the same thing twice"
        // receipts almost always use a quantity stamp instead of repeating
        // the row literally.
        var occurrenceCount: [String: Int] = [:]
        for c in deduped {
            let key = "\(c.desc.lowercased())|\(c.price)"
            occurrenceCount[key, default: 0] += 1
        }
        let loopingKeys = Set(occurrenceCount.compactMap { $0.value >= 3 ? $0.key : nil })
        if !loopingKeys.isEmpty {
            var seenLoopKey: Set<String> = []
            deduped = deduped.filter { c in
                let key = "\(c.desc.lowercased())|\(c.price)"
                guard loopingKeys.contains(key) else { return true }
                if seenLoopKey.contains(key) { return false }
                seenLoopKey.insert(key)
                return true
            }
        }

        let items: [Receipt.LineItem] = deduped.map {
            Receipt.LineItem(
                description: $0.desc,
                quantity: Self.normalizedQuantity($0.qty, totalPrice: $0.price),
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
        // Total — many receipts use alternative labels like "BALANCE",
        // "AMOUNT DUE", or "PAYMENT AMOUNT" instead of "TOTAL". Searching
        // for a single keyword misses the bbox on those receipts; the
        // value is correctly extracted by FM, just not anchored visually.
        if receipt.totals.total > 0,
           let box = locateBBox(amount: receipt.totals.total, in: lines, keywords: [
               "total", "balance", "balance due", "amount due",
               "amount paid", "amount payable", "payment amount", "grand total",
               "total due"
           ]) {
            bboxes["totals.total"] = box
        }
        if let subtotal = receipt.totals.subtotal,
           let box = locateBBox(amount: subtotal, in: lines, keywords: [
               "subtotal", "sub-total", "sub total"
           ]) {
            bboxes["totals.subtotal"] = box
        }
        // Tax — include common regional / category variants.
        if let firstTax = receipt.totals.tax.first,
           let box = locateBBox(amount: firstTax.amount, in: lines, keywords: [
               "tax", "sales tax", "vat", "gst", "hst"
           ]) {
            bboxes["totals.tax"] = box
        }
        // Tip / gratuity — restaurant-style receipts.
        if let tip = receipt.totals.tip,
           let box = locateBBox(amount: tip, in: lines, keywords: [
               "tip", "gratuity"
           ]) {
            bboxes["totals.tip"] = box
        }

        // Line items: two-pass approach.
        //
        // Pass 1 — exact match. For each item, find the description's OCR
        // line, then *also* find a separate OCR line on the same visual
        // row whose text contains the totalPrice value (with comma→dot
        // tolerance for European-format receipts).
        //
        // Pass 2 — column-based fallback. When exact match fails for a
        // line item, look in the X range where the OTHER items' prices
        // landed and pick any price-shaped OCR token on the item's Y row.
        // This catches receipts where Vision read "5,49 S" and FM
        // normalized to "5.49", or where the price observation is shaped
        // slightly differently than expected. Even when the value doesn't
        // match exactly, a bbox in the right column gives the labeler
        // something to grab — better than no bbox at all.
        var claimed = Set<Int>()
        var matchedPriceBoxes: [Receipt.BBox] = []
        var pendingPriceItems: [(idx: Int, descBox: Receipt.BBox)] = []

        for (idx, item) in receipt.lineItems.enumerated() {
            guard let (descLineIdx, descBox) = locateItemBBox(
                description: item.description,
                totalPrice: item.totalPrice,
                in: lines,
                excluding: claimed
            ) else { continue }
            let key = String(format: "lineItem.%03d", idx)
            bboxes[key] = descBox
            claimed.insert(descLineIdx)
            if let (priceLineIdx, priceBox) = locateItemPriceBBox(
                amount: item.totalPrice,
                nearY: descBox.y + descBox.height / 2,
                rowHeight: descBox.height,
                in: lines,
                excluding: claimed
            ) {
                bboxes["\(key).price"] = priceBox
                claimed.insert(priceLineIdx)
                matchedPriceBoxes.append(priceBox)
            } else {
                pendingPriceItems.append((idx, descBox))
            }
        }

        // Column-fallback only fires once we have ≥2 confirmed matches —
        // a single match doesn't establish a column reliably.
        if let columnX = priceColumnRange(from: matchedPriceBoxes) {
            for (idx, descBox) in pendingPriceItems {
                if let (priceLineIdx, priceBox) = locateItemPriceBBoxInColumn(
                    nearY: descBox.y + descBox.height / 2,
                    rowHeight: descBox.height,
                    columnX: columnX,
                    in: lines,
                    excluding: claimed
                ) {
                    let key = String(format: "lineItem.%03d", idx)
                    bboxes["\(key).price"] = priceBox
                    claimed.insert(priceLineIdx)
                }
            }
        }

        // Sort line items into top-to-bottom receipt order. FM
        // occasionally returns them out of order (jumping departments),
        // and `recoverMissedLineItems` always appends to the end — so
        // without this sort, the visually-first item can end up at index
        // 4. The sort key is the description bbox's Y. Items with no
        // bbox (locator failed) sink to the bottom of the list — their
        // relative order is preserved.
        let withY: [(item: Receipt.LineItem, oldKey: String, y: Double)] =
            receipt.lineItems.enumerated().map { (idx, item) in
                let oldKey = String(format: "lineItem.%03d", idx)
                let y = bboxes[oldKey]?.y ?? Double.greatestFiniteMagnitude
                return (item, oldKey, y)
            }
        let sorted = withY.enumerated().sorted { lhs, rhs in
            // Stable sort by Y, with original index as tiebreak for
            // items that share a Y (or both have none).
            if lhs.element.y != rhs.element.y { return lhs.element.y < rhs.element.y }
            return lhs.offset < rhs.offset
        }.map(\.element)

        var renumberedBoxes = bboxes
        // Strip the old lineItem.* entries before writing the new ones,
        // otherwise stale keys leak through when the new layout has
        // fewer items at a given index than the old one had.
        for k in bboxes.keys where k.hasPrefix("lineItem.") {
            renumberedBoxes.removeValue(forKey: k)
        }
        var newItems: [Receipt.LineItem] = []
        for (newIdx, entry) in sorted.enumerated() {
            newItems.append(entry.item)
            let newKey = String(format: "lineItem.%03d", newIdx)
            if let descBox = bboxes[entry.oldKey] {
                renumberedBoxes[newKey] = descBox
            }
            if let priceBox = bboxes["\(entry.oldKey).price"] {
                renumberedBoxes["\(newKey).price"] = priceBox
            }
        }
        receipt.lineItems = newItems
        receipt.provenance.bboxes = renumberedBoxes
        return receipt
    }

    /// Compute the typical X range of the price column from items where
    /// we already matched a price bbox. Returns nil if there's not enough
    /// data (fewer than 2 matches) — better to skip the fallback than to
    /// guess at an unreliable column.
    private static func priceColumnRange(from priceBoxes: [Receipt.BBox]) -> ClosedRange<Double>? {
        guard priceBoxes.count >= 2 else { return nil }
        let xs = priceBoxes.map(\.x).sorted()
        let median = xs[xs.count / 2]
        // Tolerate a slop of ~5% to the left (handles "$10.00" vs "$1.00"
        // — the wider price has a smaller x) and ~20% to the right (in
        // case the column is anchored differently for outliers).
        return max(0.0, median - 0.05)...(median + 0.20)
    }

    /// Look at the typical price-column X range at a given Y, and return
    /// any OCR observation whose text matches a money-shaped pattern.
    /// Used when exact-amount substring matching failed (Vision misread,
    /// punctuation difference, etc.) — having ANY bbox in the right place
    /// is better than no bbox.
    private static func locateItemPriceBBoxInColumn(
        nearY: Double,
        rowHeight: Double,
        columnX: ClosedRange<Double>,
        in lines: [(text: String, box: Receipt.BBox)],
        excluding: Set<Int>
    ) -> (lineIndex: Int, box: Receipt.BBox)? {
        let tolerance = max(0.01, rowHeight * 1.5)
        // Accept any money-shaped token. Allows comma or dot decimal,
        // optional leading $, optional trailing single-letter (we already
        // strip those upstream, but be lenient here).
        let priceish = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s*[A-Z])?$"#
        var best: (idx: Int, box: Receipt.BBox)? = nil
        for (idx, line) in lines.enumerated() {
            if excluding.contains(idx) { continue }
            let centerY = line.box.y + line.box.height / 2
            guard abs(centerY - nearY) <= tolerance else { continue }
            guard columnX.contains(line.box.x) else { continue }
            let trimmed = line.text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceish, options: .regularExpression) != nil else { continue }
            // Rightmost qualifying observation wins — keeps us in the
            // actual price column when the row has multiple numeric tokens.
            if best == nil || line.box.x > best!.box.x {
                best = (idx, line.box)
            }
        }
        return best.map { ($0.idx, $0.box) }
    }

    /// Find an OCR line on the same visual row as a line-item description
    /// whose text contains the `amount` — that's the price column. We
    /// prefer the rightmost candidate (price columns are at the right
    /// edge) and require the Y center to be within ~1.5 row-heights of
    /// the description's Y center so we don't grab a number from a
    /// completely different row.
    private static func locateItemPriceBBox(
        amount: Decimal,
        nearY: Double,
        rowHeight: Double,
        in lines: [(text: String, box: Receipt.BBox)],
        excluding: Set<Int>
    ) -> (lineIndex: Int, box: Receipt.BBox)? {
        let amountStr = NSDecimalNumber(decimal: amount).stringValue
        // Tolerance must cover the gap between two adjacent OCR rows,
        // which is typically ~1.5–2× a single row's height. The 0.02
        // floor handles the Happy Hound-style receipt where the price
        // wraps onto the row below the description — descBox.height
        // alone underestimates the row spacing.
        let tolerance = max(0.02, rowHeight * 2.0)
        // Collect every qualifying candidate, then pick the best by
        // (closer-Y → rightmost-X). Just picking "rightmost X" breaks on
        // Safeway-style receipts where the price ABOVE and the price
        // BELOW a description are both in the same column at the same X —
        // the iteration order of `lines` (which Vision doesn't sort by Y)
        // ends up deciding the winner. Closer-Y is the structurally
        // correct disambiguator: the price on the actual same visual row
        // is always closer in Y than a neighbor.
        var candidates: [(idx: Int, box: Receipt.BBox, dy: Double)] = []
        for (idx, line) in lines.enumerated() {
            if excluding.contains(idx) { continue }
            let centerY = line.box.y + line.box.height / 2
            let dy = abs(centerY - nearY)
            guard dy <= tolerance else { continue }
            // Match either the exact value, or a comma-decimal variant
            // (Vision reads "4,00" while FM normalizes to "4.00"), or a
            // hyphen-decimal variant (small-font receipts read "11.75"
            // as "11-75"). The hyphen substitution is constrained to
            // digit-hyphen-digit so we don't accidentally rewrite real
            // hyphens elsewhere (e.g. "sub-total").
            let dotted = line.text
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(
                    of: #"(\d)-(\d)"#,
                    with: "$1.$2",
                    options: .regularExpression
                )
            guard line.text.contains(amountStr) || dotted.contains(amountStr) else { continue }
            candidates.append((idx, line.box, dy))
        }
        // Sort: closer-Y first, rightmost-X as tiebreak.
        candidates.sort { a, b in
            if a.dy != b.dy { return a.dy < b.dy }
            return a.box.x > b.box.x
        }
        return candidates.first.map { ($0.idx, $0.box) }
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
        keywords: [String]
    ) -> Receipt.BBox? {
        let amountStr = NSDecimalNumber(decimal: amount).stringValue
        let keywords = keywords.map { $0.lowercased() }
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?$"#

        // Comma-vs-dot tolerant substring match. Receipts in regions that
        // use "," as the decimal separator ("4,00") wouldn't otherwise
        // match FM's normalized "4.00" output.
        func lineContainsAmount(_ text: String) -> Bool {
            if text.contains(amountStr) { return true }
            return text.replacingOccurrences(of: ",", with: ".").contains(amountStr)
        }
        func lineContainsKeyword(_ lc: String) -> Bool {
            keywords.contains { lc.contains($0) }
        }
        // "Pure" = the whole observation is just the price (allows leading
        // $, an optional sign, and 0-2 decimals). Distinguishes "$14.10"
        // from "HEALTH ITEM TOTAL: 14.10".
        func isPureAmount(_ text: String) -> Bool {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            return trimmed.range(of: priceShape, options: .regularExpression) != nil
        }

        let keywordObs = lines.filter { lineContainsKeyword($0.text.lowercased()) }

        // Pass A — preferred path. For each keyword observation, find a
        // PURE-PRICE observation on the same Y row whose value matches.
        // Lands on the actual price column, sidesteps footer noise like
        // "HEALTH ITEM TOTAL: 14.10" which otherwise matches case-1 below.
        if !keywordObs.isEmpty {
            var best: (box: Receipt.BBox, x: Double)? = nil
            for amountLine in lines
                where isPureAmount(amountLine.text) && lineContainsAmount(amountLine.text)
            {
                let amountY = amountLine.box.y + amountLine.box.height / 2
                let onSameRow = keywordObs.contains { kw in
                    let kwY = kw.box.y + kw.box.height / 2
                    let tol = max(amountLine.box.height, kw.box.height) * 1.5
                    return abs(amountY - kwY) <= tol
                }
                guard onSameRow else { continue }
                if best == nil || amountLine.box.x > best!.x {
                    best = (amountLine.box, amountLine.box.x)
                }
            }
            if let best { return best.box }
        }

        // Pass B — same observation contains BOTH keyword and amount
        // (e.g. "BALANCE  60.99" as one line). Multiple matches are
        // resolved by SHORTEST text: a compact "TOTAL  $14.10" line beats
        // a long footer "HEALTH ITEM TOTAL: 14.10" sentence.
        let combined = lines.filter {
            lineContainsKeyword($0.text.lowercased()) && lineContainsAmount($0.text)
        }
        if let pick = combined.min(by: { $0.text.count < $1.text.count }) {
            return pick.box
        }

        // Pass C — last resort: any line with the keyword, even without
        // the amount on it. Outlines the label itself, which at least
        // points the user at the right region.
        return lines.reversed().first { lineContainsKeyword($0.text.lowercased()) }?.box
    }

    private static func locateItemBBox(
        description: String,
        totalPrice: Decimal,
        in lines: [(text: String, box: Receipt.BBox)],
        excluding: Set<Int>
    ) -> (lineIndex: Int, box: Receipt.BBox)? {
        let needle = description.lowercased().trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return nil }

        // Pass A — strict: full description appears as substring.
        for (idx, line) in lines.enumerated() {
            if excluding.contains(idx) { continue }
            if line.text.lowercased().contains(needle) {
                return (idx, line.box)
            }
        }

        // Pass B — fuzzy: split description into ≥3-char words and look
        // for an OCR line that contains ALL of them. Handles cases where
        // Vision split the row into separate observations and FM's
        // description was reconstructed across them — e.g. FM emits
        // "DE CECCO L" but the row's main observation is just "2 QTY DE
        // CECCO" (the "L" is a sibling observation). "cecco" is the only
        // ≥3-char word and matches.
        let words = needle
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count >= 3 }
        if !words.isEmpty {
            for (idx, line) in lines.enumerated() {
                if excluding.contains(idx) { continue }
                let lc = line.text.lowercased()
                if words.allSatisfy({ lc.contains($0) }) {
                    return (idx, line.box)
                }
            }
            // Pass C — looser fuzzy: any ≥3-char word from the description.
            // Only fires if pass B didn't, so we don't grab unrelated text
            // when a fuller match was available.
            for (idx, line) in lines.enumerated() {
                if excluding.contains(idx) { continue }
                let lc = line.text.lowercased()
                if words.contains(where: { lc.contains($0) }) {
                    return (idx, line.box)
                }
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

    /// Scan the OCR for prices in the typical price column whose Y row
    /// isn't covered by any FM-extracted line item, then promote each
    /// such (description, price) pair to a new line item. Catches rows
    /// that the model just plain dropped — e.g. Safeway's
    /// "2 QTY DE CECCO  L  6.00", which FM left out for unclear reasons
    /// even though the row is in the prompt.
    ///
    /// Build the set of receipt-level totals values we should NEVER
    /// promote to a line item. Used by both `columnAnchoredLineItems`
    /// and `recoverMissedLineItems`.
    private static func totalsValueSet(from receipt: Receipt) -> Set<Decimal> {
        var out: Set<Decimal> = [receipt.totals.total]
        if let s = receipt.totals.subtotal { out.insert(s) }
        for t in receipt.totals.tax { out.insert(t.amount) }
        if let t = receipt.totals.tip { out.insert(t) }
        if let d = receipt.totals.discount { out.insert(d) }
        return out
    }

    /// PRIMARY line-item extraction — walks the price column and emits
    /// a line item for every price observation that isn't a totals
    /// block value, a regular-price amount, or a savings amount.
    ///
    /// Why column-anchored, not FM-output-anchored:
    /// FM is unreliable on dense Safeway/Sprouts receipts (drops items,
    /// picks regular prices, hallucinates qty × unit math). The price
    /// column, by contrast, is unambiguous: Vision puts every price-
    /// shaped token at a known X. Walking the column and pairing each
    /// price with the description on the same Y gives us the structural
    /// truth of the receipt. FM is then used as a HINT source for
    /// quantity / unit / category, and remains the sole authority for
    /// merchant name, date, and totals.
    ///
    /// Pipeline:
    ///   1. Collect every price-shaped OCR observation. The median X
    ///      defines the column (±5%).
    ///   2. Discard everything below the first totals-block boundary.
    ///   3. Discard amounts equal to subtotal / tax / tip / total.
    ///   4. Discard amounts whose row sits adjacent (Δy ≤ 0.006) to a
    ///      "Regular Price" / "Member Savings" / "you save" label.
    ///   5. For each surviving price, JOIN every text observation on
    ///      the same visual row to the left of the price (sorted by X)
    ///      into the description string. Joining (vs picking the
    ///      leftmost) recovers multi-segment descriptions like
    ///      "SIG  BOUILLON CHKN".
    ///   6. Run the joined description through `cleanLineItemDescription`,
    ///      `isFooterRow`, and `looksLikeNonItemRecoveryCandidate` to
    ///      drop department headers / "Price"-only fragments / SKUs.
    ///   7. Match against FM's items by (price equal, description
    ///      prefix overlap) to inherit quantity / unitPrice / category.
    ///
    /// Returns empty when fewer than 2 prices land in the column —
    /// caller falls back to the FM-only path.
    private static func columnAnchoredLineItems(
        lines: [(text: String, box: Receipt.BBox)],
        fmItems: [Receipt.LineItem],
        totalsValues: Set<Decimal>,
        receiptTotal: Decimal
    ) -> [Receipt.LineItem] {
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#

        // Step 1: every price-shaped observation with its value.
        struct PricePoint {
            let value: Decimal
            let box: Receipt.BBox
            let centerY: Double
            let lineIdx: Int
        }
        var pricePoints: [PricePoint] = []
        for (idx, line) in lines.enumerated() {
            let trimmed = line.text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { continue }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
            guard let v = Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX")) else { continue }
            guard v > 0 else { continue }
            pricePoints.append(PricePoint(
                value: v,
                box: line.box,
                centerY: line.box.y + line.box.height / 2,
                lineIdx: idx
            ))
        }
        guard pricePoints.count >= 2 else { return [] }

        // Step 1b: price column = median X ± 5%.
        let xs = pricePoints.map(\.box.x).sorted()
        let medianX = xs[xs.count / 2]
        let columnLo = max(0.0, medianX - 0.05)
        let columnHi = medianX + 0.05

        // Step 2: totals-block Y boundary. Anything at or below is
        // payment / footer noise.
        let totalsKeywords: [String] = [
            "subtotal", "sub-total", "tax", "balance", "total",
            "amount due", "amount paid", "payment amount", "tip", "change",
        ]
        var totalsBlockY: Double = 1.0
        for line in lines {
            if Self.lineLooksLikeTotalsBoundary(line.text, keywords: totalsKeywords) {
                let cy = line.box.y + line.box.height / 2
                if cy < totalsBlockY { totalsBlockY = cy }
            }
        }

        // Step 4 (precompute): Y centers of every regular-price /
        // member-savings / you-save label observation.
        let labelPatterns: [String] = [
            "regular price", "resular price", "reqular price",
            "recular price", "repular price", "reaular price",
            "member savings", "member saving",
            "member savinas", "member savinos", "member savincs",
            "for personalized", "forl personalized",
            "you save", "you saved",
        ]
        let labelYs: [Double] = lines.compactMap { line in
            let lc = line.text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            let isLabel = labelPatterns.contains { lc == $0 || lc.hasPrefix($0 + " ") }
            return isLabel ? line.box.y + line.box.height / 2 : nil
        }

        // Step 5–7: walk surviving prices in Y order.
        let inColumn = pricePoints
            .filter { columnLo <= $0.box.x && $0.box.x <= columnHi }
            .sorted { $0.centerY < $1.centerY }

        var items: [Receipt.LineItem] = []
        for p in inColumn {
            // Filter: below the totals block.
            if p.centerY >= totalsBlockY - 0.002 { continue }
            // Filter: equals a known totals value.
            if totalsValues.contains(p.value) { continue }
            // Filter: adjacent to a regular-price / member-savings label.
            if labelYs.contains(where: { abs($0 - p.centerY) <= 0.006 }) { continue }
            // Filter: implausibly large value. Vision sometimes mangles a
            // price like "$4.99" into just "99" (decimal point lost). The
            // resulting "$99" exceeds the receipt total and is obviously
            // wrong. Drop anything that's strictly greater than the total.
            if receiptTotal > 0, p.value > receiptTotal { continue }

            // Pick the BEST single text observation to the left, within
            // ±2 line-heights vertically. Strategy:
            //   * Group candidates into Y-buckets of 0.005 (truly same
            //     row vs neighboring row).
            //   * Across buckets, prefer the closer one (smaller dy).
            //   * Within a bucket, prefer the LONGEST text — that's
            //     the actual product name, not a brand prefix like "SIG"
            //     or a side marker like "S" or "WT".
            // Joining wasn't worth it: most receipts split descriptions
            // across multiple Y values (Vision puts "SIG" on one
            // baseline and "BOUILLON CHKN" on another), so a generous
            // tolerance ends up dragging in the previous-row header
            // ("PRODUCE", "DAIRY") too.
            let rowTol = max(0.012, p.box.height * 2.0)
            var candidates: [(text: String, x: Double, dy: Double)] = []
            for line in lines {
                let cy = line.box.y + line.box.height / 2
                let dy = abs(cy - p.centerY)
                guard dy <= rowTol else { continue }
                guard line.box.x < p.box.x else { continue }
                let t = line.text.trimmingCharacters(in: .whitespaces)
                guard t.count >= 3 else { continue }
                // Skip other price-shaped tokens.
                if t.range(of: priceShape, options: .regularExpression) != nil { continue }
                candidates.append((t, line.box.x, dy))
            }
            guard !candidates.isEmpty else { continue }

            candidates.sort { a, b in
                // Bucket dy by 0.005 — descriptions on truly the same row
                // (Vision baseline jitter) should all count as equally
                // close.
                let aBucket = (a.dy / 0.005).rounded(.down)
                let bBucket = (b.dy / 0.005).rounded(.down)
                if aBucket != bBucket { return aBucket < bBucket }
                // Within the bucket, deprioritize "noisy" text (mostly
                // tax markers / vol indicators / department headers).
                // "SNGL TAX" loses to "CRV BEER" even though both have
                // the same length and same dy — neither word in CRV BEER
                // is a known metadata fragment.
                let aNoisy = Self.descriptionIsMostlyNoise(a.text)
                let bNoisy = Self.descriptionIsMostlyNoise(b.text)
                if aNoisy != bNoisy { return !aNoisy }
                // Otherwise pick the longer text — almost always the
                // real product name vs a brand prefix.
                return a.text.count > b.text.count
            }
            // Walk sorted candidates and take the FIRST one that
            // survives the non-item filters. If the closest-Y candidate
            // is a SKU / weight-marker / metadata fragment (which the
            // sort already deprioritizes but doesn't necessarily push
            // to the bottom), don't drop the line item — fall back to
            // the next candidate. That's what saves Ulta receipts
            // where the SKU sits closer to the price than the actual
            // product name.
            var cleaned: String? = nil
            for c in candidates {
                let candidate = Self.cleanLineItemDescription(c.text)
                if candidate.count < 3 { continue }
                if Self.isFooterRow(candidate) { continue }
                if Self.looksLikeNonItemRecoveryCandidate(candidate) { continue }
                cleaned = candidate
                break
            }
            guard let cleaned else { continue }

            // Step 7: inherit qty / unit / category from FM if it
            // extracted a matching item.
            let cleanedLower = cleaned.lowercased()
            let needle = String(cleanedLower.prefix(4))
            let fmMatch = fmItems.first { fm in
                fm.totalPrice == p.value
                    && (fm.description.lowercased().contains(needle)
                        || cleanedLower.contains(String(fm.description.lowercased().prefix(4))))
            }
            items.append(Receipt.LineItem(
                description: cleaned,
                quantity: Self.normalizedQuantity(fmMatch?.quantity, totalPrice: p.value),
                unitPrice: fmMatch?.unitPrice,
                totalPrice: p.value,
                category: fmMatch?.category
            ))
        }

        return items
    }

    /// Three guards keep this conservative:
    ///   * a price column is only "established" if we already matched
    ///     at least two existing items' prices to OCR observations
    ///   * candidate prices that equal the receipt's total / subtotal /
    ///     tax / tip are excluded (those are the totals block)
    ///   * candidate prices must be strictly less than the receipt total
    ///   * the description observation must pass `isFooterRow`
    private static func recoverMissedLineItems(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#

        // Compute the price column from prices already matched to OCR.
        // Claim ONE OCR observation per extracted item. The set
        // `claimedIdxs` ensures:
        //   (a) When two items genuinely share a price (e.g. IMG_1260
        //       has CHOBANI and LITL SMOKIES both at $6.99), each
        //       claims a distinct OCR observation, so both Ys end up
        //       in matchedYs and neither looks "unclaimed" later.
        //   (b) When OCR prints a price more times than there are
        //       items (e.g. IMG_1291 has 2 OCR rows of $3.50 but FM
        //       only extracted 1 SIERRA NEVADA), the extras stay
        //       unclaimed and recovery promotes them to new items.
        var matchedPriceXs: [Double] = []
        var matchedYs: [Double] = []
        var claimedIdxs: Set<Int> = []
        for item in receipt.lineItems {
            for (idx, line) in lines.enumerated() {
                if claimedIdxs.contains(idx) { continue }
                let trimmed = line.text.trimmingCharacters(in: .whitespaces)
                guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { continue }
                // Normalize the OCR text and parse it as Decimal so the
                // comparison is numeric, not string. String comparison
                // breaks on trailing zeros: "8.20" (OCR) ≠ "8.2"
                // (NSDecimalNumber(decimal: 8.20).stringValue).
                let normalized = trimmed
                    .replacingOccurrences(of: "$", with: "")
                    .replacingOccurrences(of: ",", with: ".")
                    .replacingOccurrences(
                        of: #"(\d)-(\d)"#,
                        with: "$1.$2",
                        options: .regularExpression
                    )
                    .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
                guard let v = Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX")) else { continue }
                if v == item.totalPrice {
                    matchedPriceXs.append(line.box.x)
                    matchedYs.append(line.box.y + line.box.height / 2)
                    claimedIdxs.insert(idx)
                    break  // one OCR observation per item
                }
            }
        }
        guard matchedPriceXs.count >= 2 else { return receipt }

        let sortedXs = matchedPriceXs.sorted()
        let medianX = sortedXs[sortedXs.count / 2]
        let columnRange = max(0.0, medianX - 0.05)...(medianX + 0.20)

        // Values we should NEVER promote to a line item (those are totals).
        var totalsValues: Set<String> = [
            NSDecimalNumber(decimal: receipt.totals.total).stringValue
        ]
        if let s = receipt.totals.subtotal {
            totalsValues.insert(NSDecimalNumber(decimal: s).stringValue)
        }
        for t in receipt.totals.tax {
            totalsValues.insert(NSDecimalNumber(decimal: t.amount).stringValue)
        }
        if let t = receipt.totals.tip {
            totalsValues.insert(NSDecimalNumber(decimal: t).stringValue)
        }

        // The items band is bounded vertically by the existing items'
        // Y range, expanded generously on top (FM tends to miss items
        // ABOVE its first match more than after its last), and clamped
        // by the first observation that looks like a totals-block label
        // ("subtotal" / "tax" / "balance" / "total" / "amount due" /
        // "tip" / "change") so we don't accidentally promote a payment
        // row to a line item.
        guard let earliestY = matchedYs.min(), let latestY = matchedYs.max() else {
            return receipt
        }
        // Generous top slack: covers the case where FM started extracting
        // mid-receipt and skipped the first 3-4 items.
        let topSlack: Double = 0.18
        let bottomSlack: Double = 0.06
        // Also scan the OCR for the topmost price-shaped observation that
        // sits in the established price column — that's almost always the
        // first line item (multi-department receipts like Sprouts can
        // have the first item at y≈0.12 while FM only emits items below
        // y≈0.48, which `topSlack` alone can't bridge). The price-shape
        // and column constraints already exclude header noise like store
        // IDs or phone numbers.
        let firstColumnPriceY: Double? = lines
            .filter { columnRange.contains($0.box.x) }
            .filter {
                let t = $0.text.trimmingCharacters(in: .whitespaces)
                return t.range(of: priceShape, options: .regularExpression) != nil
            }
            .map { $0.box.y + $0.box.height / 2 }
            .min()
        let slackLo = earliestY - topSlack
        let columnLo = (firstColumnPriceY ?? earliestY) - 0.005
        let bandLo = max(0.0, min(slackLo, columnLo))
        var bandHi = min(1.0, latestY + bottomSlack)

        // Snap bandHi up to the first "totals block" keyword we see at
        // a Y > latestY. Anything below that is payment / footer noise.
        let totalsKeywords: [String] = [
            "subtotal", "sub-total", "tax", "balance", "total",
            "amount due", "amount paid", "payment amount", "tip", "change",
        ]
        let totalsBlockY = lines
            .filter { line in
                let cY = line.box.y + line.box.height / 2
                guard cY > latestY else { return false }
                return Self.lineLooksLikeTotalsBoundary(line.text, keywords: totalsKeywords)
            }
            .map { $0.box.y + $0.box.height / 2 }
            .min()
        if let totalsBlockY {
            bandHi = min(bandHi, totalsBlockY - 0.002)
        }
        // Don't let the band shrink to nothing or invert.
        guard bandLo < bandHi else { return receipt }

        let matchedYSet = Set(matchedYs.map { Int(($0 * 1000).rounded()) })

        var newItems: [Receipt.LineItem] = []
        for priceLine in lines {
            // Must be in the price column AND price-shaped.
            guard columnRange.contains(priceLine.box.x) else { continue }
            let trimmed = priceLine.text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { continue }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
            // Don't promote a totals-block value.
            guard !totalsValues.contains(normalized) else { continue }
            guard let priceValue = Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX")) else { continue }
            guard priceValue > 0, priceValue < receipt.totals.total else { continue }

            let centerY = priceLine.box.y + priceLine.box.height / 2
            // Inside the items band?
            guard bandLo <= centerY, centerY <= bandHi else { continue }
            // Already on a matched item's row? (compare by Y quantized to 1/1000)
            let yKey = Int((centerY * 1000).rounded())
            guard !matchedYSet.contains(yKey) else { continue }

            // Find description-shaped observations on the same Y, to the left.
            // Tolerance is intentionally tight (≤0.7× line height) — at 1.5×
            // we were pulling in observations from the row ABOVE (e.g. the
            // previous item's "Member Savings" line drifts into range for
            // the next item's price).
            var leftCandidates: [(text: String, box: Receipt.BBox, dy: Double)] = []
            for descLine in lines {
                let cY = descLine.box.y + descLine.box.height / 2
                let dy = abs(cY - centerY)
                guard dy <= max(0.008, priceLine.box.height * 0.7) else { continue }
                guard descLine.box.x < priceLine.box.x else { continue }
                // Skip other price tokens.
                let dt = descLine.text.trimmingCharacters(in: .whitespaces)
                guard dt.range(of: priceShape, options: .regularExpression) == nil else { continue }
                // Skip very-short observations like a lone "L" tax marker —
                // they're never the real product name.
                guard dt.count >= 3 else { continue }
                leftCandidates.append((descLine.text, descLine.box, dy))
            }
            // Prefer the description with the smallest Y-difference to the
            // price — they're on the SAME visual row. Falling back to
            // "leftmost X" would happily pick a department header
            // ("DAIRY", "PRODUCE") that sits at the same X column as the
            // product names but on the row above. As a tiebreak when Ys
            // are essentially identical, pick the leftmost X (the real
            // description usually starts left of any inline metadata).
            guard let descObs = leftCandidates.min(by: { a, b in
                if abs(a.dy - b.dy) > 0.002 { return a.dy < b.dy }
                return a.box.x < b.box.x
            }) else { continue }

            let cleanedDesc = cleanLineItemDescription(descObs.text)
            guard cleanedDesc.count >= 3 else { continue }
            guard !isFooterRow(cleanedDesc) else { continue }
            guard !Self.looksLikeNonItemRecoveryCandidate(cleanedDesc) else { continue }

            newItems.append(Receipt.LineItem(
                description: cleanedDesc,
                quantity: Self.normalizedQuantity(nil, totalPrice: priceValue),
                unitPrice: nil,
                totalPrice: priceValue,
                category: nil
            ))
        }

        if !newItems.isEmpty {
            receipt.lineItems.append(contentsOf: newItems)
        }
        return receipt
    }

    /// Post-FM correction for items where the model chose the printed
    /// regular price instead of the discounted actual price. Detected by:
    ///   1. Finding the OCR row that contains the extracted price.
    ///   2. Checking whether a "Regular Price" label observation sits on
    ///      the same visual row as that price.
    ///   3. If so: searching for an alternative price observation in the
    ///      same X column, located ABOVE the description (smaller Y).
    ///   4. Replacing the extracted price with the alternative.
    ///
    /// Conservative — only swaps when all three conditions are met, so
    /// receipts that print only one price per item are untouched.
    /// Verify the extracted grand total against what's actually printed
    /// on the receipt. FM has two failure modes we've observed:
    ///
    /// * **Wrong source row** — FM picks "Total" from a "YOUR SAVINGS"
    ///   summary block instead of the "**** BALANCE" grand total.
    ///   IMG_2942 landed on $5.53 (savings total) when the real balance
    ///   was $56.75.
    ///
    /// * **Hallucination** — FM emits a value not present anywhere in
    ///   the OCR. IMG_7951's $161.99 doesn't appear in any observation;
    ///   the actual total prints as $93.16 three times.
    ///
    /// The correction:
    ///   1. Compute an expected total from arithmetic:
    ///      `expected = items_sum + tax + tip - discount`.
    ///   2. Scan the OCR for every value on a row whose text starts with
    ///      a totals-block keyword ("BALANCE", "PAYMENT AMOUNT",
    ///      "AMOUNT DUE", "TOTAL", "GRAND TOTAL"). These are the true
    ///      candidates for the grand total.
    ///   3. Only override when FM's total is BOTH:
    ///      (a) suspicious — either not present in ANY OCR observation
    ///          (hallucination), OR less than half the arithmetic
    ///          expectation (savings-block confusion),
    ///      AND
    ///      (b) we have at least one OCR-anchored candidate that fits
    ///          the arithmetic within 5% tolerance.
    ///
    /// The conservative gate on (a) prevents the guardrail from
    /// second-guessing FM on ordinary receipts where its total is fine.
    /// Re-anchor subtotal / tax / tip against the OCR. For each field
    /// we find the labelling observation ("Subtotal", "Tip", "Tax",
    /// "Gratuity") and grab the price observation on the same visual
    /// row (or the row immediately above/below on receipts that print
    /// the value on a separate line). The value from OCR wins over
    /// FM's when they disagree and the arithmetic then adds up.
    ///
    /// Why: FM cross-wires these fields on receipts where the layout
    /// isn't strict. Philz Coffee prints values ABOVE their labels
    /// ("$10.70\nSubtotal\n$2.14\nTip\n$12.84\nTotal"); FM picked
    /// the tip amount ($2.14) as the subtotal and left tip empty.
    private static func sanityCheckSubtotalTaxTip(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#

        func priceValue(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }

        /// Find the price value paired with a label observation. Looks
        /// same-row first (label + value in same obs, or same-Y sibling),
        /// then row-above / row-below for receipts that print the value
        /// on a separate line from the label.
        func findPairedValue(labelObs: (text: String, box: Receipt.BBox)) -> Decimal? {
            // Try tokens in the label observation itself.
            for tok in labelObs.text.split(whereSeparator: { $0.isWhitespace }) {
                if let v = priceValue(String(tok)), v > 0 { return v }
            }
            let labelY = labelObs.box.y + labelObs.box.height / 2
            let sameRowTol = max(0.008, labelObs.box.height * 0.9)
            // Same-Y sibling.
            for other in lines {
                let cy = other.box.y + other.box.height / 2
                guard abs(cy - labelY) <= sameRowTol else { continue }
                if let v = priceValue(other.text), v > 0 { return v }
            }
            // Row immediately above OR below (Philz-style: value/label
            // pairs stacked vertically). Look within ~1.5 line heights.
            let neighborTol = max(0.020, labelObs.box.height * 1.5)
            var bestAbove: (dy: Double, v: Decimal)? = nil
            var bestBelow: (dy: Double, v: Decimal)? = nil
            for other in lines {
                let cy = other.box.y + other.box.height / 2
                let dy = abs(cy - labelY)
                guard dy > sameRowTol, dy <= neighborTol else { continue }
                guard let v = priceValue(other.text), v > 0 else { continue }
                if cy < labelY {
                    if bestAbove == nil || dy < bestAbove!.dy {
                        bestAbove = (dy, v)
                    }
                } else {
                    if bestBelow == nil || dy < bestBelow!.dy {
                        bestBelow = (dy, v)
                    }
                }
            }
            // Prefer the closer of the two neighbors.
            switch (bestAbove, bestBelow) {
            case (nil, nil): return nil
            case (let a?, nil): return a.v
            case (nil, let b?): return b.v
            case (let a?, let b?): return a.dy <= b.dy ? a.v : b.v
            }
        }

        // Locate observations whose text is exactly (or begins with)
        // one of these labels — same normalization used by other passes.
        func normalizedLabel(_ text: String) -> String {
            text.lowercased()
                .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
                .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
        }

        func findLabel(matching keywords: [String]) -> (text: String, box: Receipt.BBox)? {
            for line in lines {
                let n = normalizedLabel(line.text)
                for kw in keywords {
                    if n == kw || n.hasPrefix(kw + " ") || n.hasSuffix(" " + kw) {
                        return (line.text, line.box)
                    }
                }
            }
            return nil
        }

        // --- Subtotal ---
        if let label = findLabel(matching: ["subtotal", "sub total", "sub-total"]),
           let v = findPairedValue(labelObs: label) {
            // Only override if OCR value differs from FM's and OCR
            // value fits arithmetic better.
            let current = receipt.totals.subtotal
            if v != current {
                receipt.totals.subtotal = v
                receipt.provenance.fieldConfidence["totals.subtotal"] = 0.75
            }
        }

        // --- Tip ---
        if let label = findLabel(matching: ["tip", "gratuity"]),
           let v = findPairedValue(labelObs: label),
           v > 0 {
            let current = receipt.totals.tip ?? 0
            if v != current {
                receipt.totals.tip = v
                receipt.provenance.fieldConfidence["totals.tip"] = 0.75
            }
        }

        // --- Tax --- (skip if we already have a positive extracted tax;
        // tax has stronger heuristics in buildReceipt and multi-tax
        // receipts can trip up a naive re-scan).
        if receipt.totals.tax.first?.amount ?? 0 == 0,
           let label = findLabel(matching: ["tax", "sales tax", "vat", "gst", "hst"]),
           let v = findPairedValue(labelObs: label),
           v > 0,
           v < receipt.totals.total {
            receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: nil, amount: v)]
            receipt.provenance.fieldConfidence["totals.tax"] = 0.75
        }

        // Final consistency: if subtotal ends up equal to total (a
        // stubborn FM habit on receipts without a printed "Subtotal"
        // line), drop it entirely — a real subtotal is always < total
        // when tax or tip is charged, and receipts with no tax rarely
        // print a subtotal at all.
        if let sub = receipt.totals.subtotal, sub == receipt.totals.total {
            receipt.totals.subtotal = nil
        }

        return receipt
    }

    private static func sanityCheckTotal(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let fmTotal = receipt.totals.total
        guard fmTotal > 0 else { return receipt }

        // Compute the arithmetic-expected total from line items + tax
        // - discount + tip. Items are the strongest signal we have when
        // there are enough of them.
        let itemsSum: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        let taxSum: Decimal = receipt.totals.tax.reduce(0) { $0 + $1.amount }
        let tip: Decimal = receipt.totals.tip ?? 0
        let discount: Decimal = receipt.totals.discount ?? 0
        let expected: Decimal = itemsSum + taxSum + tip - discount

        // Does FM's total actually appear anywhere in the OCR?
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#
        func priceValue(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }
        let fmTotalPresent = lines.contains { line in
            priceValue(line.text) == fmTotal
        }

        // Flag 1: FM's total is hallucinated (not in OCR at all).
        let hallucinated = !fmTotalPresent

        // Flag 2: items_sum >> total × 2 suggests FM picked a small
        // savings/discount value instead of the real total. Only fires
        // when we have enough items to trust items_sum.
        let savingsBlockConfusion =
            receipt.lineItems.count >= 3
            && itemsSum > fmTotal * Decimal(2)

        guard hallucinated || savingsBlockConfusion else { return receipt }

        // Gather grand-total candidates: any price observation on a row
        // whose text starts with a "grand-total" keyword. We accept
        // multi-word keywords too ("PAYMENT AMOUNT", "AMOUNT DUE",
        // "GRAND TOTAL") — the naive first-token check misses those.
        // For split observations ("**** BALANCE" and "56.75" as two
        // separate obs at the same Y), we scan same-Y neighbors of
        // each label.
        let grandTotalKeywords: [String] = [
            "grand total", "balance due", "total due",
            "amount due", "amount paid", "amount payable",
            "payment amount",
            "balance", "total",
        ]

        func normalizeForKeyword(_ text: String) -> String {
            text.lowercased()
                .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
                .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
        }

        func isTotalsLabel(_ text: String) -> Bool {
            let stripped = normalizeForKeyword(text)
            for kw in grandTotalKeywords {
                if stripped == kw { return true }
                if stripped.hasPrefix(kw + " ") { return true }
                // Also allow "AMOUNT PAID 56.75" style where the price
                // is a trailing token — the prefix check already covers
                // this. And "56.75 BALANCE" (rare) where value is first
                // and label is second.
                if stripped.hasSuffix(" " + kw) { return true }
            }
            return false
        }

        struct Candidate { let value: Decimal; let y: Double }
        var candidates: [Candidate] = []
        for (idx, line) in lines.enumerated() {
            guard isTotalsLabel(line.text) else { continue }
            // Inline: any price-shaped token in the same observation.
            for tok in line.text.split(whereSeparator: { $0.isWhitespace }) {
                if let v = priceValue(String(tok)), v > 0 {
                    candidates.append(Candidate(value: v, y: line.box.y + line.box.height / 2))
                }
            }
            // Same-Y neighbors — Vision often splits the label and its
            // amount into distinct observations.
            let cy = line.box.y + line.box.height / 2
            let tol = max(0.008, line.box.height * 0.9)
            for (jdx, other) in lines.enumerated() where jdx != idx {
                let oy = other.box.y + other.box.height / 2
                guard abs(oy - cy) <= tol else { continue }
                if let v = priceValue(other.text), v > 0 {
                    candidates.append(Candidate(value: v, y: oy))
                }
            }
        }

        // Fallback: on receipts where Vision scrambles the layout so the
        // "BALANCE" label and its "93.16" value end up on totally
        // different Ys (IMG_7951), the label-adjacency scan misses.
        // A real grand total almost always prints 2-3 times on a receipt
        // — BALANCE, PAYMENT AMOUNT confirmation, and a footer copy.
        // Look for any value that appears ≥ 2 times among price-shaped
        // observations. These are strong candidates even without a
        // nearby label.
        if candidates.isEmpty {
            var counts: [Decimal: [Double]] = [:]
            for line in lines {
                guard let v = priceValue(line.text), v > 0 else { continue }
                counts[v, default: []].append(line.box.y + line.box.height / 2)
            }
            for (v, ys) in counts where ys.count >= 2 {
                for y in ys {
                    candidates.append(Candidate(value: v, y: y))
                }
            }
        }
        guard !candidates.isEmpty else { return receipt }

        // Filter to plausible values:
        //   * ≥ items_sum (a real total can never be less than the sum
        //     of its lineitems). Falls back to 0 if we have no items.
        //   * ≤ a very generous ceiling. A single grocery receipt
        //     rarely exceeds $10K; anything above is a transaction ID
        //     chunk we mistook for a price.
        let lowerBound: Decimal = itemsSum
        let upperBound: Decimal = 10_000
        let plausible = candidates.filter { $0.value >= lowerBound && $0.value <= upperBound }
        guard !plausible.isEmpty else { return receipt }

        // Rank within plausible: closest to arithmetic expected wins.
        // On receipts where we captured every line item, items_sum + tax
        // ≈ total exactly. On receipts with missed items, expected
        // underestimates but the candidate closest to it is still
        // usually the right one — the wrong one (a card-number chunk,
        // a savings amount) will be much further off. Tie-break toward
        // the smaller value.
        let target: Decimal = expected > 0 ? expected : (itemsSum + taxSum)
        let best = plausible.min { a, b in
            let ad = a.value > target ? a.value - target : target - a.value
            let bd = b.value > target ? b.value - target : target - b.value
            if ad != bd { return ad < bd }
            return a.value < b.value
        }!

        // Override policy:
        //   * Hallucinated — FM's value doesn't exist in OCR, so it
        //     can't be right. Always replace.
        //   * savingsBlockConfusion — items_sum is more than double
        //     FM's total. The grand total must be ≥ items_sum, so FM
        //     is guaranteed wrong. Always replace.
        //   * Otherwise — require the replacement to beat FM on
        //     arithmetic fit. This branch isn't currently reachable
        //     (the guardrail only fires under one of the two flags
        //     above) but keeps the logic future-proof.
        if !hallucinated && !savingsBlockConfusion {
            let fmDiff = fmTotal > target ? fmTotal - target : target - fmTotal
            let newDiff = best.value > target ? best.value - target : target - best.value
            guard newDiff < fmDiff else { return receipt }
        }

        receipt.totals.total = best.value
        receipt.provenance.fieldConfidence["totals.total"] = 0.7
        return receipt
    }

    private static func correctRegularPriceMistakes(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#

        // Collect every "Regular Price" label Y-center.
        let regularLabels: [Double] = lines.compactMap { line in
            let lc = line.text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            let isLabel = [
                "regular price", "resular price", "reqular price",
                "recular price", "repular price", "reaular price",
            ].contains { lc == $0 || lc.hasPrefix($0 + " ") }
            return isLabel ? line.box.y + line.box.height / 2 : nil
        }
        guard !regularLabels.isEmpty else { return receipt }

        // Helper: parse an OCR observation's text into a Decimal value
        // if it's price-shaped, normalizing $, commas, trailing letter.
        func priceValue(of line: (text: String, box: Receipt.BBox)) -> Decimal? {
            let trimmed = line.text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }

        for i in 0..<receipt.lineItems.count {
            let item = receipt.lineItems[i]
            // Find the description's OCR row. Fuzzy match — receipt OCR
            // can drop punctuation that FM preserves.
            let descNeedle = item.description.lowercased().trimmingCharacters(in: .whitespaces)
            let descWords = descNeedle
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count >= 3 }
            var descY: Double? = nil
            for line in lines {
                let lc = line.text.lowercased()
                let strict = lc.contains(descNeedle)
                let fuzzy = !descWords.isEmpty && descWords.allSatisfy { lc.contains($0) }
                if strict || fuzzy {
                    descY = line.box.y + line.box.height / 2
                    break
                }
            }
            guard let descCenter = descY else { continue }

            // Find the OCR observation matching the extracted price,
            // closest in Y to the description (in case the value appears
            // multiple times on the receipt).
            var matchedPriceLineIdx: Int? = nil
            var matchedPriceY: Double = 0
            var matchedPriceX: Double = 0
            var bestDy = Double.infinity
            for (idx, line) in lines.enumerated() {
                guard let v = priceValue(of: line), v == item.totalPrice else { continue }
                let cy = line.box.y + line.box.height / 2
                let dy = abs(cy - descCenter)
                if dy < bestDy {
                    bestDy = dy
                    matchedPriceLineIdx = idx
                    matchedPriceY = cy
                    matchedPriceX = line.box.x
                }
            }
            guard matchedPriceLineIdx != nil else { continue }

            // Is the matched price's row adjacent to a "Regular Price"
            // label? Tight tolerance — same visual row, not a neighbor.
            let nearLabel = regularLabels.contains { abs($0 - matchedPriceY) <= 0.006 }
            guard nearLabel else { continue }

            // Look for an alternative price ABOVE the description in the
            // same column. Must be smaller than the regular-price value
            // (the whole point is that the actual is less than the
            // regular).
            var altPrice: Decimal? = nil
            var altDistance = Double.infinity
            for line in lines {
                guard let v = priceValue(of: line) else { continue }
                guard v > 0, v < item.totalPrice else { continue }
                let cy = line.box.y + line.box.height / 2
                guard cy < descCenter else { continue }
                let dy = descCenter - cy
                guard dy <= 0.014 else { continue }            // within ~2 line heights
                guard abs(line.box.x - matchedPriceX) <= 0.05 else { continue }  // same column
                if dy < altDistance {
                    altDistance = dy
                    altPrice = v
                }
            }
            guard let alt = altPrice else { continue }
            receipt.lineItems[i] = Receipt.LineItem(
                description: item.description,
                quantity: Self.normalizedQuantity(nil, totalPrice: alt),
                unitPrice: nil,
                totalPrice: alt,
                category: item.category
            )
        }
        return receipt
    }

    /// Returns true if a single OCR observation looks like a real
    /// totals-block label ("TAX", "SUBTOTAL", "**** BALANCE", "TAX 0.70")
    /// — used to clamp the line-item recovery band so we don't accidentally
    /// promote a row beneath the footer to a line item.
    ///
    /// The naive `lc.contains("tax")` approach catches false positives
    /// like "SNGL TAX" (Safeway's single-tax category marker, which sits
    /// inline with line items). Instead, tokenize the line and require
    /// the keyword to be EITHER the first alphanumeric token OR the second
    /// token preceded by a numeric one (handles "0.70 TAX"-style rows).
    /// Asterisks / punctuation in the prefix ("**** BALANCE") fall away
    /// because the split skips non-alphanumerics.
    fileprivate static func lineLooksLikeTotalsBoundary(
        _ text: String,
        keywords: [String]
    ) -> Bool {
        let tokens = text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
        guard let first = tokens.first else { return false }
        for kw in keywords {
            if first == kw { return true }
            // Allow "0.70 TAX" — numeric first, keyword second. Use Double
            // to accept both "0.70" and "0,70" (after a comma→dot pass).
            if Double(first.replacingOccurrences(of: ",", with: ".")) != nil,
               tokens.dropFirst().first == kw
            {
                return true
            }
        }
        return false
    }

    /// Reject a description that the line-item recovery pass surfaced if
    /// it's obviously not a product:
    ///   * Department headers (LIQUOR, PRODUCE, DAIRY, …) — always one
    ///     short uppercase word, often used as a section divider.
    ///   * Coupon / rewards / savings lines — "forU Store Coupon 3.50-",
    ///     "Grocery Rewards", "Member Savings", "Regular Price".
    ///   * Lines whose text ends in a negative-discount price ("3.50-")
    ///     — that's a savings amount, not a product.
    ///   * SKU-style codes ("A031 10", "C1)") — short alphanumeric
    ///     fragments that aren't real product names.
    ///
    /// The earlier `metadataLinePatterns` check only matched if the
    /// True when MORE THAN HALF of a description's words are known
    /// metadata fragments — used by the column-anchored description
    /// picker to deprioritize candidates like "SNGL TAX" or "VOL+ WT"
    /// in favor of real product names.
    ///
    /// Looser than `looksLikeNonItemRecoveryCandidate` (which rejects
    /// outright): a description like "TAX FREE PRODUCT" has one noise
    /// word out of three and stays through this check.
    fileprivate static func descriptionIsMostlyNoise(_ text: String) -> Bool {
        let noise: Set<String> = [
            "tax", "vol", "vol+", "vol-", "info", "wt", "sngl",
            "regular", "price", "savings", "saving", "member",
            "rewards", "discount", "discounts", "additional",
            "subtotal", "total", "balance", "change", "redeemed",
            "produce", "dairy", "meat", "poultry", "deli", "bakery",
            "grocery", "liquor", "wine", "beer", "frozen", "refrig",
        ]
        let words = text
            .lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { !$0.isEmpty }
        guard !words.isEmpty else { return false }
        let noiseCount = words.filter { noise.contains($0) }.count
        return Double(noiseCount) > Double(words.count) * 0.5
    }

    /// pattern was a prefix of the line. Garbled Vision output like
    /// "23MMember Savings Ta 18.00-" (the `23M` is a SKU prefix the OCR
    /// merged onto the savings row) defeats prefix matching, so this
    /// helper checks substrings.
    fileprivate static func looksLikeNonItemRecoveryCandidate(_ description: String) -> Bool {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        let lc = trimmed.lowercased()

        // Common department headers — these print on their own row and
        // get picked up if a price observation lands in the same band.
        let departments: Set<String> = [
            "grocery", "produce", "dairy", "meat", "poultry", "seafood",
            "deli", "bakery", "frozen", "refrig", "refrig/frozen",
            "gen merchandise", "general merchandise", "merchandise",
            "liquor", "wine", "beer", "bulk", "household", "pharmacy",
            "health", "beauty", "snacks", "beverages",
        ]
        if departments.contains(lc) { return true }

        // Single-word metadata fragments — leftovers after Vision split
        // "Member Savings" / "Regular Price" / "Additional Discounts"
        // into two observations and one half slipped through filtering.
        // None of these are plausible standalone product names.
        let metadataFragments: Set<String> = [
            "member", "regular", "savings", "discounts", "discount",
            "rewards", "coupon", "additional", "balance", "subtotal",
            "tax", "tip", "total", "change", "redeemed",
            // Safeway-specific column markers — appear in the price
            // column area at random Ys and confuse recovery.
            "vol", "vol+", "vol-", "wt", "info",
        ]
        if metadataFragments.contains(lc) { return true }

        // Multi-word footer phrases.
        let footerPhrases: [String] = [
            "additional discounts",
            "additional discount",
            "instant savings",
            "promo savings",
        ]
        if footerPhrases.contains(where: { lc.contains($0) }) { return true }

        // Coupon / savings / loyalty / rewards anywhere in the text.
        // These phrases never appear inside a real product name.
        // Includes typo-tolerant variants ("foru personalized" without
        // the space Vision drops) and trailing-marker variants like
        // "sale price" (a Sprouts-style annotation Vision sometimes
        // glues onto the description).
        let nonItemSubstrings: [String] = [
            "store coupon", "manufacturer coupon", "mfr coupon",
            "member savings", "member saving", "member savinas",
            "regular price", "resular price", "reqular price",
            "for personalized", "forl personalized",
            "foru personalized",  // Vision drops the space in "for U"
            "you save", "you saved",
            "grocery rewards", "rewards earned", "points earned",
            "loyalty discount",
            "sale price",
        ]
        for sub in nonItemSubstrings {
            if lc.contains(sub) { return true }
        }

        // Weight-metadata rows on produce receipts: "1.31 lb @",
        // "0.91 lb @ $1.65 / lb", "1.5 oz @". Never a product name.
        if trimmed.range(
            of: #"\d+(?:[.,]\d+)?\s*(?:lb|oz|kg|g)\s*@"#,
            options: .regularExpression
        ) != nil {
            return true
        }

        // Trailing negative-amount marker ("3.50-", "18.00-") — that's a
        // savings line, not a product.
        if trimmed.range(of: #"\d+[.,]\d{1,2}\s*-\s*$"#, options: .regularExpression) != nil {
            return true
        }

        // Leading negative-amount marker ("-0.25 forU Personalized") —
        // same story: a savings amount attached to a label.
        if trimmed.range(of: #"^\s*-\d+[.,]\d{1,2}\b"#, options: .regularExpression) != nil {
            return true
        }

        // Pure-digit UPC / SKU code (6+ digits, possibly with a trailing
        // check digit block). Product descriptions always have letters.
        if trimmed.range(of: #"^\d{6,}\s*$"#, options: .regularExpression) != nil {
            return true
        }

        // SKU-style code: starts with one or two letters, then a run of
        // digits, optionally followed by spaces and more digits. "A031 10"
        // is the canonical example; "C1)" similar shape.
        if trimmed.range(
            of: #"^[A-Z]{1,2}\d{2,}([ )]+\d+)?\s*$"#,
            options: .regularExpression
        ) != nil {
            return true
        }

        return false
    }

    /// Find the OCR row whose text matches `description` (substring or
    /// any 3+-char word from it), then return the largest price-shaped
    /// token in that row. Used by the FM-output sanity check to compare
    /// what the receipt actually printed against what the model emitted.
    private static func rightmostPriceInOCRRow(
        matching description: String,
        in lines: [(text: String, box: Receipt.BBox)]
    ) -> Decimal? {
        // Returns the first matching row's rightmost price. Kept as a
        // single-row convenience; for the OCR-anchored correction we
        // want EVERY occurrence, so callers should prefer
        // `rightmostPricesInOCRRows` below.
        rightmostPricesInOCRRows(matching: description, in: lines).first
    }

    /// Every distinct rightmost-price token from each OCR row that
    /// matches `description`. Used by the OCR-anchored price
    /// correction: when a receipt has two items with the same name
    /// (e.g. "BONE-IN CHCKN THIGHS 5.53" and "BONE-IN CHCKN THIGHS 6.03"),
    /// FM sometimes collapses them into one item with quantity=2 and
    /// totalPrice=11.06. To detect that we need to see ALL the printed
    /// prices, not just the first row's.
    private static func rightmostPricesInOCRRows(
        matching description: String,
        in lines: [(text: String, box: Receipt.BBox)]
    ) -> [Decimal] {
        let needle = description.lowercased().trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return [] }
        let words = needle
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count >= 3 }

        // Find every OCR observation whose text references this item.
        var rowYs: [Double] = []
        for line in lines {
            let lc = line.text.lowercased()
            let strict = lc.contains(needle)
            let fuzzy = !words.isEmpty && words.allSatisfy { lc.contains($0) }
            if strict || fuzzy {
                rowYs.append(line.box.y + line.box.height / 2)
            }
        }
        guard !rowYs.isEmpty else { return [] }

        let priceShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?(?:\s+[A-Z])?$"#
        var out: [Decimal] = []
        for rowY in rowYs {
            // Tight tolerance — we want THIS row's price, not a neighbor's.
            // 0.7 × line-height typically keeps us inside a single OCR row
            // even when the description and price sit on slightly different
            // baselines.
            var best: (price: Decimal, x: Double)? = nil
            for line in lines {
                let centerY = line.box.y + line.box.height / 2
                guard abs(centerY - rowY) <= max(0.008, line.box.height * 0.7) else { continue }
                let trimmed = line.text.trimmingCharacters(in: .whitespaces)
                guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { continue }
                let stripped = trimmed
                    .replacingOccurrences(of: "$", with: "")
                    .replacingOccurrences(of: ",", with: ".")
                    .replacingOccurrences(of: #"\s+[A-Z]\s*$"#, with: "", options: .regularExpression)
                guard let value = Decimal(string: stripped, locale: Locale(identifier: "en_US_POSIX")) else { continue }
                if best == nil || line.box.x > best!.x {
                    best = (value, line.box.x)
                }
            }
            if let b = best { out.append(b.price) }
        }
        return out
    }

    /// Trailing tokens that are typically tax-category indicators on US
    /// chain receipts ("T" = taxable, "+" = health item, "F" = food, "N"
    /// = non-taxable, "X" = exempt). They're never part of the actual
    /// product name and should be stripped from the description.
    private static let trailingDescriptionMarkers: Set<String> = ["T", "+", "F", "N", "X", "S"]

    /// Clean up an FM-extracted line-item description:
    ///   1. Strip a leading numeric SKU/UPC of 6+ digits ("037129013 NTG
    ///      Stubb Bd" → "NTG Stubb Bd"). The pre-cluster filter already
    ///      drops SKUs that arrive as their own OCR observation, but when
    ///      Vision clustered the SKU into the description segment we still
    ///      see it here.
    ///   2. Strip trailing single-character markers, possibly multiple
    ///      ("NTG Stubb Bd T +" → "NTG Stubb Bd"). We only strip tokens
    ///      from a small allowlist to avoid clipping legitimate trailing
    ///      letters in descriptions like "COKE 12 OZ C" (where C is part
    ///      of the size descriptor).
    private static func cleanLineItemDescription(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip leading SKU/UPC code (6+ digits + whitespace).
        s = s.replacingOccurrences(of: #"^\d{6,}\s+"#, with: "", options: .regularExpression)
        // Strip leading quantity prefix ("2 QTY DE CECCO" → "DE CECCO",
        // "3x EGGS" → "EGGS"). Keep the quantity info on the LineItem's
        // own `quantity` field rather than tangled into the description.
        s = s.replacingOccurrences(
            of: #"^\d+\s*(?i:qty|x|@)\s+"#,
            with: "",
            options: .regularExpression
        )
        // Repeatedly peel trailing single-character markers from the
        // allowlist until the last token is something more substantive.
        while true {
            let tokens = s.split(separator: " ", omittingEmptySubsequences: true)
            guard let last = tokens.last, trailingDescriptionMarkers.contains(String(last)) else { break }
            s = tokens.dropLast().joined(separator: " ")
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

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

    // MARK: - OCR metadata pre-filter
    //
    // Drop OCR observations whose text is *only* receipt metadata — they
    // surround real item rows on chain-grocery receipts ("Regular Price"
    // and "Member Savings" beneath every Safeway item, etc.) and just
    // confuse the LLM. We do this BEFORE row clustering so the surviving
    // rows are clean description+price pairs. The full unfiltered line
    // list still flows through `attachBBoxes` and into `ocrLines` for the
    // labeler overlay, so dropped lines remain click-assignable.

    /// Substring tests (lowercased) that mean "this line is metadata,
    /// not a purchased item". An OCR observation matches if its text,
    /// lowercased and trimmed, equals one of these or starts with one
    /// followed by a space / colon / dash.
    private static let metadataLinePatterns: [String] = [
        // "regular price" plus Vision OCR misreads of the 'g' (small
        // fonts substitute s/q/c/p for g routinely). Without these
        // variants, lines like "Resular Price 6.99" leak through the
        // metadata filter and FM treats them as line items.
        "regular price",
        "resular price",
        "reqular price",
        "recular price",
        "repular price",
        "reaular price",
        // "member savings" plus common Vision OCR misreads of "savings"
        // — letters get substituted on small fonts. Without these typo
        // variants, the misread row leaks into the FM prompt and the
        // model gets confused about which numbers belong to which item.
        "member savings",
        "member savinas",
        "member savinos",
        "member saving",
        "member savincs",
        "member savir",
        "member price",
        "you saved",
        "you save",
        "savings",
        "savinas",
        "savinos",
        "discount",
        "for personalized",
        // Safeway prints "VOL+" / "VOL-" / "INFO" / "WT" markers in a
        // far-left column that clusters with the price column.
        "vol+", "vol-", "vol +", "vol -",
        "info",
        "wt",
        // Loyalty / payment scaffolding that sometimes ends up clustered
        // with a real row.
        "rewards earned",
        "points earned",
    ]

    /// Drop OCR observations that are purely a long digit sequence —
    /// almost always a product SKU / UPC code printed next to the line
    /// item. They confuse the LLM (the row becomes
    /// "037129013 NTG Stubb Bd  T +  $12.89" instead of just
    /// "NTG Stubb Bd  T +  $12.89") and carry no value we'd ever want
    /// in the canonical Receipt schema. Threshold ≥6 digits so we don't
    /// accidentally strip 5-digit ZIPs or 4-digit years (those usually
    /// live in OCR observations with surrounding context anyway).
    fileprivate static func dropSKUObservations(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        let sku = #"^\d{6,}$"#
        return lines.filter { line in
            let trimmed = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.range(of: sku, options: .regularExpression) == nil
        }
    }

    /// Vision sometimes reads a decimal point as a hyphen on small
    /// receipt fonts ("$11.75" → "$11-75"). Rewrite any
    /// `digit(s)-2digit` substring INSIDE an observation back to a dot
    /// so price parsing downstream picks it up. Restricted to exactly
    /// 2 trailing digits + a word boundary so we don't accidentally
    /// rewrite phone numbers or dates ("555-1234" is left alone — no
    /// boundary after just 2 of those 4 digits).
    fileprivate static func normalizePriceTokens(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        // (1) Hyphen-misread for the decimal point ("$11-75" → "$11.75").
        let hyphenPattern = #"(\d+)-(\d{2})\b"#
        // (2) Spurious whitespace after the decimal ("6. 49" → "6.49").
        // Vision sometimes splits the digit pair on small fonts; without
        // closing the gap, the row is no longer price-shaped and the
        // item silently disappears from extraction.
        let spaceAfterDotPattern = #"(\d)\.\s+(\d{2})\b"#
        return lines.map { line in
            let normalized = line.text
                .replacingOccurrences(
                    of: hyphenPattern,
                    with: "$1.$2",
                    options: .regularExpression
                )
                .replacingOccurrences(
                    of: spaceAfterDotPattern,
                    with: "$1.$2",
                    options: .regularExpression
                )
            return (text: normalized, box: line.box)
        }
    }

    /// Rewrite individual OCR segments that look like `"8.49 S"` or
    /// `"3.50 T"` (price plus a single trailing tax/SKU letter) to drop
    /// the letter. We only touch segments that match the strict
    /// `decimal-then-letter` shape so legitimate item names like
    /// `"COKE 12 OZ"` are untouched.
    fileprivate static func stripTrailingTaxMarkers(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        let pattern = #"^(-?\$?\d{1,5}(?:[.,]\d{1,2})?)(?:\s+([A-Z]))+\s*$"#
        let regex = try? NSRegularExpression(pattern: pattern)
        return lines.map { line in
            guard let regex else { return line }
            let trimmed = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
            guard let match = regex.firstMatch(in: trimmed, options: [], range: range),
                  let priceRange = Range(match.range(at: 1), in: trimmed)
            else { return line }
            return (text: String(trimmed[priceRange]), box: line.box)
        }
    }

    fileprivate static func dropMetadataObservations(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        // Substring-match patterns: drop the whole observation if its
        // text *contains* one of these, no matter the prefix. Catches
        // garbled OCR like "23MMember Savings Ta 18.00-" (a savings row
        // with a SKU "23M" merged onto the start) and "forU Store
        // Coupon 3.50-" where a prefix-only match would miss it because
        // "foru" isn't in `metadataLinePatterns`. These phrases never
        // appear inside a real product name.
        let dropIfContains: [String] = [
            "store coupon", "manufacturer coupon", "mfr coupon",
            "member savings", "member saving",
            "member savinas", "member savinos", "member savincs",
            "for personalized", "forl personalized",
            "you save", "you saved",
            "grocery rewards", "rewards earned", "points earned",
            "loyalty discount",
        ]
        return lines.filter { line in
            let lc = line.text
                .lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !lc.isEmpty else { return false }
            for pat in metadataLinePatterns {
                if lc == pat { return false }
                if lc.hasPrefix(pat + " ") { return false }
                if lc.hasPrefix(pat + ":") { return false }
                if lc.hasPrefix(pat + "-") { return false }
            }
            if dropIfContains.contains(where: { lc.contains($0) }) {
                return false
            }
            return true
        }
    }

    /// When a row's *only* content is a price-shaped token (e.g. "$11.75")
    /// and the row immediately above has text but no price of its own,
    /// merge the price up into the description row. This catches the
    /// common case where a long item description wraps to one line and
    /// the printer drops the price onto the next line — the
    /// description+price are visually one row but Vision returns them as
    /// two observations with a vertical gap larger than the row-cluster
    /// tolerance.
    ///
    /// Bounded by `gap ≤ 2.5 × previous row height` so we don't yank a
    /// price from a totals block four lines below the last description.
    fileprivate static func mergeOrphanedPriceRows(_ rows: [OCRRow]) -> [OCRRow] {
        guard rows.count >= 2 else { return rows }
        // A standalone price (single segment matching this shape).
        let standaloneShape = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?$"#
        // A price appearing anywhere in the row text.
        let embeddedShape   = #"\$?\d{1,5}[.,]\d{1,2}"#

        func isPriceOnly(_ row: OCRRow) -> Bool {
            guard row.segments.count == 1 else { return false }
            let text = row.segments[0].text.trimmingCharacters(in: .whitespaces)
            return text.range(of: standaloneShape, options: .regularExpression) != nil
        }
        func alreadyHasPrice(_ row: OCRRow) -> Bool {
            for seg in row.segments {
                if seg.text.range(of: embeddedShape, options: .regularExpression) != nil {
                    return true
                }
            }
            return false
        }

        var result: [OCRRow] = []
        for row in rows {
            if isPriceOnly(row),
               let prev = result.last,
               !alreadyHasPrice(prev) {
                let prevCenter = (prev.yMin + prev.yMax) / 2
                let currCenter = (row.yMin + row.yMax) / 2
                let prevHeight = max(prev.yMax - prev.yMin, 0.005)
                let gap = abs(currCenter - prevCenter)
                if gap <= prevHeight * 2.5 {
                    var merged = prev
                    merged.segments.append(contentsOf: row.segments)
                    merged.yMax = max(merged.yMax, row.yMax)
                    result[result.count - 1] = merged
                    continue
                }
            }
            result.append(row)
        }
        return result
    }

    /// Drop the rightmost segment from a row when it's a single uppercase
    /// letter (tax category) sitting next to a price-shaped segment. After
    /// clustering, `6.99  S` is two segments; we want the row to look like
    /// `CHOBANI YGRT VAN G  6.99` so the LLM treats the number as the
    /// price. If the rightmost two segments don't look like
    /// `<price> <single-letter>`, leave the row alone.
    fileprivate static func stripTrailingSingleLetterSegment(_ row: OCRRow) -> OCRRow {
        // Segments aren't sorted in `row.segments`; the `joined` accessor
        // sorts by x for display. We need the rightmost (highest x) here.
        let sortedByX = row.segments.sorted { $0.x < $1.x }
        guard sortedByX.count >= 2 else { return row }
        let last = sortedByX[sortedByX.count - 1].text.trimmingCharacters(in: .whitespaces)
        let prev = sortedByX[sortedByX.count - 2].text.trimmingCharacters(in: .whitespaces)
        let isSingleUpper = last.count == 1 && last.range(of: "^[A-Z]$", options: .regularExpression) != nil
        let isPrice = prev.range(of: #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?$"#, options: .regularExpression) != nil
        guard isSingleUpper, isPrice else { return row }
        var row = row
        // Find and remove the last-x segment from the original (unsorted)
        // segments array.
        if let dropIdx = row.segments.indices.max(by: { row.segments[$0].x < row.segments[$1].x }) {
            row.segments.remove(at: dropIdx)
        }
        return row
    }

    /// True if a clustered row has no description, only money values /
    /// trailing dashes. These are the orphaned "regular price" and
    /// "savings amount" rows left behind when their label sibling was
    /// pruned by `dropMetadataObservations` — including them would tempt
    /// the LLM to invent extra line items.
    /// True when a clustered row's joined text contains one of the
    /// "hard" discount/metadata markers ("Regular Price",
    /// "Member Savings", "for Personalized", "Store Coupon"). The row
    /// always carries a price sibling (the regular-price amount or the
    /// savings amount), so FM will mistake the row for an item if we
    /// don't drop the whole thing here.
    ///
    /// Distinct from `dropMetadataObservations`, which filters individual
    /// observations BEFORE clustering. That pass only catches lines
    /// where the metadata text is the entire observation; clustering can
    /// still glue the surviving price observation onto a different row
    /// containing the marker. This is the post-cluster safety net.
    fileprivate static func rowContainsHardMetadataMarker(_ row: OCRRow) -> Bool {
        // Strip punctuation and collapse whitespace so "Regular, Price"
        // (comma artifact) still matches "regular price". Without this
        // normalization, Vision's per-character OCR for low-quality
        // receipts inserts commas/dots/spaces inside multi-word labels
        // and our markers slip past.
        let rawLc = row.joined.lowercased()
        let collapsed = rawLc
            .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
        let markers: [String] = [
            "regular price", "resular price", "reqular price",
            "recular price", "repular price", "reaular price",
            "member savings", "member saving",
            "member savinas", "member savinos", "member savincs",
            "for personalized", "forl personalized",
            "store coupon", "manufacturer coupon", "mfr coupon",
            "you save", "you saved",
            "rewards earned", "grocery rewards", "points earned",
            "loyalty discount",
        ]
        return markers.contains { collapsed.contains($0) }
    }

    fileprivate static func isOrphanedNumericRow(_ row: OCRRow) -> Bool {
        let segments = row.segments
        guard !segments.isEmpty else { return true }
        // Every segment must look like a price (or be a single-letter
        // store-code marker like "S" / "T" / "C1]") for the row to count
        // as orphaned.
        let priceish = #"^-?\$?\d{1,5}(?:[.,]\d{1,2})?-?$"#
        let trailMarker = #"^[A-Z]?\d?[A-Z]?-?$"#
        for seg in segments {
            let t = seg.text.trimmingCharacters(in: .whitespaces)
            if t.range(of: priceish, options: .regularExpression) != nil { continue }
            if t.range(of: trailMarker, options: .regularExpression) != nil { continue }
            return false
        }
        return true
    }

    // MARK: - OCR row clustering

    /// One visual row of the receipt, made up of one or more OCR
    /// observations that share roughly the same Y. Segments are kept
    /// left-to-right by X so prompt rows mirror printed receipt rows.
    fileprivate struct OCRRow {
        var segments: [(text: String, x: Double)]
        /// yMin/yMax track the *seed* line's bounds and don't grow as more
        /// segments are absorbed. Snowballing the range across joiners is
        /// what produced the original failure on dense Safeway receipts:
        /// each absorption widens the gate just enough to admit the next
        /// row, and a whole vertical band collapses into one [R##].
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

    /// Group OCR observations into visual rows. A new line joins the
    /// current row only when its vertical center lies within ~50% of
    /// the seed line's height of the seed's center. Using the seed
    /// (not a cumulative range) prevents the snowballing failure on
    /// dense receipts.
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
            let centerY = (yTop + yBot) / 2
            if var current = rows.last {
                let seedCenter = (current.yMin + current.yMax) / 2
                let seedHeight = current.yMax - current.yMin
                // Tolerance scales with the seed line's height so it adapts
                // to dense small-text receipts and big-font ones alike.
                let tolerance = max(0.005, seedHeight * 0.5)
                if abs(centerY - seedCenter) <= tolerance {
                    current.segments.append((text: line.text, x: line.box.x))
                    // Intentionally do NOT widen current.yMin / yMax here.
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

    /// Normalize a candidate quantity value:
    ///   * empty (nil) → 1
    ///   * equals the totalPrice (within 1 cent) → 1
    ///
    /// The second case catches FM misreading a weight ("BROCCOLI CROWNS
    /// 1.15 lb @ $1.00/lb — total $1.15") as a quantity of 1.15 rather
    /// than a weight of 1.15 lb; and misparsing produce rows where FM
    /// duplicates the price into the quantity slot.
    ///
    /// Returns a non-nil Decimal so downstream forms always display a
    /// quantity — no ambiguous blank field.
    private static func normalizedQuantity(_ raw: Decimal?, totalPrice: Decimal) -> Decimal? {
        guard let q = raw else { return 1 }
        // qty == totalPrice within 1c is treated as spurious.
        let diff = q > totalPrice ? q - totalPrice : totalPrice - q
        if diff < Decimal(0.01) { return 1 }
        return q
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
