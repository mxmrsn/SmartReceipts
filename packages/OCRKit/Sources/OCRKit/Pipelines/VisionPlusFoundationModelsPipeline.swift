import CoreGraphics
import CoreImage
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

        // ---- 0) Preprocess: crop + dewarp the receipt out of the photo ----
        // Receipts often occupy only a small strip of the frame (rotated on
        // a table, curled, angled). Isolating the receipt via document
        // segmentation and perspective-correcting into a rectangle gives
        // Vision much cleaner input — small numeric prices that were being
        // dropped from tilted / small-receipt photos are now recognized.
        // If no document is detected, `preprocess` returns the original
        // image + an identity bbox mapper, so the pipeline degrades
        // gracefully to the pre-preprocess behavior.
        let prep = ReceiptImagePreprocessor.preprocess(image: image, orientation: orientation)

        // ---- 1) Vision OCR ----
        // NOTE: bboxes are in the CORRECTED image's coord space (not the
        // original photo's) when preprocessing fired. This is deliberate —
        // the rest of the pipeline (column-anchored price extraction, row
        // clustering, etc.) needs prices to be aligned in a vertical
        // column, which is what the corrected image guarantees. The
        // `prep.mapBBoxToOriginal` mapper is available for callers that
        // need to draw bboxes on the ORIGINAL image.
        let lines = try Self.recognizeLines(image: prep.image, orientation: prep.orientation)
        guard !lines.isEmpty else {
            throw OCRError.parseFailed("Vision returned no recognized text.")
        }

        // ---- 2) Foundation Models structured extraction ----
        guard case .available = SystemLanguageModel.default.availability else {
            throw OCRError.modelNotAvailable(
                "Apple Foundation Models unavailable. Enable Apple Intelligence in System Settings."
            )
        }

        var receipt = try await Self.extractReceipt(from: lines)
        var usedLines = lines

        // ---- 2b) Escalation: tiled hi-res re-scan ----
        // Vision silently drops small text (price cents are the first
        // casualty) when the receipt's glyphs are few pixels tall in the
        // frame. When the standard pass lands below the low-confidence
        // band, re-OCR the receipt as overlapping horizontal bands
        // upscaled 2× — text twice as tall recovers observations the
        // full-frame pass never produced — then run the whole pipeline
        // again and keep whichever result scores higher. Cost is a
        // second Vision + FM round on only the receipts that need it.
        if receipt.provenance.confidence < 0.6,
           let tiled = Self.recognizeLinesTiled(image: prep.image, orientation: prep.orientation),
           !tiled.isEmpty,
           let retry = try? await Self.extractReceipt(from: tiled),
           retry.provenance.confidence > receipt.provenance.confidence {
            receipt = retry
            usedLines = tiled
        }

        // ---- 2c) Escalation: forced perspective correction ----
        // The conservative preprocessing gate skips dewarp for medium-
        // area receipts, but a mild tilt (≈ 5°) that passes the gate
        // still drifts same-row label/value pairs a full row apart
        // across the page width, defeating every pairing heuristic
        // (IMG_2587: "**** BALANCE" and its 39.49 split by dy 0.025).
        // If confidence is still low, dewarp unconditionally, re-run,
        // and keep the winner.
        if receipt.provenance.confidence < 0.6, !prep.didCorrect {
            let forced = ReceiptImagePreprocessor.preprocessForced(image: image, orientation: orientation)
            if forced.didCorrect,
               let straightLines = try? Self.recognizeLines(image: forced.image, orientation: forced.orientation),
               !straightLines.isEmpty,
               let retry = try? await Self.extractReceipt(from: straightLines),
               retry.provenance.confidence > receipt.provenance.confidence {
                receipt = retry
                usedLines = straightLines
            }
        }

        let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds &- startNs) / 1_000_000)
        return ExtractionResult(
            receipt: receipt,
            latencyMs: elapsedMs,
            peakMemoryMB: nil,
            rawText: usedLines.map(\.text).joined(separator: "\n"),
            ocrLines: usedLines.map { OCRLine(text: $0.text, box: $0.box) }
        )
    }

    /// Standard single-pass Vision OCR: recognize, correct 90°-rotated
    /// observation geometry, sort top-to-bottom.
    private static func recognizeLines(
        image: CGImage,
        orientation: CGImagePropertyOrientation
    ) throws -> [(text: String, box: Receipt.BBox)] {
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
        let rawLines: [(text: String, box: Receipt.BBox)] = observations
            .compactMap { obs -> (text: String, box: Receipt.BBox)? in
                guard let top = obs.topCandidates(1).first else { return nil }
                return (text: top.string, box: Self.bbox(from: obs.boundingBox))
            }
        // Correct for a receipt captured 90° rotated (missing / wrong
        // EXIF, or a screenshot of a landscape phone photo). Vision
        // recognizes the characters fine either way, but the bounding
        // boxes are in image coordinates and if the receipt was on its
        // side then every "text line" observation is TALL and NARROW
        // and all lines collapse into a single narrow Y band. Column-
        // anchored extraction can't work without a vertical price
        // column, so we detect this case and rotate the observations
        // back into a portrait-oriented coordinate system.
        return Self.correctRotatedObservations(rawLines)
            .sorted { $0.box.y < $1.box.y }
    }

    /// Tiled hi-res OCR: orient the image upright, slice it into
    /// overlapping horizontal bands, upscale each band 2×, OCR each
    /// band independently, then merge observations back into full-image
    /// normalized coordinates (deduping the overlap zones).
    ///
    /// Why this recovers text the standard pass drops: Vision
    /// downsamples large inputs internally, and a tall receipt photo
    /// leaves each glyph only a handful of pixels — price cents are the
    /// first thing it silently discards ("$8.42" comes back as "$8" or
    /// nothing). A 2×-upscaled third of the image keeps the pixel count
    /// manageable while doubling glyph height.
    private static func recognizeLinesTiled(
        image: CGImage,
        orientation: CGImagePropertyOrientation
    ) -> [(text: String, box: Receipt.BBox)]? {
        guard let upright = uprightCGImage(image, orientation: orientation) else { return nil }
        let W = upright.width
        let H = upright.height
        guard W > 0, H > 0 else { return nil }

        let bands = 3
        let overlap = 0.08   // fraction of full height shared between bands

        var merged: [(text: String, box: Receipt.BBox)] = []
        for i in 0..<bands {
            let y0 = max(0.0, Double(i) / Double(bands) - overlap / 2)
            let y1 = min(1.0, Double(i + 1) / Double(bands) + overlap / 2)
            let bandFrac = y1 - y0
            let cropRect = CGRect(
                x: 0, y: (y0 * Double(H)).rounded(.down),
                width: Double(W), height: (bandFrac * Double(H)).rounded(.up)
            )
            guard let tile = upright.cropping(to: cropRect) else { continue }
            // Upscale 2× unless the band is already huge (at that point
            // resolution isn't the limiting factor and doubling would
            // just burn memory).
            let scaled: CGImage = tile.width * 2 <= 9000
                ? (upscaled(tile, factor: 2) ?? tile)
                : tile

            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            request.recognitionLanguages = ["en-US"]
            let handler = VNImageRequestHandler(cgImage: scaled, orientation: .up, options: [:])
            guard (try? handler.perform([request])) != nil else { continue }

            for obs in request.results ?? [] {
                guard let top = obs.topCandidates(1).first else { continue }
                let bb = obs.boundingBox   // tile space, origin bottom-left
                // Convert to full-image normalized top-left coords.
                let tileTopY = 1.0 - bb.origin.y - bb.height
                let box = Receipt.BBox(
                    x: bb.origin.x,
                    y: y0 + tileTopY * bandFrac,
                    width: bb.width,
                    height: bb.height * bandFrac
                )
                merged.append((text: top.string, box: box))
            }
        }
        guard !merged.isEmpty else { return nil }

        // Dedupe the overlap zones: the same printed line OCR'd by two
        // adjacent bands produces near-identical text at near-identical
        // position. Keep the first occurrence.
        var kept: [(text: String, box: Receipt.BBox)] = []
        for obs in merged.sorted(by: { $0.box.y < $1.box.y }) {
            let cy = obs.box.y + obs.box.height / 2
            let isDupe = kept.contains { k in
                let kcy = k.box.y + k.box.height / 2
                return abs(kcy - cy) < 0.006
                    && abs(k.box.x - obs.box.x) < 0.02
                    && k.text.trimmingCharacters(in: .whitespaces)
                        == obs.text.trimmingCharacters(in: .whitespaces)
            }
            if !isDupe { kept.append(obs) }
        }

        return Self.correctRotatedObservations(kept)
            .sorted { $0.box.y < $1.box.y }
    }

    /// Render the image with its EXIF orientation applied so tiling
    /// bands align with printed text rows.
    private static func uprightCGImage(
        _ image: CGImage,
        orientation: CGImagePropertyOrientation
    ) -> CGImage? {
        guard orientation != .up else { return image }
        let ci = CIImage(cgImage: image).oriented(forExifOrientation: Int32(orientation.rawValue))
        let ctx = CIContext(options: nil)
        return ctx.createCGImage(ci, from: ci.extent)
    }

    /// Integer-factor bicubic upscale via CGContext.
    private static func upscaled(_ image: CGImage, factor: Int) -> CGImage? {
        let w = image.width * factor
        let h = image.height * factor
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    /// The full post-OCR pipeline: FM structured extraction plus every
    /// sanity / repair / reconciliation pass, ending with the receipt-
    /// level confidence score. Factored out of `extract` so the tiled
    /// hi-res escalation can run the identical pipeline on a second OCR
    /// line set and compare confidence scores.
    private static func extractReceipt(
        from lines: [(text: String, box: Receipt.BBox)]
    ) async throws -> Receipt {
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
        // Merge Vision-split price fragments ("2." + "79 S" — real
        // $2.79 that Vision fractured across two observations). Runs
        // before other filters so both this pipeline path and the
        // column-anchored path (which re-normalizes) see the merged
        // form. Without merge, "2." is discarded (not price-shaped)
        // and "79 S" is treated as $79.
        let mergedFragments = Self.mergeSplitPriceFragments(lines)
        let prunedLines = Self.dropMetadataObservations(mergedFragments)
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
        func renderPrompt(_ rs: [OCRRow]) -> String {
            rs.enumerated().map { idx, row in
                let yPct = Int((row.meanY * 100).rounded())
                return String(format: "[R%02d y=%02d] %@", idx, yPct, row.joined)
            }.joined(separator: "\n")
        }
        // Budget the FM prompt. Apple's on-device model shares one fixed
        // context across input + output; very long receipts (Costco-
        // length, multi-fold) overflow it and the WHOLE extraction dies
        // with "Exceeded model context window size". Column-anchored
        // item extraction reads the full OCR line set independently of
        // this prompt, so FM only needs the edges (merchant + date at
        // the top, totals + payment at the bottom) and a representative
        // slice of items. Compact in two stages:
        //   1. Drop footer rows past the last money-bearing row
        //      (surveys, rewards marketing, store hours) — minus a
        //      small margin for the payment/date block.
        //   2. Thin item rows from the MIDDLE until the prompt fits.
        //      Middle rows are the most redundant: FM's items are only
        //      a hint source; totals sanity + column extraction carry
        //      the real signal.
        var promptRows = rows
        let promptBudget = 7000
        if renderPrompt(promptRows).count > promptBudget {
            func rowHasMoney(_ row: OCRRow) -> Bool {
                row.joined.range(of: #"\d+[.,]\d{2}\b"#, options: .regularExpression) != nil
            }
            if let lastMoney = promptRows.lastIndex(where: rowHasMoney) {
                let keepEnd = min(promptRows.count, lastMoney + 5)
                promptRows = Array(promptRows[..<keepEnd])
            }
            while renderPrompt(promptRows).count > promptBudget, promptRows.count > 24 {
                promptRows.remove(at: promptRows.count / 2)
            }
        }

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

        // Token budgets are unpredictable from character counts alone —
        // digit-heavy receipt text tokenizes far denser than prose (a
        // 2.7 KB Sprouts prompt overflowed the window). So besides the
        // proactive character budget above, retry reactively: on a
        // context-window error, thin the middle rows and try again with
        // a FRESH session (a session accumulates transcript state, so
        // reusing one after a failure compounds the overflow).
        var attemptRows = promptRows
        var rawResponseOpt: String? = nil
        var lastError: Error? = nil
        for _ in 0..<3 {
            let attemptOCR = renderPrompt(attemptRows)
            let prompt = "Receipt OCR, pre-grouped into visual rows. Each row is prefixed with [R## y=YY] where YY is vertical position (0=top..99=bottom). Within a row, segments are separated by two spaces and are listed left-to-right; on a line-item row the rightmost segment is almost always the price.\n\n\(attemptOCR)\n\nReturn ONLY the JSON object."
            if ProcessInfo.processInfo.environment["OCR_DEBUG_PROMPT"] != nil {
                FileHandle.standardError.write(Data("=== FM PROMPT ===\n\(attemptOCR)\n=== END ===\n".utf8))
            }
            let session = LanguageModelSession(instructions: Self.instructions)
            do {
                let response = try await session.respond(to: prompt, options: options)
                rawResponseOpt = response.content
                break
            } catch {
                lastError = error
                let msg = error.localizedDescription.lowercased()
                guard msg.contains("context window"), attemptRows.count > 16 else { break }
                // Drop the middle 40% of rows and retry. Header rows
                // (merchant, date) and trailing rows (totals, payment)
                // survive; the dropped items are recovered by column-
                // anchored extraction from the full OCR line set.
                let target = max(16, (attemptRows.count * 6) / 10)
                while attemptRows.count > target {
                    attemptRows.remove(at: attemptRows.count / 2)
                }
            }
        }
        guard let rawResponse = rawResponseOpt else {
            throw OCRError.parseFailed(
                "Foundation Models generation failed: \(lastError?.localizedDescription ?? "unknown error")"
            )
        }
        if ProcessInfo.processInfo.environment["OCR_DEBUG_FM"] != nil {
            FileHandle.standardError.write(Data("=== FM RAW RESPONSE ===\n\(rawResponse)\n=== END ===\n".utf8))
        }
        let extracted: ReceiptExtraction
        do {
            extracted = try Self.parseJSON(rawResponse)
        } catch {
            throw OCRError.parseFailed(
                "Could not parse model output as JSON: \(error.localizedDescription). Raw output: \(rawResponse.prefix(400))"
            )
        }

        // ---- 3) Build canonical Receipt ----
        // Use the price-normalized lines for all post-extraction work
        // (buildReceipt sanity checks, recovery, bbox attachment). The
        // raw `lines` array still has Vision's literal output like
        // "6. 49 S" with the spurious space — without normalization,
        // priceShape regexes reject the token and the item silently
        // vanishes from recovery (IMG_1326 B&J at $6.49).
        // Apply the same fragment merge to the post-extraction lines
        // used by buildReceipt / column-anchored / bbox attachment.
        //
        // We intentionally do NOT drop metadata observations here — the
        // column-anchored path needs the "Regular Price" / "Member
        // Savings" label observations to identify which price observations
        // are Safeway metadata (to skip them). Filtering the labels out
        // would leave the metadata VALUES (3.99, 1.49-, etc.) unlabeled
        // and column-anchored would pick them as if they were items.
        let postLines = Self.normalizePriceTokens(Self.mergeSplitPriceFragments(lines))
        var receipt = Self.buildReceipt(from: extracted, ocrLines: postLines)
        // Currency correction: FM defaults to USD, but travel receipts
        // slip in ("TOTAL YEN", "¥5160" — a ¥5,160 konbini receipt was
        // landing in the dataset as $5,160 with full confidence).
        // Detect yen markers in the OCR and relabel before the totals
        // sanity checks run — integer totals are NORMAL for JPY, so the
        // whole-dollar-artifact guard must know the real currency.
        if receipt.header.currency == "USD" {
            let joined = lines.map(\.text).joined(separator: " ")
            if joined.contains("¥")
                || joined.range(of: #"\b(?i:yen|jpy)\b"#, options: .regularExpression) != nil {
                receipt.header.currency = "JPY"
            }
        }
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
        let (columnItems, rejectedPoints) = Self.columnAnchoredLineItems(
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
        // A negative tax is impossible; drop it before the total sanity
        // check so it can't corroborate a wrong total or poison the
        // net-items arithmetic (IMG_8253 read tax "-1.25").
        receipt = Self.dropInvalidTax(receipt)
        receipt = Self.sanityCheckTotal(receipt: receipt, lines: postLines)
        // Same idea for subtotal / tax / tip — find the OCR label,
        // grab its adjacent value. FM sometimes cross-wires these
        // (Philz IMG: emits tip amount in the subtotal slot). If the
        // OCR is clear about which label goes with which value, we
        // should trust that over FM.
        receipt = Self.sanityCheckSubtotalTaxTip(receipt: receipt, lines: postLines)
        // Last line of defense for subtotal / tax: enforce arithmetic
        // invariants that no real receipt can violate (tax == total,
        // fractional-cent money, tax rates parked in money fields,
        // subtotal == tax). Label matching can't fix these — the OCR
        // label either wasn't found or pointed at the same junk — but
        // arithmetic against the (already sanity-checked) total can.
        receipt = Self.reconcileTotalsArithmetic(receipt: receipt, lines: postLines)
        // Sprouts (and other weight-priced grocery) receipts print
        // weight-based line items as a stack of THREE observations:
        //   BROCCOLI CROWNS            <- description
        //   1.15 lb @                  <- weight × marker
        //   $1.99 / lb                 <- unit price
        // and the actual paid total (weight × unit) appears in the
        // price column at a nearby Y. FM sometimes emits the WEIGHT
        // (1.15) as the totalPrice instead of the computed total
        // (2.29). Recompute from OCR when we can find both the weight
        // observation and the unit-price observation adjacent to the
        // item's description.
        receipt = Self.correctWeightPricedItems(receipt: receipt, lines: postLines)
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
        // Catches both FM-emitted and recovered items in one pass.
        //
        // Exception: negative-priced items are allowed to keep their
        // "Member Savings" / "Regular Price" / other-metadata
        // descriptions. Those are the LEGITIMATE descriptions for
        // discount / credit rows — filtering them out would silently
        // drop legit negative line items (Safeway IMG_2460 has many
        // "1.00- Member Savings"-style rows we want to preserve as
        // negative-value items so items_sum reconciles with total).
        receipt.lineItems.removeAll { item in
            if item.totalPrice < 0 { return false }
            guard Self.looksLikeNonItemRecoveryCandidate(item.description) else { return false }
            // Department-keyed ring-ups print the department AS the item
            // row ("DAIRY  8.99 T F", IMG_3524 Sprouts). The column path
            // has a same-row exception for these; FM-emitted items need
            // the same one here or a legitimate one-line ring-up gets
            // filtered into an empty item list.
            if Self.departmentRingUpConfirmed(
                description: item.description,
                price: item.totalPrice,
                lines: postLines
            ) { return false }
            return true
        }
        // Implausible-price filter: drop items whose price exceeds
        // 3× the receipt total. On Ace Hardware IMG_4359, FM emitted
        // "CALCOAST DUMP SKU $821.99" — that's a SKU/serial number
        // Vision concatenated onto the description text, then FM
        // treated as the price. Column-anchored already filters
        // price > total on its path, but FM's items survive when
        // column extraction produces zero items and we fall through
        // to the FM output. The × 3 threshold keeps legit cases
        // where a big item is offset by a credit/refund (IMG_4161:
        // $34.99 CARBONATOR with a −$15.50 credit and a $21.41 total
        // stays at ratio 1.63, well under 3×).
        if receipt.totals.total > 0 {
            let priceCeiling = receipt.totals.total * Decimal(3)
            receipt.lineItems.removeAll { $0.totalPrice > priceCeiling }
        }
        // Backfill bboxes by matching extracted field values back to OCR lines.
        // The bbox overlay in the labeler uses these to highlight detections.
        // Uses the normalized lines so a price like "6. 49 S" still matches
        // its bbox slot — the bbox coordinates are identical to the
        // original observation either way.
        receipt = Self.attachBBoxes(receipt: receipt, lines: postLines)

        // Checksum repair: the receipt's own arithmetic (items sum to
        // subtotal / total) is a printed checksum. When the final item
        // list misses it by EXACTLY one item's price (phantom) or one
        // rejected price point's value (wrong rejection), make that
        // single conservative move. Exact-match + uniqueness gates keep
        // coincidence repairs out.
        receipt = Self.balanceItemsAgainstChecksum(
            receipt: receipt,
            rejects: rejectedPoints,
            lines: postLines
        )
        // Late arithmetic pass, AFTER all item filtering has settled:
        // when the items independently sum to the printed total, any
        // nonzero tax / tip / discount is arithmetically impossible —
        // those fields are phantom copies (a "You Saved" amount or rate
        // value FM cross-wired into them). This must run after the item
        // list is final, unlike reconcileTotalsArithmetic which runs
        // before the weight/non-item passes mutate items.
        receipt = Self.reconcileItemsVsTotals(receipt)
        // Replace the field-average confidence with a receipt-level score
        // grounded in structural signals (items sum to total, date parsed,
        // merchant identified, no metadata leakage). Downstream consumers
        // filter or highlight low-confidence extractions.
        receipt.provenance.confidence = Self.computeReceiptConfidence(receipt)

        return receipt
    }

    // MARK: - Date validation + recovery

    /// Today's date as "YYYY-MM-DD". ISO strings compare correctly with
    /// plain string ordering, which is all the plausibility checks need.
    private static func todayISO() -> String {
        let comps = Calendar(identifier: .gregorian).dateComponents([.year, .month, .day], from: Date())
        return String(format: "%04d-%02d-%02d", comps.year ?? 0, comps.month ?? 0, comps.day ?? 0)
    }

    /// Strict validation of an FM-emitted date. Returns the normalized
    /// "YYYY-MM-DD" string, or nil if the value is malformed, not a real
    /// calendar date (month 13, day 42), in the future, or implausibly
    /// old for a personal receipt library (> 10 years back).
    ///
    /// The calendar-validity check matters in practice: FM fuses the
    /// printed time into the day slot ("03-15-2026 09:42" → "2026-09-42")
    /// often enough that shape-only validation lets junk through.
    static func validatedISODate(_ raw: String) -> String? {
        let d = raw.trimmingCharacters(in: .whitespaces)
        guard d.range(of: #"^(19|20)\d{2}-\d{2}-\d{2}$"#, options: .regularExpression) != nil else {
            return nil
        }
        let parts = d.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return validatedDateComponents(year: parts[0], month: parts[1], day: parts[2])
    }

    /// Shared plausibility gate: real month, real day-of-month (leap
    /// years handled), within [today - 10 years, today].
    private static func validatedDateComponents(year: Int, month: Int, day: Int) -> String? {
        guard (1...12).contains(month) else { return nil }
        let daysInMonth: Int = {
            switch month {
            case 1, 3, 5, 7, 8, 10, 12: return 31
            case 4, 6, 9, 11: return 30
            default:
                let isLeap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
                return isLeap ? 29 : 28
            }
        }()
        guard (1...daysInMonth).contains(day) else { return nil }
        let iso = String(format: "%04d-%02d-%02d", year, month, day)
        let today = todayISO()
        guard iso <= today else { return nil }
        let currentYear = Int(today.prefix(4)) ?? 0
        guard year >= currentYear - 10 else { return nil }
        return iso
    }

    /// Scan OCR observations (already sorted top-to-bottom) for a
    /// date-shaped token and return the first plausible one as
    /// "YYYY-MM-DD". Handles the formats receipts actually print:
    ///   12/17/23 08:50        (US M/D/YY + time)
    ///   11/06/24              (US M/D/YY)
    ///   03-15-2026 09:42      (US M-D-YYYY + time)
    ///   2025/09/08 14:51:30   (ISO-ish Y/M/D + time)
    ///   Wednesday, 16 October, 2024 02:22 PM
    ///   Oct 16, 2024
    /// Candidates failing the plausibility gate (month 13, future date,
    /// ancient date) are skipped, so a stray SKU or phone number can't
    /// win — those digit runs virtually never form a valid recent date.
    static func recoverDateFromOCR(_ lines: [(text: String, box: Receipt.BBox)]) -> String? {
        let monthNames: [String: Int] = [
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        ]
        // (pattern, group order) — group order maps regex capture groups
        // to (year, month, day) positions.
        struct DatePattern {
            let regex: NSRegularExpression
            let order: (y: Int, m: Int, d: Int)  // capture-group indices
            let twoDigitYear: Bool
        }
        let patternSpecs: [(String, (y: Int, m: Int, d: Int), Bool)] = [
            // 2025/09/08, 2025-09-08, 2025.09.08
            (#"\b(20\d{2})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b"#, (1, 2, 3), false),
            // 03/15/2026, 03-15-2026
            (#"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](20\d{2})\b"#, (3, 1, 2), false),
            // 12/17/23 — two-digit year, assume 20YY
            (#"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})\b"#, (3, 1, 2), true),
        ]
        var numericPatterns: [DatePattern] = []
        for spec in patternSpecs {
            guard let re = try? NSRegularExpression(pattern: spec.0) else { continue }
            numericPatterns.append(DatePattern(regex: re, order: spec.1, twoDigitYear: spec.2))
        }
        // "October 16, 2024" / "Oct 16 2024"
        let monthFirstRe = try? NSRegularExpression(
            pattern: #"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*,?\s+(20\d{2})\b"#)
        // "16 October, 2024" / "16 Oct 2024"
        let dayFirstRe = try? NSRegularExpression(
            pattern: #"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s*,?\s+(20\d{2})\b"#)

        func group(_ match: NSTextCheckingResult, _ i: Int, in text: String) -> String? {
            guard let r = Range(match.range(at: i), in: text) else { return nil }
            return String(text[r])
        }

        for line in lines {
            let text = line.text
            let full = NSRange(text.startIndex..<text.endIndex, in: text)
            for pat in numericPatterns {
                for match in pat.regex.matches(in: text, options: [], range: full) {
                    guard
                        let ys = group(match, pat.order.y, in: text),
                        let ms = group(match, pat.order.m, in: text),
                        let ds = group(match, pat.order.d, in: text),
                        var y = Int(ys), let m = Int(ms), let d = Int(ds)
                    else { continue }
                    if pat.twoDigitYear { y += 2000 }
                    if let iso = validatedDateComponents(year: y, month: m, day: d) {
                        return iso
                    }
                }
            }
            for (re, monthGroup, dayGroup) in [(monthFirstRe, 1, 2), (dayFirstRe, 2, 1)] {
                guard let re else { continue }
                for match in re.matches(in: text, options: [], range: full) {
                    guard
                        let name = group(match, monthGroup, in: text)?.lowercased().prefix(3),
                        let month = monthNames[String(name)],
                        let ds = group(match, dayGroup, in: text), let d = Int(ds),
                        let ys = group(match, 3, in: text), let y = Int(ys)
                    else { continue }
                    if let iso = validatedDateComponents(year: y, month: month, day: d) {
                        return iso
                    }
                }
            }
        }
        return nil
    }

    /// Single-move checksum repair against the receipt's own arithmetic.
    ///
    /// Target: the printed subtotal when we have one (it IS the items
    /// sum, pre-tax), else total − tax − tip + discount. If the final
    /// items miss the target, attempt exactly ONE repair:
    ///
    ///   * OVER by exactly one item's price → that item is a phantom
    ///     (duplicated price row, stolen footer value). Drop it — but
    ///     only when precisely one item matches the delta, so we never
    ///     guess between candidates.
    ///   * UNDER by exactly one rejected price point's value → that
    ///     rejection was wrong. Re-admit it with the nearest left-side
    ///     text as description. Again, only on a unique match.
    ///
    /// The exactness requirement (± 1¢) is the safety: an arbitrary
    /// wrong item/reject matching the residual to the cent is unlikely,
    /// and the uniqueness gate refuses ambiguous repairs outright.
    private static func balanceItemsAgainstChecksum(
        receipt input: Receipt,
        rejects: [RejectedPricePoint],
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let total = receipt.totals.total
        guard total > 0, !receipt.lineItems.isEmpty else { return receipt }

        let tax = receipt.totals.tax.first?.amount ?? 0
        let tip = receipt.totals.tip ?? 0
        let discount = receipt.totals.discount ?? 0
        // Prefer the printed subtotal as the items target; it's the
        // direct checksum. Guard against degenerate subtotals (≤ 0,
        // or ≥ total with tax present).
        var targets: [Decimal] = []
        if let s = receipt.totals.subtotal, s > 0, s <= total { targets.append(s) }
        let arithmeticTarget = total - tax - tip + discount
        if arithmeticTarget > 0, !targets.contains(arithmeticTarget) {
            targets.append(arithmeticTarget)
        }
        guard !targets.isEmpty else { return receipt }

        let itemsSum: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        let tol = Decimal(0.011)
        func approx(_ a: Decimal, _ b: Decimal) -> Bool {
            let d = a - b
            return d < tol && d > -tol
        }

        // Attempt the single-move repair against EACH target in turn and
        // keep the first that lands. Trying both targets matters when a
        // surcharge sits between subtotal and total (TASSI: item $2,160
        // + surcharge $64.80 = total $2,224.80 — only the total-based
        // target exposes the missing $2,160 line).
        for target in targets {
            let delta = itemsSum - target
            let absDelta = delta < 0 ? -delta : delta
            if absDelta <= tol { return receipt }   // books already balance

            if delta > 0 {
                // OVER: drop the unique item whose price equals the excess.
                let matches = receipt.lineItems.enumerated().filter { approx($0.element.totalPrice, delta) }
                if matches.count == 1 {
                    receipt.lineItems.remove(at: matches[0].offset)
                    return receipt
                }
                continue
            }

            // UNDER: re-admit the unique rejected price point whose value
            // equals the shortfall. Dedupe rejects by (value, centerY) —
            // the same point can be rejected at multiple gates.
            let needed = -delta
            var seen = Set<String>()
            let candidates = rejects.filter { rp in
                guard rp.value > 0, approx(rp.value, needed) else { return false }
                let key = "\(rp.value)@\(Int((rp.centerY * 1000).rounded()))"
                return seen.insert(key).inserted
            }
            guard candidates.count == 1, let point = candidates.first else { continue }

            // Nearest non-price text to the left for the description.
            let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})(?:\s+[A-Z]{1,2}){0,2}$"#
            var bestDesc: (text: String, dy: Double)? = nil
            for line in lines {
                let cy = line.box.y + line.box.height / 2
                let dy = abs(cy - point.centerY)
                guard dy <= max(0.012, point.box.height * 2.0) else { continue }
                guard line.box.x < point.box.x else { continue }
                let t = line.text.trimmingCharacters(in: .whitespaces)
                guard t.count >= 3 else { continue }
                if t.range(of: priceShape, options: .regularExpression) != nil { continue }
                if bestDesc == nil || dy < bestDesc!.dy {
                    bestDesc = (t, dy)
                }
            }
            guard let desc = bestDesc.map({ Self.cleanLineItemDescription($0.text) }),
                  desc.count >= 3 else { continue }
            receipt.lineItems.append(Receipt.LineItem(
                description: desc,
                quantity: nil,
                unitPrice: nil,
                totalPrice: point.value,
                category: nil
            ))
            return receipt
        }
        return receipt
    }

    /// A tax amount is never negative on a real receipt. FM occasionally
    /// reads a savings figure or a misaligned column value into the tax
    /// slot with a stray minus (IMG_8253: tax "-1.25"). Drop it so the
    /// bogus value can't corroborate a wrong total or skew confidence.
    private static func dropInvalidTax(_ input: Receipt) -> Receipt {
        var receipt = input
        let before = receipt.totals.tax.count
        receipt.totals.tax.removeAll { $0.amount < 0 }
        if receipt.totals.tax.count != before {
            receipt.provenance.fieldConfidence["totals.tax"] = 0.3
        }
        return receipt
    }

    /// See call site: clears phantom tax / tip / discount when the final
    /// item list already sums to the total, and clamps tax down to the
    /// arithmetic gap when items nearly reach the total on their own.
    private static func reconcileItemsVsTotals(_ input: Receipt) -> Receipt {
        var receipt = input
        let total = receipt.totals.total
        guard total > 0, !receipt.lineItems.isEmpty else { return receipt }
        let itemsSum: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        guard itemsSum > 0 else { return receipt }

        let tol: Decimal = max(Decimal(0.02), total * Decimal(0.005))
        let gap = total - itemsSum
        let absGap = gap < 0 ? -gap : gap

        let tax = receipt.totals.tax.first?.amount ?? 0
        let tip = receipt.totals.tip ?? 0
        let discount = receipt.totals.discount ?? 0
        let extras = tax + tip - discount
        let absExtras = extras < 0 ? -extras : extras

        // Items ≈ total exactly: nothing else fits. Two independently-
        // extracted signals agreeing beats any single extracted field.
        // (IMG_2678: sum == total == 32.42 yet tax AND tip both said
        // 6.43 — a savings amount cross-wired into both slots.)
        if absGap <= tol, absExtras > tol {
            receipt.totals.tax = []
            receipt.totals.tip = nil
            receipt.totals.discount = nil
            if let s = receipt.totals.subtotal, s != total {
                receipt.totals.subtotal = nil
            }
            return receipt
        }

        // Items nearly reach the total (≥ 90%): the tax can never
        // exceed the remaining gap. A bigger extracted value is a
        // misread — clamp it to the arithmetic remainder. Only when
        // tip/discount are absent so the algebra is unambiguous.
        if gap > tol, itemsSum >= total * Decimal(0.9),
           tip == 0, discount == 0, tax > gap + tol {
            receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: nil, amount: gap)]
            receipt.provenance.fieldConfidence["totals.tax"] = 0.6
        }
        return receipt
    }

    // MARK: - Confidence scoring

    /// Compute a 0.0–1.0 receipt-level confidence from structural signals.
    ///
    /// The dominant signal is "items sum to total (± tax/tip)": if that
    /// holds within a small relative tolerance, the extraction is almost
    /// certainly correct end-to-end. From there we subtract for missing /
    /// suspicious pieces (no date, no merchant, no line items, an
    /// implausibly large total, metadata leaked into descriptions).
    ///
    /// Kept intentionally simple so it's easy to reason about downstream —
    /// dashboards can bucket by e.g. ≥ 0.85 high, ≥ 0.6 medium, < 0.6 low.
    static func computeReceiptConfidence(_ receipt: Receipt) -> Double {
        let total = doubleValue(receipt.totals.total)
        let itemsSum = receipt.lineItems.reduce(0.0) { $0 + doubleValue($1.totalPrice) }
        let tax = receipt.totals.tax.reduce(0.0) { $0 + doubleValue($1.amount) }
        let tip = doubleValue(receipt.totals.tip ?? 0)
        let discount = doubleValue(receipt.totals.discount ?? 0)

        var score = 0.5

        // Structural agreement between items sum and total. The best signal
        // by far: if it balances, everything else was probably read right.
        // Skipped when NO items were extracted — a zero sum against a real
        // total isn't evidence of a bad read (cut-off photos legitimately
        // have no item region); the empty-items branch below scores those.
        if total > 0, !receipt.lineItems.isEmpty {
            let expected = itemsSum + tax + tip - discount
            let delta = abs(total - expected)
            let rel = delta / total
            if rel < 0.01 {
                score += 0.40         // exact match
            } else if rel < 0.03 {
                score += 0.32
            } else if rel < 0.06 {
                score += 0.22
            } else if rel < 0.10 {
                score += 0.10
            } else if rel < 0.20 {
                score -= 0.05
            } else if rel < 0.50 {
                score -= 0.20
            } else {
                score -= 0.35
            }
        } else {
            // No total at all — we have almost nothing to check against.
            score -= 0.25
        }

        // Every line item has a plausible positive price.
        if !receipt.lineItems.isEmpty {
            let allPricesOK = receipt.lineItems.allSatisfy { item in
                let v = doubleValue(item.totalPrice)
                return v != 0 && abs(v) < max(total * 5, 1000)
            }
            if allPricesOK {
                score += 0.05
            } else {
                score -= 0.10
            }
        } else {
            // Zero line items with COHERENT money fields (subtotal + tax
            // = total) is the cut-off-photo signature: the shot captured
            // only the totals block, and there was nothing more to
            // extract (IMG_3040 Grocery Outlet). The spend is real and
            // verified by its own arithmetic — score medium so the
            // receipt's total participates in trends, while the missing
            // items keep it out of the high band.
            let sub = receipt.totals.subtotal
            if let sub, total > 0 {
                let expected = doubleValue(sub) + tax
                if abs(expected - total) <= 0.02 {
                    score += 0.13
                } else if total > 3 {
                    score -= 0.15
                }
            } else if total > 3 {
                // Zero line items on a non-trivial receipt with no
                // corroborating subtotal is suspicious.
                score -= 0.15
            }
        }

        // Date parsed AND plausible. The "1970-01-01" sentinel means "no
        // valid date found" — it must score as missing, not as a valid
        // date (it's shaped like one, which fooled the earlier check and
        // let undated receipts report confidence 1.0).
        let dateValue = receipt.header.date.value
        if dateValue == "1970-01-01" || validatedISODate(dateValue) == nil {
            score -= 0.10
        } else {
            score += 0.05
        }

        // Merchant identified.
        let merchant = receipt.header.merchant.name.trimmingCharacters(in: .whitespaces)
        if !merchant.isEmpty, merchant.lowercased() != "unknown" {
            score += 0.05
        } else {
            score -= 0.10
        }

        // Metadata leakage in descriptions.
        let metadataMarkers = ["personalized", "lb @", "oz @", "sale price", "member savings"]
        let hasMetadataLeak = receipt.lineItems.contains { item in
            let d = item.description.lowercased()
            return metadataMarkers.contains { d.contains($0) }
        }
        if hasMetadataLeak {
            score -= 0.08
        }

        // Pure-numeric description (SKU/UPC leaked past the strip).
        let hasSKUDesc = receipt.lineItems.contains { item in
            let stripped = item.description.trimmingCharacters(in: .whitespaces)
            return stripped.count >= 6 && stripped.allSatisfy(\.isNumber)
        }
        if hasSKUDesc { score -= 0.05 }

        // Tax > 50% of total is nonsense — usually the "total" cell was
        // misread and a subtotal / balance value ended up there.
        if total > 0, tax > total * 0.5 { score -= 0.20 }

        return max(0.0, min(1.0, score))
    }

    private static func doubleValue(_ d: Decimal) -> Double {
        NSDecimalNumber(decimal: d).doubleValue
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

        // Validate FM's date (calendar-valid, not future, not ancient).
        // FM misassembles dates in specific ways we've seen in the wild:
        // Trader Joe's prints "03-15-2026 09:42" and FM emitted
        // "2026-09-42" — the YEAR fused with the TIME. When validation
        // rejects, recover by scanning the OCR text for a date-shaped
        // token; receipts nearly always print the transaction date, FM
        // just failed to assemble it. Sentinel only as a last resort.
        let dateValue: String =
            Self.validatedISODate(x.date)
            ?? Self.recoverDateFromOCR(ocrLines)
            ?? "1970-01-01"

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
        // Walk the string with a minimal JSON lexer (string / escape /
        // depth state) and remember the last offset where a line-item
        // object closed cleanly — i.e. a '}' that returns depth to the
        // lineItems-array level. Truncate there and close the array +
        // root object. A lexer is necessary: naive "}," searches break
        // when the model output was cut mid-string, contains escaped
        // quotes ('24\" PLANT PROP'), or braces inside descriptions.
        guard let itemsKey = json.range(of: "\"lineItems\"") else { return nil }

        var inString = false
        var escaped = false
        var depth = 0
        var arrayDepthAtItems: Int? = nil     // depth just after '[' of lineItems
        var lastCompleteItemEnd: String.Index? = nil
        var sawItemsArray = false

        var i = json.startIndex
        while i < json.endIndex {
            let c = json[i]
            if inString {
                if escaped {
                    escaped = false
                } else if c == "\\" {
                    escaped = true
                } else if c == "\"" {
                    inString = false
                }
            } else {
                switch c {
                case "\"": inString = true
                case "{", "[":
                    depth += 1
                    if c == "[", !sawItemsArray, i > itemsKey.upperBound {
                        sawItemsArray = true
                        arrayDepthAtItems = depth
                    }
                case "}", "]":
                    depth -= 1
                    // A '}' that lands back on the lineItems array depth
                    // is the clean end of one complete item object.
                    if c == "}", let ad = arrayDepthAtItems, depth == ad {
                        lastCompleteItemEnd = i
                    }
                default: break
                }
            }
            i = json.index(after: i)
        }

        guard sawItemsArray else { return nil }
        if let end = lastCompleteItemEnd {
            return String(json[...end]) + "]}"
        }
        // No complete item survived — close the array empty so at least
        // merchant / date / totals make it through.
        guard let arrayOpen = json.range(of: "[", range: itemsKey.upperBound..<json.endIndex) else {
            return nil
        }
        return String(json[..<arrayOpen.upperBound]) + "]}"
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
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
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
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?$"#

        // Comma-vs-dot tolerant substring match. Receipts in regions that
        // use "," as the decimal separator ("4,00") wouldn't otherwise
        // match FM's normalized "4.00" output.
        func lineContainsAmount(_ text: String) -> Bool {
            if text.contains(amountStr) { return true }
            return text.replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".").contains(amountStr)
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

    /// Label-anchored totals and tax values read straight from the OCR,
    /// independent of FM's (possibly wrong) field picks. The column
    /// election uses these as checksum targets: a candidate price column
    /// whose item sum equals a printed BALANCE / PAYMENT AMOUNT — or
    /// that value minus a printed TAX — is the real paid column.
    private static func ocrTotalsTargets(
        lines: [(text: String, box: Receipt.BBox)]
    ) -> (totals: Set<Decimal>, taxes: Set<Decimal>) {
        let totalKeywords = [
            "grand total", "balance due", "total due", "amount due",
            "amount paid", "payment amount", "balance", "total",
            "subtotal", "sub total", "sub-total",
        ]
        let taxKeywords = ["tax", "sales tax"]
        let countMarkers = [
            "number of items", "items in transaction",
            "item count", "items sold", "total items",
        ]
        // Cents required: integer tokens near totals labels are usually
        // item counts, store numbers, or points balances.
        let valueShape = #"^-?\$?\d{1,3}(?:,\d{3})*[.,]\d{2}$"#
        func value(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
            guard trimmed.range(of: valueShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }
        func normalize(_ text: String) -> String {
            text.lowercased()
                .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
                .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
        }

        var totals: Set<Decimal> = []
        var taxes: Set<Decimal> = []
        for (idx, line) in lines.enumerated() {
            let stripped = normalize(line.text)
            guard !stripped.isEmpty else { continue }
            if countMarkers.contains(where: { stripped.contains($0) }) { continue }
            let isTotal = totalKeywords.contains { stripped == $0 || stripped.hasPrefix($0 + " ") }
            let isTax = !isTotal && taxKeywords.contains { stripped == $0 || stripped.hasPrefix($0 + " ") }
            guard isTotal || isTax else { continue }
            var found: [Decimal] = []
            // Inline value ("BALANCE 75.78" as one observation).
            for tok in line.text.split(whereSeparator: { $0.isWhitespace }) {
                if let v = value(String(tok)), v > 0 { found.append(v) }
            }
            // Same-row neighbors — Vision usually splits the label and
            // its amount into separate observations.
            let cy = line.box.y + line.box.height / 2
            let tol = max(0.008, line.box.height * 0.9)
            for (jdx, other) in lines.enumerated() where jdx != idx {
                let oy = other.box.y + other.box.height / 2
                guard abs(oy - cy) <= tol else { continue }
                if let v = value(other.text), v > 0 { found.append(v) }
            }
            for v in found {
                if isTotal { totals.insert(v) } else { taxes.insert(v) }
            }
        }
        return (totals, taxes)
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
    /// A price observation that column-anchored extraction REJECTED, and
    /// where. The arithmetic-balance pass uses these: when the final
    /// items fall short of the receipt's own subtotal/total by exactly
    /// one rejected value, that rejection was wrong — the value both
    /// exists in the price column and completes the checksum.
    struct RejectedPricePoint {
        let value: Decimal
        let centerY: Double
        let box: Receipt.BBox
    }

    /// Returns empty items when fewer than 2 prices land in the column —
    /// caller falls back to the FM-only path.
    private static func columnAnchoredLineItems(
        lines: [(text: String, box: Receipt.BBox)],
        fmItems: [Receipt.LineItem],
        totalsValues: Set<Decimal>,
        receiptTotal: Decimal
    ) -> (items: [Receipt.LineItem], rejects: [RejectedPricePoint]) {
        // Require decimal cents in column-anchored price detection.
        // Integer-only prices like "79 S" are almost always OCR
        // fragments of a longer price ("2.79 S" split by Vision into
        // "2." and "79 S") rather than legitimate whole-dollar
        // amounts. Allowing them lets the fragment pass as a large
        // price and inflates items_sum (IMG_2460 Safeway "$79 ADOBO"
        // is really $2.79). Real whole-dollar amounts are rare
        // enough that this trade-off pays off across the sample.
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})(?:\s+[A-Z]{1,2}){0,2}$"#

        // Step 1: every price-shaped observation with its value.
        // Also flag whether the observation carried a tax-category
        // marker ("4.79 S", "7.99 B") — Safeway prints both an
        // actual paid price WITH marker and a regular price WITHOUT
        // marker for every item; the marker tells us which is the
        // one that was actually charged.
        struct PricePoint {
            let value: Decimal
            let box: Receipt.BBox
            let centerY: Double
            let lineIdx: Int
            let hasTaxMarker: Bool
        }
        let taxMarkerSuffix = try? NSRegularExpression(
            pattern: #"\d\s+[A-Z]{1,2}\s*$"#,
            options: []
        )
        var pricePoints: [PricePoint] = []
        for (idx, line) in lines.enumerated() {
            let trimmed = line.text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { continue }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
            guard let v = Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX")) else { continue }
            // Accept negative prices — they're legitimate line items
            // (voided items, exchange credits, Safety Captain discounts,
            // Ace Hardware EXCHNG credits, etc.). Column-anchored
            // extraction should pick them up and record them as
            // negative-valued LineItems so items_sum reconciles with
            // total.
            guard v != 0 else { continue }
            var marker = false
            if let re = taxMarkerSuffix {
                let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
                marker = re.firstMatch(in: trimmed, options: [], range: range) != nil
            }
            pricePoints.append(PricePoint(
                value: v,
                box: line.box,
                centerY: line.box.y + line.box.height / 2,
                lineIdx: idx,
                hasTaxMarker: marker
            ))
        }
        guard pricePoints.count >= 2 else { return ([], []) }
        var rejects: [RejectedPricePoint] = []

        // Step 2 (moved before column election — the election's checksum
        // simulation needs the boundary): totals-block Y. Anything at or
        // below is payment / footer noise.
        let totalsKeywords: [String] = [
            "subtotal", "sub-total", "tax", "balance", "total",
            "amount due", "amount paid", "payment amount", "tip", "change",
            // Restaurant / fast-food subtotal-summary rows. These
            // aren't the grand total but they DO sit in the totals
            // block above tax, and they carry a total-shaped price —
            // if we don't mark them as totals boundary, column-
            // anchored treats them as a line item (IMG_1250 In-N-Out:
            // "DRIVE-Take Out 14.15" got picked up as an item for
            // $14.15 above the actual $16.21 total).
            "drive-take out", "drive thru", "drive-thru", "dine-in",
            "dine in", "for here", "to go", "take out", "takeout",
        ]
        // A totals label only bounds the item region when actual item
        // prices sit ABOVE it. Receipts photographed with the payment
        // slip laid on top of the receipt put a "Total:" at the TOP of
        // the frame (IMG_4253 Sprouts) — taking the minimum label Y
        // filtered out every real item, column extraction returned
        // empty, and the FM fallback's corrupt values leaked through.
        // Values that print ≥ 3 times are almost certainly the grand
        // total (Subtotal / Net Sales / Total / payment copies all carry
        // it). Treated as totals values below so a flipped receipt's
        // stack of identical totals can't masquerade as "item prices
        // above" the boundary.
        let repeatedValues: Set<Decimal> = {
            var counts: [Decimal: Int] = [:]
            for p in pricePoints { counts[p.value, default: 0] += 1 }
            return Set(counts.filter { $0.value >= 3 }.keys)
        }()
        var totalsBlockY: Double = 1.0
        for line in lines {
            if Self.lineLooksLikeTotalsBoundary(line.text, keywords: totalsKeywords) {
                let cy = line.box.y + line.box.height / 2
                // Require ≥ 1 genuine ITEM price above — not a totals
                // value and not a value repeated ≥ 3× (both are the
                // total echoed through the totals block). The slip-above
                // case has ZERO prices above its top-of-frame "Total:"
                // label; a real boundary always has item prices above.
                // The item-price qualifier additionally rescues FLIPPED
                // receipts (Whole Foods photographed 180°, IMG_8245):
                // there the totals block sits ABOVE the items in OCR Y,
                // and the only prices above a totals label are echoes of
                // the total — so no boundary is set, the total echoes
                // filter as totals values, and the real items below
                // survive to reconcile against the total.
                let itemPricesAbove = pricePoints.filter {
                    $0.centerY < cy - 0.002
                        && !totalsValues.contains($0.value)
                        && !repeatedValues.contains($0.value)
                }.count
                guard itemPricesAbove >= 1 else { continue }
                if cy < totalsBlockY { totalsBlockY = cy }
            }
        }

        // Step 4 (precompute): Y centers of every metadata label
        // observation, classed by semantics. Fuzzy-matches against
        // canonical phrases — with a trailing amount stripped first —
        // so OCR typos ("Reguler Price", "Menber Savings -0.79") still
        // get flagged; otherwise column-anchored picks the metadata
        // values (or their text) as line items on faded Safeway
        // receipts. Savings-class labels poison BOTH signs of adjacent
        // price (the paid value lives elsewhere); discount-class labels
        // ("Item Discount") own a legitimate NEGATIVE item amount, so
        // only positive same-row prices are junk there.
        // NOTE: registration deliberately does NOT use the amount-
        // stripped fuzzy match. A merged label ("Member Savings -1.30"
        // in one observation) carries its value inside its own text —
        // there is no separate same-row price point to poison, and on
        // dense three-row Safeway layouts the row pitch is tight enough
        // that a stripped-match registry started poisoning the PAID
        // price on the neighboring row (IMG_5046 lost $9 of items).
        // Merged labels are handled where they actually bite: the
        // description filter (`looksLikeNonItemRecoveryCandidate`) and
        // the FM prompt filter (`dropMetadataObservations`).
        struct MetaLabel { let y: Double; let isSavingsClass: Bool }
        let metaLabels: [MetaLabel] = lines.compactMap { line in
            let lc = line.text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            guard !lc.isEmpty else { return nil }
            let y = line.box.y + line.box.height / 2
            if lc.count <= 30, fuzzyMatchesAny(lc, phrases: savingsClassPhrases) {
                return MetaLabel(y: y, isSavingsClass: true)
            }
            if lc.count <= 30, fuzzyMatchesAny(lc, phrases: discountClassPhrases) {
                return MetaLabel(y: y, isSavingsClass: false)
            }
            // Legacy shortcuts for phrases the fuzzy set doesn't cover.
            let extras = ["for personalized", "forl personalized"]
            if extras.contains(where: { lc == $0 || lc.hasPrefix($0 + " ") }) {
                return MetaLabel(y: y, isSavingsClass: true)
            }
            return nil
        }

        // Step 1b: identify the price column.
        //
        // Median X works for most receipts, but Safeway-style chains that
        // print BOTH a "Regular Price / Member Savings" metadata column
        // AND a real paid-price column produce a bimodal X distribution
        // (metadata at x≈0.58, actual prices at x≈0.66 on IMG_1788).
        // With more metadata rows than real rows (2:1 ratio: each item
        // has a regular-price + savings sibling), the median lands
        // BETWEEN the columns and the ±0.05 band excludes the real
        // prices entirely.
        //
        // Legacy heuristic: detect a bimodal X distribution by scanning
        // sorted X values for the single largest gap and prefer the
        // RIGHTMOST cluster. That breaks on Safeway's two-column
        // "Price | You Pay" layout (IMG_6463): the columns sit only
        // ~0.04 apart while stray left-side points (weights, SKU
        // fragments) produce a LARGER gap elsewhere, so the merged
        // right side elects the regular-price column and every
        // discounted item extracts at its pre-savings price.
        //
        // Checksum election, tried first: the receipt itself says which
        // column is real — the paid column SUMS to a printed totals
        // value (BALANCE / PAYMENT AMOUNT), optionally plus tax. Split
        // the X axis into clusters, simulate each cluster's item sum
        // under the standard filters, and elect the rightmost cluster
        // whose sum matches a label-anchored totals target to the cent.
        // Falls back to the legacy pick when nothing matches (missing
        // rows, misread digits) — no behavior change there.
        //
        // Column X is derived from ITEM-REGION prices only (above the
        // totals block). On short receipts the totals block prints more
        // values than there are items (Subtotal / Savings / Net / Tax /
        // Total — 5+ values vs 2 items on Whole Foods IMG_8129), and
        // those right-aligned totals values pull the median X off the
        // item column, excluding the real item prices from the ±0.05
        // band entirely. The totals values get filtered as totals-block
        // anyway, so they have no business defining where the item
        // column is. Fall back to all points if the item region is too
        // sparse to be reliable.
        let itemRegionXs = pricePoints
            .filter { $0.centerY < totalsBlockY - 0.002 }
            .map(\.box.x)
            .sorted()
        let xs = itemRegionXs.count >= 2 ? itemRegionXs : pricePoints.map(\.box.x).sorted()
        let legacyMedianX: Double = {
            guard xs.count >= 4 else { return xs[xs.count / 2] }
            var largestGap: Double = 0
            var gapIdx = -1
            for i in 1..<xs.count {
                let g = xs[i] - xs[i - 1]
                if g > largestGap { largestGap = g; gapIdx = i }
            }
            // A gap of 0.03+ is a real column boundary (thermal-printer
            // columns are typically ≥ 0.06 apart in normalized coords).
            if largestGap >= 0.03, gapIdx >= 0 {
                let rightCluster = Array(xs[gapIdx..<xs.count])
                // Only use the rightmost cluster if it's got enough
                // members that it's plausibly the item column — otherwise
                // a single stray X (rare) would hijack the pick.
                if rightCluster.count >= 2 {
                    return rightCluster[rightCluster.count / 2]
                }
            }
            return xs[xs.count / 2]
        }()

        let debugCol = ProcessInfo.processInfo.environment["OCR_DEBUG_COL"] != nil
        func dbg(_ s: String) {
            if debugCol { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        }

        // Simulate the item sum a column pick would produce, mirroring
        // the pair-dedup + totals-block + totals-value (with the
        // multiplicity exception) + metadata-label filters the real
        // walk applies below. Also reports how many points survive —
        // election requires a real column, not two junk values that
        // happen to add up.
        func simulateColumn(around columnX: Double) -> (sum: Decimal, survivors: Int) {
            let lo = max(0.0, columnX - 0.05)
            let hi = columnX + 0.05
            let raw = pricePoints
                .filter { lo <= $0.box.x && $0.box.x <= hi }
                .sorted { $0.centerY < $1.centerY }
            var kept: [PricePoint] = []
            var i = 0
            while i < raw.count {
                let cur = raw[i]
                if i + 1 < raw.count {
                    let nxt = raw[i + 1]
                    let sameSign = (cur.value > 0 && nxt.value > 0) || (cur.value < 0 && nxt.value < 0)
                    if abs(nxt.centerY - cur.centerY) < 0.005 && sameSign {
                        let keeper: PricePoint = {
                            if cur.hasTaxMarker && !nxt.hasTaxMarker { return cur }
                            if !cur.hasTaxMarker && nxt.hasTaxMarker { return nxt }
                            return cur.value <= nxt.value ? cur : nxt
                        }()
                        kept.append(keeper)
                        i += 2
                        continue
                    }
                }
                kept.append(cur)
                i += 1
            }
            var sum: Decimal = 0
            var survivors = 0
            for p in kept {
                if p.centerY >= totalsBlockY - 0.002 { continue }
                if totalsValues.contains(p.value) {
                    let dupesAbove = kept.filter {
                        $0.value == p.value && $0.centerY < totalsBlockY - 0.002
                    }.count
                    let distinctAbove = Set(kept.filter {
                        $0.centerY < totalsBlockY - 0.002
                    }.map(\.value)).count
                    if dupesAbove < 2 || distinctAbove < 2 { continue }
                }
                let poisoned = metaLabels.contains { m in
                    abs(m.y - p.centerY) <= 0.006 && (p.value > 0 || m.isSavingsClass)
                }
                if poisoned { continue }
                sum += p.value
                survivors += 1
            }
            return (sum, survivors)
        }

        let ocrTargets = Self.ocrTotalsTargets(lines: lines)
        func sumMatchesTarget(_ sum: Decimal) -> Bool {
            guard sum > 0 else { return false }
            let eps = Decimal(string: "0.02")!
            for t in ocrTargets.totals {
                if abs(t - sum) <= eps { return true }
                for tax in ocrTargets.taxes where tax > 0 && tax < t {
                    if abs(t - (sum + tax)) <= eps { return true }
                }
            }
            return false
        }

        let medianX: Double = {
            guard !ocrTargets.totals.isEmpty else { return legacyMedianX }
            // If the legacy pick already balances against a printed
            // total, keep it — election exists to rescue receipts whose
            // column choice is provably wrong, not to second-guess ones
            // that already reconcile. This is the no-regression gate.
            let legacySim = simulateColumn(around: legacyMedianX)
            if sumMatchesTarget(legacySim.sum) { return legacyMedianX }
            // X clusters, split at gaps > 0.015. Safeway's "Price" and
            // "You Pay" columns sit as little as ~0.02 apart, so the
            // split must be finer than the inter-column gap. Election
            // only makes sense with ≥ 2 plausible columns.
            var clusters: [[Double]] = []
            var cur: [Double] = [xs[0]]
            for x in xs.dropFirst() {
                if let last = cur.last, x - last > 0.015 {
                    clusters.append(cur)
                    cur = [x]
                } else {
                    cur.append(x)
                }
            }
            clusters.append(cur)
            let viable = clusters.filter { $0.count >= 3 }
            guard viable.count >= 2 else { return legacyMedianX }
            for cluster in viable.reversed() {
                let m = cluster[cluster.count / 2]
                // Paid columns never sit LEFT of the legacy pick's
                // column — a left-side cluster whose junk happens to
                // sum to a target (weights, SKU fragments) must not
                // win. Small tolerance for jitter around equality.
                guard m >= legacyMedianX - 0.01 else { continue }
                let sim = simulateColumn(around: m)
                dbg("COL-CAND: x=\(m) n=\(cluster.count) sum=\(sim.sum) survivors=\(sim.survivors) targets=\(ocrTargets.totals.sorted()) taxes=\(ocrTargets.taxes.sorted())")
                guard sim.survivors >= 3 else { continue }
                if sumMatchesTarget(sim.sum) {
                    dbg("COL-ELECT: x=\(m) sum=\(sim.sum) survivors=\(sim.survivors) (legacy=\(legacyMedianX) legacySum=\(legacySim.sum))")
                    return m
                }
            }
            // Tax-marker election, when nothing checksums exactly (one
            // misread digit breaks ±1¢ matching). Safeway-style paid
            // columns suffix every value with a tax letter ("3.99 S");
            // the sibling regular-price column never does. A heavily-
            // marked cluster strictly RIGHT of a nearly-unmarked legacy
            // pick is definitively the paid column even without a
            // checksum: paid prices print rightmost, and markers only
            // ever decorate charged amounts.
            func markerStats(_ cluster: [Double]) -> (n: Int, marked: Int) {
                guard let lo = cluster.first, let hi = cluster.last else { return (0, 0) }
                var n = 0, marked = 0
                for p in pricePoints where p.box.x >= lo - 0.0001 && p.box.x <= hi + 0.0001 {
                    n += 1
                    if p.hasTaxMarker { marked += 1 }
                }
                return (n, marked)
            }
            let legacyCluster = clusters.first {
                ($0.first ?? 1) - 0.0001 <= legacyMedianX && legacyMedianX <= ($0.last ?? -1) + 0.0001
            } ?? []
            let ls = markerStats(legacyCluster)
            let legacyMostlyUnmarked = ls.n == 0 || ls.marked * 5 <= ls.n
            if legacyMostlyUnmarked {
                for cluster in viable.reversed() {
                    let m = cluster[cluster.count / 2]
                    guard m > legacyMedianX + 0.01 else { continue }
                    let s = markerStats(cluster)
                    guard s.n >= 5, s.marked * 2 >= s.n else { continue }
                    dbg("COL-ELECT-MARKER: x=\(m) n=\(s.n) marked=\(s.marked) (legacy=\(legacyMedianX) legacyMarked=\(ls.marked)/\(ls.n))")
                    return m
                }
            }
            return legacyMedianX
        }()
        let columnLo = max(0.0, medianX - 0.05)
        let columnHi = medianX + 0.05

        // Step 5–7: walk surviving prices in Y order.
        let inColumnRaw = pricePoints
            .filter { columnLo <= $0.box.x && $0.box.x <= columnHi }
            .sorted { $0.centerY < $1.centerY }

        // Safeway (and some other chains) print BOTH the actual paid
        // price AND the regular price for every item, stacked in the
        // same column just a few points apart:
        //     4.79 S     ← actual paid price (S = tax marker)
        //     4.79       ← regular price (no marker)
        //     MRTN CRSE SEA SLT
        // Or for discounted items:
        //     4.49 S     ← actual paid ($4.49)
        //     4.99       ← regular ($4.99, marked down $0.50)
        //     TRPCNA OJ
        // Column-anchored extraction was picking up BOTH prices,
        // doubling every item in the extracted list (IMG_5591 turned
        // 5 real items into 10, sum inflated $21 → $43). Dedupe by
        // scanning pairs of consecutive prices in the column: if
        // they're within 0.012 Y of each other, they're the paired
        // "actual/regular" and only one should survive. Preference:
        //   1. keep the price WITH a tax marker (S/T/B/F/N — that's
        //      the paid line)
        //   2. tiebreak toward the SMALLER value (the discounted
        //      paid amount is less than the regular)
        var inColumn: [PricePoint] = []
        var i = 0
        while i < inColumnRaw.count {
            let cur = inColumnRaw[i]
            if i + 1 < inColumnRaw.count {
                let nxt = inColumnRaw[i + 1]
                let dy = abs(nxt.centerY - cur.centerY)
                // 0.005 threshold — Safeway's actual/regular pairs sit
                // 0.001–0.003 apart (visually the same line, different
                // OCR baselines), while distinct items on other layouts
                // (Target IMG_6855) are 0.012+ apart. A stricter gate
                // keeps the dedup targeted at genuine pairs.
                // Only pair-dedup two same-sign values. A negative-value
                // item (credit / voided) and a positive item shouldn't
                // ever be paired even at the same Y — they're distinct
                // rows.
                let sameSign = (cur.value > 0 && nxt.value > 0) || (cur.value < 0 && nxt.value < 0)
                if dy < 0.005 && sameSign {
                    // A pair. Keep whichever has the tax marker; if
                    // both/neither have markers, keep the smaller
                    // value (the paid price).
                    let keeper: PricePoint = {
                        if cur.hasTaxMarker && !nxt.hasTaxMarker { return cur }
                        if !cur.hasTaxMarker && nxt.hasTaxMarker { return nxt }
                        return cur.value <= nxt.value ? cur : nxt
                    }()
                    let loser = keeper.lineIdx == cur.lineIdx ? nxt : cur
                    rejects.append(RejectedPricePoint(
                        value: loser.value, centerY: loser.centerY, box: loser.box
                    ))
                    inColumn.append(keeper)
                    i += 2
                    continue
                }
            }
            inColumn.append(cur)
            i += 1
        }

        var items: [Receipt.LineItem] = []
        func reject(_ p: PricePoint) {
            rejects.append(RejectedPricePoint(value: p.value, centerY: p.centerY, box: p.box))
        }
        if debugCol {
            let msg = "COL: pricePoints=\(pricePoints.count) medianX=\(medianX) inColumn=\(inColumn.count) totalsBlockY=\(totalsBlockY) metaLabels=\(metaLabels.count)\n"
            FileHandle.standardError.write(Data(msg.utf8))
            for p in inColumn {
                FileHandle.standardError.write(Data("  point v=\(p.value) cy=\(p.centerY) x=\(p.box.x)\n".utf8))
            }
        }
        // Implausibly-large filter setup. The ceiling protects against
        // decimal-lost misreads ("$99" from "$4.99") — but it derives
        // from FM's UNVERIFIED total. When FM grabs a tiny bogus value
        // (the "$1.00" YOUR-SAVINGS total on Safeway), a trusted ceiling
        // silently kills every real item, column extraction comes back
        // empty, and the total sanity check loses the items-sum signal
        // it needs to detect the bad total — a vicious circle
        // (IMG_3555: 5-item $27.52 receipt extracted as 0 items @ $1).
        // Rule: if the ceiling would eliminate MORE THAN HALF of the
        // in-column points, the total is what's wrong — don't apply it.
        let hasNegatives = pricePoints.contains { $0.value < 0 }
        let ceilingFactor: Decimal = hasNegatives ? 3 : 1
        let priceCeiling: Decimal? = {
            guard receiptTotal > 0 else { return nil }
            // Checksum override: when the column's unceilinged sum
            // already matches a printed totals value, every point in it
            // is corroborated by the receipt's own arithmetic — an FM
            // total small enough to shave points off it is what's wrong
            // (IMG_5026: FM grabbed the $5.39 savings total, whose
            // ceiling killed the $6.00 and $6.99 items of a $22.94
            // BALANCE that the column summed to exactly).
            if sumMatchesTarget(simulateColumn(around: medianX).sum) { return nil }
            let c = receiptTotal * ceilingFactor
            let over = inColumn.filter { $0.value > c }.count
            return over * 2 <= inColumn.count ? c : nil
        }()
        for p in inColumn {
            // Filter: below the totals block. NOT added to the rejects
            // ledger — payment / change / tender rows must never be
            // re-admitted as items, even when they'd balance the books.
            if p.centerY >= totalsBlockY - 0.002 { dbg("SKIP totals-block v=\(p.value)"); continue }
            // Filter: equals a known totals value. Also not re-admittable.
            // Exception: FM sometimes stuffs an ITEM price into its total
            // field (La Baguette IMG_2169: total "13.00" = the sandwich).
            // A genuine totals value never prints TWICE above the totals
            // block — when this value does, those occurrences are item
            // rows; keep them and let sanityCheckTotal fix the total.
            if totalsValues.contains(p.value) {
                // A value that appears ≥ 3× AND equals a printed total
                // is the total echoed through Subtotal / Net Sales /
                // Total / payment rows — never three identical items.
                // Force-skip ONLY when no totals boundary was found
                // (the flipped-receipt path: echoes sit among the items
                // and would otherwise be re-admitted via the two-
                // sandwiches exception below — IMG_8245: 33.20 ×3).
                // When a boundary IS set, a value can legitimately be
                // both an item price and the subtotal on a single-item
                // receipt (IMG_8040 Lytt: $84.95 item == subtotal ==
                // total, printed 3×); leave those to the exception.
                if repeatedValues.contains(p.value), totalsBlockY >= 0.999 {
                    dbg("SKIP totals-echo v=\(p.value)"); continue
                }
                let dupesAbove = inColumn.filter {
                    $0.value == p.value && $0.centerY < totalsBlockY - 0.002
                }.count
                // Require OTHER price values above too: a single-item
                // receipt echoes its lone price through Items Subtotal /
                // Subtotal rows (Hashi: 0.24 × 3) — those echoes are NOT
                // items. Two same-priced sandwiches on a multi-price
                // receipt (La Baguette) still qualify.
                let distinctAbove = Set(inColumn.filter {
                    $0.centerY < totalsBlockY - 0.002
                }.map(\.value)).count
                if dupesAbove < 2 || distinctAbove < 2 {
                    dbg("SKIP totals-value v=\(p.value)"); continue
                }
            }
            // Filter: adjacent to a metadata label. Savings-class labels
            // ("Regular Price", "Member Savings") poison both signs —
            // the paid amount lives on the item row. Discount-class
            // labels ("Item Discount") keep their NEGATIVE amount: on
            // Old Navy-style receipts that IS the line item; only a
            // positive value on such a row is junk.
            let labelPoisoned = metaLabels.contains { m in
                abs(m.y - p.centerY) <= 0.006 && (p.value > 0 || m.isSavingsClass)
            }
            if labelPoisoned { dbg("SKIP label-adj v=\(p.value)"); reject(p); continue }
            // Filter: implausibly large value. Vision sometimes mangles a
            // price like "$4.99" into just "99" (decimal point lost). The
            // resulting "$99" exceeds the receipt total and is obviously
            // wrong. With credits present, big positive items (offset by
            // negative ones) are legit (IMG_4161 Ace: CARBONATOR $34.99
            // + -$15.50 credit = $19.49 net). `priceCeiling` is nil when
            // the FM total itself is implausible — see setup above.
            if let c = priceCeiling, p.value > c { dbg("SKIP over-ceiling v=\(p.value)"); continue }

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
            // 2.5 line-heights: restaurant printers (La Baguette, Philz)
            // stagger the value column up to half a row from the
            // description column, which put the real description just
            // past the old 2.0× band. Dense grocery receipts are
            // protected by the dy-bucketing below — same-row candidates
            // always outrank a farther row.
            let rowTol = max(0.014, p.box.height * 2.5)
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
            guard !candidates.isEmpty else { dbg("SKIP no-candidates v=\(p.value)"); reject(p); continue }

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
            // If the closest-row candidate is a money-summary label
            // ("Total Bottle Deposit", "TAX", "Balance …"), the price
            // BELONGS to that label — skip the price point entirely
            // rather than letting the nearest item description above
            // claim it. TJ IMG_7496 extracted a third "BEANS GREEN
            // $0.10" because the deposit-summary's $0.10 fell through
            // to the beans row when the label was filtered out.
            if let closest = candidates.first {
                let closestBucket = (closest.dy / 0.005).rounded(.down)
                let owned = candidates.contains { c in
                    (c.dy / 0.005).rounded(.down) == closestBucket
                        && Self.labelOwnsPrice(c.text)
                }
                if owned { dbg("SKIP label-owned v=\(p.value)"); reject(p); continue }
            }

            // Walk sorted candidates and take the FIRST one that
            // survives the non-item filters. If the closest-Y candidate
            // is a SKU / weight-marker / metadata fragment (which the
            // sort already deprioritizes but doesn't necessarily push
            // to the bottom), don't drop the line item — fall back to
            // the next candidate. That's what saves Ulta receipts
            // where the SKU sits closer to the price than the actual
            // product name.
            // For NEGATIVE-value prices, allow "Member Savings" /
            // "Regular Price"-style descriptions to pass the non-item
            // filter: those are legitimate labels for discount and
            // credit line items.
            let allowMetadataDesc = p.value < 0
            var cleaned: String? = nil
            for c in candidates {
                let candidate = Self.cleanLineItemDescription(c.text)
                if candidate.count < 3 { continue }
                if Self.isFooterRow(candidate) { continue }
                if !allowMetadataDesc, Self.looksLikeNonItemRecoveryCandidate(candidate) { continue }
                cleaned = candidate
                break
            }
            // Department-keyed ring-ups: cashiers ring generic items
            // under the department key, printing "DAIRY  8.99 T F" as
            // the actual purchase line (IMG_3524 Sprouts, total $8.99,
            // one dairy item). The department name is a legitimate
            // description there — but ONLY when it shares the price's
            // row; a department HEADER on its own line above the items
            // never does.
            if cleaned == nil {
                let sameRowDept = candidates.first { c in
                    c.dy <= 0.006 && Self.isDepartmentName(c.text)
                }
                if let dept = sameRowDept {
                    cleaned = Self.cleanLineItemDescription(dept.text)
                }
            }
            guard let cleaned, cleaned.count >= 3 else { dbg("SKIP no-desc v=\(p.value)"); reject(p); continue }

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

        return (items, rejects)
    }

    /// True when a text observation is a money-summary label that owns
    /// the price printed next to it ("Total Bottle Deposit", "TAX",
    /// "Balance to pay"). Used by column-anchored extraction to skip
    /// prices whose closest-row text is such a label — attributing them
    /// to the nearest ITEM description above would invent a line item.
    private static func labelOwnsPrice(_ text: String) -> Bool {
        let n = text.lowercased()
            .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        let prefixes = [
            "total", "subtotal", "sub total", "balance",
            "tax", "change", "payment", "amount due", "amount paid",
        ]
        return prefixes.contains { n == $0 || n.hasPrefix($0 + " ") }
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
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?(?:\s+[A-Z]{1,2}){0,2}$"#

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
                    .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                    .replacingOccurrences(
                        of: #"(\d)-(\d)"#,
                        with: "$1.$2",
                        options: .regularExpression
                    )
                    .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
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
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
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
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?(?:\s+[A-Z]{1,2}){0,2}$"#

        func priceValue(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }

        /// Extract the first price-shaped token from any whitespace-
        /// separated position in the text. Handles Vision merges like
        /// "19.49 TAX: $" where the SUB-TOTAL's value ($19.49) is
        /// embedded before the next label word.
        func firstPriceToken(_ text: String) -> Decimal? {
            for tok in text.split(whereSeparator: { $0.isWhitespace }) {
                if let v = priceValue(String(tok)), v > 0 { return v }
            }
            return nil
        }

        /// Find the price value paired with a label observation. Looks
        /// same-row first (label + value in same obs, or same-Y sibling
        /// preferring the closest sibling to the LEFT/RIGHT of the
        /// label in reading order), then row-above / row-below for
        /// receipts that print the value on a separate line from the
        /// label.
        func findPairedValue(labelObs: (text: String, box: Receipt.BBox)) -> Decimal? {
            // Try tokens in the label observation itself.
            for tok in labelObs.text.split(whereSeparator: { $0.isWhitespace }) {
                if let v = priceValue(String(tok)), v > 0 { return v }
            }
            let labelY = labelObs.box.y + labelObs.box.height / 2
            let labelXEnd = labelObs.box.x + labelObs.box.width
            // Tight same-Y tolerance: 0.006 or 0.5x label height,
            // whichever is smaller. On stacked totals (Costco: values
            // in a column with labels alongside, each label/value pair
            // only ~0.010 apart), a looser tolerance would pull in the
            // NEIGHBORING row's value.
            let sameRowTol = min(0.006, max(labelObs.box.height * 0.5, 0.004))
            // Same-Y siblings — sort by Y-distance first (closest same-
            // row wins), then X distance (Vision often merges "value
            // nextLabel" so the closer-X sibling in tie has the value
            // for THIS label). IMG_4161's "SUB-TOTAL:$" + "19.49 TAX:
            // $" + "1.92" is the classic case: the middle observation
            // has the subtotal's value ($19.49) embedded.
            var sameRow: [(text: String, x: Double, dy: Double)] = []
            for other in lines {
                let cy = other.box.y + other.box.height / 2
                let dy = abs(cy - labelY)
                guard dy <= sameRowTol else { continue }
                if other.box.x == labelObs.box.x, other.box.y == labelObs.box.y { continue }
                sameRow.append((other.text, other.box.x, dy))
            }
            sameRow.sort { a, b in
                // Bucket dy so tiny baseline jitter doesn't dominate.
                let aB = (a.dy / 0.003).rounded(.down)
                let bB = (b.dy / 0.003).rounded(.down)
                if aB != bB { return aB < bB }
                let ad = a.x >= labelXEnd ? a.x - labelXEnd : labelXEnd - a.x + 10
                let bd = b.x >= labelXEnd ? b.x - labelXEnd : labelXEnd - b.x + 10
                return ad < bd
            }
            for sib in sameRow {
                if let v = firstPriceToken(sib.text) { return v }
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
                guard let v = firstPriceToken(other.text) else { continue }
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
            switch (bestAbove, bestBelow) {
            case (nil, nil): return nil
            case (let a?, nil): return a.v
            case (nil, let b?): return b.v
            case (let a?, let b?):
                // Slight preference for ABOVE — Philz Coffee prints
                // values ABOVE their labels (Subtotal → $10.70 above,
                // Tip → $2.14 above), and picking the "just-closer"
                // BELOW value would grab the NEXT label's value. Only
                // prefer BELOW when it's meaningfully closer (>5pt
                // tighter than above).
                return a.dy <= b.dy + 0.005 ? a.v : b.v
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
            // Allow prefix and (limited) suffix matching:
            //   * "Subtotal $19.49" — prefix match ✓
            //   * "Items Subtotal" — suffix match, but ONLY when the
            //     first word is a WORD (not a number). "Items" is a
            //     word so this matches; "19.49 TAX: $" has first
            //     token "19.49" (a number) which would misattribute
            //     the subtotal value to the TAX label.
            func firstTokenIsWord(_ n: String) -> Bool {
                let tokens = n.split(whereSeparator: { $0 == " " })
                guard let first = tokens.first else { return false }
                return Double(first) == nil
            }
            for line in lines {
                let n = normalizedLabel(line.text)
                for kw in keywords {
                    if n == kw || n.hasPrefix(kw + " ") {
                        return (line.text, line.box)
                    }
                    if n.hasSuffix(" " + kw) && firstTokenIsWord(n) {
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

        // --- Tax --- Two paths:
        //   1. If we have an OCR tax label, use its adjacent value —
        //      strongest signal, wins over anything buildReceipt
        //      computed arithmetically.
        //   2. If no OCR tax label AND the current tax equals the tip
        //      we just extracted, the buildReceipt tax was a false
        //      positive from arithmetic (real total was sub + tip,
        //      but buildReceipt saw the tip amount and attributed it
        //      to tax). Clear it.
        let extractedTax = receipt.totals.tax.first?.amount ?? 0
        let taxLabel = findLabel(matching: ["tax", "sales tax", "vat", "gst", "hst"])
        let itemsSumForTax: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        if let label = taxLabel, let v = findPairedValue(labelObs: label),
           v > 0, v != extractedTax,
           // Tax plausibility. Target prints "T = CA TAX 9.375% on 28.79"
           // — the 28.79 is the TAXABLE BASE, not the tax, and it sits
           // right on the label row. Reject any candidate that (a) echoes
           // the subtotal / items sum or (b) exceeds 30% of the total —
           // no US sales tax comes anywhere near that; a value that big
           // is always a base amount or a misread.
           v != (receipt.totals.subtotal ?? -1),
           v != itemsSumForTax,
           v <= receipt.totals.total * Decimal(0.3) {
            receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: nil, amount: v)]
            receipt.provenance.fieldConfidence["totals.tax"] = 0.75
        } else if taxLabel == nil,
                  extractedTax > 0,
                  let extractedTip = receipt.totals.tip,
                  extractedTax == extractedTip {
            // No tax label exists, but buildReceipt computed a tax
            // that happens to equal the tip we found. This is the
            // "sub + tip = total on a no-tax receipt" case (Philz
            // Coffee IMG_7869); clear the ghost tax.
            receipt.totals.tax = []
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

    /// Enforce arithmetic invariants on subtotal / tax that no real
    /// receipt can violate. Runs AFTER label-based sanity checks — this
    /// pass is for the cases where the labels were missing or pointed at
    /// the same cross-wired junk FM emitted. The total is trusted here:
    /// `sanityCheckTotal` has already cross-checked it against OCR.
    ///
    /// Observed corruption classes from the full-dataset audit:
    ///   * tax == total  (Safeway tax-exempt groceries; FM copied the
    ///     total into the tax slot — structurally impossible whenever
    ///     any line item exists, since total = base + tax)
    ///   * subtotal == tax  (ULTA: sub=1.47 tax=1.47 on a $25.97
    ///     receipt; the real subtotal is total − tax = $24.50)
    ///   * tax RATE parked in a money field (Burger King prints
    ///     "9.375 TAX" — the % sign OCR'd away — and FM used 9.375 as
    ///     the SUBTOTAL, then invented tax = total − 9.375 = 6.335, a
    ///     fractional-cent amount no register ever prints)
    ///   * tax > 30% of total (base-echo or misread survivors)
    private static func reconcileTotalsArithmetic(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let total = receipt.totals.total
        guard total > 0 else { return receipt }

        let itemsSum: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        let taxCap: Decimal = total * Decimal(0.3)

        func fractionalCents(_ v: Decimal) -> Bool {
            var cents = v * 100
            var rounded = Decimal()
            NSDecimalRound(&rounded, &cents, 0, .plain)
            return rounded != cents
        }
        func approxEqual(_ a: Decimal, _ b: Decimal) -> Bool {
            let d = a - b
            return d < Decimal(0.005) && d > Decimal(-0.005)
        }

        // Zero-valued subtotal is "not printed", not "$0.00".
        if let s = receipt.totals.subtotal, s == 0 {
            receipt.totals.subtotal = nil
        }

        // --- Rate parked in the subtotal slot (fractional cents in a
        // plausible percent range). Recover both fields from the rate:
        // subtotal = total / (1 + r/100), tax = the remainder.
        if let s = receipt.totals.subtotal, fractionalCents(s),
           s >= Decimal(2), s <= Decimal(12) {
            let rate = NSDecimalNumber(decimal: s).doubleValue
            let totalD = NSDecimalNumber(decimal: total).doubleValue
            let subD = (totalD / (1.0 + rate / 100.0) * 100).rounded() / 100
            let newSub = Decimal(subD)
            let newTax = total - newSub
            if newTax > 0, newTax <= taxCap {
                receipt.totals.subtotal = newSub
                receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: s, amount: newTax)]
                receipt.provenance.fieldConfidence["totals.subtotal"] = 0.6
                receipt.provenance.fieldConfidence["totals.tax"] = 0.6
            }
        }

        let tax = receipt.totals.tax.first?.amount ?? 0

        // --- tax == total with items present: impossible; the receipt
        // is tax-exempt (or FM copied the wrong cell). Clear it.
        if tax > 0, approxEqual(tax, total), !receipt.lineItems.isEmpty {
            receipt.totals.tax = []
        }

        // --- subtotal == tax: one of them is a copy of the other. The
        // real subtotal is total − tax when that remainder dominates
        // the tax (subtotal ≥ tax on any real receipt — tax rates are
        // nowhere near 50%).
        if let s = receipt.totals.subtotal,
           let x = receipt.totals.tax.first?.amount,
           approxEqual(s, x) {
            let candidate = total - x
            if candidate >= x, x <= taxCap {
                receipt.totals.subtotal = candidate
                receipt.provenance.fieldConfidence["totals.subtotal"] = 0.6
            } else {
                // Tax itself is implausible too — drop both; arithmetic
                // below may rebuild the tax from items.
                receipt.totals.subtotal = nil
                receipt.totals.tax = []
            }
        }

        // --- Fractional-cent tax that survived: registers never print
        // one. Rebuild from subtotal if possible, else drop.
        if let x = receipt.totals.tax.first?.amount, fractionalCents(x) {
            if let s = receipt.totals.subtotal, s < total, total - s <= taxCap {
                receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: nil, amount: total - s)]
                receipt.provenance.fieldConfidence["totals.tax"] = 0.6
            } else {
                receipt.totals.tax = []
            }
        }

        // --- Tax still implausibly large: reassign the arithmetic
        // remainder, choosing tax vs tip by which label the OCR
        // actually prints (Philz: sub 10.75 + 2.15 on a "Tip" receipt
        // is a tip, not tax).
        if let x = receipt.totals.tax.first?.amount, x > taxCap {
            let joined = lines.map { $0.text.lowercased() }.joined(separator: "\n")
            let hasTipLabel = joined.range(of: #"\b(tip|gratuity)\b"#, options: .regularExpression) != nil
            let hasTaxLabel = joined.range(of: #"\b(tax|vat|gst|hst)\b"#, options: .regularExpression) != nil
            var remainder: Decimal? = nil
            if let s = receipt.totals.subtotal, s < total, s >= total / 2 {
                remainder = total - s
            } else if itemsSum > 0, itemsSum < total, total - itemsSum <= taxCap {
                remainder = total - itemsSum
            }
            if let r = remainder, hasTipLabel, !hasTaxLabel {
                receipt.totals.tax = []
                receipt.totals.tip = r
                receipt.provenance.fieldConfidence["totals.tip"] = 0.6
            } else if let r = remainder, r <= taxCap {
                receipt.totals.tax = [Receipt.TaxLine(label: "Tax", rate: nil, amount: r)]
                receipt.provenance.fieldConfidence["totals.tax"] = 0.6
            } else {
                receipt.totals.tax = []
            }
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
        // there are enough of them. Track positives separately for the
        // savings-block trigger: receipts with credit lines (negative
        // items) push items_sum lower which would hide FM totals that
        // came from a savings-block value.
        let itemsSum: Decimal = receipt.lineItems.reduce(0) { $0 + $1.totalPrice }
        let positivesSum: Decimal = receipt.lineItems.reduce(0) {
            $1.totalPrice > 0 ? $0 + $1.totalPrice : $0
        }
        let taxSum: Decimal = receipt.totals.tax.reduce(0) { $0 + $1.amount }
        let tip: Decimal = receipt.totals.tip ?? 0
        let discount: Decimal = receipt.totals.discount ?? 0
        let expected: Decimal = itemsSum + taxSum + tip - discount

        // Does FM's total actually appear anywhere in the OCR?
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?(?:\s+[A-Z]{1,2}){0,2}$"#
        func priceValue(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }
        let fmTotalPresent = lines.contains { line in
            priceValue(line.text) == fmTotal
        }

        // Flag 1: FM's total is hallucinated (not in OCR at all).
        let hallucinated = !fmTotalPresent

        // Flag 2: items_sum >> total × 2 suggests FM picked a tiny
        // value (item count "12", small savings amount) instead of
        // the real total. Post-dedup, items_sum is reliable enough
        // that a 2× ratio is meaningful evidence FM is wrong.
        // (Previously we tried × 3 to avoid firing on receipts
        // where items were legit 2× the total due to over-
        // extraction — but that class of false positive is now
        // gone since column-anchored extraction dedupes
        // actual/regular price pairs before items_sum is computed.)
        // Count gate relaxed for receipts WITHOUT negative items: a
        // 1-item Safeway run whose FM total is the "$1.00" savings
        // total (IMG_8046: ROTINI $4.58 against total $1) can never
        // reach 3 items, but positives at 2× the total is just as
        // conclusive there. Receipts WITH credits keep the 3-item gate
        // — a big item offset by a refund legitimately exceeds the
        // total (IMG_4161 Ace).
        let hasNegativeItems = receipt.lineItems.contains { $0.totalPrice < 0 }
        let savingsBlockConfusion =
            positivesSum > fmTotal * Decimal(2)
            && (receipt.lineItems.count >= 3 || (!hasNegativeItems && !receipt.lineItems.isEmpty))

        // Flag 3: fmTotal implausibly large vs items — 5× or more.
        // IMG_5353 landed on $8851 for an 11-item Safeway; that's a
        // digit-run misread (probably a card number or transaction ID
        // Vision concatenated). Real totals for small item counts
        // don't blow past items_sum × 5.
        let implausiblyLarge =
            receipt.lineItems.count >= 3
            && fmTotal > itemsSum * Decimal(5)
            && itemsSum > 0

        // Flag 4: fmTotal is a small whole-number and less than
        // items_sum — highly suspicious that FM picked the item
        // COUNT ("TOTAL NUMBER OF ITEMS SOLD = 15") instead of the
        // real dollar total. Real totals almost always have cents
        // (dollars.cents format); a small whole-number total is
        // suspect. Threshold at $50 so we don't misfire on the
        // occasional legit round-dollar amount, and require ≥ 3
        // items so we're confident there's a real receipt to
        // compare against.
        var fmTotalRounded: Decimal = 0
        var fmTotalMut = fmTotal
        NSDecimalRound(&fmTotalRounded, &fmTotalMut, 0, .down)
        let isWholeInteger = fmTotalRounded == fmTotal
        let looksLikeItemCount =
            receipt.lineItems.count >= 3
            && isWholeInteger
            && fmTotal < Decimal(50)
            && fmTotal < itemsSum

        // Flag 5: a large whole-dollar total with no cents. US receipts
        // essentially always print cents on the grand total; a candidate
        // like "8854" is a transaction-ID / store-number digit run that
        // happened to sit near a "TOTAL ..." row (IMG_3389: real total
        // $5.26, FM picked the "8854" footer artifact). Yen receipts DO
        // print integer totals, so skip when the currency isn't dollar-
        // denominated.
        let wholeDollarLarge =
            isWholeInteger
            && fmTotal >= Decimal(200)
            && receipt.header.currency == "USD"

        // Flag 6: FM's total equals one of the ITEM prices on a multi-
        // item receipt. A grand total never coincides with a single
        // item's price when other items exist — FM grabbed an item row
        // (La Baguette IMG_2169: total "13.00" = the salami sandwich,
        // real Total $32.02 printed once). Only meaningful when there's
        // more than the total's worth of items extracted.
        let totalEqualsItemPrice =
            receipt.lineItems.count >= 2
            && receipt.lineItems.contains { $0.totalPrice == fmTotal }
            && itemsSum > fmTotal

        // NOTE: candidates are gathered BEFORE the flag guard because
        // Flag 7 needs them: a label-anchored value matching the items
        // checksum exactly, while FM's total doesn't, is conclusive on
        // its own (IMG_8572: BALANCE 14.97 == items sum, FM said 7.98 —
        // the savings-block total at 1.88×, under the 2× confusion
        // gate; IMG_6301: TOTAL 0.24 == items, FM took Cash $20.00).

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
            // Item-COUNT rows start with "total" but their number is a
            // count, not money ("TOTAL NUMBER OF ITEMS 5 6", "Items in
            // Transaction: 13"). Treating them as totals labels let the
            // adjacent digit-run win candidate selection on IMG_3389.
            let countMarkers = [
                "number of items", "items in transaction",
                "item count", "items sold", "total items",
            ]
            if countMarkers.contains(where: { stripped.contains($0) }) { return false }
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
            var foundSameRow = false
            for (jdx, other) in lines.enumerated() where jdx != idx {
                let oy = other.box.y + other.box.height / 2
                guard abs(oy - cy) <= tol else { continue }
                if let v = priceValue(other.text), v > 0 {
                    candidates.append(Candidate(value: v, y: oy))
                    foundSameRow = true
                }
            }
            // Staggered-column printers (La Baguette, Philz) offset the
            // value up to a full row from its label. Only when the
            // same-row scan found NOTHING for this label, take the
            // nearest price-shaped neighbor above and below within a
            // wider band — plausibility + ranking below filter junk.
            if !foundSameRow {
                let staggerTol = max(0.022, line.box.height * 2.0)
                var bestAbove: (dy: Double, v: Decimal, y: Double)? = nil
                var bestBelow: (dy: Double, v: Decimal, y: Double)? = nil
                for (jdx, other) in lines.enumerated() where jdx != idx {
                    let oy = other.box.y + other.box.height / 2
                    let dy = abs(oy - cy)
                    guard dy > tol, dy <= staggerTol else { continue }
                    guard let v = priceValue(other.text), v > 0 else { continue }
                    if oy < cy {
                        if bestAbove == nil || dy < bestAbove!.dy { bestAbove = (dy, v, oy) }
                    } else {
                        if bestBelow == nil || dy < bestBelow!.dy { bestBelow = (dy, v, oy) }
                    }
                }
                if let a = bestAbove { candidates.append(Candidate(value: a.v, y: a.y)) }
                if let b = bestBelow { candidates.append(Candidate(value: b.v, y: b.y)) }
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
            // A value repeating because the customer bought TWO of the
            // same item is not a total — exclude values that match a
            // current line-item price when we have a multi-item list
            // (La Baguette: two $13 sandwiches must not elect $13).
            let itemPrices = Set(receipt.lineItems.map(\.totalPrice))
            for (v, ys) in counts where ys.count >= 2 {
                if receipt.lineItems.count >= 2, itemPrices.contains(v) { continue }
                for y in ys {
                    candidates.append(Candidate(value: v, y: y))
                }
            }
        }
        guard !candidates.isEmpty else { return receipt }

        // Flag 7: a label-anchored candidate agrees with the items
        // checksum to the cent while FM's total doesn't. Arithmetic
        // corroboration from two independent sources (price column +
        // totals label) beats FM's field choice. Requires ≥ 2 items so
        // a single mis-extracted item can't force a re-anchor.
        let expectedFromItems = itemsSum + taxSum + tip - discount
        func candidateMatches(_ target: Decimal) -> Bool {
            candidates.contains { c in
                let d = c.value - target
                return d < Decimal(0.02) && d > Decimal(-0.02)
            }
        }
        func fmDiffers(from target: Decimal) -> Bool {
            let d = fmTotal - target
            return d > Decimal(0.02) || d < Decimal(-0.02)
        }
        let betterArithmeticCandidate =
            !receipt.lineItems.isEmpty
            && expectedFromItems > 0
            && candidateMatches(expectedFromItems)
            && fmDiffers(from: expectedFromItems)

        // Flag 7b: net-of-savings items. Whole Foods (and other loyalty
        // programs) print each line item at its ALREADY-DISCOUNTED price
        // and show "Total Savings -$2.30" purely for information. FM
        // stuffs that savings figure into `discount`, which then double-
        // counts: items already net → itemsSum + tax IS the total, but
        // expectedFromItems subtracts the discount again and matches
        // nothing. When itemsSum + tax (ignoring discount) hits a
        // printed total candidate that FM missed, the items are net and
        // the discount field is spurious. (IMG_8129: items 14.47 + tax
        // 0.72 = 15.19 == printed Total, FM picked the 16.17 subtotal.)
        let expectedNet = itemsSum + taxSum + tip
        let itemsAreNet =
            !receipt.lineItems.isEmpty
            && discount != 0
            && expectedNet > 0
            && !candidateMatches(expectedFromItems)
            && candidateMatches(expectedNet)
            && fmDiffers(from: expectedNet)

        guard hallucinated || savingsBlockConfusion || implausiblyLarge || looksLikeItemCount || wholeDollarLarge || totalEqualsItemPrice || betterArithmeticCandidate || itemsAreNet else { return receipt }

        // Filter to plausible values. Both a lower AND upper bound
        // relative to items_sum:
        //   * Lower bound of items_sum × 0.3 — the real total is
        //     often less than items_sum (over-extracted lines) but
        //     rarely more than 3× smaller. Below that is almost
        //     always a savings amount or per-unit price.
        //   * Upper bound of max(items_sum × 5, $200) — bounds the
        //     ceiling against digit-run misreads ("8893" adjacent to
        //     "TOTAL NUMBER OF ITEMS SOLD" got picked up as a
        //     candidate). $200 floor lets sparse extractions still
        //     find their real totals.
        let lowerBound: Decimal = itemsSum > 0 ? itemsSum * Decimal(0.3) : 0
        let upperBoundBase: Decimal = itemsSum > 0 ? itemsSum * Decimal(5) : Decimal(10_000)
        let upperBound: Decimal = max(upperBoundBase, Decimal(200))
        let plausible = candidates.filter { $0.value >= lowerBound && $0.value <= upperBound }
        guard !plausible.isEmpty else { return receipt }

        // Rank the plausible candidates by "closest to arithmetic
        // target" — either items_sum + tax (when items look reliable)
        // or just the max of items_sum and the median candidate value
        // (when items may be over/under-extracted). Ties break toward
        // the smaller value.
        //
        // We ALSO score against the median plausible candidate: real
        // totals are typically the value that appears multiple times
        // (BALANCE + PAYMENT AMOUNT + footer), so the mode/median of
        // the label-anchored set is a strong signal even when
        // arithmetic is off.
        let sortedVals = plausible.map(\.value).sorted()
        let candidateMedian = sortedVals[sortedVals.count / 2]
        let target: Decimal = {
            // Net-items receipts: the discount double-counts, so aim at
            // itemsSum + tax (what the items actually add up to) rather
            // than `expected`, which re-subtracts the informational
            // savings and would steer toward the wrong candidate.
            if itemsAreNet { return expectedNet }
            if expected > 0 { return expected }
            if itemsSum > 0 { return max(itemsSum + taxSum, candidateMedian) }
            return candidateMedian
        }()

        // A real grand total prints 2-3 times (BALANCE, PAYMENT AMOUNT,
        // footer copy). When any candidate value appears at ≥ 2 distinct
        // Y positions, restrict the pool to the most-repeated values —
        // repetition is stronger evidence than distance-to-target,
        // because the arithmetic target itself is unreliable exactly
        // when items were badly under-extracted (TJ IMG_4623: 4 of 18
        // items extracted, target $10.81, real total $63.37 printed
        // twice).
        var freqYs: [Decimal: Set<Int>] = [:]
        for c in plausible {
            freqYs[c.value, default: []].insert(Int((c.y * 1000).rounded()))
        }
        let maxFreq = plausible.map { freqYs[$0.value]?.count ?? 1 }.max() ?? 1
        let pool = maxFreq >= 2
            ? plausible.filter { (freqYs[$0.value]?.count ?? 1) == maxFreq }
            : plausible

        let best: Candidate = pool.min { a, b in
            let ad = a.value > target ? a.value - target : target - a.value
            let bd = b.value > target ? b.value - target : target - b.value
            if ad != bd { return ad < bd }
            return a.value < b.value
        }!

        // Override policy:
        //   * Hallucinated — FM's value doesn't exist in OCR, so it
        //     can't be right. Always replace.
        //   * savingsBlockConfusion — items_sum is much larger than
        //     FM's total. FM is guaranteed wrong. Always replace.
        //   * implausiblyLarge — FM's value is huge vs items (digit-
        //     run misread). Always replace with something reasonable.
        //   * Otherwise — require the replacement to beat FM on
        //     arithmetic fit. Not currently reachable but future-proof.
        if !hallucinated && !savingsBlockConfusion && !implausiblyLarge && !looksLikeItemCount && !wholeDollarLarge && !totalEqualsItemPrice && !betterArithmeticCandidate {
            let fmDiff = fmTotal > target ? fmTotal - target : target - fmTotal
            let newDiff = best.value > target ? best.value - target : target - best.value
            guard newDiff < fmDiff else { return receipt }
        }

        receipt.totals.total = best.value
        receipt.provenance.fieldConfidence["totals.total"] = 0.7
        // When we re-anchored because the items are net-of-savings, the
        // discount field was the informational "Total Savings" figure
        // and double-counts against the now-correct total. Clear it so
        // downstream confidence sees itemsSum + tax == total.
        if itemsAreNet {
            receipt.totals.discount = nil
        }
        return receipt
    }

    /// Recompute totalPrice from weight × unit for Sprouts-style
    /// weight-priced items.
    ///
    /// Sprouts prints each weight item as three stacked observations:
    ///
    ///     BROCCOLI CROWNS      <- description
    ///     1.15 lb @            <- weight X.XX + " lb @"
    ///     $1.99 / lb           <- unit price " $Y.YY / lb"
    ///
    /// with the paid total (X.XX × Y.YY) shown in the price column
    /// at a nearby Y. FM sometimes grabs the weight (1.15) as the
    /// totalPrice instead of the computed total ($2.29). Detect and
    /// correct.
    ///
    /// For each extracted line item, find its description observation
    /// in the OCR. Look within ±0.03 Y for BOTH:
    ///   (a) a weight observation matching `\d+\.\d{1,2}\s*lb\s*@`
    ///   (b) a unit-price observation matching `\$?\d+\.\d{1,2}\s*/\s*(?:lb|1b)`
    /// If both exist, compute weight × unit and compare to the item's
    /// current totalPrice. When they differ by more than 1%, replace.
    private static func correctWeightPricedItems(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        guard !receipt.lineItems.isEmpty else { return receipt }

        // Regex captures: 1 = weight value.
        let weightRe = try? NSRegularExpression(
            pattern: #"(\d+(?:\.\d{1,2})?)\s*lb\s*@"#,
            options: [.caseInsensitive]
        )
        // Regex captures: 1 = unit price. Accepts "$1.99 / lb" as one
        // observation OR just "$1.99 /" when Vision splits the "lb"
        // suffix onto its own observation (Sprouts small-font).
        let unitRe = try? NSRegularExpression(
            pattern: #"\$?(\d+(?:[.,]\d{1,2})?)\s*/(?:\s*(?:lb|1b|ib))?\s*$"#,
            options: [.caseInsensitive]
        )
        guard let weightRe, let unitRe else { return receipt }

        func matchFirst(_ re: NSRegularExpression, in text: String) -> Decimal? {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard let m = re.firstMatch(in: text, options: [], range: range),
                  m.numberOfRanges >= 2,
                  let captureRange = Range(m.range(at: 1), in: text)
            else { return nil }
            let raw = String(text[captureRange]).replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
            return Decimal(string: raw, locale: Locale(identifier: "en_US_POSIX"))
        }

        for i in receipt.lineItems.indices {
            let item = receipt.lineItems[i]
            let desc = item.description.lowercased().trimmingCharacters(in: .whitespaces)
            guard !desc.isEmpty else { continue }
            let descWords = desc
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count >= 3 }

            // Find the description's OCR row(s).
            var descYs: [Double] = []
            for line in lines {
                let lc = line.text.lowercased()
                let strict = lc.contains(desc)
                let fuzzy = !descWords.isEmpty && descWords.allSatisfy { lc.contains($0) }
                if strict || fuzzy {
                    descYs.append(line.box.y + line.box.height / 2)
                }
            }
            guard let descY = descYs.min() else { continue }

            // Search ±0.03 Y around the description for weight AND unit.
            var weight: Decimal? = nil
            var unit: Decimal? = nil
            for line in lines {
                let cy = line.box.y + line.box.height / 2
                guard abs(cy - descY) <= 0.03 else { continue }
                if weight == nil, let w = matchFirst(weightRe, in: line.text) {
                    weight = w
                }
                if unit == nil, let u = matchFirst(unitRe, in: line.text) {
                    unit = u
                }
                if weight != nil, unit != nil { break }
            }
            guard let weight, let unit, weight > 0, unit > 0 else { continue }

            let computed = weight * unit
            // Round to cents.
            var rounded: Decimal = 0
            var mut = computed
            NSDecimalRound(&rounded, &mut, 2, .plain)
            // Only override when the current totalPrice EQUALS the
            // weight (within 5c) — that's the specific "FM picked the
            // weight as the price" signal we want to catch. On dense
            // Sprouts receipts where multiple items' weight/unit
            // observations overlap in the ±0.03 Y search window, this
            // conservative gate keeps us from clobbering already-
            // correct items (ORG CURLY PARSLEY was $5.97 = 3 × $1.99,
            // shouldn't be recomputed to 1.15 × 1.99 = $2.29 just
            // because BROCCOLI's weight/unit sit nearby).
            let weightDiff = item.totalPrice > weight
                ? item.totalPrice - weight
                : weight - item.totalPrice
            guard weightDiff < Decimal(0.05) else { continue }
            // Also require the computed result to differ from current —
            // no need to touch items where FM already emitted the same
            // value from weight coincidence.
            let diff = item.totalPrice > rounded ? item.totalPrice - rounded : rounded - item.totalPrice
            guard diff > Decimal(0.01) else { continue }

            receipt.lineItems[i] = Receipt.LineItem(
                description: item.description,
                quantity: weight,
                unitPrice: unit,
                totalPrice: rounded,
                category: item.category
            )
        }
        return receipt
    }

    private static func correctRegularPriceMistakes(
        receipt input: Receipt,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Receipt {
        var receipt = input
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?(?:\s+[A-Z]{1,2}){0,2}$"#

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
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
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
            if Double(first.replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")) != nil,
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
    /// Department names as printed by chain groceries. Normally these
    /// are section HEADERS (non-items), but a department-keyed ring-up
    /// prints one as the actual purchase line ("DAIRY  8.99 T F").
    fileprivate static let departmentNames: Set<String> = [
        "grocery", "produce", "dairy", "meat", "poultry", "seafood",
        "deli", "bakery", "frozen", "refrig", "refrig/frozen",
        "gen merchandise", "general merchandise", "merchandise",
        "liquor", "wine", "beer", "bulk", "household", "pharmacy",
        "health", "beauty", "snacks", "beverages",
    ]

    /// True when `description` is a department name that the receipt
    /// prints on the SAME row as `price` — i.e. a department-keyed
    /// ring-up ("DAIRY  8.99 T F"), not a section header. Checks both
    /// merged observations (name and price in one text) and split
    /// same-Y observations.
    fileprivate static func departmentRingUpConfirmed(
        description: String,
        price: Decimal,
        lines: [(text: String, box: Receipt.BBox)]
    ) -> Bool {
        guard isDepartmentName(description) else { return false }
        let needle = description.lowercased().trimmingCharacters(in: .whitespaces)
        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})(?:\s+[A-Z]{1,2}){0,2}$"#
        func value(_ text: String) -> Decimal? {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard trimmed.range(of: priceShape, options: .regularExpression) != nil else { return nil }
            let normalized = trimmed
                .replacingOccurrences(of: "$", with: "")
                .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
            return Decimal(string: normalized, locale: Locale(identifier: "en_US_POSIX"))
        }
        for line in lines {
            let lc = line.text.lowercased()
            guard lc.contains(needle) else { continue }
            // Merged form: tokens of this observation carry the price.
            for tok in line.text.split(whereSeparator: { $0.isWhitespace }) {
                if value(String(tok)) == price { return true }
            }
            // Split form: a separate same-Y price observation.
            let cy = line.box.y + line.box.height / 2
            for other in lines {
                let oy = other.box.y + other.box.height / 2
                guard abs(oy - cy) <= 0.006 else { continue }
                if value(other.text) == price { return true }
            }
        }
        return false
    }

    fileprivate static func isDepartmentName(_ text: String) -> Bool {
        let lc = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return departmentNames.contains(lc)
    }

    fileprivate static func looksLikeNonItemRecoveryCandidate(_ description: String) -> Bool {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        let lc = trimmed.lowercased()
        // Also strip leading/trailing punctuation and collapse whitespace
        // for the fragment check: "**** BALANCE" should match "balance",
        // "..TAX.." should match "tax", etc.
        let stripped = lc
            .replacingOccurrences(of: #"[^a-z0-9 ]"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)

        // Common department headers — these print on their own row and
        // get picked up if a price observation lands in the same band.
        if Self.departmentNames.contains(lc) { return true }

        // Single-word metadata fragments — leftovers after Vision split
        // "Member Savings" / "Regular Price" / "Additional Discounts"
        // into two observations and one half slipped through filtering.
        // None of these are plausible standalone product names.
        let metadataFragments: Set<String> = [
            "member", "regular", "savings", "discounts", "discount",
            "rewards", "coupon", "additional", "balance", "subtotal",
            "tax", "tip", "total", "change", "redeemed",
            // Column-header fragments — Safeway prints "You Pay" +
            // "Price" as a two-line header above the item block, and
            // the "Price" observation can get picked as a description
            // for the first item.
            "price", "amount", "qty", "item", "you", "pay",
            // Safeway-specific column markers — appear in the price
            // column area at random Ys and confuse recovery.
            "vol", "vol+", "vol-", "wt", "info",
            // Unit markers — Ace/Hassett print "$31.99 EA *" (unit
            // price row) above the extended total; the orphaned "EA"
            // observation claims the unit price as its own line item,
            // doubling the product (IMG_5255: 1 CO2 refill became 2).
            "ea", "each", "ea *",
        ]
        if metadataFragments.contains(lc) || metadataFragments.contains(stripped) { return true }

        // Multi-word footer phrases.
        let footerPhrases: [String] = [
            "additional discounts",
            "additional discount",
            "instant savings",
            "promo savings",
            // Restaurant order-type rows — In-N-Out prints
            // "DRIVE-Take Out $14.15" as the subtotal display; not
            // a purchased item. Same for other quick-serve receipts.
            "drive-take out", "drive thru", "drive-thru",
            "dine-in", "dine in", "for here", "to go",
            "take out", "takeout", "take-out",
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
            "item discount",
            "total discount",
        ]
        for sub in nonItemSubstrings {
            if lc.contains(sub) { return true }
        }

        // Typo'd savings labels, possibly with the amount glued on
        // ("Menber Savings -0.79", "Meaber Savings", "Reguler Price
        // 4.28") — the substring list above only catches exact
        // spellings. Fuzzy-match after stripping a trailing amount.
        let strippedAmt = Self.strippingTrailingAmount(lc)
        if strippedAmt.count <= 30,
           Self.fuzzyMatchesAny(strippedAmt, phrases: Self.savingsClassPhrases) {
            return true
        }

        // Pure percentage descriptions ("6.0000%", "9.375 %") — tax-rate
        // rows next to the totals block, never products. ULTA's
        // "6.0000%  24.50" row was extracting the SUBTOTAL as a $24.50
        // line item named after the tax rate.
        if trimmed.range(of: #"^\d+(?:[.,]\d+)?\s*%$"#, options: .regularExpression) != nil {
            return true
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

        let priceShape = #"^-?\$?-?\d{1,3}(?:,\d{3})*(?:[.,]\d{1,2})?(?:\s+[A-Z]{1,2}){0,2}$"#
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
                    .replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                    .replacingOccurrences(of: #"(?:\s+[A-Z]{1,2})+\s*$"#, with: "", options: .regularExpression)
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
    /// Merge two adjacent OCR observations that are actually one split
    /// price — Vision sometimes fractures "$2.79 S" into "2." at one X
    /// and "79 S" at the next (same Y, tiny X gap). Without this repair
    /// the "2." fragment is discarded (not price-shaped) and "79 S"
    /// gets treated as $79, blowing up items_sum on Safeway receipts.
    ///
    /// Merge conditions:
    ///   * LEFT observation is a bare dollars fragment: optional "$",
    ///     1-3 digits, OPTIONAL trailing "." or "," ("2.", "$2.", "$8").
    ///   * RIGHT observation is a bare cents fragment: OPTIONAL leading
    ///     "." or ",", exactly 2 digits, optional single-letter tax
    ///     marker (". 29", "49", "79 S").
    ///   * SEPARATOR EVIDENCE: at least one side carries the decimal
    ///     separator (left trailing dot OR right leading dot). Without
    ///     it, "2" + "49" is more likely a quantity next to an
    ///     unrelated number than a split $2.49 — don't guess.
    ///   * Same Y within 0.005.
    ///   * X gap between LEFT.rightEdge and RIGHT.leftEdge ≤ 0.02.
    /// When matched, the LEFT observation's text becomes the merged
    /// price (e.g. "$2.29", "2.79 S") and the RIGHT observation is
    /// dropped. LEFT's bounding box is extended to cover the RIGHT
    /// observation so bbox display and column-X detection still work.
    /// (Trader Joe's splits "$2.29" into "$2." and ". 29" routinely;
    /// Safeway splits "2.79 S" into "2." and "79 S".)
    fileprivate static func mergeSplitPriceFragments(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        guard let leftRe = try? NSRegularExpression(
            pattern: #"^\s*(\$?)(\d{1,3})([.,]?)\s*$"#, options: []
        ), let rightRe = try? NSRegularExpression(
            pattern: #"^\s*([.,]?)\s*(\d{2})(\s*[A-Za-z])?\s*$"#, options: []
        ) else { return lines }

        func group(_ m: NSTextCheckingResult, _ i: Int, in text: String) -> String {
            guard m.range(at: i).location != NSNotFound,
                  let r = Range(m.range(at: i), in: text) else { return "" }
            return String(text[r])
        }

        var out = lines
        var claimed: Set<Int> = []
        for i in 0..<out.count {
            if claimed.contains(i) { continue }
            let l = out[i]
            let lRange = NSRange(l.text.startIndex..<l.text.endIndex, in: l.text)
            guard let lm = leftRe.firstMatch(in: l.text, options: [], range: lRange) else { continue }
            let dollar = group(lm, 1, in: l.text)
            let leftValue = group(lm, 2, in: l.text)
            let leftSep = group(lm, 3, in: l.text)
            guard !leftValue.isEmpty else { continue }
            let leftCy = l.box.y + l.box.height / 2
            let leftXEnd = l.box.x + l.box.width

            for j in 0..<out.count where j != i && !claimed.contains(j) {
                let r = out[j]
                let rCy = r.box.y + r.box.height / 2
                guard abs(rCy - leftCy) <= 0.005 else { continue }
                guard r.box.x >= l.box.x else { continue }
                let gap = r.box.x - leftXEnd
                guard gap <= 0.02 else { continue }
                let rRange = NSRange(r.text.startIndex..<r.text.endIndex, in: r.text)
                guard let rm = rightRe.firstMatch(in: r.text, options: [], range: rRange) else { continue }
                let rightSep = group(rm, 1, in: r.text)
                let digits = group(rm, 2, in: r.text)
                guard digits.count == 2 else { continue }
                // Separator evidence on at least one side.
                guard !leftSep.isEmpty || !rightSep.isEmpty else { continue }
                let suffix = group(rm, 3, in: r.text)
                // Merge into LEFT.
                let mergedText = "\(dollar)\(leftValue).\(digits)\(suffix)"
                let mergedBox = Receipt.BBox(
                    x: l.box.x,
                    y: min(l.box.y, r.box.y),
                    width: (r.box.x + r.box.width) - l.box.x,
                    height: max(l.box.y + l.box.height, r.box.y + r.box.height) - min(l.box.y, r.box.y)
                )
                out[i] = (text: mergedText, box: mergedBox)
                claimed.insert(j)
                break
            }
        }
        // Drop the claimed RIGHT observations.
        var result: [(text: String, box: Receipt.BBox)] = []
        for (idx, l) in out.enumerated() where !claimed.contains(idx) {
            result.append(l)
        }
        return result
    }

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
        // (3) Trailing-minus notation ("1.00-" → "-1.00"). Safeway
        // prints Member Savings / discount amounts with the minus AFTER
        // the digits; without rewrite these fail priceShape and never
        // reach column extraction as valid negative-priced line items.
        // Only rewrite when the trailing-minus is on its own (nothing
        // after it) to avoid corrupting "1.00-BASED" or similar
        // product descriptions.
        let trailingMinusPattern = #"^\s*(\d+(?:[.,]\d{1,2})?)-\s*$"#
        // (3) "$" misread as "8" or "5" — the S-with-vertical-bar shape
        // of a dollar sign often OCRs as one of those digits.
        // Detected specifically as the paired SALE/REG pattern Ace
        // Hardware prints:  "$21.99 $21.99" comes through as
        // "821.99 521.99" (or "521.99 821.99"). Both tokens strip to
        // the same underlying value, which is a very strong signal
        // that we're looking at misread dollar signs. Rewriting the
        // observation to a valid single price lets column-anchored
        // extraction see the item's real price.
        let paired58 = try? NSRegularExpression(
            pattern: #"^\s*[58](\d{1,4}[.,]\d{2})\s+[58](\d{1,4}[.,]\d{2})\s*$"#,
            options: []
        )
        return lines.map { line in
            var text = line.text
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
                .replacingOccurrences(
                    of: trailingMinusPattern,
                    with: "-$1",
                    options: .regularExpression
                )
            if let paired58 {
                let range = NSRange(text.startIndex..<text.endIndex, in: text)
                if let m = paired58.firstMatch(in: text, options: [], range: range),
                   m.numberOfRanges >= 3,
                   let g1 = Range(m.range(at: 1), in: text),
                   let g2 = Range(m.range(at: 2), in: text)
                {
                    let v1 = String(text[g1]).replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                    let v2 = String(text[g2]).replacingOccurrences(of: #",(?=\d{3}(?:\D|$))"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: ",", with: ".")
                    // Only rewrite when the two stripped values match —
                    // that's the specific "same price twice" pattern
                    // (SALE == REG on a non-discounted item). Different
                    // values would mean a real discount and we can't
                    // safely pick one over the other automatically.
                    if v1 == v2 {
                        text = "$" + v1
                    }
                }
            }
            return (text: text, box: line.box)
        }
    }

    /// Rewrite individual OCR segments that look like `"8.49 S"`,
    /// `"3.50 T"`, or `"7.49 FT"` (price plus a trailing tax/SKU marker
    /// of one or two letters) to drop the marker. We only touch segments
    /// that match the strict `decimal-then-letters` shape so legitimate
    /// item names like `"COKE 12 OZ"` are untouched. Two-letter markers
    /// cover Whole Foods' combined categories ("FT" = Food+Taxable).
    fileprivate static func stripTrailingTaxMarkers(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        let pattern = #"^(-?\$?\d{1,5}(?:[.,]\d{1,2})?)(?:\s+([A-Z]{1,2}))+\s*$"#
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
        let debug = ProcessInfo.processInfo.environment["OCR_DEBUG_META"] != nil
        return lines.filter { line in
            let lc = line.text
                .lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !lc.isEmpty else { return false }
            for pat in metadataLinePatterns {
                if lc == pat {
                    if debug { FileHandle.standardError.write(Data("DROP eq: |\(line.text)|\n".utf8)) }
                    return false
                }
                if lc.hasPrefix(pat + " ") || lc.hasPrefix(pat + ":") || lc.hasPrefix(pat + "-") {
                    if debug { FileHandle.standardError.write(Data("DROP prefix: |\(line.text)|\n".utf8)) }
                    return false
                }
            }
            if dropIfContains.contains(where: { lc.contains($0) }) {
                if debug { FileHandle.standardError.write(Data("DROP contains: |\(line.text)|\n".utf8)) }
                return false
            }
            // Fuzzy fallback. The explicit typo list catches the specific
            // OCR misreads we've seen ("resular price", "member savincs")
            // but not every possible one — Safeway thermal-fade produces
            // "Member Sovings" ("a"→"o"), "Menber Savings" ("m"→"n"),
            // "Reguler Price" ("a"→"e"), and endless friends. Reject any
            // line within edit distance 2 of a canonical metadata phrase.
            // We only apply this when the line is short (≤ 30 chars) and
            // the length is close to the phrase (guardrail against
            // matching a long real product name).
            if lc.count <= 30, fuzzyMatchesMetadata(lc) {
                if debug { FileHandle.standardError.write(Data("DROP fuzzy: |\(line.text)|\n".utf8)) }
                return false
            }
            // Same fuzzy check with a trailing amount stripped — faded
            // Safeway prints merge the savings label AND its value into
            // one observation ("Menber Savings -0.79") whose length
            // blows the fuzzy gate. SAVINGS-class only: discount-class
            // rows ("Item Discount 20.0%") are real negative line items
            // on Old Navy-style receipts and must stay in the prompt.
            let strippedAmt = strippingTrailingAmount(lc)
            if strippedAmt != lc, strippedAmt.count <= 30,
               fuzzyMatchesAny(strippedAmt, phrases: savingsClassPhrases) {
                if debug { FileHandle.standardError.write(Data("DROP fuzzy-amt: |\(line.text)|\n".utf8)) }
                return false
            }
            if debug && (lc.contains("price") || lc.contains("sav")) {
                FileHandle.standardError.write(Data("KEEP: |\(line.text)| lc=|\(lc)| bytes=\(Array(lc.utf8))\n".utf8))
            }
            return true
        }
    }

    /// Canonical metadata phrases used by the fuzzy check. We accept ≤ 2
    /// character edits between the OCR text and any of these, provided
    /// the two lengths are within 2 of each other (otherwise a long real
    /// product name could match by prefix).
    ///
    /// Two classes with different item semantics:
    ///   * SAVINGS-class rows ANNOTATE a price the receipt already
    ///     charged elsewhere — Safeway's "Regular Price 4.99" /
    ///     "Member Savings -1.30" siblings of a paid "3.49 S". Their
    ///     amounts must never become line items.
    ///   * DISCOUNT-class rows ("Item Discount -$10.00" on Old Navy)
    ///     carry a real negative amount that DOES belong in the item
    ///     list — the printed subtotal only reconciles with them
    ///     included.
    private static let savingsClassPhrases: [String] = [
        "regular price",
        "member savings",
        "member saving",
        "membership savings",
        "manufacturer coupon",
        "store coupon",
        "you saved",
        "you save",
        "grocery rewards",
        "rewards earned",
        "loyalty discount",
        "sale price",
        "reg price",
    ]
    private static let discountClassPhrases: [String] = [
        "item discount",
        "total discount",
    ]
    private static let fuzzyMetadataPhrases: [String] =
        savingsClassPhrases + discountClassPhrases

    private static func fuzzyMatchesAny(_ lc: String, phrases: [String]) -> Bool {
        for phrase in phrases {
            if abs(lc.count - phrase.count) > 2 { continue }
            if levenshteinDistance(lc, phrase) <= 2 { return true }
        }
        return false
    }

    private static func fuzzyMatchesMetadata(_ lc: String) -> Bool {
        fuzzyMatchesAny(lc, phrases: fuzzyMetadataPhrases)
    }

    /// "menber savings -0.79" → "menber savings": strip ONE trailing
    /// amount-like token (optional sign, $, %, trailing minus) so a
    /// metadata label still matches when OCR merges the label and its
    /// value into a single observation. The merged form defeats both
    /// the prefix check (typo in the label) and the fuzzy check (the
    /// amount blows the length-within-2 gate). Returns the input
    /// unchanged when no amount is present or nothing meaningful
    /// would remain.
    fileprivate static func strippingTrailingAmount(_ lc: String) -> String {
        let pattern = #"[\s:]+[-+]?\$?\d{1,5}(?:[.,]\d{1,3})?%?-?\s*$"#
        guard let r = lc.range(of: pattern, options: .regularExpression) else { return lc }
        let head = String(lc[..<r.lowerBound]).trimmingCharacters(in: .whitespaces)
        return head.count >= 3 ? head : lc
    }

    /// Iterative Levenshtein — small strings (< 32 chars) so allocation
    /// cost is negligible compared to the Vision + FM latency.
    private static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        let s1 = Array(a)
        let s2 = Array(b)
        if s1.isEmpty { return s2.count }
        if s2.isEmpty { return s1.count }
        var prev = Array(0...s2.count)
        var curr = Array(repeating: 0, count: s2.count + 1)
        for i in 1...s1.count {
            curr[0] = i
            for j in 1...s2.count {
                let cost = s1[i - 1] == s2[j - 1] ? 0 : 1
                curr[j] = min(
                    curr[j - 1] + 1,        // insertion
                    prev[j] + 1,            // deletion
                    prev[j - 1] + cost      // substitution
                )
            }
            swap(&prev, &curr)
        }
        return prev[s2.count]
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

    /// Detect the "receipt captured 90° rotated" case and rotate the
    /// Vision observation coordinates back into portrait orientation.
    /// Every observation's text was recognized correctly by Vision
    /// (per-observation orientation is Vision's job), but the bboxes
    /// are in image coordinates and if the receipt was on its side
    /// then all "text lines" are tall/narrow observations that
    /// collapse into a single narrow Y band. Column-anchored
    /// extraction can't work without a vertical price column.
    ///
    /// Detection uses the aspect-ratio signature (width/height ratio
    /// of each observation): normal-orientation text lines are much
    /// wider than tall (typical 5–8×), but text on a 90°-rotated
    /// image lands in a bounding box that's much TALLER than wide
    /// (aspect < 0.5). If ≥ 70% of observations have h > w we're
    /// almost certainly looking at a rotated capture.
    ///
    /// Direction (CW vs CCW) is decided by trying both transforms
    /// and picking the one that produces a larger Y-span (portrait
    /// content spans the full Y axis).
    fileprivate static func correctRotatedObservations(
        _ lines: [(text: String, box: Receipt.BBox)]
    ) -> [(text: String, box: Receipt.BBox)] {
        guard lines.count >= 8 else { return lines }
        var tallCount = 0
        for l in lines {
            if l.box.height > l.box.width { tallCount += 1 }
        }
        let tallFraction = Double(tallCount) / Double(lines.count)
        guard tallFraction >= 0.70 else { return lines }

        // Two possible rotations to un-rotate the observations. Under a
        // 90° CW rotation of coordinates around image center:
        //     (x, y, w, h) → (1 - y - h, x, h, w)
        // Under a 90° CCW rotation:
        //     (x, y, w, h) → (y, 1 - x - w, h, w)
        func rotateCW(_ b: Receipt.BBox) -> Receipt.BBox {
            Receipt.BBox(
                x: max(0, 1 - b.y - b.height),
                y: b.x,
                width: b.height,
                height: b.width
            )
        }
        func rotateCCW(_ b: Receipt.BBox) -> Receipt.BBox {
            Receipt.BBox(
                x: b.y,
                y: max(0, 1 - b.x - b.width),
                width: b.height,
                height: b.width
            )
        }

        let cw = lines.map { (text: $0.text, box: rotateCW($0.box)) }
        let ccw = lines.map { (text: $0.text, box: rotateCCW($0.box)) }

        // Both rotations produce the same Y-span (rotating the whole
        // set doesn't change its Y-range), so we can't pick based on
        // range alone. But we CAN pick based on where content-header
        // signals end up. The store name / masthead is typically the
        // first meaningful text on a receipt and should land near the
        // TOP after correction. Vision returns observations sorted by
        // Y (already-sorted assumption for the pre-rotation set), so
        // the first few observations correspond to the top of the
        // physical receipt. Whichever rotation puts those at the top
        // of the new coordinate system wins.
        func earlyContentYMean(_ rotated: [(text: String, box: Receipt.BBox)]) -> Double {
            // The rotation transform doesn't change observation ORDER
            // in the input list; the input list order came from Vision
            // and is roughly top-to-bottom in ORIGINAL image coords.
            // For a receipt rotated 90° in the image, the "original
            // top-to-bottom" order is actually left-to-right of the
            // receipt content — so `rotated[0]` is likely a corner
            // observation. Instead, pick observations by content:
            // header/store text is usually longer than 4 chars and
            // uppercase-heavy. Fall back to the first 20% of items
            // sorted by post-rotation Y.
            let sortedByY = rotated.sorted { $0.box.y < $1.box.y }
            let head = sortedByY.prefix(max(3, rotated.count / 5))
            guard !head.isEmpty else { return 0.5 }
            let ys = head.map { $0.box.y + $0.box.height / 2 }
            return ys.reduce(0, +) / Double(ys.count)
        }

        // Prefer the rotation that puts the "top block" of content
        // closer to Y=0. Both have identical *distributions*, but
        // the actual observation-to-Y assignment differs.
        let cwHeaderY = earlyContentYMean(cw)
        let ccwHeaderY = earlyContentYMean(ccw)

        // The heuristic above is weak; strengthen with a merchant-
        // header check. Common brand or store words that appear at
        // the top of most receipts.
        let brandHints: Set<String> = [
            "safeway", "trader joe", "walmart", "target", "costco",
            "sprouts", "philz", "mobil", "ulta", "chipotle", "starbucks",
            "amazon", "receipt", "welcome", "thank you",
        ]
        func brandY(_ rotated: [(text: String, box: Receipt.BBox)]) -> Double? {
            var ys: [Double] = []
            for l in rotated {
                let lc = l.text.lowercased()
                if brandHints.contains(where: { lc.contains($0) }) {
                    ys.append(l.box.y + l.box.height / 2)
                }
            }
            guard !ys.isEmpty else { return nil }
            return ys.min()
        }
        let cwBrandY = brandY(cw)
        let ccwBrandY = brandY(ccw)

        // Pick: brand-anchored preference beats the header-mean
        // heuristic when available.
        let chosen: [(text: String, box: Receipt.BBox)]
        switch (cwBrandY, ccwBrandY) {
        case (let a?, let b?):
            chosen = a <= b ? cw : ccw
        case (_?, nil):
            chosen = cw
        case (nil, _?):
            chosen = ccw
        case (nil, nil):
            chosen = cwHeaderY <= ccwHeaderY ? cw : ccw
        }
        return chosen
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
