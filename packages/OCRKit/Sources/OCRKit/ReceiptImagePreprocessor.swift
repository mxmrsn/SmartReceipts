import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import Vision

/// Preprocess a receipt photo before feeding it to OCR.
///
/// Receipts are white(-ish) rectangles, but photos of them almost never
/// arrive that way: the receipt is one strip in a much larger frame, often
/// laid on a dark surface, sometimes rotated 90°, sometimes curled at the
/// ends, sometimes photographed from an angle so the far end is smaller.
/// When the receipt covers only, say, 20% of the frame, Vision's text
/// recognizer has less signal per printed glyph and starts silently
/// dropping small numeric observations (the price column is the first to
/// go). We've seen this on IMG_5785 — 7 grocery items have OCR'd
/// descriptions but no price observations at all, even though the printed
/// prices are clearly visible.
///
/// The fix is to isolate the receipt itself: detect its quadrilateral
/// with `VNDetectDocumentSegmentationRequest`, then perspective-correct
/// it into an upright rectangle that fills the output image. Text is now
/// bigger, straight, and printed on a uniform background — exactly the
/// input Vision does best on.
///
/// This runs BEFORE `VNRecognizeTextRequest`. Bounding boxes coming out
/// of Vision are in the corrected image's coordinate space, so we also
/// return a mapper that transforms them back into the original image's
/// coordinate space; the caller stitches that in so downstream consumers
/// (labeler overlays, iOS UI) don't have to know about preprocessing.
///
/// If no document is detected (or the request fails), the original image
/// and an identity mapper are returned. Preprocessing is a lossless
/// improvement — the pipeline works either way.
public enum ReceiptImagePreprocessor {

    /// Result of the preprocessing step.
    public struct Preprocessed {
        /// Image to feed into `VNRecognizeTextRequest`. Either the
        /// perspective-corrected receipt or the original input.
        public let image: CGImage
        /// Orientation to pass to Vision when using `image`. The corrected
        /// image is always upright, so this becomes `.up` on success.
        public let orientation: CGImagePropertyOrientation
        /// Maps a normalized bbox (`Receipt.BBox`, top-left origin) from
        /// the corrected image's coordinate space back into the ORIGINAL
        /// input image's coordinate space. Identity when no correction
        /// was applied.
        public let mapBBoxToOriginal: (Receipt.BBox) -> Receipt.BBox
        /// True if perspective correction was applied. Useful for logging
        /// / diagnostics — not required by the pipeline.
        public let didCorrect: Bool
    }

    /// Try to isolate the receipt in `image` via document segmentation +
    /// perspective correction. Falls back to the original image if
    /// segmentation returns no reasonable quad, or if the receipt is
    /// already well-framed (upright and filling most of the frame — in
    /// that case correction is near-identity and only introduces
    /// resampling artifacts).
    public static func preprocess(
        image: CGImage,
        orientation: CGImagePropertyOrientation
    ) -> Preprocessed {
        guard let quad = detectDocumentQuad(image: image, orientation: orientation) else {
            return Preprocessed(
                image: image,
                orientation: orientation,
                mapBBoxToOriginal: { $0 },
                didCorrect: false
            )
        }

        // Decide whether correction is actually worth doing.
        //
        // Empirically, perspective correction is a big win only when the
        // receipt is very small in the frame (< ~20% area) OR when it's
        // strongly rotated relative to the frame axes (> ~30°). Medium-
        // sized receipts (20-70% area, roughly axis-aligned) generally
        // suffer from correction: OCR was already reading the text fine,
        // and resampling + aspect-ratio change subtly moves the price
        // column X positions and confuses downstream column-anchored
        // extraction and FM total-picking. The old rotation-detection
        // heuristic (`correctRotatedObservations`) handles small tilts
        // adequately without a full image transform.
        if !shouldCorrect(quad: quad) {
            return Preprocessed(
                image: image,
                orientation: orientation,
                mapBBoxToOriginal: { $0 },
                didCorrect: false
            )
        }

        // The quad we get from Vision is in Vision's coord space: normalized,
        // origin bottom-left, and — critically — pre-orientation. Callers pass
        // `orientation` so Vision applies it internally, and the quad is
        // reported in the RE-ORIENTED image's coordinate frame. We build the
        // corrected image from a `CIImage` that we ALSO orient via
        // `.oriented(...)` so the quad and the CIImage share the same frame.
        let ciImage = CIImage(cgImage: image).oriented(forExifOrientation: Int32(orientation.rawValue))

        guard let corrected = applyPerspectiveCorrection(ci: ciImage, quad: quad) else {
            return Preprocessed(
                image: image,
                orientation: orientation,
                mapBBoxToOriginal: { $0 },
                didCorrect: false
            )
        }

        // Build the inverse mapper. `corrected.image` is a rectangle whose
        // corners correspond to `quad` in the ORIENTED input space. To map
        // a point back to the ORIGINAL input space we compose:
        //   (bbox in corrected) → (point in oriented) → (point in original)
        let orientedExtent = ciImage.extent
        let inverseOrientation = invertOrientation(orientation, extent: orientedExtent)
        let mapper = makeBBoxMapper(
            quad: quad,
            orientedExtent: orientedExtent,
            inverseOrientation: inverseOrientation
        )

        return Preprocessed(
            image: corrected.image,
            orientation: .up,
            mapBBoxToOriginal: mapper,
            didCorrect: true
        )
    }

    // MARK: - Document detection

    /// The 4 corners of the detected document. Points are in the ORIENTED
    /// image coord space: normalized [0,1], origin bottom-left (Vision
    /// convention). This is Vision's native output — we don't convert.
    private struct DocumentQuad {
        let topLeft: CGPoint
        let topRight: CGPoint
        let bottomLeft: CGPoint
        let bottomRight: CGPoint
    }

    private static func detectDocumentQuad(
        image: CGImage,
        orientation: CGImagePropertyOrientation
    ) -> DocumentQuad? {
        let request = VNDetectDocumentSegmentationRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let obs = (request.results as [VNRectangleObservation]?)?.first else { return nil }

        // Reject implausibly small detections. A receipt should occupy at
        // least a modest fraction of the frame; a 5%-area quad is more
        // likely a QR code or business card than the receipt itself.
        let area = quadArea(
            tl: obs.topLeft, tr: obs.topRight,
            bl: obs.bottomLeft, br: obs.bottomRight
        )
        guard area > 0.05 else { return nil }

        // Reject wildly non-quadrilateral detections. Adjacent sides
        // shouldn't differ by more than 10× — real receipts are long
        // strips (up to ~5:1) but not infinite ones.
        let widthTop = distance(obs.topLeft, obs.topRight)
        let widthBottom = distance(obs.bottomLeft, obs.bottomRight)
        let heightLeft = distance(obs.topLeft, obs.bottomLeft)
        let heightRight = distance(obs.topRight, obs.bottomRight)
        let avgWidth = (widthTop + widthBottom) * 0.5
        let avgHeight = (heightLeft + heightRight) * 0.5
        guard avgWidth > 0.01, avgHeight > 0.01 else { return nil }
        let sideRatio = max(avgWidth / avgHeight, avgHeight / avgWidth)
        guard sideRatio < 10 else { return nil }

        return DocumentQuad(
            topLeft: obs.topLeft,
            topRight: obs.topRight,
            bottomLeft: obs.bottomLeft,
            bottomRight: obs.bottomRight
        )
    }

    // MARK: - Perspective correction

    private struct CorrectedImage {
        let image: CGImage
        /// Size of the corrected output in pixels.
        let pixelSize: CGSize
    }

    private static func applyPerspectiveCorrection(
        ci: CIImage,
        quad: DocumentQuad
    ) -> CorrectedImage? {
        let extent = ci.extent

        // Convert the normalized quad into image-space points (pixels,
        // origin bottom-left — same as CIImage's coord system).
        let tl = CGPoint(x: quad.topLeft.x * extent.width,     y: quad.topLeft.y * extent.height)
        let tr = CGPoint(x: quad.topRight.x * extent.width,    y: quad.topRight.y * extent.height)
        let bl = CGPoint(x: quad.bottomLeft.x * extent.width,  y: quad.bottomLeft.y * extent.height)
        let br = CGPoint(x: quad.bottomRight.x * extent.width, y: quad.bottomRight.y * extent.height)

        let filter = CIFilter(name: "CIPerspectiveCorrection")
        filter?.setValue(ci, forKey: kCIInputImageKey)
        filter?.setValue(CIVector(cgPoint: tl), forKey: "inputTopLeft")
        filter?.setValue(CIVector(cgPoint: tr), forKey: "inputTopRight")
        filter?.setValue(CIVector(cgPoint: bl), forKey: "inputBottomLeft")
        filter?.setValue(CIVector(cgPoint: br), forKey: "inputBottomRight")
        guard let output = filter?.outputImage else { return nil }

        let context = CIContext(options: nil)
        guard let cg = context.createCGImage(output, from: output.extent) else { return nil }
        return CorrectedImage(image: cg, pixelSize: output.extent.size)
    }

    // MARK: - BBox mapper

    private static func makeBBoxMapper(
        quad: DocumentQuad,
        orientedExtent: CGRect,
        inverseOrientation: @escaping (CGPoint) -> CGPoint
    ) -> (Receipt.BBox) -> Receipt.BBox {
        // The corrected image is a rectangle whose four corners map to the
        // four quad corners in the ORIENTED input frame. To map a bbox
        // corner (u, v) from the corrected image's normalized coords back:
        //   1) bilinear-interpolate over the quad to get the point in the
        //      oriented frame
        //   2) apply the inverse orientation to get the point in the
        //      original frame
        //
        // The corrected image has origin top-left (Receipt.BBox convention);
        // Vision's quad has origin bottom-left. So (u_top, v_top) in the
        // corrected image maps to (u, 1 - v_top) in Vision's frame.
        //
        // Corner correspondence (Vision quad → corrected image):
        //   topLeft     → (0, 1) in Vision coords → (0, 0) top-left
        //   topRight    → (1, 1) in Vision coords → (1, 0) top-right
        //   bottomLeft  → (0, 0) in Vision coords → (0, 1) bottom-left
        //   bottomRight → (1, 0) in Vision coords → (1, 1) bottom-right
        let tl = quad.topLeft
        let tr = quad.topRight
        let bl = quad.bottomLeft
        let br = quad.bottomRight

        return { box in
            // Corrected image coords, top-left origin, normalized.
            let uL = box.x
            let uR = box.x + box.width
            let vT = box.y
            let vB = box.y + box.height

            // Map each corner through bilinear + inverse orientation.
            let p0 = mapCorrectedToOriginal(u: uL, v: vT, tl: tl, tr: tr, bl: bl, br: br,
                                            extent: orientedExtent, inverse: inverseOrientation)
            let p1 = mapCorrectedToOriginal(u: uR, v: vT, tl: tl, tr: tr, bl: bl, br: br,
                                            extent: orientedExtent, inverse: inverseOrientation)
            let p2 = mapCorrectedToOriginal(u: uR, v: vB, tl: tl, tr: tr, bl: bl, br: br,
                                            extent: orientedExtent, inverse: inverseOrientation)
            let p3 = mapCorrectedToOriginal(u: uL, v: vB, tl: tl, tr: tr, bl: bl, br: br,
                                            extent: orientedExtent, inverse: inverseOrientation)

            // Axis-aligned bounding box of the four mapped corners.
            let minX = min(min(p0.x, p1.x), min(p2.x, p3.x))
            let maxX = max(max(p0.x, p1.x), max(p2.x, p3.x))
            let minY = min(min(p0.y, p1.y), min(p2.y, p3.y))
            let maxY = max(max(p0.y, p1.y), max(p2.y, p3.y))
            return Receipt.BBox(
                x: minX, y: minY,
                width: max(0.0001, maxX - minX),
                height: max(0.0001, maxY - minY)
            )
        }
    }

    private static func mapCorrectedToOriginal(
        u: Double, v: Double,
        tl: CGPoint, tr: CGPoint, bl: CGPoint, br: CGPoint,
        extent: CGRect,
        inverse: (CGPoint) -> CGPoint
    ) -> CGPoint {
        // (u, v) is in the corrected image, top-left origin.
        // In Vision quad coords (bottom-left origin), v_vision = 1 - v.
        // Bilinear over the quad:
        //   P = (1-u)(1-v') * BL + u(1-v') * BR + (1-u)v' * TL + u*v' * TR
        // where v' = 1 - v.
        let vP = 1.0 - v
        let x =
            (1.0 - u) * (1.0 - vP) * Double(bl.x) +
            u         * (1.0 - vP) * Double(br.x) +
            (1.0 - u) * vP         * Double(tl.x) +
            u         * vP         * Double(tr.x)
        let y =
            (1.0 - u) * (1.0 - vP) * Double(bl.y) +
            u         * (1.0 - vP) * Double(br.y) +
            (1.0 - u) * vP         * Double(tl.y) +
            u         * vP         * Double(tr.y)
        // (x, y) is normalized [0,1], Vision's bottom-left origin, in the
        // ORIENTED input frame. Convert to a normalized top-left-origin
        // point and hand to the inverse-orientation mapper.
        let orientedTopLeft = CGPoint(x: x, y: 1.0 - y)
        return inverse(orientedTopLeft)
    }

    // MARK: - Orientation inverse

    /// Given the orientation that was passed INTO Vision, return a function
    /// that maps a normalized (top-left origin) point in the ORIENTED image
    /// back to normalized coords in the ORIGINAL image.
    ///
    /// EXIF orientation values:
    ///   .up (1)         : as-is
    ///   .upMirrored (2) : mirror horizontally
    ///   .down (3)       : rotate 180°
    ///   .downMirrored (4): mirror vertically
    ///   .leftMirrored (5): transpose (swap x/y)
    ///   .right (6)      : rotate 90° CW (image was on its side, right edge is top)
    ///   .rightMirrored (7): antitranspose
    ///   .left (8)       : rotate 90° CCW
    ///
    /// The oriented frame has swapped dimensions for .right / .left /
    /// .leftMirrored / .rightMirrored. We work in normalized coordinates,
    /// so the swap doesn't matter for the math — only the mapping does.
    private static func invertOrientation(
        _ orientation: CGImagePropertyOrientation,
        extent _: CGRect
    ) -> (CGPoint) -> CGPoint {
        switch orientation {
        case .up:
            return { $0 }
        case .upMirrored:
            return { CGPoint(x: 1.0 - $0.x, y: $0.y) }
        case .down:
            return { CGPoint(x: 1.0 - $0.x, y: 1.0 - $0.y) }
        case .downMirrored:
            return { CGPoint(x: $0.x, y: 1.0 - $0.y) }
        case .left:
            // Oriented image was rotated 90° CCW to become upright.
            // Inverse: rotate 90° CW.
            // In normalized coords: (x, y)_oriented → (y, 1 - x)_original
            return { CGPoint(x: $0.y, y: 1.0 - $0.x) }
        case .leftMirrored:
            return { CGPoint(x: 1.0 - $0.y, y: 1.0 - $0.x) }
        case .right:
            // Oriented image was rotated 90° CW to become upright.
            // Inverse: rotate 90° CCW.
            // (x, y)_oriented → (1 - y, x)_original
            return { CGPoint(x: 1.0 - $0.y, y: $0.x) }
        case .rightMirrored:
            return { CGPoint(x: $0.y, y: $0.x) }
        @unknown default:
            return { $0 }
        }
    }

    // MARK: - Guardrail heuristics

    /// True when the detected quad is a case where perspective correction
    /// is empirically a win: the receipt is very small in the frame, OR
    /// it's rotated far from axis-aligned. Everything else — an ordinary
    /// upright-and-close phone photo — is left alone, so OCR sees the
    /// original pixels without resampling.
    private static func shouldCorrect(quad: DocumentQuad) -> Bool {
        let area = quadArea(
            tl: quad.topLeft, tr: quad.topRight,
            bl: quad.bottomLeft, br: quad.bottomRight
        )
        // Top-edge tilt from the horizontal axis of the oriented frame.
        let dx = Double(quad.topRight.x - quad.topLeft.x)
        let dy = Double(quad.topRight.y - quad.topLeft.y)
        let tiltRadians = abs(atan2(dy, dx))

        // Small receipts — enough context wasted around them that cropping
        // and upscaling gives Vision a real signal-to-noise improvement.
        // Threshold picked from bench: IMG_5785 (area 0.14) wins big;
        // receipts at 0.23+ regress if their existing OCR was already
        // adequate, so we don't try to "help" them.
        if area < 0.20 { return true }

        // Strongly rotated receipts — the receipt is on its side or worse
        // relative to the frame. Vision's OCR technically reads rotated
        // text, but small numeric prices tend to drop out; correcting is
        // worth the resampling cost.
        // 30° = 0.52 rad. A landscape receipt in a portrait photo has
        // tilt near π/2 (90°); a slightly-tilted upright one is under 15°.
        if tiltRadians > 0.52 { return true }

        return false
    }

    // MARK: - Geometry helpers

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> Double {
        let dx = Double(a.x - b.x)
        let dy = Double(a.y - b.y)
        return (dx * dx + dy * dy).squareRoot()
    }

    /// Area of the quadrilateral (shoelace formula, in normalized units).
    private static func quadArea(tl: CGPoint, tr: CGPoint, bl: CGPoint, br: CGPoint) -> Double {
        // Corners in CCW order: bl -> br -> tr -> tl
        let pts: [CGPoint] = [bl, br, tr, tl]
        var s = 0.0
        for i in 0..<pts.count {
            let a = pts[i]
            let b = pts[(i + 1) % pts.count]
            s += Double(a.x) * Double(b.y) - Double(b.x) * Double(a.y)
        }
        return abs(s) * 0.5
    }
}
