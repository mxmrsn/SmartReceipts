import CoreGraphics
import Foundation
import Vision

/// Baseline pipeline: Apple Vision text recognition + lightweight regex/heuristic
/// parsing into the canonical `Receipt` schema. The "floor" against which all
/// other pipelines are benchmarked.
///
/// This is intentionally simple at M1. The heuristics here will not perfectly
/// extract every receipt — that is the point. The benchmark harness measures
/// how big the gap to `vision-fm` and the MLX VLMs actually is.
public struct VisionOnlyPipeline: OCRPipeline {
    public static let id = "vision-regex"
    public static let displayName = "Apple Vision + Regex"
    public static let modelVersion = "vision-accurate.2"

    public init() {}

    public func extract(image: CGImage) async throws -> ExtractionResult {
        let startNs = DispatchTime.now().uptimeNanoseconds

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["en-US"]

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw OCRError.visionRequestFailed(error.localizedDescription)
        }

        let observations: [VNRecognizedTextObservation] = request.results ?? []
        // Sort top-to-bottom so the rawText shown to the user matches the
        // reading order the parser sees.
        let lines: [(text: String, box: Receipt.BBox)] = observations
            .compactMap { obs -> (text: String, box: Receipt.BBox)? in
                guard let top = obs.topCandidates(1).first else { return nil }
                return (text: top.string, box: Self.bbox(from: obs.boundingBox))
            }
            .sorted { $0.box.y < $1.box.y }
        let rawText = lines.map(\.text).joined(separator: "\n")

        let parsed = ReceiptHeuristicParser.parse(lines: lines)

        let elapsedMs = Int((DispatchTime.now().uptimeNanoseconds &- startNs) / 1_000_000)

        let receipt = Receipt(
            imageId: UUID(),
            header: parsed.header,
            lineItems: parsed.lineItems,
            totals: parsed.totals,
            payment: parsed.payment,
            provenance: Receipt.Provenance(
                pipelineId: Self.id,
                modelVersion: Self.modelVersion,
                confidence: parsed.overallConfidence,
                fieldConfidence: parsed.fieldConfidence,
                bboxes: parsed.bboxes
            )
        )

        return ExtractionResult(
            receipt: receipt,
            latencyMs: elapsedMs,
            peakMemoryMB: nil,
            rawText: rawText
        )
    }

    /// Vision's bounding boxes are in normalized image coords with origin at
    /// bottom-left and `width`/`height` derived from corner points. Flip Y so
    /// the canonical schema's origin is top-left (matching SwiftUI / web).
    private static func bbox(from vn: CGRect) -> Receipt.BBox {
        Receipt.BBox(
            x: Double(vn.minX),
            y: Double(1.0 - vn.maxY),
            width: Double(vn.width),
            height: Double(vn.height)
        )
    }
}
