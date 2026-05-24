import CoreGraphics
import Foundation
import ImageIO

/// Uniform interface for any OCR + structured-extraction pipeline.
///
/// Implementations should be stateless from the caller's perspective; any heavy
/// resources (loaded models, etc.) must be lazily initialized internally and
/// guarded for concurrent access.
public protocol OCRPipeline: Sendable {
    /// Stable identifier, e.g. "vision-regex". Used in `Receipt.provenance.pipelineId`
    /// and as the lookup key in `OCRPipelineRegistry`.
    static var id: String { get }

    /// Human-readable name shown in the labeler / DEBUG iOS settings.
    static var displayName: String { get }

    /// Version string for the model(s) backing this pipeline. Stored in provenance
    /// so benchmark results are reproducible across model updates.
    static var modelVersion: String { get }

    /// Extract receipt data from an image. `orientation` tells Vision how to
    /// interpret the input — critical for HEIC/JPEG photos which carry EXIF
    /// rotation metadata; without it, bbox coordinates land outside the
    /// displayed image.
    func extract(image: CGImage, orientation: CGImagePropertyOrientation) async throws -> ExtractionResult
}

public extension OCRPipeline {
    /// Convenience for callers that always pass upright images (e.g. iOS
    /// `VNDocumentCameraViewController` output).
    func extract(image: CGImage) async throws -> ExtractionResult {
        try await extract(image: image, orientation: .up)
    }
}

public struct ExtractionResult: Codable, Sendable {
    public var receipt: Receipt
    public var latencyMs: Int
    public var peakMemoryMB: Int?
    public var rawText: String?
    /// Every individual text observation from the OCR step, with its bbox.
    /// The labeler renders these as faint overlays and lets the user click
    /// any line to (re)assign it to a field — much more accurate than
    /// matching extracted values back to lines by string search.
    public var ocrLines: [OCRLine]

    public init(
        receipt: Receipt,
        latencyMs: Int,
        peakMemoryMB: Int? = nil,
        rawText: String? = nil,
        ocrLines: [OCRLine] = []
    ) {
        self.receipt = receipt
        self.latencyMs = latencyMs
        self.peakMemoryMB = peakMemoryMB
        self.rawText = rawText
        self.ocrLines = ocrLines
    }
}

/// One text observation from the OCR step. Coordinates are normalized [0,1]
/// in image space with origin top-left (matches the canonical schema).
public struct OCRLine: Codable, Sendable, Hashable, Identifiable {
    public let id: UUID
    public var text: String
    public var box: Receipt.BBox

    public init(id: UUID = UUID(), text: String, box: Receipt.BBox) {
        self.id = id
        self.text = text
        self.box = box
    }
}

public enum OCRError: LocalizedError, Sendable {
    case visionRequestFailed(String)
    case unsupportedPlatform(String)
    case modelNotAvailable(String)
    case parseFailed(String)

    public var errorDescription: String? {
        switch self {
        case .visionRequestFailed(let msg): return "Vision OCR failed: \(msg)"
        case .unsupportedPlatform(let msg): return "Unsupported platform: \(msg)"
        case .modelNotAvailable(let msg):   return "Model not available: \(msg)"
        case .parseFailed(let msg):         return "Parsing failed: \(msg)"
        }
    }
}
