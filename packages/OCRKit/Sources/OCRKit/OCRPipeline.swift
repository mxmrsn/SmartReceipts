import CoreGraphics
import Foundation

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

    func extract(image: CGImage) async throws -> ExtractionResult
}

public struct ExtractionResult: Codable, Sendable {
    public var receipt: Receipt
    public var latencyMs: Int
    public var peakMemoryMB: Int?
    public var rawText: String?

    public init(receipt: Receipt, latencyMs: Int, peakMemoryMB: Int? = nil, rawText: String? = nil) {
        self.receipt = receipt
        self.latencyMs = latencyMs
        self.peakMemoryMB = peakMemoryMB
        self.rawText = rawText
    }
}

public enum OCRError: Error, Sendable {
    case visionRequestFailed(String)
    case unsupportedPlatform(String)
    case modelNotAvailable(String)
    case parseFailed(String)
}
