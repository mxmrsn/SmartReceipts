import Foundation

/// Central lookup for all available OCR pipelines.
///
/// The iOS app, the macOS labeler, and the `ocr-cli` Swift CLI all consult this
/// registry. Adding a new pipeline = conform to `OCRPipeline` + register here.
public enum OCRPipelineRegistry {

    /// All pipelines available on the current platform.
    public static let all: [any OCRPipeline] = [
        VisionOnlyPipeline()
        // Future:
        // VisionPlusFoundationModelsPipeline()
        // MLXQwen25VLPipeline()
        // MLXSmolVLMPipeline()
    ]

    public static func pipeline(id: String) -> (any OCRPipeline)? {
        all.first { type(of: $0).id == id }
    }

    public static var ids: [String] {
        all.map { type(of: $0).id }
    }
}
