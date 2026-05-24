import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Central lookup for all available OCR pipelines.
///
/// The iOS app, the macOS labeler, and the `ocr-cli` Swift CLI all consult this
/// registry. Adding a new pipeline = conform to `OCRPipeline` + register here.
public enum OCRPipelineRegistry {

    /// All pipelines available on the current platform, in preference order
    /// (best first). The Foundation-Models pipeline registers on platforms
    /// that have the framework at build time; whether it actually works at
    /// runtime depends on Apple Intelligence being enabled — which the
    /// pipeline checks at `extract` time.
    public static let all: [any OCRPipeline] = {
        var pipelines: [any OCRPipeline] = []
        #if canImport(FoundationModels)
        if #available(macOS 26.0, iOS 26.0, *) {
            pipelines.append(VisionPlusFoundationModelsPipeline())
        }
        #endif
        pipelines.append(VisionOnlyPipeline())
        return pipelines
    }()

    public static func pipeline(id: String) -> (any OCRPipeline)? {
        all.first { type(of: $0).id == id }
    }

    public static var ids: [String] {
        all.map { type(of: $0).id }
    }

    /// First-preference pipeline for callers that don't care about
    /// benchmarking specific ids — typically the iOS Capture flow and the
    /// labeler's pre-labeling step.
    public static var preferred: any OCRPipeline {
        all.first ?? VisionOnlyPipeline()
    }
}
