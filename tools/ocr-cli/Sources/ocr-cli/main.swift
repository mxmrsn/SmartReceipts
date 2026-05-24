import AppKit
import CoreGraphics
import Foundation
import OCRKit

/// `ocr-cli` — the Python-bench → Swift-pipeline bridge.
///
/// Usage:
///   ocr-cli --pipeline <id> --image <path> [--json]
///   ocr-cli --list
///
/// On success, prints a JSON envelope to stdout:
///   { "ok": true,  "pipeline": "...", "result": <ExtractionResult> }
/// On failure, prints:
///   { "ok": false, "pipeline": "...", "error": "..." }   (exit 1)

@main
struct OCRCLI {
    static func main() async {
        let argv = Array(CommandLine.arguments.dropFirst())

        if argv.contains("--list") {
            let ids = OCRPipelineRegistry.ids
            let payload: [String: Any] = ["ok": true, "pipelines": ids]
            print(try! jsonString(payload))
            return
        }

        guard let pipelineId = value(for: "--pipeline", in: argv),
              let imagePath = value(for: "--image", in: argv) else {
            fail(error: "Usage: ocr-cli --pipeline <id> --image <path> [--json]\n       ocr-cli --list", pipelineId: nil)
            return
        }

        guard let pipeline = OCRPipelineRegistry.pipeline(id: pipelineId) else {
            fail(error: "Unknown pipeline '\(pipelineId)'. Available: \(OCRPipelineRegistry.ids.joined(separator: ", "))", pipelineId: pipelineId)
            return
        }

        guard let cgImage = loadCGImage(at: imagePath) else {
            fail(error: "Could not load image at \(imagePath)", pipelineId: pipelineId)
            return
        }

        do {
            let result = try await pipeline.extract(image: cgImage)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let resultData = try encoder.encode(result)
            let resultJSON = try JSONSerialization.jsonObject(with: resultData)

            let envelope: [String: Any] = [
                "ok": true,
                "pipeline": pipelineId,
                "result": resultJSON
            ]
            print(try jsonString(envelope))
        } catch {
            fail(error: "Extraction failed: \(error)", pipelineId: pipelineId)
        }
    }

    private static func value(for flag: String, in argv: [String]) -> String? {
        guard let i = argv.firstIndex(of: flag), i + 1 < argv.count else { return nil }
        return argv[i + 1]
    }

    private static func loadCGImage(at path: String) -> CGImage? {
        let url = URL(fileURLWithPath: path)
        guard let nsImage = NSImage(contentsOf: url) else { return nil }
        var rect = CGRect(origin: .zero, size: nsImage.size)
        return nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }

    private static func fail(error: String, pipelineId: String?) {
        let envelope: [String: Any] = [
            "ok": false,
            "pipeline": pipelineId ?? NSNull(),
            "error": error
        ]
        let msg = (try? jsonString(envelope)) ?? error
        FileHandle.standardError.write(Data(msg.utf8))
        FileHandle.standardError.write(Data("\n".utf8))
        exit(1)
    }

    private static func jsonString(_ obj: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
