import Foundation
import OCRKit

/// Reads and writes label JSON files under `dataset/labels/{imageId}.json`.
struct LabelStore: Sendable {

    let datasetDirectory: URL

    var labelsDirectory: URL {
        datasetDirectory.appending(path: "labels", directoryHint: .isDirectory)
    }

    var imagesDirectory: URL {
        datasetDirectory.appending(path: "images", directoryHint: .isDirectory)
    }

    func ensureDirectoriesExist() throws {
        try FileManager.default.createDirectory(at: labelsDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
    }

    func labelURL(for imageId: UUID) -> URL {
        labelsDirectory.appending(
            path: "\(imageId.uuidString).json",
            directoryHint: .notDirectory
        )
    }

    func labelExists(for imageId: UUID) -> Bool {
        FileManager.default.fileExists(atPath: labelURL(for: imageId).path(percentEncoded: false))
    }

    func load(imageId: UUID) throws -> LabelDocument {
        let url = labelURL(for: imageId)
        let data = try Data(contentsOf: url)
        return try LabelDocument.decoded(from: data)
    }

    func save(_ document: LabelDocument, imageId: UUID) throws {
        try ensureDirectoriesExist()
        let data = try document.encoded()
        try data.write(to: labelURL(for: imageId), options: .atomic)
    }

    /// List every image in `dataset/images/` with supported extensions, case-insensitive.
    func discoverImages() -> [URL] {
        let exts: Set<String> = ["jpg", "jpeg", "png", "heic"]
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: imagesDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        return entries
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
