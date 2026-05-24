import Foundation
import UIKit

/// On-disk storage for captured receipt images under `Documents/receipts/{uuid}.jpg`.
///
/// Images live outside SwiftData so the store stays small and migrations are
/// cheap. The SwiftData `Receipt` holds only the relative path; resolution
/// happens here.
enum ReceiptImageStorage {

    private static let subdirectory = "receipts"
    private static let jpegQuality: CGFloat = 0.85

    static func documentsDirectory() throws -> URL {
        try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
    }

    static func ensureReceiptsDirectory() throws -> URL {
        let dir = try documentsDirectory().appending(path: subdirectory, directoryHint: .isDirectory)
        if !FileManager.default.fileExists(atPath: dir.path(percentEncoded: false)) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// Persist `image` and return the relative path to store in the Receipt model.
    @discardableResult
    static func save(_ image: UIImage, id: UUID) throws -> String {
        guard let data = image.jpegData(compressionQuality: jpegQuality) else {
            throw NSError(domain: "ReceiptImageStorage", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Could not encode image as JPEG."
            ])
        }
        let dir = try ensureReceiptsDirectory()
        let url = dir.appending(path: "\(id.uuidString).jpg", directoryHint: .notDirectory)
        try data.write(to: url, options: .atomic)
        return "\(subdirectory)/\(id.uuidString).jpg"
    }

    static func absoluteURL(for relativePath: String) throws -> URL {
        try documentsDirectory().appending(path: relativePath, directoryHint: .notDirectory)
    }

    static func load(relativePath: String) -> UIImage? {
        guard let url = try? absoluteURL(for: relativePath) else { return nil }
        return UIImage(contentsOfFile: url.path(percentEncoded: false))
    }

    static func delete(relativePath: String) {
        guard let url = try? absoluteURL(for: relativePath) else { return }
        try? FileManager.default.removeItem(at: url)
    }
}
