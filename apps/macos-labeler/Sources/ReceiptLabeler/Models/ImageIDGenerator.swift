import CryptoKit
import Foundation

/// Produces a stable UUID for a given image filename. Used so the user can
/// drop messy original filenames (e.g. `IMG_1234.heic`) into `dataset/images/`
/// and we still get reproducible, schema-conformant `imageId` values that
/// resolve back to the source file later.
enum ImageIDGenerator {

    /// Derive a deterministic UUID v5-style from the image's basename (no
    /// extension). Two files with the same stem map to the same UUID — that's
    /// intentional so a user re-encoding from .heic → .jpg keeps the same id.
    static func uuid(forFilename filename: String) -> UUID {
        let stem = (filename as NSString).deletingPathExtension
        let data = Data(stem.utf8)
        let digest = SHA256.hash(data: data)
        var bytes = Array(digest.prefix(16))
        bytes[6] = (bytes[6] & 0x0F) | 0x50  // RFC 4122 version 5
        bytes[8] = (bytes[8] & 0x3F) | 0x80  // RFC 4122 variant
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }

    static func uuid(forURL url: URL) -> UUID {
        uuid(forFilename: url.lastPathComponent)
    }
}
