import AppKit
import Foundation
import ImageIO

/// Reads EXIF / TIFF / filesystem date stamps from an image. Used as a
/// fallback when the receipt has no printed date — many receipts (especially
/// gas pump and ATM receipts) skip the date entirely.
enum ImageMetadata {

    enum DateSource: String, Sendable {
        case exif
        case tiff
        case file
    }

    /// Best-effort image creation date. Tries:
    /// 1. EXIF `DateTimeOriginal` (when the photo was actually taken)
    /// 2. TIFF `DateTime` (file metadata)
    /// 3. Filesystem creation date
    static func creationDate(at url: URL) -> (date: Date, source: DateSource)? {
        if let cgSource = CGImageSourceCreateWithURL(url as CFURL, nil),
           let properties = CGImageSourceCopyPropertiesAtIndex(cgSource, 0, nil) as? [CFString: Any] {

            if let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any],
               let dateString = exif[kCGImagePropertyExifDateTimeOriginal] as? String,
               let date = parseExifDate(dateString) {
                return (date, .exif)
            }
            if let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any],
               let dateString = tiff[kCGImagePropertyTIFFDateTime] as? String,
               let date = parseExifDate(dateString) {
                return (date, .tiff)
            }
        }

        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path(percentEncoded: false)),
           let date = attrs[.creationDate] as? Date {
            return (date, .file)
        }
        return nil
    }

    private static let exifFormatter: DateFormatter = {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyy:MM:dd HH:mm:ss"
        df.timeZone = TimeZone(identifier: "UTC")
        return df
    }()

    private static func parseExifDate(_ s: String) -> Date? {
        exifFormatter.date(from: s)
    }

    static func formatISODate(_ d: Date) -> String {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyy-MM-dd"
        df.timeZone = TimeZone(identifier: "UTC")
        return df.string(from: d)
    }

    // MARK: - EXIF orientation

    /// EXIF orientation tag for the image at `url`. HEIC and JPEG photos
    /// commonly carry orientation = 6 (90° rotation needed). We pass this to
    /// Vision so its normalized bbox coords match the upright display.
    static func orientation(at url: URL) -> CGImagePropertyOrientation {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let raw = props[kCGImagePropertyOrientation] as? UInt32,
              let value = CGImagePropertyOrientation(rawValue: raw)
        else {
            return .up
        }
        return value
    }
}
