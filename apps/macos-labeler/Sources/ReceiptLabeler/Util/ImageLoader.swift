import AppKit
import Foundation
import ImageIO

/// Loads images upright (EXIF orientation already applied) so that Vision's
/// normalized bbox coords map cleanly onto the displayed image. If we let
/// NSImage handle this lazily it returns the oriented size but the raw
/// unrotated CGImage, which makes Vision normalize against the wrong aspect
/// and our bounding boxes land outside the visible image.
///
/// Tiny in-memory thumbnail cache. Decoding HEIC/JPEG is expensive; once we
/// have a thumbnail at the requested pixel size we keep it around for the
/// lifetime of the process.
@MainActor
final class ImageLoader {
    static let shared = ImageLoader()

    private var thumbCache: [Key: NSImage] = [:]

    private struct Key: Hashable {
        let url: URL
        let pixelSize: CGFloat
    }

    /// Full-size upright image for the detail pane. Large maxPixelSize so OCR
    /// still has detail; ImageIO downsamples if the source is huge.
    func fullImage(for url: URL) -> NSImage? {
        loadUpright(at: url, maxPixelSize: 6000)
    }

    func thumbnail(for url: URL, pixelSize: CGFloat = 160) -> NSImage? {
        let key = Key(url: url, pixelSize: pixelSize)
        if let cached = thumbCache[key] { return cached }
        // Request 2x for Retina sharpness.
        guard let thumb = loadUpright(at: url, maxPixelSize: pixelSize * 2) else { return nil }
        thumbCache[key] = thumb
        return thumb
    }

    // MARK: - ImageIO upright loader

    private func loadUpright(at url: URL, maxPixelSize: CGFloat) -> NSImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return NSImage(contentsOf: url)  // ultimate fallback
        }
        let options: [CFString: Any] = [
            kCGImageSourceShouldCacheImmediately: true,
            // Apply EXIF orientation to the produced thumbnail bitmap.
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return NSImage(contentsOf: url)
        }
        // Use pixel dimensions as logical size so SwiftUI displays + Vision
        // normalization agree on aspect ratio.
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }
}
