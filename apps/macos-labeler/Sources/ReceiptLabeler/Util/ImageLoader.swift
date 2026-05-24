import AppKit
import Foundation

/// Tiny in-memory thumbnail cache. Decoding HEIC/JPEG is expensive; once we
/// have a thumbnail at the requested pixel size we keep it around for the
/// lifetime of the process. Roughly 1000 thumbnails @ ~64KB each = 64 MB.
@MainActor
final class ImageLoader {
    static let shared = ImageLoader()

    private var thumbCache: [Key: NSImage] = [:]

    private struct Key: Hashable {
        let url: URL
        let pixelSize: CGFloat
    }

    func thumbnail(for url: URL, pixelSize: CGFloat = 160) -> NSImage? {
        let key = Key(url: url, pixelSize: pixelSize)
        if let cached = thumbCache[key] { return cached }
        guard let image = NSImage(contentsOf: url) else { return nil }
        let thumb = image.resized(toFit: NSSize(width: pixelSize, height: pixelSize))
        thumbCache[key] = thumb
        return thumb
    }

    /// Full-size image for the detail pane. Not cached — typically only one is
    /// in view at a time.
    func fullImage(for url: URL) -> NSImage? {
        NSImage(contentsOf: url)
    }
}

private extension NSImage {
    /// Downscale to fit within `target` while preserving aspect ratio.
    func resized(toFit target: NSSize) -> NSImage {
        let original = self.size
        guard original.width > 0, original.height > 0 else { return self }
        let scale = min(target.width / original.width, target.height / original.height, 1.0)
        let newSize = NSSize(width: original.width * scale, height: original.height * scale)

        let out = NSImage(size: newSize)
        out.lockFocus()
        NSGraphicsContext.current?.imageInterpolation = .medium
        self.draw(
            in: NSRect(origin: .zero, size: newSize),
            from: NSRect(origin: .zero, size: original),
            operation: .copy,
            fraction: 1.0
        )
        out.unlockFocus()
        return out
    }
}
