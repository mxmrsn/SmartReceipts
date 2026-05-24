import SwiftUI
import UIKit
import VisionKit

/// SwiftUI wrapper around `VNDocumentCameraViewController` — Apple's built-in
/// document scanner with auto edge detection, perspective correction, and
/// multi-page support.
///
/// The wrapper does not dismiss itself. Present it via `.sheet(isPresented:)`
/// and toggle the binding to `false` from within the `onComplete` callback.
struct DocumentScannerView: UIViewControllerRepresentable {

    let onComplete: (Result<[UIImage], Error>) -> Void

    func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let vc = VNDocumentCameraViewController()
        vc.delegate = context.coordinator
        return vc
    }

    func updateUIViewController(_ uiViewController: VNDocumentCameraViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onComplete: onComplete)
    }

    @MainActor
    final class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        let onComplete: (Result<[UIImage], Error>) -> Void

        init(onComplete: @escaping (Result<[UIImage], Error>) -> Void) {
            self.onComplete = onComplete
        }

        func documentCameraViewController(
            _ controller: VNDocumentCameraViewController,
            didFinishWith scan: VNDocumentCameraScan
        ) {
            var pages: [UIImage] = []
            pages.reserveCapacity(scan.pageCount)
            for i in 0..<scan.pageCount {
                pages.append(scan.imageOfPage(at: i))
            }
            onComplete(.success(pages))
        }

        func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) {
            onComplete(.failure(CancellationError()))
        }

        func documentCameraViewController(
            _ controller: VNDocumentCameraViewController,
            didFailWithError error: any Error
        ) {
            onComplete(.failure(error))
        }
    }
}

extension UIImage {
    /// Stack `images` vertically into a single image, scaling each to the
    /// widest page width. Used for multi-page (e.g. long CVS-style) receipts
    /// before handing off to OCR.
    static func concatenatedVertically(_ images: [UIImage]) -> UIImage? {
        guard !images.isEmpty else { return nil }
        if images.count == 1 { return images[0] }

        let maxWidth = images.map(\.size.width).max() ?? 0
        guard maxWidth > 0 else { return nil }

        var totalHeight: CGFloat = 0
        let scaledHeights: [CGFloat] = images.map { img in
            let h = img.size.height * (maxWidth / img.size.width)
            totalHeight += h
            return h
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: maxWidth, height: totalHeight), format: format)
        return renderer.image { _ in
            var y: CGFloat = 0
            for (img, h) in zip(images, scaledHeights) {
                img.draw(in: CGRect(x: 0, y: y, width: maxWidth, height: h))
                y += h
            }
        }
    }
}
