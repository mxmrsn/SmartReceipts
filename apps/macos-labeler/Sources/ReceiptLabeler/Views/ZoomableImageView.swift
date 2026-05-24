import AppKit
import SwiftUI

/// Scroll-and-zoom image viewer with fit-to-window default, a slider, +/−
/// buttons, ⌘0/⌘1/⌘+/⌘− shortcuts, and a percent readout. Built on top of
/// `ImageLoader` so we never decode the same image twice.
struct ZoomableImageView: View {

    let url: URL

    @State private var image: NSImage?
    @State private var zoom: CGFloat = 1.0
    @State private var fitZoom: CGFloat = 1.0
    @State private var containerSize: CGSize = .zero

    private let minZoom: CGFloat = 0.05
    private let maxZoom: CGFloat = 4.0

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                ScrollView([.horizontal, .vertical]) {
                    if let image {
                        Image(nsImage: image)
                            .resizable()
                            .interpolation(.medium)
                            .frame(
                                width: image.size.width * zoom,
                                height: image.size.height * zoom
                            )
                            .padding(16)
                            .frame(
                                minWidth: geo.size.width,
                                minHeight: geo.size.height,
                                alignment: .center
                            )
                    } else {
                        ProgressView()
                            .frame(
                                minWidth: geo.size.width,
                                minHeight: geo.size.height
                            )
                    }
                }
                .background(Color(nsColor: .controlBackgroundColor))
                .task(id: url) {
                    image = ImageLoader.shared.fullImage(for: url)
                    containerSize = geo.size
                    recomputeFit(initial: true)
                }
                .onChange(of: geo.size) { _, newSize in
                    containerSize = newSize
                    recomputeFit(initial: false)
                }
            }
            Divider()
            controlBar
        }
    }

    // MARK: - Controls

    private var controlBar: some View {
        HStack(spacing: 12) {
            Button { adjustZoom(by: 1 / 1.25) } label: {
                Image(systemName: "minus.magnifyingglass")
            }
            .help("Zoom out (⌘−)")
            .keyboardShortcut("-", modifiers: [.command])

            Button { adjustZoom(by: 1.25) } label: {
                Image(systemName: "plus.magnifyingglass")
            }
            .help("Zoom in (⌘+)")
            .keyboardShortcut("=", modifiers: [.command])

            Slider(value: $zoom, in: minZoom...maxZoom)
                .frame(minWidth: 120, idealWidth: 180, maxWidth: 240)

            Button("Fit") {
                recomputeFit(initial: false)
                zoom = fitZoom
            }
            .help("Fit window (⌘0)")
            .keyboardShortcut("0", modifiers: [.command])

            Button("100%") { zoom = 1.0 }
                .help("Actual size (⌘1)")
                .keyboardShortcut("1", modifiers: [.command])

            Spacer(minLength: 8)

            Text("\(Int((zoom * 100).rounded()))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 48, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    // MARK: - Sizing

    private func adjustZoom(by factor: CGFloat) {
        zoom = (zoom * factor).clamped(to: minZoom...maxZoom)
    }

    /// Compute fit-to-window scale. If `initial` is true, also set `zoom`
    /// to the fit value (used on first load of an image).
    private func recomputeFit(initial: Bool) {
        guard let image, containerSize.width > 0, containerSize.height > 0 else {
            fitZoom = 1.0
            if initial { zoom = 1.0 }
            return
        }
        let padding: CGFloat = 32
        let avail = CGSize(
            width: max(containerSize.width - padding, 1),
            height: max(containerSize.height - padding, 1)
        )
        let scaleW = avail.width / image.size.width
        let scaleH = avail.height / image.size.height
        // Fill the larger axis: portrait receipts fill the pane width and
        // scroll vertically; landscape ones fill height. Cap at 1.0 so we
        // never enlarge past native resolution.
        fitZoom = max(minZoom, min(max(scaleW, scaleH), 1.0))
        if initial {
            zoom = fitZoom
        }
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
