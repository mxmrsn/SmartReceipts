import AppKit
import SwiftUI

/// Scroll-and-zoom image viewer with fit-to-window default, a slider, +/−
/// buttons, ⌘0/⌘1/⌘+/⌘− shortcuts, and a percent readout. Built on top of
/// `ImageLoader` so we never decode the same image twice.
struct ZoomableImageView<Overlay: View>: View {

    let url: URL
    let overlay: Overlay

    @State private var image: NSImage?
    @State private var zoom: CGFloat = 1.0
    @State private var fitZoom: CGFloat = 1.0
    @State private var containerSize: CGSize = .zero
    @State private var showOverlay: Bool = true
    /// True once the user has touched the slider, +/- buttons, or 100%.
    /// While this is false we keep snapping `zoom` to `fitZoom` on every
    /// layout change — which is the only way to win the race against
    /// SwiftUI's layout settling after a new receipt loads. Reset to false
    /// when navigating to a new URL.
    @State private var userZoomed: Bool = false

    private let minZoom: CGFloat = 0.05
    private let maxZoom: CGFloat = 4.0

    init(url: URL, @ViewBuilder overlay: () -> Overlay) {
        self.url = url
        self.overlay = overlay()
    }

    init(url: URL, overlay: Overlay) {
        self.url = url
        self.overlay = overlay
    }

    var body: some View {
        GeometryReader { geo in
            ScrollView([.horizontal, .vertical]) {
                if let image {
                    // Image with the overlay attached via .overlay() — this
                    // guarantees the overlay's GeometryReader sees the same
                    // size as the image's actual display frame, eliminating
                    // the "overlay 2x bigger than image" bug we'd otherwise
                    // get with a sibling ZStack.
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.medium)
                        .frame(
                            width: image.size.width * zoom,
                            height: image.size.height * zoom
                        )
                        .overlay {
                            if showOverlay {
                                overlay
                            }
                        }
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
            // Keep containerSize in lock-step with the GeometryReader's
            // measurement. `initial: true` seeds it on first appearance.
            // applyFit() also re-snaps zoom while userZoomed is false, so
            // late layout passes correct any too-small initial fit.
            .onChange(of: geo.size, initial: true) { _, newSize in
                containerSize = newSize
                applyFit()
            }
            .task(id: url) {
                userZoomed = false
                image = ImageLoader.shared.fullImage(for: url)
                applyFit()
            }
            .onChange(of: image) { _, _ in
                applyFit()
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                Divider()
                controlBar
            }
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

            Slider(value: $zoom, in: minZoom...maxZoom) { editing in
                if !editing { userZoomed = true }
            }
            .frame(minWidth: 120, idealWidth: 180, maxWidth: 240)

            Button("Fit") {
                userZoomed = false
                applyFit()
            }
            .help("Fit window (⌘0)")
            .keyboardShortcut("0", modifiers: [.command])

            Button("100%") {
                userZoomed = true
                zoom = 1.0
            }
            .help("Actual size (⌘1)")
            .keyboardShortcut("1", modifiers: [.command])

            Divider().frame(height: 18)

            Toggle(isOn: $showOverlay) {
                Image(systemName: showOverlay ? "viewfinder" : "viewfinder.slash")
            }
            .toggleStyle(.button)
            .help("Toggle detected-field bounding boxes (⌘B)")
            .keyboardShortcut("b", modifiers: [.command])

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
        userZoomed = true
        zoom = (zoom * factor).clamped(to: minZoom...maxZoom)
    }

    /// Recompute fit-to-window scale and apply it to `zoom` if the user
    /// hasn't manually zoomed for this receipt yet. Runs on every layout
    /// change (geo.size, image, url) — that way late layout settling can
    /// still correct an initial too-small fit before the user has touched
    /// anything.
    private func applyFit() {
        guard let image, containerSize.width > 0, containerSize.height > 0 else { return }
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
        if !userZoomed {
            zoom = fitZoom
        }
    }
}

extension ZoomableImageView where Overlay == EmptyView {
    init(url: URL) {
        self.url = url
        self.overlay = EmptyView()
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
