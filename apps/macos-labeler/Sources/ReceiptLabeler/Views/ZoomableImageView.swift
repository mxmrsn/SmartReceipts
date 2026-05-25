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

    /// Programmatic scroll offset for the underlying ScrollView. We update
    /// this when zooming so the point under the cursor stays under the cursor.
    @State private var scrollPosition = ScrollPosition(x: 0, y: 0)
    /// The latest scroll offset reported by `onScrollGeometryChange`. We need
    /// to read it to compute the delta when zooming around the cursor.
    @State private var currentScrollOffset: CGPoint = .zero
    /// Where the mouse currently is, in *image-natural* coordinates
    /// (before zoom is applied). Set/cleared by `onContinuousHover` on the
    /// image. Used as the focal point when the user changes zoom.
    @State private var hoverInImage: CGPoint? = nil
    /// Snapshot of `zoom` at the moment a pinch gesture began. Each
    /// MagnifyGesture event reports `magnification` as the *total* scale
    /// since gesture start, so we need a stable baseline to multiply against.
    @State private var pinchStartZoom: CGFloat = 1.0
    /// Snapshot of the live scroll offset at gesture start. We anchor every
    /// frame of the pinch off this baseline rather than the live offset,
    /// because `onScrollGeometryChange` is async — by the time the next
    /// pinch frame fires, the offset we just wrote hasn't been reported
    /// back, and using the stale live value compounds an error per frame.
    @State private var pinchStartScroll: CGPoint = .zero
    /// Focal point of the pinch in image-natural coords, captured at
    /// gesture start. Computed from `MagnifyGesture.startAnchor`, which
    /// is the exact midpoint between the two fingers — more reliable than
    /// relying on the last hover position because hover events stop firing
    /// while the fingers are on the trackpad.
    @State private var pinchStartFocal: CGPoint = .zero
    /// True while a pinch gesture is in flight. Used to seed the three
    /// snapshots above exactly once per gesture (on the first onChanged).
    @State private var isPinching: Bool = false

    private let minZoom: CGFloat = 0.05
    private let maxZoom: CGFloat = 4.0
    /// Padding around the image inside the scroll content. Needs to match
    /// the `.padding(...)` modifier below for the zoom math to be correct.
    private let imagePadding: CGFloat = 16

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
                        // Track cursor in image-natural coords. `location`
                        // here is in the image's local frame (already
                        // dimensioned by `image.size * zoom`), so dividing
                        // by zoom recovers the natural coords. This is what
                        // we anchor zoom around so the bbox the user is
                        // hovering stays under the cursor as zoom changes.
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let location):
                                hoverInImage = CGPoint(
                                    x: location.x / zoom,
                                    y: location.y / zoom
                                )
                            case .ended:
                                hoverInImage = nil
                            }
                        }
                        // Trackpad two-finger pinch / spread → zoom around
                        // the cursor. MagnifyGesture.Value.magnification is
                        // the *cumulative* scale since the gesture started,
                        // so we multiply against a snapshot of `zoom` taken
                        // on the first onChanged of each gesture.
                        .gesture(pinchGesture)
                        .padding(imagePadding)
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
            .scrollPosition($scrollPosition)
            // Stash the current scroll offset on every change so we can
            // compute "where the cursor lands in scroll content" without
            // a second GeometryReader.
            .onScrollGeometryChange(for: CGPoint.self) { geom in
                geom.contentOffset
            } action: { _, newOffset in
                currentScrollOffset = newOffset
            }
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

            // Slider drives zoom directly. We can't pivot around the cursor
            // here because the cursor is on the slider, not the image — so
            // slider drags fall back to anchoring at the image center.
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
                setZoomAroundCursor(1.0)
                userZoomed = true
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
        setZoomAroundCursor((zoom * factor).clamped(to: minZoom...maxZoom))
        userZoomed = true
    }

    /// Where the image's top-left sits in scroll-content coordinates at a
    /// given zoom. When the image+padding is SMALLER than the viewport the
    /// outer `.frame(minWidth:alignment: .center)` centers the image, so
    /// the origin sits at `(viewport - image*zoom)/2` instead of just at
    /// `(imagePadding)`. As zoom grows past the threshold the centering
    /// disappears and the origin jumps to `(imagePadding)`. Without
    /// compensating for that jump in the zoom math, a focal point under
    /// the cursor visibly drifts toward the left edge while crossing the
    /// threshold — the "shifts left" behavior the user was seeing.
    private func imageOrigin(at z: CGFloat) -> CGPoint {
        guard let image, containerSize.width > 0, containerSize.height > 0 else {
            return CGPoint(x: imagePadding, y: imagePadding)
        }
        let scaledW = image.size.width * z
        let scaledH = image.size.height * z
        let contentW = scaledW + 2 * imagePadding
        let contentH = scaledH + 2 * imagePadding
        let x: CGFloat = contentW < containerSize.width
            ? (containerSize.width - scaledW) / 2
            : imagePadding
        let y: CGFloat = contentH < containerSize.height
            ? (containerSize.height - scaledH) / 2
            : imagePadding
        return CGPoint(x: x, y: y)
    }

    /// Two-finger pinch on the trackpad. We snapshot zoom + scroll + focal
    /// point once on the gesture's first frame, then on each subsequent
    /// frame compute the new zoom and the new scroll offset *as absolute
    /// values* from those snapshots. That avoids the drift that comes from
    /// reading the live scroll offset every frame — `onScrollGeometryChange`
    /// is async, so the live value lags one frame behind every scroll
    /// write and the focal point would walk away from the cursor.
    private var pinchGesture: some Gesture {
        MagnifyGesture(minimumScaleDelta: 0.01)
            .onChanged { value in
                guard let img = image else { return }
                if !isPinching {
                    pinchStartZoom = zoom
                    pinchStartScroll = currentScrollOffset
                    // startAnchor is a UnitPoint (0..1) inside the gesture's
                    // host view — the Image, sized `image.size * zoom`. So
                    // `startAnchor * image.size` is the focal point in
                    // *natural* image coords, independent of zoom.
                    let anchor = value.startAnchor
                    pinchStartFocal = CGPoint(
                        x: anchor.x * img.size.width,
                        y: anchor.y * img.size.height
                    )
                    isPinching = true
                }

                let newZoom = (pinchStartZoom * value.magnification)
                    .clamped(to: minZoom...maxZoom)
                // Compute origins BEFORE mutating zoom (newOrigin uses the
                // about-to-be-current zoom value, oldOrigin uses the
                // gesture-start snapshot).
                let oldOrigin = imageOrigin(at: pinchStartZoom)
                let newOrigin = imageOrigin(at: newZoom)
                zoom = newZoom
                userZoomed = true

                // Keep the focal point's screen position fixed across the
                // whole gesture, including any centered→top-left layout
                // shift that crosses the viewport-width threshold:
                //   newScroll = startScroll + (newOrigin − oldOrigin) + focal * Δzoom
                let dx = (newOrigin.x - oldOrigin.x) + pinchStartFocal.x * (newZoom - pinchStartZoom)
                let dy = (newOrigin.y - oldOrigin.y) + pinchStartFocal.y * (newZoom - pinchStartZoom)
                scrollPosition.scrollTo(point: CGPoint(
                    x: pinchStartScroll.x + dx,
                    y: pinchStartScroll.y + dy
                ))
            }
            .onEnded { _ in
                isPinching = false
            }
    }

    /// Change zoom while keeping the point currently under the mouse fixed
    /// on screen — the same UX as Preview.app / Acorn / Photoshop. If the
    /// mouse isn't over the image (e.g. we got here from a keyboard
    /// shortcut while pointer is on the controlbar), zoom anchors at the
    /// scroll viewport's center instead.
    ///
    /// Derivation: in scroll-content coordinates the cursor is at
    /// `(currentScrollOffset + viewportCursor)`. The image's top-left
    /// inside the scroll content is at `(imagePadding, imagePadding)`.
    /// The point in image-natural coords under the cursor is
    /// `hoverInImage = ((scrollOffset + viewportCursor) - imagePadding) / oldZoom`.
    /// To keep that natural-image point fixed on screen after zooming to
    /// `newZoom`, the new scrollOffset must be:
    ///   newOffset = imagePadding + hoverInImage * newZoom - viewportCursor
    /// Subtracting the old offset gives a delta of
    ///   delta = hoverInImage * (newZoom − oldZoom)
    /// which is what we apply.
    private func setZoomAroundCursor(_ newZoom: CGFloat) {
        let oldZoom = zoom
        let clamped = newZoom.clamped(to: minZoom...maxZoom)
        guard clamped != oldZoom else { return }
        // Snapshot the image origin BEFORE updating zoom — origin depends
        // on the current zoom via the centering rule in imageOrigin(at:).
        let oldOrigin = imageOrigin(at: oldZoom)
        let newOrigin = imageOrigin(at: clamped)
        zoom = clamped

        guard let hover = hoverInImage else { return }
        // newScroll = oldScroll + (newOrigin − oldOrigin) + focal * Δzoom
        // The origin term compensates for the centered→top-left layout shift.
        let dx = (newOrigin.x - oldOrigin.x) + hover.x * (clamped - oldZoom)
        let dy = (newOrigin.y - oldOrigin.y) + hover.y * (clamped - oldZoom)
        let target = CGPoint(
            x: currentScrollOffset.x + dx,
            y: currentScrollOffset.y + dy
        )
        scrollPosition.scrollTo(point: target)
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
