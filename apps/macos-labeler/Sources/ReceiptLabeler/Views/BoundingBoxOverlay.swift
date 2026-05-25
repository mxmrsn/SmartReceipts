import OCRKit
import SwiftUI

/// Two-layer overlay on top of the receipt image:
///
/// 1. **All OCR observations** rendered as faint outlines — click any to
///    open a popover to assign it to a field, OR if a field is currently
///    selected, click to assign directly without the popover.
/// 2. **Claimed fields** rendered as solid colored rectangles. The one
///    matching `draft.selectedBBoxKey` gets a highlighted ring, four
///    corner resize handles, and a drag-to-move handler.
struct BoundingBoxOverlay: View {

    @Bindable var draft: LabelDraft

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                // Layer 1: all OCR lines as faint click-targets
                ForEach(draft.ocrLines) { line in
                    OCRLineButton(
                        line: line,
                        size: geo.size,
                        selectedKey: draft.selectedBBoxKey,
                        onAssign: { field in
                            draft.assign(line: line, to: field)
                            draft.selectedBBoxKey = field.bboxKey
                        },
                        onAssignToSelected: {
                            guard let key = draft.selectedBBoxKey else { return }
                            // Line-item keys are dynamic (lineItem.NNN[.price]),
                            // so FieldTarget can't lookup them. Dispatch by
                            // key prefix instead.
                            if key.hasPrefix("lineItem.") {
                                draft.assignLine(line, toLineItemKey: key)
                            } else if let field = LabelDraft.FieldTarget.allCases.first(where: { $0.bboxKey == key }) {
                                draft.assign(line: line, to: field)
                            }
                        }
                    )
                }
                // Layer 2: solid colored rectangles for claimed fields
                ForEach(claimedBoxes) { item in
                    ClaimedBoxView(
                        item: item,
                        canvasSize: geo.size,
                        isSelected: item.bboxKey == draft.selectedBBoxKey,
                        onSelect: { draft.selectedBBoxKey = item.bboxKey },
                        onMoveEnd: { delta in
                            draft.translateBBox(key: item.bboxKey, by: delta)
                        },
                        onResizeEnd: { corner, delta in
                            draft.resizeBBox(key: item.bboxKey, corner: corner, by: delta)
                        }
                    )
                }
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
        }
    }

    private var claimedBoxes: [ClaimedItem] {
        var out: [ClaimedItem] = []
        let bboxes = draft.effectiveBBoxes
        if let b = bboxes["merchant.name"] {
            out.append(ClaimedItem(id: "merchant", bboxKey: "merchant.name", label: "Merchant", color: .blue, bbox: b))
        }
        if let b = bboxes["date.value"] {
            out.append(ClaimedItem(id: "date", bboxKey: "date.value", label: "Date", color: .green, bbox: b))
        }
        if let b = bboxes["totals.subtotal"] {
            out.append(ClaimedItem(id: "subtotal", bboxKey: "totals.subtotal", label: "Subtotal", color: .purple, bbox: b))
        }
        if let b = bboxes["totals.tax"] {
            out.append(ClaimedItem(id: "tax", bboxKey: "totals.tax", label: "Tax", color: .pink, bbox: b))
        }
        if let b = bboxes["totals.tip"] {
            out.append(ClaimedItem(id: "tip", bboxKey: "totals.tip", label: "Tip", color: .mint, bbox: b))
        }
        if let b = bboxes["totals.total"] {
            out.append(ClaimedItem(id: "total", bboxKey: "totals.total", label: "Total", color: .red, bbox: b))
        }
        // Two streams of line-item bboxes: description rows (yellow) and
        // price columns (orange). Distinct colors so a user scanning the
        // overlay can tell at a glance which is which.
        let itemBoxes = bboxes
            .filter { $0.key.hasPrefix("lineItem.") }
            .sorted { $0.key < $1.key }
        for (idx, kv) in itemBoxes.enumerated() {
            let isPrice = kv.key.hasSuffix(".price")
            out.append(ClaimedItem(
                id: "li-\(idx)",
                bboxKey: kv.key,
                label: isPrice ? "$" : "Item",
                color: isPrice ? .orange : .yellow,
                bbox: kv.value
            ))
        }
        return out
    }

    fileprivate struct ClaimedItem: Identifiable {
        let id: String
        let bboxKey: String
        let label: String
        let color: Color
        let bbox: OCRKit.Receipt.BBox
    }
}

// MARK: - Claimed-box view with selection, move, and corner-resize

private struct ClaimedBoxView: View {
    let item: BoundingBoxOverlay.ClaimedItem
    let canvasSize: CGSize
    let isSelected: Bool
    let onSelect: () -> Void
    /// Commit a move (translation) on drag-end. Delta is in normalized
    /// image coordinates.
    let onMoveEnd: (CGSize) -> Void
    /// Commit a corner-resize on drag-end. 0=TL, 1=TR, 2=BL, 3=BR.
    let onResizeEnd: (Int, CGSize) -> Void

    @State private var dragOffset: CGSize = .zero
    /// Which corner is being resized (nil = none, the whole body is the
    /// drag target for translate).
    @State private var resizingCorner: Int? = nil
    @State private var resizeOffset: CGSize = .zero

    var body: some View {
        let rect = liveRect()

        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(item.color, lineWidth: isSelected ? 3 : 2)
                .background(item.color.opacity(isSelected ? 0.18 : 0.10))
                .frame(width: rect.width, height: rect.height)
                // Body drag handles translation. Auto-selects on first
                // drag so users don't need to click-then-drag.
                .gesture(translateGesture)

            Text(item.label)
                .font(.system(size: 9, weight: .semibold))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(item.color)
                .foregroundStyle(.white)
                .cornerRadius(2)
                .offset(x: 0, y: -14)
                .allowsHitTesting(false)

            if isSelected {
                ForEach(0..<4, id: \.self) { corner in
                    cornerHandle(corner: corner, in: rect)
                }
            }
        }
        .offset(x: rect.minX, y: rect.minY)
        .help("\(item.label) — drag the body to move, drag a corner to resize")
    }

    // MARK: - Live rect (applies the in-flight drag/resize visually)

    private func liveRect() -> CGRect {
        let baseX = item.bbox.x * canvasSize.width
        let baseY = item.bbox.y * canvasSize.height
        let baseW = item.bbox.width * canvasSize.width
        let baseH = item.bbox.height * canvasSize.height

        // Apply translate
        var x = baseX + dragOffset.width
        var y = baseY + dragOffset.height
        var w = baseW
        var h = baseH

        // Apply resize for whichever corner is active
        if let corner = resizingCorner {
            let dx = resizeOffset.width
            let dy = resizeOffset.height
            switch corner {
            case 0:  // TL
                x = baseX + dx;  y = baseY + dy
                w = max(8, baseW - dx);  h = max(8, baseH - dy)
            case 1:  // TR
                y = baseY + dy
                w = max(8, baseW + dx);  h = max(8, baseH - dy)
            case 2:  // BL
                x = baseX + dx
                w = max(8, baseW - dx);  h = max(8, baseH + dy)
            case 3:  // BR
                w = max(8, baseW + dx);  h = max(8, baseH + dy)
            default: break
            }
        }
        return CGRect(x: x, y: y, width: w, height: h)
    }

    // MARK: - Body translate gesture

    private var translateGesture: some Gesture {
        // minimumDistance: 0 so the box tracks the cursor immediately
        // on mouse-down with no "dead zone". A click without any motion
        // fires onChanged once with translation = .zero, onEnded with
        // translation = .zero — the bbox stays put and onSelect fires,
        // so it doubles as a click-to-select.
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if !isSelected { onSelect() }
                dragOffset = value.translation
            }
            .onEnded { value in
                if !isSelected { onSelect() }
                let normalized = CGSize(
                    width: value.translation.width / canvasSize.width,
                    height: value.translation.height / canvasSize.height
                )
                onMoveEnd(normalized)
                dragOffset = .zero
            }
    }

    // MARK: - Corner handle

    @ViewBuilder
    private func cornerHandle(corner: Int, in rect: CGRect) -> some View {
        let pos: CGPoint = {
            switch corner {
            case 0: return CGPoint(x: 0, y: 0)
            case 1: return CGPoint(x: rect.width, y: 0)
            case 2: return CGPoint(x: 0, y: rect.height)
            case 3: return CGPoint(x: rect.width, y: rect.height)
            default: return .zero
            }
        }()
        Circle()
            .fill(item.color)
            .overlay(Circle().stroke(Color.white, lineWidth: 1.5))
            .frame(width: 14, height: 14)
            .offset(x: pos.x - 7, y: pos.y - 7)
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { value in
                        resizingCorner = corner
                        resizeOffset = value.translation
                    }
                    .onEnded { value in
                        let normalized = CGSize(
                            width: value.translation.width / canvasSize.width,
                            height: value.translation.height / canvasSize.height
                        )
                        onResizeEnd(corner, normalized)
                        resizingCorner = nil
                        resizeOffset = .zero
                    }
            )
            .help("Drag to resize")
    }
}

// MARK: - Per-OCR-line click button

private struct OCRLineButton: View {
    let line: OCRLine
    let size: CGSize
    let selectedKey: String?
    let onAssign: (LabelDraft.FieldTarget) -> Void
    let onAssignToSelected: () -> Void

    @State private var hovering = false
    @State private var showMenu = false

    var body: some View {
        let rect = CGRect(
            x: line.box.x * size.width,
            y: line.box.y * size.height,
            width: line.box.width * size.width,
            height: line.box.height * size.height
        )
        ZStack {
            Rectangle()
                .fill(hovering ? Color.accentColor.opacity(0.25) : Color.cyan.opacity(0.10))
            Rectangle()
                .strokeBorder(
                    hovering ? Color.accentColor : Color.cyan.opacity(0.75),
                    lineWidth: hovering ? 2.0 : 0.8
                )
        }
        .frame(width: rect.width, height: rect.height)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture {
            if selectedKey != nil {
                onAssignToSelected()
            } else {
                showMenu = true
            }
        }
        .help(selectedKey != nil ? "Click to set as \(selectedKey ?? "")\n\(line.text)" : "\(line.text) — click to assign")
        .popover(isPresented: $showMenu, arrowEdge: .bottom) {
            assignmentMenu
                .padding(8)
                .frame(minWidth: 220)
        }
        .offset(x: rect.minX, y: rect.minY)
    }

    private var assignmentMenu: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(line.text)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .padding(.bottom, 4)
            Divider()
            assignButton("Set as Merchant", systemImage: "building.2", color: .blue) { onAssign(.merchant) }
            assignButton("Set as Date", systemImage: "calendar", color: .green) { onAssign(.date) }
            assignButton("Set as Total", systemImage: "creditcard", color: .red) { onAssign(.total) }
            assignButton("Set as Subtotal", systemImage: "minus.circle", color: .purple) { onAssign(.subtotal) }
            assignButton("Set as Tax", systemImage: "percent", color: .pink) { onAssign(.tax) }
            assignButton("Set as Tip", systemImage: "dollarsign.arrow.circlepath", color: .mint) { onAssign(.tip) }
            Divider()
            assignButton("Add as line item", systemImage: "plus", color: .orange) { onAssign(.lineItem) }
        }
    }

    @ViewBuilder
    private func assignButton(_ title: String, systemImage: String, color: Color, action: @escaping () -> Void) -> some View {
        Button {
            action()
            showMenu = false
        } label: {
            HStack {
                Image(systemName: systemImage)
                    .foregroundStyle(color)
                    .frame(width: 16)
                Text(title)
                Spacer()
            }
            .padding(.vertical, 3)
            .padding(.horizontal, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
