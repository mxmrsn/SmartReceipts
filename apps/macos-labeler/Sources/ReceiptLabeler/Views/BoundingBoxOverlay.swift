import OCRKit
import SwiftUI

/// Two-layer overlay on top of the receipt image:
///
/// 1. **All OCR observations** rendered as faint outlines — click any to
///    open a popover to assign it to a field, OR if a field is currently
///    selected, click to assign directly without the popover.
/// 2. **Claimed fields** rendered as solid colored rectangles. The one
///    matching `draft.selectedBBoxKey` gets a highlighted ring and a
///    drag-gesture handler — drag it to move the bbox to a new location.
struct BoundingBoxOverlay: View {

    @Bindable var draft: LabelDraft

    var body: some View {
        // GeometryReader inside an `.overlay()` of an Image reports the
        // image's actual display size — required for normalized coords to
        // map cleanly onto the visible image.
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
                            guard let key = draft.selectedBBoxKey,
                                  let field = LabelDraft.FieldTarget.allCases.first(where: { $0.bboxKey == key })
                            else { return }
                            draft.assign(line: line, to: field)
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
                        onDragEnd: { delta in
                            draft.translateBBox(key: item.bboxKey, by: delta)
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
        if let b = bboxes["totals.total"] {
            out.append(ClaimedItem(id: "total", bboxKey: "totals.total", label: "Total", color: .red, bbox: b))
        }
        let itemBoxes = bboxes
            .filter { $0.key.hasPrefix("lineItem.") }
            .sorted { $0.key < $1.key }
        for (idx, kv) in itemBoxes.enumerated() {
            out.append(ClaimedItem(id: "li-\(idx)", bboxKey: kv.key, label: "Item", color: .yellow, bbox: kv.value))
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

// MARK: - Claimed-box view with selection + drag

private struct ClaimedBoxView: View {
    let item: BoundingBoxOverlay.ClaimedItem
    let canvasSize: CGSize
    let isSelected: Bool
    let onSelect: () -> Void
    let onDragEnd: (CGSize) -> Void

    @State private var dragOffset: CGSize = .zero

    var body: some View {
        let rect = CGRect(
            x: item.bbox.x * canvasSize.width,
            y: item.bbox.y * canvasSize.height,
            width: item.bbox.width * canvasSize.width,
            height: item.bbox.height * canvasSize.height
        )

        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(item.color, lineWidth: isSelected ? 3 : 2)
                .background(item.color.opacity(isSelected ? 0.18 : 0.10))
                .frame(width: rect.width, height: rect.height)
            // Corner pips on the selected box give it a "draggable handle" feel.
            if isSelected {
                ForEach(0..<4, id: \.self) { corner in
                    Circle()
                        .fill(item.color)
                        .frame(width: 8, height: 8)
                        .offset(
                            x: corner % 2 == 0 ? -4 : (rect.width - 4),
                            y: corner < 2 ? -4 : (rect.height - 4)
                        )
                }
            }
            Text(item.label)
                .font(.system(size: 9, weight: .semibold))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(item.color)
                .foregroundStyle(.white)
                .cornerRadius(2)
                .offset(x: 0, y: -14)
        }
        .offset(x: rect.minX + dragOffset.width, y: rect.minY + dragOffset.height)
        .contentShape(Rectangle())
        // Tap selects; drag (only when already selected) moves. exclusively
        // ensures a stationary click never accidentally triggers a drag and
        // a real drag never falsely fires as a tap.
        .gesture(
            TapGesture()
                .onEnded { _ in onSelect() }
                .exclusively(before:
                    DragGesture(minimumDistance: 4)
                        .onChanged { value in
                            guard isSelected else { return }
                            dragOffset = value.translation
                        }
                        .onEnded { value in
                            guard isSelected else {
                                dragOffset = .zero
                                return
                            }
                            let normalized = CGSize(
                                width: value.translation.width / canvasSize.width,
                                height: value.translation.height / canvasSize.height
                            )
                            onDragEnd(normalized)
                            dragOffset = .zero
                        }
                )
        )
        .help("\(item.label): tap to select, drag to move")
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
            // If a field is selected, single-click assigns directly. Else show menu.
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
