import OCRKit
import SwiftUI

/// Two-layer overlay on top of the receipt image:
///
/// 1. **All OCR observations** rendered as faint outlines. Vision knows the
///    exact pixel position of every line of text it recognized — this layer
///    surfaces all of them so the user can SEE what was detected.
/// 2. **Claimed fields** rendered as solid colored rectangles on top
///    (Merchant/blue, Date/green, Total/red, etc.) — pulled from
///    `draft.effectiveBBoxes` so they update live as the user reassigns.
///
/// Clicking any OCR line opens a Menu that lets you assign that line to a
/// field. This is more reliable than substring-matching extracted values
/// back to lines: Vision tells us EXACTLY where the text is; we just let
/// the user pick which line maps to which field.
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
                        assignedField: assignedField(for: line),
                        onAssign: { field in
                            draft.assign(line: line, to: field)
                        }
                    )
                }
                // Layer 2: solid colored rectangles for claimed fields
                ForEach(claimedBoxes) { item in
                    claimedView(item, in: geo.size)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }

    // MARK: - Claimed-bbox rendering

    @ViewBuilder
    private func claimedView(_ item: ClaimedItem, in size: CGSize) -> some View {
        let rect = CGRect(
            x: item.bbox.x * size.width,
            y: item.bbox.y * size.height,
            width: item.bbox.width * size.width,
            height: item.bbox.height * size.height
        )
        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(item.color, lineWidth: 2)
                .background(item.color.opacity(0.10))
                .frame(width: rect.width, height: rect.height)
                .allowsHitTesting(false)  // pass clicks through to the OCRLineButton beneath
            Text(item.label)
                .font(.system(size: 9, weight: .semibold))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(item.color)
                .foregroundStyle(.white)
                .cornerRadius(2)
                .offset(x: 0, y: -14)
                .allowsHitTesting(false)
        }
        .offset(x: rect.minX, y: rect.minY)
    }

    private var claimedBoxes: [ClaimedItem] {
        var out: [ClaimedItem] = []
        let bboxes = draft.effectiveBBoxes
        if let b = bboxes["merchant.name"] {
            out.append(ClaimedItem(id: "merchant", label: "Merchant", color: .blue, bbox: b))
        }
        if let b = bboxes["date.value"] {
            out.append(ClaimedItem(id: "date", label: "Date", color: .green, bbox: b))
        }
        if let b = bboxes["totals.subtotal"] {
            out.append(ClaimedItem(id: "subtotal", label: "Subtotal", color: .purple, bbox: b))
        }
        if let b = bboxes["totals.total"] {
            out.append(ClaimedItem(id: "total", label: "Total", color: .red, bbox: b))
        }
        let itemBoxes = bboxes
            .filter { $0.key.hasPrefix("lineItem.") }
            .sorted { $0.key < $1.key }
        for (idx, kv) in itemBoxes.enumerated() {
            out.append(ClaimedItem(id: "li-\(idx)", label: "Item", color: .yellow, bbox: kv.value))
        }
        return out
    }

    /// If `line` matches one of the claimed bboxes exactly, return the field
    /// kind so the click menu can show the current assignment.
    private func assignedField(for line: OCRLine) -> String? {
        for (key, box) in draft.effectiveBBoxes {
            if box == line.box {
                return key
            }
        }
        return nil
    }

    private struct ClaimedItem: Identifiable {
        let id: String
        let label: String
        let color: Color
        let bbox: OCRKit.Receipt.BBox
    }
}

// MARK: - Per-OCR-line click button

private struct OCRLineButton: View {
    let line: OCRLine
    let size: CGSize
    let assignedField: String?
    let onAssign: (LabelDraft.FieldTarget) -> Void

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
                .fill(hovering ? Color.accentColor.opacity(0.30) : Color.cyan.opacity(0.18))
            Rectangle()
                .strokeBorder(
                    hovering ? Color.accentColor : Color.cyan,
                    lineWidth: hovering ? 2.5 : 1.2
                )
        }
        .frame(width: rect.width, height: rect.height)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { showMenu = true }
        .help(line.text)
        .popover(isPresented: $showMenu, arrowEdge: .bottom) {
            assignmentMenu
                .padding(8)
                .frame(minWidth: 200)
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
