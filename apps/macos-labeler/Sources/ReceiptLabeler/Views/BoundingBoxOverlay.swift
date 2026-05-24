import OCRKit
import SwiftUI

/// Renders colored bounding boxes over the receipt image showing where the
/// pipeline detected each field. Coordinates come straight from the canonical
/// Receipt's provenance bboxes (normalized [0,1] image-space, origin top-left).
///
/// Read-only for now — drag-to-edit lands in a later pass. Hover any box to
/// see the field name + the OCR text it was sourced from.
struct BoundingBoxOverlay: View {

    let receipt: OCRKit.Receipt

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                ForEach(boxItems) { item in
                    boxView(item, in: geo.size)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .allowsHitTesting(false)  // pass clicks through to image controls
        }
    }

    // MARK: - Box rendering

    @ViewBuilder
    private func boxView(_ item: BoxItem, in size: CGSize) -> some View {
        let rect = CGRect(
            x: item.bbox.x * size.width,
            y: item.bbox.y * size.height,
            width: item.bbox.width * size.width,
            height: item.bbox.height * size.height
        )
        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(item.color, lineWidth: 2)
                .background(item.color.opacity(0.08))
                .frame(width: rect.width, height: rect.height)
            Text(item.label)
                .font(.system(size: 9, weight: .semibold))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(item.color)
                .foregroundStyle(.white)
                .cornerRadius(2)
                .offset(x: 0, y: -14)
        }
        .offset(x: rect.minX, y: rect.minY)
    }

    // MARK: - Aggregate which boxes to draw

    private var boxItems: [BoxItem] {
        var out: [BoxItem] = []
        let bboxes = receipt.provenance.bboxes
        if let b = bboxes["merchant.name"] {
            out.append(BoxItem(id: "merchant", label: "Merchant", color: .blue, bbox: b))
        }
        if let b = bboxes["date.value"] {
            out.append(BoxItem(id: "date", label: "Date", color: .green, bbox: b))
        }
        if let b = bboxes["totals.subtotal"] {
            out.append(BoxItem(id: "subtotal", label: "Subtotal", color: .purple, bbox: b))
        }
        if let b = bboxes["totals.total"] {
            out.append(BoxItem(id: "total", label: "Total", color: .red, bbox: b))
        }
        // Line item bboxes use keys like "lineItem.0", "lineItem.1", …
        let itemBoxes = bboxes
            .filter { $0.key.hasPrefix("lineItem.") }
            .sorted { $0.key < $1.key }
        for (idx, kv) in itemBoxes.enumerated() {
            out.append(BoxItem(id: "li-\(idx)", label: "Item", color: .yellow, bbox: kv.value))
        }
        return out
    }

    private struct BoxItem: Identifiable {
        let id: String
        let label: String
        let color: Color
        let bbox: OCRKit.Receipt.BBox
    }
}
