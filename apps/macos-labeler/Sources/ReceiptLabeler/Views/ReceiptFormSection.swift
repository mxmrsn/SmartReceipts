import OCRKit
import SwiftUI

/// The editable form on the right of the LabelingView.
///
/// Labels stack ABOVE inputs so every input gets the full column width.
/// Saved-vendor quick-pick sits next to the Merchant field. When the date
/// came from EXIF metadata (rather than the receipt text), a small badge
/// surfaces that on the Date row. Per-field confidence is rendered as a
/// colored percentage inside the Auto / Edited pill.
struct ReceiptFormSection: View {

    @Bindable var draft: LabelDraft
    let vendorStore: VendorStore
    /// True if a label file currently exists on disk for this image. Drives
    /// the wording / behaviour of the Discard button — discard reverts to
    /// disk when there is a saved version, falls back to re-extract otherwise.
    let hasSavedVersion: Bool
    let onSaveDraft: () -> Void
    let onVerify: () -> Void
    let onReject: () -> Void
    /// Re-run the OCR pipeline on this image, replacing the live draft.
    let onReExtract: () -> Void
    /// Throw away unsaved edits and revert to the on-disk version.
    let onDiscard: () -> Void

    @State private var showRawOCR: Bool = false
    @State private var newVendorAddedTick: Date = .distantPast
    @State private var showReExtractConfirm: Bool = false
    @State private var showDiscardConfirm: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                draftToolbar
                // Tells the user which field is currently selected for
                // drag-edit on the image. Doubles as the "drop target" hint
                // when they click a faint OCR line — it'll be assigned here.
                if let key = draft.selectedBBoxKey {
                    HStack(spacing: 6) {
                        Image(systemName: "scope")
                            .foregroundStyle(Color.accentColor)
                        Text("Editing: ")
                            .foregroundStyle(.secondary)
                        + Text(prettyName(for: key))
                            .font(.callout.weight(.semibold))
                        Spacer()
                        Text("drag the box on the image, or tap an OCR line")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Button {
                            draft.selectedBBoxKey = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.accentColor.opacity(0.15))
                    .cornerRadius(6)
                }
                if draft.isPreLabel {
                    PreLabelBanner(pipelineId: draft.pipelineId, confidence: draft.basis.provenance.confidence)
                }
                headerSection
                if let rawText = draft.rawText, !rawText.isEmpty {
                    rawOCRSection(rawText)
                }
                lineItemsSection
                notesSection
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                Divider()
                actionBar
            }
        }
        // When the user selects a field that doesn't have a bbox yet — common
        // for dates that came from EXIF metadata — create a default-sized
        // placeholder box at image center so they can immediately drag/resize
        // it into place.
        .onChange(of: draft.selectedBBoxKey) { _, newKey in
            if let key = newKey {
                draft.ensureBBox(for: key)
            }
        }
    }

    // MARK: - Draft toolbar (Re-extract / Discard)

    /// Compact action row that lives at the very top of the form. Two
    /// destructive-ish operations on the current draft:
    ///   - Re-extract: throws away the live draft and re-runs the OCR
    ///     pipeline on the source image. Useful after improving a pipeline
    ///     or when you want a fresh starting point.
    ///   - Discard:    reverts to the on-disk version (if any), throwing
    ///     away unsaved edits. Falls back to a re-extract when there's no
    ///     saved version yet.
    /// Both go through .confirmationDialog so an accidental click doesn't
    /// destroy work in progress.
    private var draftToolbar: some View {
        HStack(spacing: 12) {
            Button {
                showReExtractConfirm = true
            } label: {
                Label("Re-extract", systemImage: "arrow.clockwise.circle")
                    .font(.caption.weight(.medium))
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderless)
            .help("Re-run the OCR pipeline on this image. Discards current edits.")

            Button {
                showDiscardConfirm = true
            } label: {
                Label(
                    hasSavedVersion ? "Discard changes" : "Reset",
                    systemImage: "arrow.uturn.backward.circle"
                )
                .font(.caption.weight(.medium))
                .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderless)
            .help(
                hasSavedVersion
                    ? "Revert to the last version saved to disk."
                    : "No saved version yet — re-runs the OCR pipeline."
            )

            Spacer()
        }
        .confirmationDialog(
            "Re-extract this receipt?",
            isPresented: $showReExtractConfirm,
            titleVisibility: .visible
        ) {
            Button("Re-extract", role: .destructive) { onReExtract() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Re-runs the OCR pipeline against the image. Current edits and bounding-box adjustments will be lost.")
        }
        .confirmationDialog(
            hasSavedVersion ? "Discard current changes?" : "Reset to a fresh extraction?",
            isPresented: $showDiscardConfirm,
            titleVisibility: .visible
        ) {
            Button(hasSavedVersion ? "Discard" : "Reset", role: .destructive) {
                onDiscard()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                hasSavedVersion
                    ? "Reverts to the version saved on disk. Unsaved edits will be lost."
                    : "No saved version exists yet — this re-runs the OCR pipeline."
            )
        }
    }

    // MARK: - Header section

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Receipt")

            LabeledFieldRow(
                label: "Merchant",
                showBadge: draft.isPreLabel,
                edited: draft.merchantWasEdited,
                confidence: draft.basis.provenance.fieldConfidence["merchant.name"],
                bboxKey: "merchant.name",
                selectedKey: $draft.selectedBBoxKey
            ) {
                HStack(spacing: 6) {
                    TextField("Merchant name", text: $draft.merchantName)
                        .textFieldStyle(.roundedBorder)
                    vendorMenu
                }
            }

            LabeledFieldRow(
                label: "Date",
                showBadge: draft.isPreLabel,
                edited: draft.dateWasEdited,
                confidence: draft.basis.provenance.fieldConfidence["date.value"],
                bboxKey: "date.value",
                selectedKey: $draft.selectedBBoxKey,
                suffix: dateSourceTag
            ) {
                DatePicker("", selection: $draft.receiptDate, displayedComponents: .date)
                    .labelsHidden()
                    .datePickerStyle(.compact)
            }

            HStack(alignment: .top, spacing: 12) {
                LabeledFieldRow(
                    label: "Currency",
                    showBadge: draft.isPreLabel,
                    edited: draft.currencyWasEdited,
                    confidence: nil,
                    bboxKey: nil,
                    selectedKey: $draft.selectedBBoxKey
                ) {
                    TextField("USD", text: $draft.currency)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 90, alignment: .leading)
                }
                LabeledFieldRow(
                    label: "Subtotal",
                    showBadge: draft.isPreLabel,
                    edited: draft.subtotalWasEdited,
                    confidence: draft.basis.provenance.fieldConfidence["totals.subtotal"],
                    bboxKey: "totals.subtotal",
                    selectedKey: $draft.selectedBBoxKey
                ) {
                    TextField("0.00", value: optionalDecimalBinding(\.subtotal), format: .number.precision(.fractionLength(0...2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
            }
            HStack(alignment: .top, spacing: 12) {
                LabeledFieldRow(
                    label: "Tax",
                    showBadge: draft.isPreLabel,
                    edited: draft.taxWasEdited,
                    confidence: draft.basis.provenance.fieldConfidence["totals.tax"],
                    bboxKey: "totals.tax",
                    selectedKey: $draft.selectedBBoxKey
                ) {
                    TextField("0.00", value: optionalDecimalBinding(\.tax), format: .number.precision(.fractionLength(0...2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
                LabeledFieldRow(
                    label: "Tip",
                    showBadge: draft.isPreLabel,
                    edited: draft.tipWasEdited,
                    confidence: draft.basis.provenance.fieldConfidence["totals.tip"],
                    bboxKey: "totals.tip",
                    selectedKey: $draft.selectedBBoxKey
                ) {
                    TextField("0.00", value: optionalDecimalBinding(\.tip), format: .number.precision(.fractionLength(0...2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
                LabeledFieldRow(
                    label: "Total",
                    showBadge: draft.isPreLabel,
                    edited: draft.totalWasEdited,
                    confidence: draft.basis.provenance.fieldConfidence["totals.total"],
                    bboxKey: "totals.total",
                    selectedKey: $draft.selectedBBoxKey
                ) {
                    TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(0...2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
    }

    /// Bridge a TextField's non-optional `value:` binding to an optional
    /// Decimal property on `draft`. We treat a typed 0 as "user wants to
    /// keep 0 as the value" (some receipts legitimately have $0.00 tax);
    /// the empty-field state maps to nil via the formatter.
    private func optionalDecimalBinding(_ keyPath: ReferenceWritableKeyPath<LabelDraft, Decimal?>) -> Binding<Decimal> {
        Binding(
            get: { draft[keyPath: keyPath] ?? 0 },
            set: { draft[keyPath: keyPath] = $0 }
        )
    }

    // MARK: - Vendor picker

    private var vendorMenu: some View {
        Menu {
            if !vendorStore.topVendors.isEmpty {
                Section("Quick-pick") {
                    ForEach(vendorStore.topVendors.prefix(12)) { vendor in
                        Button(vendor.name) {
                            draft.merchantName = vendor.name
                            vendorStore.record(vendor.name)
                        }
                    }
                }
                Divider()
            }
            Button {
                if let v = vendorStore.record(draft.merchantName) {
                    newVendorAddedTick = Date()
                    _ = v
                }
            } label: {
                Label("Save “\(draft.merchantName.isEmpty ? "…" : draft.merchantName)”", systemImage: "plus")
            }
            .disabled(draft.merchantName.trimmingCharacters(in: .whitespaces).count < 2)

            if !vendorStore.vendors.isEmpty {
                Divider()
                Menu("Manage…") {
                    ForEach(vendorStore.topVendors) { vendor in
                        Button(role: .destructive) {
                            vendorStore.remove(vendor)
                        } label: {
                            Label("Delete “\(vendor.name)” (\(vendor.usageCount))", systemImage: "trash")
                        }
                    }
                }
            }
        } label: {
            Image(systemName: "building.2")
                .symbolRenderingMode(.hierarchical)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 26, height: 26)
        .help("Saved vendors")
    }

    private var dateSourceTag: AnyView? {
        guard let src = draft.dateSource, !src.isEmpty else { return nil }
        return AnyView(
            Text("from \(src.uppercased())")
                .font(.system(size: 9, weight: .medium))
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.orange.opacity(0.18))
                .foregroundStyle(.orange)
                .cornerRadius(3)
                .help("Date was not found on the receipt; filled in from \(src) metadata.")
        )
    }

    // MARK: - Raw OCR

    private func rawOCRSection(_ rawText: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            DisclosureGroup(isExpanded: $showRawOCR) {
                ScrollView {
                    Text(rawText)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                }
                .frame(maxHeight: 240)
                .background(Color.gray.opacity(0.06))
                .cornerRadius(4)
                .padding(.top, 6)
            } label: {
                HStack(spacing: 6) {
                    Text("Raw OCR text")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                    Text("(\(rawText.count) chars)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    // MARK: - Line items

    private var lineItemsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                sectionTitle("Line items")
                if draft.isPreLabel {
                    EditBadge(edited: draft.lineItemsWereEdited, confidence: nil)
                }
                Spacer()
                Text("\(draft.lineItems.count)")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            ForEach(Array(draft.lineItems.enumerated()), id: \.element.id) { idx, item in
                LineItemRow(
                    item: item,
                    index: idx,
                    selectedKey: $draft.selectedBBoxKey,
                    onRemove: {
                        if let i = draft.lineItems.firstIndex(where: { $0.id == item.id }) {
                            draft.lineItems.remove(at: i)
                        }
                    }
                )
                .padding(.vertical, 2)
            }

            Button {
                draft.lineItems.append(LineItemDraft(blank: ()))
            } label: {
                Label("Add line item", systemImage: "plus.circle")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.tint)
        }
    }

    // MARK: - Notes

    private var notesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionTitle("Notes")
            TextEditor(text: $draft.notes)
                .frame(minHeight: 60, maxHeight: 120)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(Color.gray.opacity(0.07))
                .cornerRadius(4)
        }
    }

    // MARK: - Action bar

    private var actionBar: some View {
        HStack(spacing: 10) {
            Button("Reject", role: .destructive) { onReject() }
            Spacer()
            Button("Save Draft") { onSaveDraft() }
                .keyboardShortcut("s", modifiers: [.command])
            Button("Verify ✓") { onVerify() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(!draft.isSavable)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
    }

    // MARK: - Helpers

    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }

    private func prettyName(for bboxKey: String) -> String {
        switch bboxKey {
        case "merchant.name":     return "Merchant"
        case "date.value":        return "Date"
        case "totals.total":      return "Total"
        case "totals.subtotal":   return "Subtotal"
        case "totals.tax":        return "Tax"
        case "totals.tip":        return "Tip"
        case let k where k.hasPrefix("lineItem."):
            // lineItem.005 → "Line item #5"
            // lineItem.005.price → "Line item #5 — price"
            let rest = k.dropFirst("lineItem.".count)
            let isPrice = rest.hasSuffix(".price")
            let idxPart = isPrice ? rest.dropLast(".price".count) : Substring(rest)
            let idxLabel = Int(idxPart).map { "#\($0 + 1)" } ?? ""
            return isPrice ? "Line item \(idxLabel) — price" : "Line item \(idxLabel)"
        default: return bboxKey
        }
    }
}

// MARK: - Banner

private struct PreLabelBanner: View {
    let pipelineId: String
    let confidence: Double

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
                .foregroundStyle(.tint)
            Text("Auto-detected by ")
                .foregroundStyle(.secondary)
            + Text(pipelineId)
                .font(.callout.weight(.semibold))
            Spacer()
            ConfidenceChip(value: confidence, label: "overall")
        }
        .font(.callout)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.10))
        .cornerRadius(6)
        .help("Each header field shows an Auto / Edited pill with confidence. ⌘⏎ Verify · ⇧⌘S Save Draft.")
    }
}

// MARK: - Labeled field row (label + badge + content)

private struct LabeledFieldRow<Content: View>: View {
    let label: String
    let showBadge: Bool
    let edited: Bool
    let confidence: Double?
    let bboxKey: String?
    @Binding var selectedKey: String?
    var suffix: AnyView? = nil
    @ViewBuilder var content: () -> Content

    private var isSelected: Bool {
        guard let bboxKey, let selectedKey else { return false }
        return bboxKey == selectedKey
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                    .textCase(.uppercase)
                if isSelected {
                    Image(systemName: "scope")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                }
                if showBadge {
                    EditBadge(edited: edited, confidence: confidence)
                }
                if let suffix { suffix }
                Spacer(minLength: 0)
            }
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.25) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .contentShape(Rectangle())
        // simultaneousGesture so the tap selects EVEN when the user clicks
        // directly on the TextField — without this, the TextField's own
        // tap-to-edit handler swallows the event and selection never fires.
        .simultaneousGesture(
            TapGesture()
                .onEnded { _ in
                    if let bboxKey {
                        selectedKey = (selectedKey == bboxKey) ? nil : bboxKey
                    }
                }
        )
    }
}

// MARK: - Edit badge with confidence

private struct EditBadge: View {
    let edited: Bool
    let confidence: Double?

    var body: some View {
        HStack(spacing: 4) {
            Text(edited ? "Edited" : "Auto")
                .font(.system(size: 9, weight: .semibold))
            if let confidence, !edited {
                Text("· \(Int((confidence * 100).rounded()))%")
                    .font(.system(size: 9, weight: .medium))
                    .monospacedDigit()
            }
        }
        .padding(.horizontal, 5)
        .padding(.vertical, 1)
        .background(badgeBackground)
        .foregroundStyle(badgeForeground)
        .cornerRadius(3)
    }

    private var badgeBackground: Color {
        if edited { return Color.blue.opacity(0.18) }
        guard let c = confidence else { return Color.gray.opacity(0.18) }
        if c >= 0.75 { return Color.green.opacity(0.18) }
        if c >= 0.50 { return Color.yellow.opacity(0.25) }
        return Color.red.opacity(0.22)
    }

    private var badgeForeground: Color {
        if edited { return Color.blue }
        guard let c = confidence else { return Color.secondary }
        if c >= 0.75 { return Color.green }
        if c >= 0.50 { return Color.orange }
        return Color.red
    }
}

// MARK: - Confidence chip (used in banner)

private struct ConfidenceChip: View {
    let value: Double
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text("\(Int((value * 100).rounded()))% \(label)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .help("Overall extraction confidence")
    }

    private var color: Color {
        if value >= 0.75 { return .green }
        if value >= 0.50 { return .orange }
        return .red
    }
}

// MARK: - Line item row
//
// Two distinct click-zones so the user can target either bbox independently:
//   - Description zone (left)  → key `lineItem.NNN`        (yellow box)
//   - Price zone (right total) → key `lineItem.NNN.price`  (orange box)
//
// Each zone uses the same simultaneousGesture pattern as `LabeledFieldRow`
// so taps still register when the user clicks directly on the TextField.

private struct LineItemRow: View {
    @Bindable var item: LineItemDraft
    let index: Int
    @Binding var selectedKey: String?
    let onRemove: () -> Void

    private var descKey: String { LabelDraft.lineItemBBoxKey(index: index, isPrice: false) }
    private var priceKey: String { LabelDraft.lineItemBBoxKey(index: index, isPrice: true) }

    private var descSelected: Bool { selectedKey == descKey }
    private var priceSelected: Bool { selectedKey == priceKey }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Description zone — selects `lineItem.NNN` (description bbox).
            HStack(spacing: 8) {
                LineItemZone(
                    isSelected: descSelected,
                    accent: .yellow,
                    icon: "text.alignleft",
                    selectionHint: "Item",
                    onTap: { toggle(descKey) }
                ) {
                    TextField("Description", text: $item.itemDescription)
                        .textFieldStyle(.roundedBorder)
                }
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "minus.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red.opacity(0.85))
            }
            // Qty + price row. Only the price area is a click-zone — qty
            // doesn't have its own bbox (yet).
            HStack(spacing: 8) {
                TextField(
                    "Qty",
                    value: Binding(
                        get: { item.quantity ?? 0 },
                        set: { item.quantity = $0 == 0 ? nil : $0 }
                    ),
                    format: .number.precision(.fractionLength(0...2))
                )
                .frame(maxWidth: 70)
                .textFieldStyle(.roundedBorder)
                Spacer()
                LineItemZone(
                    isSelected: priceSelected,
                    accent: .orange,
                    icon: "dollarsign.circle",
                    selectionHint: "Price",
                    onTap: { toggle(priceKey) }
                ) {
                    HStack(spacing: 4) {
                        Text("$").foregroundStyle(.secondary)
                        TextField("Total", value: $item.totalPrice, format: .number.precision(.fractionLength(0...2)))
                            .frame(maxWidth: 100)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }
            .font(.callout)
        }
        .padding(8)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(6)
    }

    private func toggle(_ key: String) {
        selectedKey = (selectedKey == key) ? nil : key
    }
}

/// One click-to-select sub-row inside a LineItemRow. Visually tracks the
/// selection ring + an accent-colored chip showing which bbox it targets.
private struct LineItemZone<Content: View>: View {
    let isSelected: Bool
    let accent: Color
    let icon: String
    let selectionHint: String
    let onTap: () -> Void
    @ViewBuilder var content: () -> Content

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(isSelected ? Color.accentColor : accent)
                .frame(width: 12)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
            if isSelected {
                Text(selectionHint)
                    .font(.system(size: 9, weight: .semibold))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.accentColor.opacity(0.18))
                    .foregroundStyle(Color.accentColor)
                    .cornerRadius(3)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 5)
                .fill(isSelected ? Color.accentColor.opacity(0.18) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 5)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 1.5)
        )
        .contentShape(Rectangle())
        // simultaneousGesture lets a tap on the inner TextField still
        // bubble up to select the zone (otherwise the TextField swallows it).
        .simultaneousGesture(
            TapGesture()
                .onEnded { _ in onTap() }
        )
    }
}
