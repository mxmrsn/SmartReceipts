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
    let onSaveDraft: () -> Void
    let onVerify: () -> Void
    let onReject: () -> Void

    @State private var showRawOCR: Bool = false
    @State private var newVendorAddedTick: Date = .distantPast

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
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
    }

    // MARK: - Header section

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Receipt")

            LabeledFieldRow(
                label: "Merchant",
                showBadge: draft.isPreLabel,
                edited: draft.merchantWasEdited,
                confidence: draft.basis.provenance.fieldConfidence["merchant.name"]
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
                    confidence: nil
                ) {
                    TextField("USD", text: $draft.currency)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 90, alignment: .leading)
                }
                LabeledFieldRow(
                    label: "Total",
                    showBadge: draft.isPreLabel,
                    edited: draft.totalWasEdited,
                    confidence: draft.basis.provenance.fieldConfidence["totals.total"]
                ) {
                    TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
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

            ForEach(draft.lineItems) { item in
                LineItemRow(item: item, onRemove: {
                    if let idx = draft.lineItems.firstIndex(where: { $0.id == item.id }) {
                        draft.lineItems.remove(at: idx)
                    }
                })
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
                .keyboardShortcut("s", modifiers: [.command, .shift])
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
    var suffix: AnyView? = nil
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                if showBadge {
                    EditBadge(edited: edited, confidence: confidence)
                }
                if let suffix { suffix }
                Spacer(minLength: 0)
            }
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
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

private struct LineItemRow: View {
    @Bindable var item: LineItemDraft
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                TextField("Description", text: $item.itemDescription)
                    .textFieldStyle(.roundedBorder)
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "minus.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red.opacity(0.85))
            }
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
                Text("$")
                    .foregroundStyle(.secondary)
                TextField("Total", value: $item.totalPrice, format: .number.precision(.fractionLength(2)))
                    .frame(maxWidth: 120)
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
            }
            .font(.callout)
        }
        .padding(8)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(6)
    }
}
