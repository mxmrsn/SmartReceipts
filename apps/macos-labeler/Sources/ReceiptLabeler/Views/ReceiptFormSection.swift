import OCRKit
import SwiftUI

/// The editable form on the right of the LabelingView.
///
/// When the draft is sourced from a pipeline pre-label, the top banner
/// announces that explicitly and each field shows an "Auto" pill that flips
/// to "Edited" the moment the user touches it. The raw OCR text is available
/// as a disclosure group so the user can cross-check the extraction.
struct ReceiptFormSection: View {

    @Bindable var draft: LabelDraft
    let onSaveDraft: () -> Void
    let onVerify: () -> Void
    let onReject: () -> Void

    @State private var showRawOCR: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            if draft.isPreLabel {
                PreLabelBanner(pipelineId: draft.pipelineId)
            }
            Form {
                Section("Receipt") {
                    LabeledFieldRow(
                        label: "Merchant",
                        showBadge: draft.isPreLabel,
                        edited: draft.merchantWasEdited
                    ) {
                        TextField("Merchant", text: $draft.merchantName)
                            .textFieldStyle(.roundedBorder)
                    }
                    LabeledFieldRow(
                        label: "Date",
                        showBadge: draft.isPreLabel,
                        edited: draft.dateWasEdited
                    ) {
                        DatePicker("", selection: $draft.receiptDate, displayedComponents: .date)
                            .labelsHidden()
                    }
                    LabeledFieldRow(
                        label: "Currency",
                        showBadge: draft.isPreLabel,
                        edited: draft.currencyWasEdited
                    ) {
                        TextField("USD", text: $draft.currency)
                            .frame(width: 80)
                            .textFieldStyle(.roundedBorder)
                    }
                    LabeledFieldRow(
                        label: "Total",
                        showBadge: draft.isPreLabel,
                        edited: draft.totalWasEdited
                    ) {
                        TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(2)))
                            .frame(width: 120)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
                    }
                }

                if let rawText = draft.rawText, !rawText.isEmpty {
                    Section {
                        DisclosureGroup("Raw OCR text", isExpanded: $showRawOCR) {
                            ScrollView {
                                Text(rawText)
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(8)
                            }
                            .frame(maxHeight: 220)
                            .background(Color.gray.opacity(0.06))
                            .cornerRadius(4)
                        }
                    }
                }

                Section {
                    ForEach(draft.lineItems) { item in
                        LineItemRow(item: item, onRemove: {
                            if let idx = draft.lineItems.firstIndex(where: { $0.id == item.id }) {
                                draft.lineItems.remove(at: idx)
                            }
                        })
                    }
                    Button {
                        draft.lineItems.append(LineItemDraft(blank: ()))
                    } label: {
                        Label("Add line item", systemImage: "plus.circle")
                    }
                    .buttonStyle(.plain)
                } header: {
                    HStack(spacing: 6) {
                        Text("Line items")
                        if draft.isPreLabel {
                            EditBadge(edited: draft.lineItemsWereEdited)
                        }
                        Spacer()
                        Text("\(draft.lineItems.count)")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                }

                Section("Notes") {
                    TextEditor(text: $draft.notes)
                        .frame(minHeight: 60)
                        .font(.body)
                }
            }
            .formStyle(.grouped)

            Divider()
            actionBar
        }
    }

    private var actionBar: some View {
        HStack(spacing: 8) {
            Button(role: .destructive) { onReject() } label: {
                Label("Reject", systemImage: "xmark.circle")
            }
            Spacer()
            Button("Save Draft") { onSaveDraft() }
                .keyboardShortcut("s", modifiers: [.command, .shift])
            Button("Verify ✓") { onVerify() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(!draft.isSavable)
        }
        .padding(12)
        .background(.bar)
    }
}

// MARK: - Banner

private struct PreLabelBanner: View {
    let pipelineId: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "sparkles")
                .foregroundStyle(.tint)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text("Auto-detected by \(pipelineId)")
                    .font(.subheadline.weight(.semibold))
                Text("Review the fields below — “Auto” marks untouched, “Edited” marks your changes. ⌘⏎ Verify · ⇧⌘S Save Draft.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.10))
    }
}

// MARK: - Labeled field row with optional badge

private struct LabeledFieldRow<Content: View>: View {
    let label: String
    let showBadge: Bool
    let edited: Bool
    @ViewBuilder var content: () -> Content

    var body: some View {
        HStack {
            HStack(spacing: 6) {
                Text(label)
                if showBadge {
                    EditBadge(edited: edited)
                }
            }
            Spacer()
            content()
        }
    }
}

private struct EditBadge: View {
    let edited: Bool

    var body: some View {
        Text(edited ? "Edited" : "Auto")
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(edited ? Color.blue.opacity(0.18) : Color.gray.opacity(0.18))
            .foregroundStyle(edited ? Color.blue : Color.secondary)
            .cornerRadius(3)
    }
}

// MARK: - Line item row

private struct LineItemRow: View {
    @Bindable var item: LineItemDraft
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                TextField("Description", text: $item.itemDescription)
                    .textFieldStyle(.roundedBorder)
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red)
            }
            HStack(spacing: 6) {
                TextField(
                    "Qty",
                    value: Binding(
                        get: { item.quantity ?? 0 },
                        set: { item.quantity = $0 == 0 ? nil : $0 }
                    ),
                    format: .number.precision(.fractionLength(0...2))
                )
                .frame(width: 60)
                .textFieldStyle(.roundedBorder)
                Spacer()
                Text("$").foregroundStyle(.secondary)
                TextField("Total", value: $item.totalPrice, format: .number.precision(.fractionLength(2)))
                    .frame(width: 100)
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
            }
            .font(.callout)
        }
    }
}
