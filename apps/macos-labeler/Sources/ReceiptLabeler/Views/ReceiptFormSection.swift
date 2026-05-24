import OCRKit
import SwiftUI

/// The editable form on the right of the LabelingView.
///
/// Labels stack ABOVE inputs so every input gets the full column width —
/// otherwise extracted values clip on narrow split panes.
///
/// When the draft is sourced from a pipeline pre-label, the top banner
/// announces that explicitly and each field shows an "Auto" pill that flips
/// to "Edited" the moment the user touches it. Raw OCR text is available as
/// a disclosure group so the user can cross-check the extraction.
struct ReceiptFormSection: View {

    @Bindable var draft: LabelDraft
    let onSaveDraft: () -> Void
    let onVerify: () -> Void
    let onReject: () -> Void

    @State private var showRawOCR: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if draft.isPreLabel {
                    PreLabelBanner(pipelineId: draft.pipelineId)
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

    // MARK: - Sections

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Receipt")

            LabeledFieldRow(label: "Merchant", showBadge: draft.isPreLabel, edited: draft.merchantWasEdited) {
                TextField("Merchant name", text: $draft.merchantName)
                    .textFieldStyle(.roundedBorder)
            }

            LabeledFieldRow(label: "Date", showBadge: draft.isPreLabel, edited: draft.dateWasEdited) {
                DatePicker("", selection: $draft.receiptDate, displayedComponents: .date)
                    .labelsHidden()
                    .datePickerStyle(.compact)
            }

            HStack(alignment: .top, spacing: 12) {
                LabeledFieldRow(label: "Currency", showBadge: draft.isPreLabel, edited: draft.currencyWasEdited) {
                    TextField("USD", text: $draft.currency)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 90, alignment: .leading)
                }
                LabeledFieldRow(label: "Total", showBadge: draft.isPreLabel, edited: draft.totalWasEdited) {
                    TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
    }

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
                .background(Color.gray.opacity(0.07))
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

    private var lineItemsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                sectionTitle("Line items")
                if draft.isPreLabel {
                    EditBadge(edited: draft.lineItemsWereEdited)
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

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
                .foregroundStyle(.tint)
            Text("Auto-detected by ")
                .foregroundStyle(.secondary)
            + Text(pipelineId)
                .font(.callout.weight(.semibold))
            Spacer()
            Text("review and modify")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .font(.callout)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.10))
        .cornerRadius(6)
        .help("Each header field shows an Auto / Edited pill. ⌘⏎ to Verify · ⇧⌘S to Save Draft.")
    }
}

// MARK: - Labeled field row (label above input)

private struct LabeledFieldRow<Content: View>: View {
    let label: String
    let showBadge: Bool
    let edited: Bool
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                if showBadge {
                    EditBadge(edited: edited)
                }
                Spacer(minLength: 0)
            }
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
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
