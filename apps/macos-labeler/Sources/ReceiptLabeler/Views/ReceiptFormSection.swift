import OCRKit
import SwiftUI

/// The editable form on the right of the LabelingView. Bound to a LabelDraft;
/// emits the three save actions (Save Draft / Verify / Reject) as callbacks.
struct ReceiptFormSection: View {

    @Bindable var draft: LabelDraft
    let onSaveDraft: () -> Void
    let onVerify: () -> Void
    let onReject: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Receipt") {
                    TextField("Merchant", text: $draft.merchantName)
                        .textFieldStyle(.roundedBorder)
                    DatePicker("Date", selection: $draft.receiptDate, displayedComponents: .date)
                    HStack {
                        Text("Currency")
                        Spacer()
                        TextField("USD", text: $draft.currency)
                            .frame(width: 80)
                            .textFieldStyle(.roundedBorder)
                    }
                    HStack {
                        Text("Total")
                        Spacer()
                        TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(2)))
                            .frame(width: 120)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
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
                    HStack {
                        Text("Line items")
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
                TextField("Qty",
                          value: Binding(
                              get: { item.quantity ?? 0 },
                              set: { item.quantity = $0 == 0 ? nil : $0 }
                          ),
                          format: .number.precision(.fractionLength(0...2)))
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
