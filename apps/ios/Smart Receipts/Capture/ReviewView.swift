import OCRKit
import SwiftUI
import UIKit

/// Editable review form shown after OCR. Pre-filled from the canonical
/// `ExtractionResult`; user can fix any field before saving. Low-confidence
/// fields are visually flagged.
struct ReviewView: View {

    let image: UIImage
    let extraction: OCRKit.ExtractionResult
    let onCancel: () -> Void
    let onSave: (ReceiptDraft) -> Void

    @State private var draft: ReceiptDraft

    init(
        image: UIImage,
        extraction: OCRKit.ExtractionResult,
        onCancel: @escaping () -> Void,
        onSave: @escaping (ReceiptDraft) -> Void
    ) {
        self.image = image
        self.extraction = extraction
        self.onCancel = onCancel
        self.onSave = onSave
        _draft = State(initialValue: ReceiptDraft(from: extraction.receipt))
    }

    var body: some View {
        Form {
            Section {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 200)
                    .cornerRadius(8)
                    .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
            }

            Section("Receipt") {
                LowConfidenceField(
                    label: "Merchant",
                    confidence: draft.fieldConfidence["merchant.name"]
                ) {
                    TextField("Merchant name", text: $draft.merchantName)
                        .textInputAutocapitalization(.words)
                }

                LowConfidenceField(
                    label: "Date",
                    confidence: draft.fieldConfidence["date.value"]
                ) {
                    DatePicker("", selection: $draft.receiptDate, displayedComponents: .date)
                        .labelsHidden()
                }

                LowConfidenceField(
                    label: "Total",
                    confidence: draft.fieldConfidence["totals.total"]
                ) {
                    HStack(spacing: 4) {
                        Text(currencySymbol(draft.currency))
                            .foregroundStyle(.secondary)
                        TextField("0.00", value: $draft.total, format: .number.precision(.fractionLength(2)))
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }

            if !draft.lineItems.isEmpty {
                Section("Line items") {
                    ForEach($draft.lineItems) { $item in
                        LineItemRow(item: $item, currency: draft.currency)
                    }
                    .onDelete { offsets in
                        draft.lineItems.remove(atOffsets: offsets)
                    }
                }
            }

            Section {
                LabeledContent("Pipeline") {
                    Text(extraction.receipt.provenance.pipelineId)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Latency") {
                    Text("\(extraction.latencyMs) ms")
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Confidence") {
                    Text(extraction.receipt.provenance.confidence.formatted(.percent.precision(.fractionLength(0))))
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Provenance")
            } footer: {
                Text("Fields flagged in yellow had low extraction confidence. Verify before saving.")
            }
        }
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel", role: .cancel) { onCancel() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") { onSave(draft) }
                    .disabled(!draft.isSavable)
            }
        }
    }

    private func currencySymbol(_ code: String) -> String {
        Locale.current.localizedCurrencySymbol(forCurrencyCode: code) ?? code
    }
}

// MARK: - Line item row

private struct LineItemRow: View {
    @Binding var item: LineItemDraft
    let currency: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            TextField("Description", text: $item.itemDescription)
            HStack {
                if let q = Binding($item.quantity) {
                    TextField("Qty", value: q, format: .number)
                        .keyboardType(.decimalPad)
                        .frame(width: 60)
                }
                Spacer()
                TextField("Total", value: $item.totalPrice, format: .number.precision(.fractionLength(2)))
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Confidence-aware field wrapper

private struct LowConfidenceField<Content: View>: View {
    let label: String
    let confidence: Double?
    @ViewBuilder var content: () -> Content

    private var isLow: Bool { (confidence ?? 1) < 0.5 }

    var body: some View {
        HStack {
            HStack(spacing: 6) {
                Text(label)
                if isLow {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                        .imageScale(.small)
                        .accessibilityLabel("Low confidence")
                }
            }
            Spacer()
            content()
        }
    }
}

// MARK: - Locale helper

extension Locale {
    fileprivate func localizedCurrencySymbol(forCurrencyCode code: String) -> String? {
        let id = Locale.Components(languageCode: language.languageCode, languageRegion: region)
        var components = id
        components.currency = .init(code)
        let locale = Locale(components: components)
        return locale.currencySymbol
    }
}
