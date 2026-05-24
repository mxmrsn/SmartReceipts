import OCRKit
import SwiftUI
import UIKit

/// Read-only detail view for a saved Receipt. Phase 1: no edit; that lands in M7.
struct ReceiptDetailView: View {
    let receipt: Receipt

    var body: some View {
        List {
            Section {
                if let image = ReceiptImageStorage.load(relativePath: receipt.imageRelativePath) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(8)
                        .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                } else {
                    Text("Image unavailable")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Receipt") {
                LabeledContent("Merchant", value: receipt.merchantName ?? "—")
                if let d = receipt.receiptDate {
                    LabeledContent("Date") {
                        Text(d, format: .dateTime.year().month(.abbreviated).day())
                    }
                }
                if let t = receipt.total {
                    LabeledContent("Total") {
                        Text(t, format: .currency(code: receipt.currency))
                            .monospacedDigit()
                    }
                }
            }

            if !receipt.lineItems.isEmpty {
                Section("Line items") {
                    ForEach(receipt.lineItems) { item in
                        HStack {
                            Text(item.itemDescription)
                            Spacer()
                            Text(item.totalPrice, format: .currency(code: receipt.currency))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Section("Provenance") {
                LabeledContent("Pipeline", value: receipt.pipelineId)
                LabeledContent("Confidence") {
                    Text(receipt.overallConfidence.formatted(.percent.precision(.fractionLength(0))))
                }
                LabeledContent("Captured") {
                    Text(receipt.capturedAt, format: .dateTime.year().month().day().hour().minute())
                }
            }
        }
        .navigationTitle(receipt.merchantName ?? "Receipt")
        .navigationBarTitleDisplayMode(.inline)
    }
}
