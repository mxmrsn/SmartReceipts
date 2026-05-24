import SwiftData
import SwiftUI

/// List of all saved receipts, newest first. Tap to view detail; swipe to delete.
struct LibraryView: View {

    @Environment(\.modelContext) private var modelContext
    @Query(sort: [SortDescriptor(\Receipt.receiptDate, order: .reverse), SortDescriptor(\Receipt.capturedAt, order: .reverse)])
    private var receipts: [Receipt]

    @State private var searchText: String = ""

    var body: some View {
        NavigationStack {
            Group {
                if filtered.isEmpty {
                    EmptyState(hasAnyReceipts: !receipts.isEmpty)
                } else {
                    List {
                        ForEach(filtered) { receipt in
                            NavigationLink(value: receipt) {
                                ReceiptRow(receipt: receipt)
                            }
                        }
                        .onDelete(perform: delete)
                    }
                }
            }
            .navigationTitle("Library")
            .searchable(text: $searchText, prompt: "Merchant")
            .navigationDestination(for: Receipt.self) { receipt in
                ReceiptDetailView(receipt: receipt)
            }
        }
    }

    private var filtered: [Receipt] {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return receipts }
        return receipts.filter { ($0.merchantName ?? "").localizedCaseInsensitiveContains(trimmed) }
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            let receipt = filtered[index]
            ReceiptImageStorage.delete(relativePath: receipt.imageRelativePath)
            modelContext.delete(receipt)
        }
        try? modelContext.save()
    }
}

// MARK: - Row

private struct ReceiptRow: View {
    let receipt: Receipt

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(receipt.merchantName ?? "Unknown merchant")
                    .font(.body.weight(.medium))
                    .lineLimit(1)
                Text(receipt.receiptDate ?? receipt.capturedAt, format: .dateTime.month(.abbreviated).day().year())
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(totalString)
                .font(.body.weight(.semibold))
                .monospacedDigit()
        }
    }

    private var totalString: String {
        guard let total = receipt.total else { return "—" }
        return total.formatted(.currency(code: receipt.currency))
    }
}

// MARK: - Empty state

private struct EmptyState: View {
    let hasAnyReceipts: Bool

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: hasAnyReceipts ? "magnifyingglass" : "tray")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.secondary)
            Text(hasAnyReceipts ? "No matches" : "No receipts yet")
                .font(.headline)
            if !hasAnyReceipts {
                Text("Scan one from the Capture tab to get started.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
