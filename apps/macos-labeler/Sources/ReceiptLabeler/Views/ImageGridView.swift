import SwiftUI

/// Sidebar: scrollable list of dataset entries. Each row shows a thumbnail,
/// the source filename, and a status pill. Selection drives the LabelingView.
struct ImageGridView: View {

    @Bindable var controller: DatasetController
    @State private var statusFilter: LabelStatus? = nil

    var body: some View {
        VStack(spacing: 0) {
            filterBar
            Divider()
            List(selection: bindingSelection) {
                ForEach(filtered) { entry in
                    EntryRow(entry: entry)
                        .tag(entry.imageId)
                }
            }
            .listStyle(.sidebar)
        }
    }

    // MARK: - Filter

    private var filterBar: some View {
        HStack(spacing: 8) {
            FilterChip(label: "All", count: controller.entries.count, isSelected: statusFilter == nil) {
                statusFilter = nil
            }
            ForEach(LabelStatus.allCases, id: \.self) { status in
                FilterChip(
                    label: status.displayName,
                    count: controller.entries.filter { $0.status == status }.count,
                    isSelected: statusFilter == status
                ) {
                    statusFilter = (statusFilter == status) ? nil : status
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var filtered: [DatasetEntry] {
        guard let statusFilter else { return controller.entries }
        return controller.entries.filter { $0.status == statusFilter }
    }

    private var bindingSelection: Binding<UUID?> {
        Binding(
            get: { controller.selectedID },
            // Defer the controller mutation past the current List update cycle
            // to avoid the "reentrant operation in NSTableView delegate" warning.
            set: { newValue in
                Task { @MainActor in
                    controller.select(newValue)
                }
            }
        )
    }
}

// MARK: - Row

private struct EntryRow: View {
    let entry: DatasetEntry

    var body: some View {
        HStack(spacing: 8) {
            Thumbnail(url: entry.imageURL)
                .frame(width: 56, height: 56)
                .background(Color(nsColor: .quaternaryLabelColor))
                .cornerRadius(4)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.sourceFilename)
                    .font(.system(.body, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                StatusBadge(status: entry.status)
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }
}

private struct Thumbnail: View {
    let url: URL

    var body: some View {
        if let img = ImageLoader.shared.thumbnail(for: url, pixelSize: 56) {
            Image(nsImage: img)
                .resizable()
                .aspectRatio(contentMode: .fit)
        } else {
            Image(systemName: "doc")
                .foregroundStyle(.secondary)
        }
    }
}

private struct StatusBadge: View {
    let status: LabelStatus

    var body: some View {
        Text(status.displayName)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(background)
            .foregroundStyle(foreground)
            .cornerRadius(3)
    }

    private var background: Color {
        switch status {
        case .unlabeled: Color.gray.opacity(0.2)
        case .draft:     Color.blue.opacity(0.2)
        case .verified:  Color.green.opacity(0.2)
        case .rejected:  Color.red.opacity(0.2)
        }
    }

    private var foreground: Color {
        switch status {
        case .unlabeled: .secondary
        case .draft:     .blue
        case .verified:  .green
        case .rejected:  .red
        }
    }
}

// MARK: - Filter chip

private struct FilterChip: View {
    let label: String
    let count: Int
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(label).font(.caption.weight(.medium))
                Text("\(count)").font(.caption2).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(isSelected ? Color.accentColor.opacity(0.25) : Color.gray.opacity(0.12))
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}
