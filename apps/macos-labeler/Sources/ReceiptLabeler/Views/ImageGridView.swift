import SwiftUI

/// Sidebar: scrollable list of dataset entries. Each row shows a small
/// thumbnail, the source filename, and a status pill. The top bar holds a
/// single Menu dropdown for status filtering — replaces the row of chips
/// which couldn't fit in a narrow sidebar.
struct ImageGridView: View {

    @Bindable var controller: DatasetController
    @State private var statusFilter: LabelStatus? = nil

    var body: some View {
        VStack(spacing: 0) {
            filterBar
            Divider()
            List(selection: bindingSelection) {
                ForEach(filtered) { entry in
                    EntryRow(entry: entry, lastSaveEvent: controller.lastSaveEvent)
                        .tag(entry.imageId)
                }
            }
            .listStyle(.sidebar)
        }
    }

    // MARK: - Filter bar

    private var filterBar: some View {
        HStack(spacing: 8) {
            Menu {
                Button {
                    statusFilter = nil
                } label: {
                    Label("All  (\(controller.entries.count))", systemImage: statusFilter == nil ? "checkmark" : "")
                }
                Divider()
                ForEach(LabelStatus.allCases, id: \.self) { status in
                    Button {
                        statusFilter = status
                    } label: {
                        let count = controller.entries.filter { $0.status == status }.count
                        Label("\(status.displayName)  (\(count))", systemImage: statusFilter == status ? "checkmark" : "")
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                    Text(activeFilterLabel)
                        .lineLimit(1)
                }
                .font(.caption.weight(.medium))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            Spacer(minLength: 4)

            Text("\(filtered.count) / \(controller.entries.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var activeFilterLabel: String {
        statusFilter?.displayName ?? "All"
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
    let lastSaveEvent: DatasetController.SaveEvent?

    /// Opacity of the pulse ring overlay. Set to 1 instantly on save, then
    /// animated back to 0 over 700ms.
    @State private var pulseOpacity: Double = 0
    /// Tint of the pulse ring — green/red/blue depending on the save status
    /// so the user can tell at a glance which action they just performed.
    @State private var pulseColor: Color = .accentColor

    var body: some View {
        HStack(spacing: 8) {
            Thumbnail(url: entry.imageURL)
                .frame(width: 44, height: 44)
                .background(Color(nsColor: .quaternaryLabelColor))
                .cornerRadius(3)

            VStack(alignment: .leading, spacing: 2) {
                Text(entry.sourceFilename)
                    .font(.system(.callout, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                StatusBadge(status: entry.status)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 1)
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .stroke(pulseColor, lineWidth: 2.5)
                .opacity(pulseOpacity)
                .allowsHitTesting(false)
        }
        // When the controller bumps lastSaveEvent and it points at this
        // row, snap the ring to fully visible, then fade it out. The
        // double dispatch is intentional: it forces SwiftUI to render
        // the opacity=1 state before starting the fade, so the animation
        // doesn't get collapsed away in a single update cycle.
        .onChange(of: lastSaveEvent) { _, new in
            guard let new, new.imageID == entry.imageId else { return }
            pulseColor = pulseTint(for: new.status)
            pulseOpacity = 1.0
            DispatchQueue.main.async {
                withAnimation(.easeOut(duration: 0.7)) {
                    pulseOpacity = 0.0
                }
            }
        }
    }

    private func pulseTint(for status: LabelStatus) -> Color {
        switch status {
        case .draft:     .accentColor
        case .verified:  .green
        case .rejected:  .red
        case .unlabeled: .gray
        }
    }
}

private struct Thumbnail: View {
    let url: URL

    var body: some View {
        if let img = ImageLoader.shared.thumbnail(for: url, pixelSize: 44) {
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
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
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
