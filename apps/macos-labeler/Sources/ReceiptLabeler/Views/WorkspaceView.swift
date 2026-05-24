import SwiftUI

/// Top-level layout: NavigationSplitView with image grid on the left and
/// labeling workspace on the right. Status bar at the bottom shows progress.
struct WorkspaceView: View {

    @Bindable var controller: DatasetController

    var body: some View {
        NavigationSplitView {
            ImageGridView(controller: controller)
                .navigationSplitViewColumnWidth(min: 240, ideal: 320, max: 500)
        } detail: {
            detailPane
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    pickDatasetDirectory()
                } label: {
                    Label("Open folder…", systemImage: "folder")
                }
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    controller.reload()
                } label: {
                    Label("Reload", systemImage: "arrow.clockwise")
                }
                .keyboardShortcut("r", modifiers: [.command])
            }
        }
        .safeAreaInset(edge: .bottom) {
            StatusBar(message: controller.statusMessage, isBusy: controller.isExtracting)
        }
        .navigationTitle("Receipt Labeler")
        .navigationSubtitle(controller.datasetDirectory.lastPathComponent)
        .onAppear { controller.reload() }
    }

    @ViewBuilder
    private var detailPane: some View {
        if let entry = controller.selectedEntry {
            LabelingView(controller: controller, entry: entry)
                .id(entry.imageId)
        } else {
            EmptyDetail(hasEntries: !controller.entries.isEmpty)
        }
    }

    private func pickDatasetDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = controller.datasetDirectory
        panel.title = "Select a dataset folder (containing images/ and labels/)"
        if panel.runModal() == .OK, let url = panel.url {
            controller.datasetDirectory = url
            controller.reload()
        }
    }
}

// MARK: - Status bar

private struct StatusBar: View {
    let message: String?
    let isBusy: Bool

    var body: some View {
        HStack(spacing: 8) {
            if isBusy {
                ProgressView().controlSize(.small)
            }
            Text(message ?? "Ready")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.bar)
    }
}

// MARK: - Empty detail

private struct EmptyDetail: View {
    let hasEntries: Bool

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: hasEntries ? "doc.text.viewfinder" : "tray")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.secondary)
            Text(hasEntries ? "Select a receipt" : "No images found")
                .font(.headline)
            if !hasEntries {
                Text("Drop receipts into the dataset's images/ folder, then click Reload.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
