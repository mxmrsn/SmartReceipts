import AppKit
import Foundation
import OCRKit
import Observation

/// Top-level state for the labeler:
/// - which dataset directory we're pointed at
/// - the discovered entries (image + label status)
/// - the currently-selected entry
/// - pre-labeling: kicks off VisionOnlyPipeline for newly-opened images
@Observable
@MainActor
final class DatasetController {

    var datasetDirectory: URL
    var entries: [DatasetEntry] = []
    var selectedID: UUID?
    var statusMessage: String?

    /// Pre-label draft fetched from OCRKit when the user selects an unlabeled image.
    var pendingDraft: OCRKit.Receipt?
    var isExtracting: Bool = false

    var store: LabelStore { LabelStore(datasetDirectory: datasetDirectory) }

    init(datasetDirectory: URL) {
        self.datasetDirectory = datasetDirectory
    }

    // MARK: - Loading

    func reload() {
        do {
            try store.ensureDirectoriesExist()
        } catch {
            statusMessage = "Could not access dataset: \(error.localizedDescription)"
            entries = []
            return
        }

        let urls = store.discoverImages()
        let next: [DatasetEntry] = urls.map { url in
            let id = ImageIDGenerator.uuid(forURL: url)
            let loaded = try? store.load(imageId: id)
            let status: LabelStatus = loaded?.label.status ?? .unlabeled
            return DatasetEntry(
                imageId: id,
                imageURL: url,
                status: status,
                label: loaded
            )
        }
        entries = next
        statusMessage = "\(next.count) images · \(next.filter { $0.status == .verified }.count) verified"

        if let selectedID, !next.contains(where: { $0.imageId == selectedID }) {
            self.selectedID = nil
        }
    }

    // MARK: - Selection

    func select(_ id: UUID?) {
        selectedID = id
        pendingDraft = nil
    }

    var selectedEntry: DatasetEntry? {
        guard let selectedID else { return nil }
        return entries.first(where: { $0.imageId == selectedID })
    }

    // MARK: - Pre-labeling

    /// Run the registered baseline pipeline (VisionOnlyPipeline) over the
    /// selected entry's image and stash the result as a `pendingDraft` for
    /// the form to merge in.
    func runPreLabel(for entry: DatasetEntry) async {
        guard !isExtracting else { return }
        isExtracting = true
        defer { isExtracting = false }

        guard let nsImage = NSImage(contentsOf: entry.imageURL),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else {
            statusMessage = "Could not load image: \(entry.sourceFilename)"
            return
        }

        do {
            let pipeline = VisionOnlyPipeline()
            let result = try await pipeline.extract(image: cgImage)
            var canonical = result.receipt
            canonical.imageId = entry.imageId  // anchor to the dataset-stable id
            pendingDraft = canonical
            statusMessage = "Pre-label generated for \(entry.sourceFilename) in \(result.latencyMs) ms"
        } catch {
            statusMessage = "Pre-label failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Saving

    func save(_ document: LabelDocument, for entry: DatasetEntry) {
        var doc = document
        if doc.label.sourceFilename == nil {
            doc.label.sourceFilename = entry.sourceFilename
        }
        do {
            try store.save(doc, imageId: entry.imageId)
            statusMessage = "Saved \(doc.label.status.displayName) — \(entry.sourceFilename)"
            reload()
            select(entry.imageId)  // keep selection on the same entry
        } catch {
            statusMessage = "Save failed: \(error.localizedDescription)"
        }
    }
}
