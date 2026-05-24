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

    /// Full ExtractionResult from the most recent pre-label run. The form
    /// reads both `.receipt` (for field values) and `.rawText` (for the raw OCR
    /// section so the user can verify what the OCR actually saw).
    var pendingExtraction: OCRKit.ExtractionResult?
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
        pendingExtraction = nil
    }

    var selectedEntry: DatasetEntry? {
        guard let selectedID else { return nil }
        return entries.first(where: { $0.imageId == selectedID })
    }

    // MARK: - Pre-labeling

    /// Run the best available pipeline over the selected entry's image and
    /// stash the result for the form to merge in. If the preferred pipeline
    /// (Foundation Models when Apple Intelligence is enabled) fails, fall
    /// back to the Vision+regex baseline so the user is never stuck.
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

        let preferred = OCRPipelineRegistry.preferred
        let fallback = VisionOnlyPipeline()

        do {
            let result = try await preferred.extract(image: cgImage)
            stashResult(result, for: entry, pipelineDisplayName: type(of: preferred).displayName)
            return
        } catch {
            // If preferred IS the fallback, no point retrying.
            if type(of: preferred).id == VisionOnlyPipeline.id {
                statusMessage = "Pre-label failed: \(error.localizedDescription)"
                return
            }
            // Try the fallback. Tell the user we degraded.
            statusMessage = "\(type(of: preferred).displayName) unavailable (\(error.localizedDescription)) — falling back…"
            do {
                let result = try await fallback.extract(image: cgImage)
                stashResult(result, for: entry, pipelineDisplayName: "\(VisionOnlyPipeline.displayName) (fallback)")
            } catch {
                statusMessage = "Both pipelines failed: \(error.localizedDescription)"
            }
        }
    }

    private func stashResult(_ result: OCRKit.ExtractionResult, for entry: DatasetEntry, pipelineDisplayName: String) {
        var canonical = result.receipt
        canonical.imageId = entry.imageId  // anchor to the dataset-stable id
        pendingExtraction = OCRKit.ExtractionResult(
            receipt: canonical,
            latencyMs: result.latencyMs,
            peakMemoryMB: result.peakMemoryMB,
            rawText: result.rawText
        )
        statusMessage = "Pre-label by \(pipelineDisplayName) — \(result.latencyMs) ms · \(entry.sourceFilename)"
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
