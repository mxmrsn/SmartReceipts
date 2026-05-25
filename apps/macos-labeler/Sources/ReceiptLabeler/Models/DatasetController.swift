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
    var pendingDateSource: ImageMetadata.DateSource?
    /// When set, the preferred pipeline (e.g. vision-fm) failed and we
    /// fell back. The form surfaces this in the banner with a tooltip
    /// showing the error for debugging FM unreliability.
    var pendingPreferredError: String?
    var isExtracting: Bool = false

    /// Bumped every time `save(_:for:)` writes a label successfully. The
    /// sidebar's EntryRow watches this and pulses a colored outline around
    /// the row whose `imageID` matches — handy visual confirmation when
    /// saving via ⌘S, since the action otherwise has no on-screen tell.
    /// Includes `at: Date()` so back-to-back saves of the same row still
    /// register as distinct events (Equatable on identity, not just id).
    var lastSaveEvent: SaveEvent? = nil

    struct SaveEvent: Equatable, Sendable {
        let imageID: UUID
        let status: LabelStatus
        let at: Date
    }

    /// Saved-vendor directory used by the LabelingView's quick-pick menu.
    let vendorStore = VendorStore()

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
        pendingDateSource = nil
        pendingPreferredError = nil
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

        // ImageLoader returns an upright NSImage (EXIF orientation already
        // baked into the pixels). Its cgImage matches its .size, which matches
        // the SwiftUI display, which matches Vision's normalization — so the
        // bbox overlay aligns with the visible image.
        guard let nsImage = ImageLoader.shared.fullImage(for: entry.imageURL),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else {
            statusMessage = "Could not load image: \(entry.sourceFilename)"
            return
        }

        let preferred = OCRPipelineRegistry.preferred
        let fallback = VisionOnlyPipeline()

        do {
            let result = try await preferred.extract(image: cgImage, orientation: .up)
            pendingPreferredError = nil
            stashResult(result, for: entry, pipelineDisplayName: type(of: preferred).displayName)
            return
        } catch {
            // If preferred IS the fallback, no point retrying.
            if type(of: preferred).id == VisionOnlyPipeline.id {
                statusMessage = "Pre-label failed: \(error.localizedDescription)"
                return
            }
            // Capture the FM error and try the fallback.
            pendingPreferredError = error.localizedDescription
            // Also dump to stderr for easier off-screen diagnostics.
            FileHandle.standardError.write(Data("[\(entry.sourceFilename)] FM failed: \(error.localizedDescription)\n".utf8))
            statusMessage = "\(type(of: preferred).displayName) unavailable — falling back to \(VisionOnlyPipeline.displayName)…"
            do {
                let result = try await fallback.extract(image: cgImage, orientation: .up)
                stashResult(result, for: entry, pipelineDisplayName: "\(VisionOnlyPipeline.displayName) (fallback)")
            } catch {
                statusMessage = "Both pipelines failed: \(error.localizedDescription)"
            }
        }
    }

    private func stashResult(_ result: OCRKit.ExtractionResult, for entry: DatasetEntry, pipelineDisplayName: String) {
        var canonical = result.receipt
        canonical.imageId = entry.imageId  // anchor to the dataset-stable id

        // Date fallback: if the pipeline returned the sentinel "1970-01-01"
        // (meaning it couldn't find a date on the receipt), read EXIF /
        // file metadata. Track the source for the form to badge.
        var dateSource: ImageMetadata.DateSource? = nil
        if canonical.header.date.value == "1970-01-01" {
            if let (date, source) = ImageMetadata.creationDate(at: entry.imageURL) {
                canonical.header.date.value = ImageMetadata.formatISODate(date)
                canonical.provenance.fieldConfidence["date.value"] = 0.6  // medium — metadata, not OCR
                dateSource = source
            }
        }

        pendingExtraction = OCRKit.ExtractionResult(
            receipt: canonical,
            latencyMs: result.latencyMs,
            peakMemoryMB: result.peakMemoryMB,
            rawText: result.rawText
        )
        pendingDateSource = dateSource

        let dateNote = dateSource.map { " · date from \($0.rawValue.uppercased())" } ?? ""
        statusMessage = "Pre-label by \(pipelineDisplayName) — \(result.latencyMs) ms\(dateNote) · \(entry.sourceFilename)"
    }

    // MARK: - Saving

    func save(_ document: LabelDocument, for entry: DatasetEntry) {
        var doc = document
        if doc.label.sourceFilename == nil {
            doc.label.sourceFilename = entry.sourceFilename
        }
        do {
            try store.save(doc, imageId: entry.imageId)
            // Auto-add merchant to the vendor store on Verify so subsequent
            // receipts from the same place get a one-click apply.
            if doc.label.status == .verified {
                vendorStore.record(doc.receipt.header.merchant.name)
            }
            statusMessage = "Saved \(doc.label.status.displayName) — \(entry.sourceFilename)"
            reload()
            select(entry.imageId)  // keep selection on the same entry
            // Trigger the sidebar-row pulse for this entry.
            lastSaveEvent = SaveEvent(
                imageID: entry.imageId,
                status: doc.label.status,
                at: Date()
            )
        } catch {
            statusMessage = "Save failed: \(error.localizedDescription)"
        }
    }
}
