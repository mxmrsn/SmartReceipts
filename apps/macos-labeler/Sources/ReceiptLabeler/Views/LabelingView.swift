import AppKit
import OCRKit
import SwiftUI

/// Detail pane: zoomable receipt image on the left, editable canonical
/// Receipt form on the right. Loads either the existing label or a fresh
/// OCRKit pre-label and merges either back into a `LabelDraft` for editing.
struct LabelingView: View {

    @Bindable var controller: DatasetController
    let entry: DatasetEntry

    @State private var draft: LabelDraft?

    var body: some View {
        if let draft {
            workspace(draft: draft)
        } else {
            ProgressView("Loading…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .task(id: entry.imageId) {
                    await prepareDraft()
                }
        }
    }

    // MARK: - Layout

    private func workspace(draft: LabelDraft) -> some View {
        HSplitView {
            ZoomableImageView(
                url: entry.imageURL,
                overlay: BoundingBoxOverlay(draft: draft)
            )
            .frame(minWidth: 380, idealWidth: 720)
            .layoutPriority(1)
            ReceiptFormSection(
                draft: draft,
                vendorStore: controller.vendorStore,
                onSaveDraft: { save(draft: draft, as: .draft) },
                onVerify:    { save(draft: draft, as: .verified) },
                onReject:    { save(draft: draft, as: .rejected) }
            )
            .frame(minWidth: 320, idealWidth: 360, maxWidth: 500)
        }
    }

    // MARK: - Draft lifecycle

    private func prepareDraft() async {
        if let doc = entry.label {
            draft = LabelDraft(from: doc)
            return
        }

        await controller.runPreLabel(for: entry)
        if let extraction = controller.pendingExtraction {
            draft = LabelDraft(
                fromPreLabel: extraction.receipt,
                sourceFilename: entry.sourceFilename,
                pipelineId: extraction.receipt.provenance.pipelineId,
                rawText: extraction.rawText,
                dateSource: controller.pendingDateSource?.rawValue,
                ocrLines: extraction.ocrLines,
                preferredPipelineFailed: controller.pendingPreferredError != nil,
                preferredPipelineError: controller.pendingPreferredError
            )
        } else {
            let blank = blankCanonicalReceipt(imageId: entry.imageId)
            draft = LabelDraft(
                fromPreLabel: blank,
                sourceFilename: entry.sourceFilename,
                pipelineId: "none",
                rawText: nil
            )
        }
    }

    private func save(draft: LabelDraft, as status: LabelStatus) {
        let document = draft.snapshot(asStatus: status, labeler: NSUserName())
        controller.save(document, for: entry)
        self.draft = nil   // force reload from disk so status flips in sidebar
    }

    // MARK: - Helpers

    private func blankCanonicalReceipt(imageId: UUID) -> OCRKit.Receipt {
        OCRKit.Receipt(
            imageId: imageId,
            header: .init(
                merchant: .init(name: ""),
                date: .init(value: "1970-01-01"),
                currency: "USD"
            ),
            lineItems: [],
            totals: .init(tax: [], total: 0),
            provenance: .init(
                pipelineId: "none",
                modelVersion: "none",
                confidence: 0,
                fieldConfidence: [:],
                bboxes: [:]
            )
        )
    }
}
