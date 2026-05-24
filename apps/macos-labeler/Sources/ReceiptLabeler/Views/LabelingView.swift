import AppKit
import OCRKit
import SwiftUI

/// Detail pane: receipt image on the left, editable canonical Receipt form on
/// the right. Loads either the existing label or a fresh OCRKit pre-label and
/// merges either back into a `LabelDraft` for editing.
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
            imagePane
                .frame(minWidth: 320)
            ReceiptFormSection(
                draft: draft,
                onSaveDraft: { save(draft: draft, as: .draft) },
                onVerify:    { save(draft: draft, as: .verified) },
                onReject:    { save(draft: draft, as: .rejected) }
            )
            .frame(minWidth: 360)
        }
    }

    private var imagePane: some View {
        ScrollView([.horizontal, .vertical]) {
            if let image = ImageLoader.shared.fullImage(for: entry.imageURL) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .padding(16)
            } else {
                Text("Could not load image")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Draft lifecycle

    private func prepareDraft() async {
        // If we have an on-disk label, use it as the basis.
        if let doc = entry.label {
            draft = LabelDraft(from: doc)
            return
        }

        // Otherwise run the baseline pipeline to pre-fill.
        await controller.runPreLabel(for: entry)
        if let pending = controller.pendingDraft {
            draft = LabelDraft(
                fromPreLabel: pending,
                sourceFilename: entry.sourceFilename,
                pipelineId: pending.provenance.pipelineId
            )
        } else {
            // Fall back: an empty draft anchored to this image's id.
            let blank = blankCanonicalReceipt(imageId: entry.imageId)
            draft = LabelDraft(
                fromPreLabel: blank,
                sourceFilename: entry.sourceFilename,
                pipelineId: "none"
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
