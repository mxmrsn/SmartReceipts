import AppKit
import OCRKit
import SwiftUI

/// Reference cell so an `NSEvent.addLocalMonitorForEvents` closure can look
/// up the current draft. SwiftUI `View` is a struct, so we can't capture
/// `@State` by reference in the long-lived monitor closure — but we can
/// hand it a class instance and mutate the class's `draft` whenever the
/// view re-evaluates with a new draft.
@MainActor
private final class KeyMonitorBag {
    weak var draft: LabelDraft?
    var monitor: Any?
}

/// Detail pane: zoomable receipt image on the left, editable canonical
/// Receipt form on the right. Loads either the existing label or a fresh
/// OCRKit pre-label and merges either back into a `LabelDraft` for editing.
struct LabelingView: View {

    @Bindable var controller: DatasetController
    let entry: DatasetEntry

    @State private var draft: LabelDraft?
    @State private var keyBag = KeyMonitorBag()
    /// When set, the next `prepareDraft()` skips the "load from disk"
    /// shortcut and re-runs the OCR pipeline against the image — even if
    /// a saved label exists. The flag is consumed (reset to false) by
    /// `prepareDraft()` itself, so it only affects the next preparation.
    @State private var forceReExtract: Bool = false

    var body: some View {
        Group {
            if let draft {
                workspace(draft: draft)
                    .onAppear {
                        keyBag.draft = draft
                        installKeyMonitor()
                    }
                    // After Save Draft we swap to a brand-new LabelDraft, but
                    // SwiftUI keeps the same workspace identity so onAppear
                    // doesn't refire. Re-pointing the bag here keeps the
                    // monitor talking to the live draft.
                    .onChange(of: ObjectIdentifier(draft)) { _, _ in
                        keyBag.draft = draft
                    }
            } else {
                ProgressView("Loading…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .task(id: entry.imageId) {
                        await prepareDraft()
                    }
            }
        }
        .onDisappear { uninstallKeyMonitor() }
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
                hasSavedVersion: hasSavedVersion,
                // IMPORTANT: don't capture `draft` in these closures. The
                // .keyboardShortcut bindings (⌘S, ⌘⏎) survive across body
                // re-renders, dispatching whichever closure was captured at
                // the FIRST registration — even after we replace self.draft
                // post-save. Capturing the parameter `draft` here strong-pins
                // the stale instance, so the second save would snapshot the
                // pre-first-save state and silently drop any bboxes you
                // added between saves. Calling `save(as:)` (no draft arg)
                // lets it read `self.draft` from @State at fire time,
                // which always returns the live LabelDraft.
                onSaveDraft:  { save(as: .draft) },
                onVerify:     { save(as: .verified) },
                onReject:     { save(as: .rejected) },
                onReExtract:  { reExtract() },
                onDiscard:    { discardChanges() }
            )
            .frame(minWidth: 320, idealWidth: 360, maxWidth: 500)
        }
    }

    // MARK: - Re-extract / Discard

    /// True if there's a label file on disk for this entry, so "Discard"
    /// has something to revert to. Reads through the controller's live
    /// `entries` array rather than the captured `entry`, because
    /// DatasetEntry's `==` deliberately ignores the `.label` payload —
    /// the view's `entry` value can be stale after a Save Draft cycle.
    private var hasSavedVersion: Bool {
        controller.entries.first { $0.imageId == entry.imageId }?.label != nil
    }

    /// Throw the current draft away and re-run the OCR pipeline. Useful
    /// when the user wants to start fresh, or after switching the
    /// preferred pipeline. The flag-driven path goes through
    /// `prepareDraft()` so the loading-spinner UX matches a first-open.
    private func reExtract() {
        forceReExtract = true
        draft = nil
    }

    /// Replace the current draft with the on-disk version, throwing away
    /// any unsaved edits. If there's no saved version yet, fall back to a
    /// fresh extraction.
    private func discardChanges() {
        let freshLabel = controller.entries.first { $0.imageId == entry.imageId }?.label
        if let doc = freshLabel {
            draft = LabelDraft(from: doc)
        } else {
            reExtract()
        }
    }

    // MARK: - Draft lifecycle

    private func prepareDraft() async {
        // Honor a pending re-extract: skip the "load from disk" path even if
        // a saved label exists. Reset the flag so it only fires once.
        if !forceReExtract, let doc = entry.label {
            draft = LabelDraft(from: doc)
            return
        }
        forceReExtract = false

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

    private func save(as status: LabelStatus) {
        // CRITICAL: read self.draft here rather than accepting a `draft`
        // parameter. The callers wired in workspace() are SwiftUI button
        // closures whose .keyboardShortcut binding outlives any single body
        // pass — if we captured `draft` in the closure, the second ⌘S press
        // would snapshot the *pre-first-save* draft instance and silently
        // drop edits made between saves. Reading from @State at fire time
        // always returns the live LabelDraft.
        guard let draft else { return }
        let document = draft.snapshot(asStatus: status, labeler: NSUserName())
        controller.save(document, for: entry)
        // Re-seat the draft directly from the document we just saved.
        //
        // We used to set `self.draft = nil` here so `.task(id: entry.imageId)`
        // would refire `prepareDraft()` and pick up the new label. Two problems:
        //
        //   1. DatasetEntry's `==` ignores `.label`, so SwiftUI's diffing thinks
        //      the post-save entry is unchanged when status didn't flip
        //      (Save Draft over an existing draft) and never re-passes it.
        //   2. With `entry.label` still showing the pre-save value (often nil
        //      for a freshly pre-labeled receipt), `prepareDraft` would fall
        //      through to `runPreLabel`, which re-ran the FM pipeline and
        //      *replayed the original extraction* — bringing deleted line items
        //      back and discarding the user's edits.
        //
        // Building the new draft from the just-snapshotted document is more
        // direct and immune to view-diff timing. Status flips in the sidebar
        // come from the controller's reloaded `entries` and don't need the
        // detail pane to remount.
        self.draft = LabelDraft(from: document)
    }

    // MARK: - Key monitor (Return / Escape end bbox edit)
    //
    // SwiftUI's `.onKeyPress(.return)` requires the view to actually have
    // keyboard focus, and `.focusable()` alone doesn't claim focus — so the
    // pure-SwiftUI path silently never fired. Dropping down to AppKit and
    // installing a *local* NSEvent monitor catches the key at the window
    // level regardless of who SwiftUI thinks is focused, and lets us pass
    // the event through cleanly when a TextField is editing so normal
    // text-field Enter behavior still works.

    private func installKeyMonitor() {
        guard keyBag.monitor == nil else { return }
        keyBag.monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [keyBag] event in
            // 36 = Return, 76 = numpad Enter, 53 = Escape
            let isCommit = event.keyCode == 36 || event.keyCode == 76 || event.keyCode == 53
            guard isCommit else { return event }

            // Don't steal Enter from an editing text field — that breaks
            // committing the merchant / description / total inputs.
            if let fr = event.window?.firstResponder, isTextEditor(fr) {
                return event
            }

            guard let draft = keyBag.draft, draft.selectedBBoxKey != nil else {
                return event
            }
            // Defer the mutation off AppKit's event-dispatch stack so we
            // don't fight Observation's transaction guarantees.
            DispatchQueue.main.async {
                draft.selectedBBoxKey = nil
            }
            return nil  // consume — no system beep, no further dispatch
        }
    }

    private func uninstallKeyMonitor() {
        if let m = keyBag.monitor {
            NSEvent.removeMonitor(m)
            keyBag.monitor = nil
        }
    }

    /// True if the responder is actively editing text. SwiftUI's `TextField`
    /// is backed by an `NSTextField` whose field editor (the view that
    /// becomes first responder while the user is typing) is an `NSTextView`
    /// (which is itself a subclass of `NSText`). Either form means "the
    /// user is typing — don't intercept Enter."
    private func isTextEditor(_ responder: NSResponder) -> Bool {
        if responder is NSTextView { return true }
        if responder is NSText     { return true }
        return false
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
