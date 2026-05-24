import OCRKit
import SwiftData
import SwiftUI
import UIKit

/// Entry point of the Capture tab. Drives a small state machine:
///
///     idle → scanning → processing → reviewing → idle (after save)
///
/// On save, persists a `Receipt` via SwiftData and stores the JPEG via
/// `ReceiptImageStorage`.
struct CaptureView: View {

    @Environment(\.modelContext) private var modelContext

    @State private var step: Step = .idle
    @State private var showingScanner = false
    @State private var errorMessage: String?
    @State private var lastSavedAt: Date?

    enum Step {
        case idle
        case processing(UIImage)
        case reviewing(UIImage, OCRKit.ExtractionResult)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                switch step {
                case .idle:
                    idleView
                case .processing(let image):
                    ProcessingPanel(image: image)
                        .task(id: image) {
                            await runExtraction(on: image)
                        }
                case .reviewing(let image, let result):
                    ReviewView(
                        image: image,
                        extraction: result,
                        onCancel: { reset() },
                        onSave: { draft in save(draft: draft, image: image, extraction: result) }
                    )
                }
            }
            .navigationTitle("Capture")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showingScanner) {
                DocumentScannerView { result in
                    showingScanner = false
                    handleScanResult(result)
                }
                .ignoresSafeArea()
            }
            .alert("Something went wrong", isPresented: errorAlertBinding) {
                Button("OK") { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }

    // MARK: - Substeps

    private var idleView: some View {
        VStack(spacing: 24) {
            Image(systemName: "doc.text.viewfinder")
                .font(.system(size: 72, weight: .light))
                .foregroundStyle(.tint)
            Text("Scan a receipt")
                .font(.title2.weight(.semibold))
            Text("Tap below to launch the document scanner. Multi-page is supported — long receipts get concatenated automatically.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button {
                showingScanner = true
            } label: {
                Label("Scan Receipt", systemImage: "camera.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 32)

            if let savedAt = lastSavedAt {
                Label("Saved at \(savedAt.formatted(date: .omitted, time: .standard))", systemImage: "checkmark.circle.fill")
                    .font(.footnote)
                    .foregroundStyle(.green)
            }
        }
        .padding(.vertical, 48)
    }

    // MARK: - Flow

    private func handleScanResult(_ result: Result<[UIImage], Error>) {
        switch result {
        case .success(let pages):
            guard let combined = UIImage.concatenatedVertically(pages) else {
                errorMessage = "No pages were captured."
                return
            }
            step = .processing(combined)
        case .failure(let error):
            if error is CancellationError { return }
            errorMessage = error.localizedDescription
        }
    }

    private func runExtraction(on image: UIImage) async {
        guard let cgImage = image.cgImage else {
            errorMessage = "Could not prepare image for OCR."
            step = .idle
            return
        }
        do {
            let pipeline = VisionOnlyPipeline()
            let result = try await pipeline.extract(image: cgImage)
            step = .reviewing(image, result)
        } catch {
            errorMessage = "OCR failed: \(error.localizedDescription)"
            step = .idle
        }
    }

    private func save(draft: ReceiptDraft, image: UIImage, extraction: OCRKit.ExtractionResult) {
        do {
            let id = UUID()
            let imagePath = try ReceiptImageStorage.save(image, id: id)
            let canonical = draft.applying(to: extraction.receipt, newID: id)
            let receipt = try ReceiptMapping.makePersistedReceipt(
                from: canonical,
                capturedAt: Date(),
                imageRelativePath: imagePath
            )
            modelContext.insert(receipt)
            try modelContext.save()
            lastSavedAt = Date()
            reset()
        } catch {
            errorMessage = "Could not save: \(error.localizedDescription)"
        }
    }

    private func reset() {
        step = .idle
    }

    private var errorAlertBinding: Binding<Bool> {
        Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )
    }
}

// MARK: - Processing panel

private struct ProcessingPanel: View {
    let image: UIImage

    var body: some View {
        VStack(spacing: 16) {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 240)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary, lineWidth: 1)
                )
            ProgressView()
            Text("Reading receipt…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}
