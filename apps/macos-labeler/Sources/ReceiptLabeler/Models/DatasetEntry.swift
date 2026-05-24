import Foundation
import OCRKit

/// One image in the dataset, paired with whatever label state we have for it.
/// Status is derived from the on-disk label file (if any).
struct DatasetEntry: Identifiable, Hashable {
    let imageId: UUID
    let imageURL: URL
    var status: LabelStatus
    /// Loaded receipt payload from the label file, if a label exists.
    var label: LabelDocument?

    var id: UUID { imageId }

    var sourceFilename: String { imageURL.lastPathComponent }

    static func == (lhs: DatasetEntry, rhs: DatasetEntry) -> Bool {
        lhs.imageId == rhs.imageId && lhs.status == rhs.status
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(imageId)
        hasher.combine(status)
    }
}
