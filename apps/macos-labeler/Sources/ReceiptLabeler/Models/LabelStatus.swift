import Foundation

enum LabelStatus: String, Codable, Sendable, CaseIterable {
    case unlabeled
    case draft
    case verified
    case rejected

    var displayName: String {
        switch self {
        case .unlabeled: "Unlabeled"
        case .draft:     "Draft"
        case .verified:  "Verified"
        case .rejected:  "Rejected"
        }
    }
}
