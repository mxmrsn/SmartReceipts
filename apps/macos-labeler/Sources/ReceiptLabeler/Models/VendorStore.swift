import Foundation
import Observation

/// Persistent saved-merchant directory. The user verifies a receipt, the
/// merchant name auto-adds here; next time they hit a similar receipt they
/// can one-click apply.
///
/// On disk: ~/Library/Application Support/ReceiptLabeler/vendors.json
@Observable
@MainActor
final class VendorStore {

    private(set) var vendors: [Vendor] = []

    struct Vendor: Codable, Identifiable, Hashable, Sendable {
        let id: UUID
        var name: String
        var usageCount: Int
        var lastUsedAt: Date

        init(id: UUID = UUID(), name: String, usageCount: Int = 1, lastUsedAt: Date = Date()) {
            self.id = id
            self.name = name
            self.usageCount = usageCount
            self.lastUsedAt = lastUsedAt
        }
    }

    private let storeURL: URL

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appending(path: "ReceiptLabeler", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.storeURL = dir.appending(path: "vendors.json", directoryHint: .notDirectory)
        load()
    }

    /// Top vendors sorted by usage count, then recency.
    var topVendors: [Vendor] {
        vendors.sorted {
            if $0.usageCount != $1.usageCount { return $0.usageCount > $1.usageCount }
            return $0.lastUsedAt > $1.lastUsedAt
        }
    }

    /// Record a merchant. If it already exists (case-insensitive), bump
    /// usageCount and lastUsedAt. Returns the resulting vendor.
    @discardableResult
    func record(_ rawName: String) -> Vendor? {
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.count >= 2 else { return nil }

        if let idx = vendors.firstIndex(where: { $0.name.lowercased() == name.lowercased() }) {
            vendors[idx].usageCount += 1
            vendors[idx].lastUsedAt = Date()
            // Normalize spelling to most-recent capitalization
            vendors[idx].name = name
            save()
            return vendors[idx]
        } else {
            let v = Vendor(name: name)
            vendors.append(v)
            save()
            return v
        }
    }

    func remove(_ vendor: Vendor) {
        vendors.removeAll { $0.id == vendor.id }
        save()
    }

    func clear() {
        vendors = []
        save()
    }

    // MARK: - Persistence

    private func load() {
        guard let data = try? Data(contentsOf: storeURL) else { return }
        if let decoded = try? JSONDecoder().decode([Vendor].self, from: data) {
            vendors = decoded
        }
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(vendors) else { return }
        try? data.write(to: storeURL, options: .atomic)
    }
}
