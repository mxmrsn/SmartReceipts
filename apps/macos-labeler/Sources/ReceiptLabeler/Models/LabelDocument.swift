import Foundation
import OCRKit

/// A label file as it sits on disk: canonical Receipt + a `_label` block
/// holding labeler metadata (status, source filename, reviewer, etc.).
///
/// We can't make this a plain `Codable` extension of `OCRKit.Receipt` because
/// the canonical schema is `additionalProperties: false`. So we encode/decode
/// manually, merging the `_label` key in/out.
struct LabelDocument: Sendable {
    var receipt: OCRKit.Receipt
    var label: LabelMetadata

    struct LabelMetadata: Codable, Sendable {
        var status: LabelStatus
        var sourceFilename: String?
        var labeler: String?
        var verifiedAt: Date?
        var sourcePipeline: String?
        var notes: String?

        static func draft(sourceFilename: String, sourcePipeline: String?) -> LabelMetadata {
            LabelMetadata(
                status: .draft,
                sourceFilename: sourceFilename,
                labeler: nil,
                verifiedAt: nil,
                sourcePipeline: sourcePipeline,
                notes: nil
            )
        }
    }

    // MARK: - JSON round-trip

    func encoded() throws -> Data {
        // Round-trip the receipt through canonical JSON to get a [String: Any]
        // we can stuff the `_label` block into without violating the strict
        // `additionalProperties` of the canonical schema.
        let canonicalData = try receipt.canonicalJSON()
        guard var dict = try JSONSerialization.jsonObject(with: canonicalData) as? [String: Any] else {
            throw NSError(domain: "LabelDocument", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Canonical Receipt JSON was not an object."
            ])
        }
        let labelData = try LabelDocument.labelEncoder.encode(label)
        let labelObject = try JSONSerialization.jsonObject(with: labelData)
        dict["_label"] = labelObject

        return try JSONSerialization.data(
            withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    static func decoded(from data: Data) throws -> LabelDocument {
        guard var dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "LabelDocument", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Label file was not a JSON object."
            ])
        }
        let labelBlock = (dict.removeValue(forKey: "_label") as? [String: Any]) ?? [:]
        let canonicalData = try JSONSerialization.data(withJSONObject: dict)
        let receipt = try OCRKit.Receipt.from(json: canonicalData)

        let labelData = try JSONSerialization.data(withJSONObject: labelBlock)
        let label = try LabelDocument.labelDecoder.decode(LabelMetadata.self, from: labelData)
        return LabelDocument(receipt: receipt, label: label)
    }

    private static let labelEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()
    private static let labelDecoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
