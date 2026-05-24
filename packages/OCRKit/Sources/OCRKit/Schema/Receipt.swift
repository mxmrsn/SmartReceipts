import Foundation

/// Canonical receipt extraction output. Mirrors `shared/schema/receipt.schema.json`.
///
/// Hand-maintained against the JSON schema; the round-trip test in `OCRKitTests`
/// catches drift. When this type is regenerated via Sourcery later, the JSON file
/// becomes the source of truth.
public struct Receipt: Codable, Sendable, Equatable {
    public var imageId: UUID
    public var header: Header
    public var lineItems: [LineItem]
    public var totals: Totals
    public var payment: Payment?
    public var provenance: Provenance

    public init(
        imageId: UUID,
        header: Header,
        lineItems: [LineItem],
        totals: Totals,
        payment: Payment? = nil,
        provenance: Provenance
    ) {
        self.imageId = imageId
        self.header = header
        self.lineItems = lineItems
        self.totals = totals
        self.payment = payment
        self.provenance = provenance
    }
}

extension Receipt {
    public struct Header: Codable, Sendable, Equatable {
        public var merchant: Merchant
        public var date: ReceiptDate
        public var transactionId: String?
        public var currency: String

        public init(
            merchant: Merchant,
            date: ReceiptDate,
            transactionId: String? = nil,
            currency: String = "USD"
        ) {
            self.merchant = merchant
            self.date = date
            self.transactionId = transactionId
            self.currency = currency
        }
    }

    public struct Merchant: Codable, Sendable, Equatable {
        public var name: String
        public var address: String?
        public var phone: String?
        public var taxId: String?

        public init(name: String, address: String? = nil, phone: String? = nil, taxId: String? = nil) {
            self.name = name
            self.address = address
            self.phone = phone
            self.taxId = taxId
        }
    }

    public struct ReceiptDate: Codable, Sendable, Equatable {
        /// ISO calendar date, YYYY-MM-DD.
        public var value: String
        /// 24-hour time of day, HH:mm.
        public var time: String?

        public init(value: String, time: String? = nil) {
            self.value = value
            self.time = time
        }
    }

    public struct LineItem: Codable, Sendable, Equatable {
        public var description: String
        public var quantity: Decimal?
        public var unitPrice: Decimal?
        public var totalPrice: Decimal
        public var category: Category?

        public init(
            description: String,
            quantity: Decimal? = nil,
            unitPrice: Decimal? = nil,
            totalPrice: Decimal,
            category: Category? = nil
        ) {
            self.description = description
            self.quantity = quantity
            self.unitPrice = unitPrice
            self.totalPrice = totalPrice
            self.category = category
        }
    }

    public enum Category: String, Codable, Sendable, CaseIterable, Equatable {
        case food = "Food"
        case fuel = "Fuel"
        case groceries = "Groceries"
        case office = "Office"
        case transport = "Transport"
        case lodging = "Lodging"
        case entertainment = "Entertainment"
        case health = "Health"
        case other = "Other"
    }

    public struct Totals: Codable, Sendable, Equatable {
        public var subtotal: Decimal?
        public var tax: [TaxLine]
        public var discount: Decimal?
        public var tip: Decimal?
        public var serviceCharge: Decimal?
        public var total: Decimal

        public init(
            subtotal: Decimal? = nil,
            tax: [TaxLine] = [],
            discount: Decimal? = nil,
            tip: Decimal? = nil,
            serviceCharge: Decimal? = nil,
            total: Decimal
        ) {
            self.subtotal = subtotal
            self.tax = tax
            self.discount = discount
            self.tip = tip
            self.serviceCharge = serviceCharge
            self.total = total
        }
    }

    public struct TaxLine: Codable, Sendable, Equatable {
        public var label: String
        public var rate: Decimal?
        public var amount: Decimal

        public init(label: String, rate: Decimal? = nil, amount: Decimal) {
            self.label = label
            self.rate = rate
            self.amount = amount
        }
    }

    public struct Payment: Codable, Sendable, Equatable {
        public enum Method: String, Codable, Sendable, Equatable {
            case cash, card, check, other
        }
        public var method: Method?
        public var cardLast4: String?

        public init(method: Method? = nil, cardLast4: String? = nil) {
            self.method = method
            self.cardLast4 = cardLast4
        }
    }

    public struct Provenance: Codable, Sendable, Equatable {
        public var pipelineId: String
        public var modelVersion: String
        public var confidence: Double
        public var fieldConfidence: [String: Double]
        public var bboxes: [String: BBox]

        public init(
            pipelineId: String,
            modelVersion: String,
            confidence: Double,
            fieldConfidence: [String: Double] = [:],
            bboxes: [String: BBox] = [:]
        ) {
            self.pipelineId = pipelineId
            self.modelVersion = modelVersion
            self.confidence = confidence
            self.fieldConfidence = fieldConfidence
            self.bboxes = bboxes
        }
    }

    /// Bounding box in normalized image coordinates ([0,1] for all four values).
    public struct BBox: Codable, Sendable, Equatable, Hashable {
        public var x: Double
        public var y: Double
        public var width: Double
        public var height: Double

        public init(x: Double, y: Double, width: Double, height: Double) {
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        }
    }
}

// MARK: - JSON helpers

extension Receipt {
    public static let canonicalEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return e
    }()

    public static let canonicalDecoder: JSONDecoder = JSONDecoder()

    public func canonicalJSON() throws -> Data {
        try Self.canonicalEncoder.encode(self)
    }

    public static func from(json data: Data) throws -> Receipt {
        try canonicalDecoder.decode(Receipt.self, from: data)
    }
}
