import Foundation
import XCTest
@testable import OCRKit

final class ReceiptRoundTripTests: XCTestCase {

    func testEncodeDecodeRoundTrip() throws {
        let original = Receipt.fixture
        let data = try original.canonicalJSON()
        let decoded = try Receipt.from(json: data)
        XCTAssertEqual(decoded, original)
    }

    func testCanonicalJSONIsStable() throws {
        let r = Receipt.fixture
        let a = try r.canonicalJSON()
        let b = try Receipt.from(json: a).canonicalJSON()
        XCTAssertEqual(a, b)
    }
}

extension Receipt {
    static let fixture: Receipt = Receipt(
        imageId: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
        header: Header(
            merchant: Merchant(name: "Joe's Coffee", address: "123 Main St", phone: "555-1234"),
            date: ReceiptDate(value: "2026-05-24", time: "09:14"),
            transactionId: "TX-001",
            currency: "USD"
        ),
        lineItems: [
            LineItem(description: "Latte", quantity: 1, unitPrice: 4.50, totalPrice: 4.50, category: .food),
            LineItem(description: "Croissant", quantity: 2, unitPrice: 3.25, totalPrice: 6.50, category: .food)
        ],
        totals: Totals(
            subtotal: 11.00,
            tax: [TaxLine(label: "Sales Tax", rate: 0.0875, amount: 0.96)],
            tip: 2.00,
            total: 13.96
        ),
        payment: Payment(method: .card, cardLast4: "1234"),
        provenance: Provenance(
            pipelineId: "vision-regex",
            modelVersion: "vision-accurate.1",
            confidence: 0.42,
            fieldConfidence: ["merchant.name": 0.5, "totals.total": 0.6],
            bboxes: ["merchant.name": BBox(x: 0.1, y: 0.05, width: 0.8, height: 0.05)]
        )
    )
}
