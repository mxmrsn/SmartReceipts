// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ReceiptLabeler",
    platforms: [
        .macOS("26.0")
    ],
    dependencies: [
        .package(path: "../../packages/OCRKit")
    ],
    targets: [
        .executableTarget(
            name: "ReceiptLabeler",
            dependencies: [
                .product(name: "OCRKit", package: "OCRKit")
            ],
            path: "Sources/ReceiptLabeler"
        )
    ]
)
