// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ocr-cli",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../../packages/OCRKit")
    ],
    targets: [
        .executableTarget(
            name: "ocr-cli",
            dependencies: [
                .product(name: "OCRKit", package: "OCRKit")
            ],
            path: "Sources/ocr-cli"
        )
    ]
)
