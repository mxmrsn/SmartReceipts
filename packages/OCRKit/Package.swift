// swift-tools-version: 6.0
import PackageDescription

// The OCRKitTests target is intentionally NOT declared here. `XCTest` / `Testing`
// require the Xcode toolchain, but CI / developer machines often only have
// Command Line Tools selected. Tests under `Tests/OCRKitTests/` are kept on disk
// and will be runnable from the iOS Xcode workspace once integrated, or by
// re-adding the testTarget below after running `sudo xcode-select -s /Applications/Xcode.app`.
let package = Package(
    name: "OCRKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "OCRKit", targets: ["OCRKit"])
    ],
    targets: [
        .target(
            name: "OCRKit",
            path: "Sources/OCRKit"
        )
    ]
)
