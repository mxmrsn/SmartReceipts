import AppKit
import Foundation
import SwiftUI

@main
struct ReceiptLabelerApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    @State private var controller: DatasetController = ReceiptLabelerApp.makeController()

    var body: some Scene {
        WindowGroup {
            WorkspaceView(controller: controller)
                .frame(minWidth: 1000, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }

    /// Default to `<repo>/dataset/` as detected from the binary's location.
    /// Repo layout: apps/macos-labeler/.build/.../ReceiptLabeler. Walk up
    /// until we see a `dataset/` directory.
    private static func makeController() -> DatasetController {
        let url = defaultDatasetURL()
        return DatasetController(datasetDirectory: url)
    }

    private static func defaultDatasetURL() -> URL {
        // Honor an override via env var; otherwise climb the binary path.
        if let env = ProcessInfo.processInfo.environment["RECEIPTS_DATASET_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env, isDirectory: true)
        }
        let fm = FileManager.default
        var current = URL(fileURLWithPath: Bundle.main.bundlePath)
            .deletingLastPathComponent()
        for _ in 0..<12 {
            let candidate = current.appending(path: "dataset", directoryHint: .isDirectory)
            if fm.fileExists(atPath: candidate.path(percentEncoded: false)) {
                return candidate
            }
            if current.path == "/" { break }
            current.deleteLastPathComponent()
        }
        // Fallback: ~/dataset
        return fm.homeDirectoryForCurrentUser.appending(path: "dataset", directoryHint: .isDirectory)
    }
}

/// Tiny NSApplicationDelegate so the SwiftPM-launched app shows up in the Dock
/// and becomes a proper foreground process (otherwise it can launch hidden).
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
