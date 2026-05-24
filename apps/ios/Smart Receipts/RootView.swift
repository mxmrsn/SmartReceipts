import SwiftUI

/// Top-level tab bar. Capture is the primary action, Library and Dashboard sit
/// alongside.
struct RootView: View {
    var body: some View {
        TabView {
            CaptureView()
                .tabItem { Label("Capture", systemImage: "doc.viewfinder") }

            LibraryView()
                .tabItem { Label("Library", systemImage: "tray.full") }

            DashboardView()
                .tabItem { Label("Dashboard", systemImage: "chart.bar.fill") }
        }
    }
}
