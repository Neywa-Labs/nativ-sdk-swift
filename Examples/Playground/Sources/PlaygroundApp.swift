import SwiftUI

@main
struct PlaygroundApp: App {
    @State private var mode = ExampleView.Mode.text
    @State private var chat = ChatModel()
    @State private var images = ImageGenerationModel()
    @State private var exportsImage = false
    @State private var showsSettings = false
    @State private var switching = true

    var body: some Scene {
        #if os(macOS)
            Window("Nativ SDK Playground", id: "playground") {
                playground
                    .frame(minWidth: 680, minHeight: 480)
            }
            .defaultSize(width: 860, height: 760)
            .windowResizability(.contentMinSize)
            .windowToolbarStyle(.unified)
            .commands {
                CommandGroup(replacing: .appSettings) {
                    Button("Settings…") { showsSettings = true }
                        .keyboardShortcut(",", modifiers: .command)
                }
                CommandGroup(after: .newItem) {
                    Button("New Chat") { Task { await chat.reset() } }
                        .keyboardShortcut("n", modifiers: .command)
                        .disabled(mode == .image || chat.phase != .ready || chat.messages.isEmpty)
                }
                CommandGroup(after: .importExport) {
                    Button("Save Image…") { exportsImage = true }
                        .keyboardShortcut("s", modifiers: .command)
                        .disabled(mode != .image || images.output == nil || images.isBusy)
                }
                CommandMenu("Task") {
                    Picker("Task", selection: $mode) {
                        ForEach(ExampleView.Mode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .disabled(switching || !chat.canSwitchModel || images.isBusy)
                }
            }
        #else
            WindowGroup {
                NavigationStack {
                    playground
                }
            }
        #endif
    }

    private var playground: some View {
        ExampleView(
            mode: $mode, chat: chat, images: images, exportsImage: $exportsImage,
            showsSettings: $showsSettings, switching: $switching)
    }
}
