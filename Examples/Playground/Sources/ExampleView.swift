import SwiftUI

struct ExampleView: View {
    enum Mode: String, CaseIterable {
        case text = "Chat"
        case vision = "Vision"
        case image = "Images"
    }

    @Binding var mode: Mode
    var chat: ChatModel
    var images: ImageGenerationModel
    @Binding var exportsImage: Bool
    @Binding var showsSettings: Bool
    @Binding var switching: Bool

    var body: some View {
        Group {
            if mode == .image {
                ImageGenerationView(
                    model: images, exportsImage: $exportsImage, showsSettings: $showsSettings)
            } else {
                ChatView(chat: chat, showsSettings: $showsSettings)
            }
        }
        .background(.background)
        .navigationTitle("Nativ SDK Playground")
        #if os(macOS)
            .navigationSubtitle(mode == .image ? "Qwen-Image-2.1" : chat.model.name)
            .toolbar {
                if #available(macOS 26.0, *) {
                    modeToolbarItem.sharedBackgroundVisibility(.hidden)
                } else {
                    modeToolbarItem
                }
            }
        #else
            .navigationTitle("Playground")
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .top, spacing: 0) {
                VStack(spacing: 8) {
                    modePicker
                    Text(mode == .image ? "Qwen-Image-2.1" : chat.model.name)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.bar)
            }
        #endif
        .onChange(of: mode) { _, _ in showsSettings = false }
        .task(id: mode) {
            switching = true
            defer { switching = false }
            if mode == .image {
                await chat.unload()
            } else {
                images.unload()
                await chat.select(mode == .text ? .text : .vision)
                await chat.load()
            }
        }
    }

    private var modeToolbarItem: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            modePicker.frame(width: 240)
        }
    }

    private var modePicker: some View {
        Picker("Mode", selection: $mode) {
            ForEach(Mode.allCases, id: \.self) { mode in
                Text(mode.rawValue).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .disabled(switching || !chat.canSwitchModel || images.isBusy)
        .help("Switch tasks and release the previous model")
        .accessibilityIdentifier("ExampleMode")
    }
}
