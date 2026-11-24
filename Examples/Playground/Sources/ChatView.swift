import ImageIO
import SwiftUI

struct ChatView: View {
    @Bindable var chat: ChatModel
    @Binding var showsSettings: Bool

    var body: some View {
        Group {
            if chat.messages.isEmpty {
                ChatWelcomeView(chat: chat)
            } else {
                ChatTranscriptView(chat: chat)
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                if let notice = chat.notice {
                    WorkspaceNotice(
                        message: notice,
                        retry: chat.phase == .failed ? { Task { await chat.load() } } : nil,
                        dismiss: { chat.notice = nil }
                    )
                    .frame(maxWidth: 760)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 8)
                }
                ChatComposerView(chat: chat)
            }
            .frame(maxWidth: .infinity)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Chat Settings", systemImage: "slider.horizontal.3") {
                    showsSettings.toggle()
                }
                .help("Chat settings")
                .popover(isPresented: $showsSettings, arrowEdge: .bottom) {
                    ChatSettingsView(chat: chat)
                        .presentationCompactAdaptation(.popover)
                }
                Button("New Chat", systemImage: "square.and.pencil") {
                    Task { await chat.reset() }
                }
                .disabled(chat.phase != .ready || chat.messages.isEmpty)
                .help("New chat (⌘N)")
            }
        }
        .onDisappear { chat.stop() }
    }
}

private struct ChatWelcomeView: View {
    var chat: ChatModel

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 24) {
                    VStack(spacing: 14) {
                        Image(
                            systemName: chat.model == .text ? "bubble.left.and.bubble.right" : "eye"
                        )
                        .font(.system(size: 36, weight: .regular))
                        .foregroundStyle(.tint)
                        .frame(width: 76, height: 76)
                        .background(.tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 22))
                        .accessibilityHidden(true)
                        Text(chat.model == .text ? "What’s on your mind?" : "Look a little closer")
                            .font(.largeTitle.weight(.semibold))
                        Text(
                            chat.model == .text
                                ? "Ask a question, explore an idea, or try something new."
                                : "Describe the sample image, then ask a follow-up question."
                        )
                        .font(.body)
                        .foregroundStyle(.secondary)
                    }
                    if chat.model == .text {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 160))], spacing: 10) {
                            Button("Explain a concept") {
                                chat.draft = "Explain how a rainbow forms in simple terms."
                            }
                            Button("Help me write") {
                                chat.draft = "Help me write a friendly thank-you note."
                            }
                            Button("Try the weather tool") {
                                chat.draft = "What’s the weather in Paris?"
                                chat.send()
                            }
                            .help("Uses simulated weather data")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(chat.phase != .ready)
                    } else {
                        if let url = ChatModel.sampleImageURL {
                            ChatImageView(url: url)
                        }
                        Button("Describe the sample image", systemImage: "eye") {
                            chat.describeSampleImage()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(chat.phase != .ready)
                    }
                    if chat.phase == .loading {
                        ChatLoadingView(
                            progress: chat.downloadProgress, usesDraft: chat.usesDraftModel)
                    } else {
                        Label("Replies are generated on your device", systemImage: "lock")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .multilineTextAlignment(.center)
                .frame(maxWidth: 640)
                .padding(32)
                .frame(maxWidth: .infinity)
                .frame(minHeight: geometry.size.height)
            }
        }
    }
}

private struct ChatLoadingView: View {
    var progress: Double
    var usesDraft: Bool

    var body: some View {
        VStack(spacing: 8) {
            if progress > 0 && progress < 1 {
                ProgressView(value: progress) {
                    Text(
                        "Downloading model… \(progress, format: .percent.precision(.fractionLength(0)))"
                    )
                }
                .frame(maxWidth: 280)
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(usesDraft ? "Loading model and DSpark draft…" : "Loading model…")
                }
            }
            Text("Downloaded models are saved for next time.")
                .foregroundStyle(.secondary)
        }
        .font(.caption)
    }
}

private struct ChatTranscriptView: View {
    var chat: ChatModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 28) {
                    ForEach(chat.messages) { message in
                        ChatMessageView(
                            message: message,
                            isStreaming: chat.isResponding && message.id == chat.messages.last?.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 720)
                .padding(.horizontal, 32)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: chat.messages.count) { _, _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .onChange(of: chat.messages.last?.text) { _, _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
    }
}

private struct ChatComposerView: View {
    @Bindable var chat: ChatModel
    @FocusState private var focused: Bool

    var body: some View {
        ComposerSurface {
            VStack(alignment: .leading, spacing: 12) {
                if let url = chat.draftImageURL {
                    HStack(alignment: .top, spacing: 10) {
                        ChatImageView(url: url).frame(maxWidth: 120, maxHeight: 64)
                        Button("Remove image", systemImage: "xmark") { chat.draftImageURL = nil }
                            .labelStyle(.iconOnly)
                            .buttonStyle(.borderless)
                            .disabled(chat.phase != .ready)
                    }
                }
                HStack(alignment: .bottom, spacing: 16) {
                    TextField(
                        chat.model == .text ? "Message LFM2.5…" : "Ask about an image…",
                        text: $chat.draft, axis: .vertical
                    )
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(2 ... 6)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 6)
                    .focused($focused)
                    .disabled(chat.phase != .ready)
                    .accessibilityLabel("Message")
                    .accessibilityIdentifier("MessageInput")
                    if chat.isResponding {
                        Button(
                            chat.phase == .stopping ? "Stopping…" : "Stop", systemImage: "stop.fill"
                        ) {
                            chat.stop()
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.bordered)
                        .buttonBorderShape(.capsule)
                        .controlSize(.large)
                        .disabled(chat.phase == .stopping)
                        .keyboardShortcut(.escape, modifiers: [])
                        .help("Stop response (Escape)")
                    } else {
                        Button("Send", systemImage: "arrow.up") { chat.send() }
                            .labelStyle(.iconOnly)
                            .buttonStyle(.borderedProminent)
                            .buttonBorderShape(.capsule)
                            .controlSize(.large)
                            .disabled(!chat.canSend)
                            .keyboardShortcut(.return, modifiers: .command)
                            .help("Send message (⌘Return)")
                            .accessibilityIdentifier("SendMessage")
                    }
                }
                HStack(spacing: 12) {
                    Text(statusLabel)
                    Spacer(minLength: 8)
                    #if os(macOS)
                        Text("⌘ Return to send")
                    #endif
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        #if os(macOS)
            .onAppear { focused = chat.phase == .ready }
            .onChange(of: chat.phase) { _, phase in
                if phase == .ready { focused = true }
            }
        #endif
    }

    private var statusLabel: String {
        switch chat.phase {
        case .ready: chat.usesDraftModel ? "Ready · DSpark enabled" : "Ready"
        case .responding: "Generating response…"
        case .stopping: "Stopping…"
        case .loading: "Loading model…"
        case .resetting: "Starting a new chat…"
        case .failed: "Model unavailable"
        case .idle: "Preparing model…"
        }
    }
}

private struct ChatSettingsView: View {
    @Bindable var chat: ChatModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Chat Settings").font(.headline)
            LabeledContent("Model", value: chat.model.name)
                .foregroundStyle(.secondary)
            Divider()
            Toggle(
                "Use Draft Model",
                isOn: Binding(
                    get: { chat.usesDraftModel },
                    set: { enabled in Task { await chat.setUsesDraftModel(enabled) } })
            )
            .disabled(!chat.canSwitchModel)
            .accessibilityIdentifier("UseDraftModel")
            Text(
                "DSpark can speed up replies using an additional model download. Changing this setting starts a new conversation."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(width: 320)
    }
}

private struct ChatMessageView: View {
    var message: ChatModel.Message
    var isStreaming = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.isUser { Spacer(minLength: 64) }
            VStack(alignment: .leading, spacing: 10) {
                Text(message.isUser ? "You" : "LFM2.5")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(
                        message.isUser ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tint))
                if let url = message.imageURL { ChatImageView(url: url) }
                if !message.reasoning.isEmpty {
                    DisclosureGroup("Thinking") {
                        Text(message.reasoning)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .padding(.top, 6)
                    }
                    .font(.callout)
                    .foregroundStyle(.secondary)
                }
                ForEach(message.toolActivity) { activity in
                    DisclosureGroup {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Arguments: \(activity.arguments)")
                            if let result = activity.result {
                                Text("Result: \(result)")
                            } else {
                                Text("Waiting for tool result…")
                            }
                            Text("This demo uses simulated data.")
                        }
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(.top, 8)
                    } label: {
                        Label(
                            "Simulated weather · \(activity.result == nil ? "Running…" : "Complete")",
                            systemImage: "wrench.and.screwdriver"
                        )
                        .font(.callout)
                    }
                    .padding(12)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                }
                if message.text.isEmpty && message.toolActivity.isEmpty && isStreaming {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Thinking…").foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Generating response")
                } else if !message.text.isEmpty {
                    if message.isUser {
                        Text(message.text).textSelection(.enabled)
                    } else {
                        Text(.init(message.text)).textSelection(.enabled)
                            .lineSpacing(4)
                    }
                }
                if let metrics = message.metrics, !isStreaming {
                    DisclosureGroup("Response details") {
                        VStack(alignment: .leading, spacing: 6) {
                            if metrics.generationSeconds > 0 {
                                Text(
                                    "\(Double(metrics.generatedTokenCount) / metrics.generationSeconds, specifier: "%.1f") tokens/s"
                                )
                            }
                            if let draft = metrics.speculative {
                                Text(
                                    "DSpark · accepted \(draft.acceptedTokenCount)/\(draft.proposedTokenCount) draft tokens"
                                )
                                .help(
                                    "Verification counts include draft tokens beyond the end of the response."
                                )
                            }
                        }
                        .textSelection(.enabled)
                        .padding(.top, 6)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .padding(message.isUser ? 16 : 0)
            .frame(maxWidth: message.isUser ? 560 : .infinity, alignment: .leading)
            .background {
                if message.isUser {
                    RoundedRectangle(cornerRadius: 18).fill(.quaternary.opacity(0.6))
                }
            }
            if !message.isUser { Spacer(minLength: 0) }
        }
        .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
    }
}

private struct ChatImageView: View {
    var url: URL

    var body: some View {
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        {
            Image(decorative: image, scale: 1)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 384, maxHeight: 192)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .accessibilityLabel("Sample image: a red circle, blue square, and green triangle")
        }
    }
}
