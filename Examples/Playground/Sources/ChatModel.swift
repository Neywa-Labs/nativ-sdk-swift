import Foundation
import NativSDK
import Observation

@MainActor
@Observable
final class ChatModel {
    enum Model: String, CaseIterable, Identifiable {
        case text, vision

        var id: Self { self }
        var title: String { self == .text ? "Text" : "Vision" }
        var name: String { self == .text ? "LFM2.5 · 2.6B" : "LFM2.5 VL · 3B" }
        var repository: String {
            self == .text ? "LiquidAI/LFM2.5-2.6B" : "LiquidAI/LFM2.5-VL-3B"
        }
        var draftRepository: String { repository + "-DSpark" }
    }

    static let sampleImageURL = Bundle.main.url(forResource: "Shapes", withExtension: "png")
    static let imagePrompt = "Describe the shapes and their colors from left to right."

    enum Phase {
        case idle, loading, ready, responding, stopping, resetting, failed
    }

    struct Message: Identifiable {
        let id = UUID()
        let isUser: Bool
        var text: String
        var reasoning = ""
        var toolActivity: [ToolActivity] = []
        var imageURL: URL?
        var metrics: ModelGenerationSummary?
    }

    struct ToolActivity: Identifiable {
        let id = UUID()
        let name: String
        let arguments: String
        var result: String?
    }

    private struct TranscriptEntry: Sendable {
        let role: Chat.Message.Role
        let content: String
        let tool: Chat.Message.Tool?
        var images: [URL] = []
    }

    private(set) var phase = Phase.idle
    private(set) var messages: [Message] = []
    private(set) var downloadProgress = 0.0
    private(set) var model = Model.text
    private(set) var usesDraftModel = false
    var draft = ""
    var draftImageURL: URL?
    var notice: String?

    @ObservationIgnored private var session: ModelSession?
    @ObservationIgnored private var request: ModelGeneration?

    var isResponding: Bool { phase == .responding || phase == .stopping }
    var canSend: Bool {
        phase == .ready && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canSwitchModel: Bool { phase == .ready || phase == .failed || phase == .idle }

    func select(_ model: Model) async {
        guard canSwitchModel, model != self.model else { return }
        phase = .loading
        await session?.close()
        session = nil
        self.model = model
        messages.removeAll()
        draft = ""
        draftImageURL = nil
        phase = .idle
        await load()
    }

    func unload() async {
        guard canSwitchModel else { return }
        phase = .resetting
        await session?.close()
        session = nil
        request = nil
        messages.removeAll()
        draft = ""
        draftImageURL = nil
        notice = nil
        phase = .idle
    }

    func setUsesDraftModel(_ enabled: Bool) async {
        guard canSwitchModel, enabled != usesDraftModel else { return }
        await unload()
        usesDraftModel = enabled
        await load()
    }

    private var instructions: String {
        model == .text
            ? """
            You are a helpful assistant. Give clear, concise answers.
            Use get_weather for weather questions. Its data is simulated;
            always make that clear when reporting the result.
            """
            : "You are a helpful assistant. Describe images accurately and give concise answers."
    }

    private var options: ModelGenerationOptions {
        .init(
            sampling: .init(
                temperature: model == .text ? 0.1 : 0.2, topK: 50,
                // DSpark does not support repetition penalties.
                repetition: usesDraftModel
                    ? nil : .init(model == .text ? 1.1 : 1.0, contextSize: 64)),
            output: .init(maxTokens: 1024, toolCallPolicy: .init(validation: .strict)))
    }

    func describeSampleImage() {
        guard model == .vision, phase == .ready else { return }
        guard let url = Self.sampleImageURL else {
            notice =
                "The sample image is missing from the app bundle. Regenerate the Xcode project."
            return
        }
        draftImageURL = url
        draft = Self.imagePrompt
        send()
    }

    func load() async {
        guard session == nil, phase != .loading else { return }
        if let reason = PlaygroundRuntime.unavailableReason {
            phase = .failed
            notice = reason
            return
        }
        phase = .loading
        notice = nil
        downloadProgress = 0
        do {
            let progress: @Sendable (Progress) -> Void = { [weak self] progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor [weak self] in
                    guard let self, self.phase == .loading else { return }
                    self.downloadProgress = fraction
                }
            }
            // Only the text checkpoint opens <think> before generation.
            let metadata = ModelMetadata(reasoningConfig: model == .text ? .alwaysOnThinking : nil)
            if usesDraftModel {
                let loadedModel = try await NativDraftModel.loadModel(
                    target: .repository(model.repository),
                    draft: .repository(model.draftRepository),
                    metadata: metadata, progress: progress)
                try Task.checkCancellation()
                session = ModelSession(
                    model: loadedModel, instructions: instructions, options: options)
            } else {
                let loadedModel = try await NativModel.loadModel(
                    from: .repository(model.repository), metadata: metadata, progress: progress)
                try Task.checkCancellation()
                session = ModelSession(
                    model: loadedModel, instructions: instructions, options: options)
            }
            phase = .ready
        } catch {
            if Task.isCancelled {
                phase = .idle
            } else {
                phase = .failed
                notice = "Couldn’t load the model. \(error.localizedDescription)"
            }
        }
    }

    func send() {
        guard canSend, let session else { return }
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let imageURL = draftImageURL
        let usesTools = model == .text
        let committedCount = messages.count
        messages.append(Message(isUser: true, text: prompt, imageURL: imageURL))
        draft = ""
        draftImageURL = nil
        notice = nil
        phase = .responding

        Task {
            var previousTranscript: [TranscriptEntry]?
            defer {
                request = nil
                if phase != .failed { phase = .ready }
            }
            do {
                previousTranscript = try await session.withTranscript { transcript in
                    // This example supplies only bundled image URLs, so snapshots remain Sendable.
                    transcript.filter { $0.role != .system }.map {
                        TranscriptEntry(
                            role: $0.role, content: $0.content, tool: $0.tool,
                            images: $0.images.compactMap {
                                if case .url(let url) = $0 { return url }
                                return nil
                            })
                    }
                }
                var input = [
                    TranscriptEntry(
                        role: .user, content: prompt, tool: nil, images: imageURL.map { [$0] } ?? []
                    )
                ]
                var toolRounds = 0
                while true {
                    if phase == .stopping { throw CancellationError() }
                    let messageIndex = messages.count
                    messages.append(Message(isUser: false, text: ""))
                    let response = try await session.generate(
                        input: Self.makeInput(input, usesTools: usesTools))
                    request = response
                    // Stop can be pressed while the session is admitting the request.
                    if phase == .stopping { response.cancel() }
                    var calls: [ToolCall] = []
                    for try await event in response {
                        switch event {
                        case .chunk(let text): messages[messageIndex].text += text
                        case .reasoning(let text): messages[messageIndex].reasoning += text
                        case .toolCall(let call):
                            calls.append(call)
                            let arguments = try JSONEncoder().encode(call.function.arguments)
                            messages[messageIndex].toolActivity.append(
                                ToolActivity(
                                    name: call.function.name,
                                    arguments: String(decoding: arguments, as: UTF8.self)))
                        case .rejectedToolCall(let rejection):
                            throw RejectedToolCallError(rejection)
                        case .protocolError(let detail): throw DemoError(detail)
                        default: break
                        }
                    }
                    // Wait for the assistant turn to commit before submitting tool results.
                    let metrics = try await response.metrics()
                    request = nil
                    if phase == .stopping { throw CancellationError() }
                    messages[messageIndex].metrics = metrics
                    guard !calls.isEmpty else { break }
                    guard toolRounds < 4 else {
                        throw DemoError("The model exceeded the limit of four tool rounds.")
                    }
                    toolRounds += 1
                    input = []
                    for (index, call) in calls.enumerated() {
                        if phase == .stopping { throw CancellationError() }
                        let output = try await call.execute(with: DemoWeatherTool.tool)
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.sortedKeys]
                        let result = String(decoding: try encoder.encode(output), as: UTF8.self)
                        messages[messageIndex].toolActivity[index].result = result
                        input.append(
                            TranscriptEntry(
                                role: .tool, content: result,
                                tool: Chat.Message.tool(
                                    result, id: call.id, name: call.function.name
                                ).tool))
                    }
                }
            } catch {
                request?.cancel()
                _ = try? await request?.metrics()
                let stopped = phase == .stopping || error is CancellationError
                // Each tool round commits separately. Roll back the whole user turn on failure.
                if let previousTranscript {
                    do {
                        try await session.replaceTranscript(Self.makeMessages(previousTranscript))
                        try await session.clearCache()
                    } catch {
                        self.session = nil
                        phase = .failed
                        messages.removeAll()
                        draft = prompt
                        draftImageURL = imageURL
                        notice = "Couldn’t restore the conversation. Retry to load a new session."
                        return
                    }
                }
                messages.removeSubrange(committedCount...)
                draft = prompt
                draftImageURL = imageURL
                notice =
                    stopped
                    ? "Response stopped. Your message is ready to resend."
                    : "Couldn’t finish the response. \(error.localizedDescription)"
            }
        }
    }

    // Build fresh, transferable messages because Chat.Message can also contain non-Sendable media.
    nonisolated private static func makeMessages(_ entries: [TranscriptEntry])
        -> sending [Chat.Message]
    {
        entries.map {
            Chat.Message(
                role: $0.role, content: $0.content,
                images: $0.images.map { .url($0) }, tool: $0.tool)
        }
    }

    nonisolated private static func makeInput(_ entries: [TranscriptEntry], usesTools: Bool)
        -> sending UserInput
    {
        UserInput(
            chat: makeMessages(entries), tools: usesTools ? [DemoWeatherTool.tool.schema] : nil)
    }

    func stop() {
        guard phase == .responding else { return }
        phase = .stopping
        request?.cancel()
    }

    func reset() async {
        guard phase == .ready else { return }
        phase = .resetting
        defer { phase = .ready }
        do {
            try await session?.reset()
            messages.removeAll()
            draft = ""
            draftImageURL = nil
            notice = nil
        } catch {
            notice = "Couldn’t clear the conversation. \(error.localizedDescription)"
        }
    }
}
