import Foundation
import ImageIO
import NativSDK
import Observation

@MainActor
@Observable
final class ImageGenerationModel {
    static let repository = "Qwen/Qwen-Image-2.1"
    static let samplePrompt = "A cobalt blue ceramic teapot on a wooden table in soft window light."

    enum Phase {
        case idle, loading, ready, generating, stopping
    }

    struct Output {
        let image: CGImage
        let png: Data
        let prompt: String
        let seed: UInt64
        let size: Int
        let steps: Int
    }

    var prompt = samplePrompt
    var size = 512
    var steps = 30
    private var configuredSeed = 42
    var seed: Int {
        get { configuredSeed }
        set { configuredSeed = min(999_999, max(0, newValue)) }
    }
    var notice: String?
    private(set) var phase = Phase.idle
    private(set) var downloadProgress = 0.0
    private(set) var startedAt: Date?
    private(set) var output: Output?

    @ObservationIgnored private var model: NativImageModel?
    @ObservationIgnored private var request: ModelOperation<ModelImageResult>?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var operationID: UUID?

    var isBusy: Bool { phase == .loading || phase == .generating || phase == .stopping }
    var canGenerate: Bool {
        !isBusy && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func generate() {
        guard canGenerate else { return }
        if let reason = PlaygroundRuntime.unavailableReason {
            notice = reason
            return
        }
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let size = size
        let steps = steps
        let seed = UInt64(seed)
        let id = UUID()
        operationID = id
        notice = nil
        downloadProgress = 0
        startedAt = Date()
        phase = model == nil ? .loading : .generating
        task = Task { [self] in
            defer {
                request = nil
                task = nil
                operationID = nil
                startedAt = nil
                phase = model == nil ? .idle : .ready
            }
            do {
                if model == nil {
                    let loaded = try await NativImageModel.loadModel(
                        from: .repository(Self.repository)
                    ) { [weak self] progress in
                        let fraction = progress.fractionCompleted
                        Task { @MainActor [weak self] in
                            guard let self, self.operationID == id, self.phase == .loading else {
                                return
                            }
                            self.downloadProgress = fraction
                        }
                    }
                    try Task.checkCancellation()
                    model = loaded
                }
                try Task.checkCancellation()
                guard let model else { return }
                phase = .generating
                let operation = model.generate(
                    prompt: prompt,
                    parameters: .init(
                        seed: seed, steps: steps, width: size, height: size, guidance: 1))
                request = operation
                let result = try await operation.result()
                try Task.checkCancellation()
                let pixels = result.image
                let png = try await Task.detached(priority: .userInitiated) {
                    try pixels.pngData()
                }.value
                try Task.checkCancellation()
                guard let source = CGImageSourceCreateWithData(png as CFData, nil),
                    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else {
                    throw DemoError("Couldn’t display the generated image.")
                }
                output = Output(
                    image: image, png: png, prompt: prompt, seed: result.seed,
                    size: result.width, steps: result.steps)
            } catch {
                request?.cancel()
                _ = try? await request?.result()
                notice =
                    Task.isCancelled || error is CancellationError
                    ? "Generation stopped. Your prompt and previous image are kept."
                    : "Couldn’t generate the image. \(error.localizedDescription)"
            }
        }
    }

    func stop() {
        guard isBusy, phase != .stopping else { return }
        phase = .stopping
        task?.cancel()
        request?.cancel()
    }

    func unload() {
        guard !isBusy else { return }
        model = nil
        phase = .idle
    }
}
