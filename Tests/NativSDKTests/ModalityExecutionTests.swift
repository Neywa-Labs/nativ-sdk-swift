import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

struct ExecutionFixtureTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
}
struct ExecutionFixtureProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([1, 2])) }
}
private final class ModalityProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []
    func record(_ event: String) { lock.withLock { events.append(event) } }
    var log: [String] { lock.withLock { events } }
}
private final class AudioFixture: Module, AudioGeneratingLanguageModel {
    let probe: ModalityProbe
    let outputAudioSampleRate = 24000
    init(_ probe: ModalityProbe) { self.probe = probe }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [] }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        probe.record("thinker")
        let logits = MLXArray.zeros([1, inputs.dim(-1), 8])
        logits[.ellipsis, 3] = MLXArray(Float(10))
        return logits
    }
    func synthesizeAudio(input: LMInput, generatedTokens: [Int], options: ModelAudioOptions) throws
        -> MLXArray
    {
        probe.record("audio")
        return MLXArray(generatedTokens.map { Float($0) / 10 }, [1, 1, generatedTokens.count])
    }
}
private final class ImageFixture: PromptImageGenerationModel {
    let supportsImageEditing: Bool
    let probe: ModalityProbe
    let onGenerate: (@Sendable () -> Void)?
    init(editing: Bool, probe: ModalityProbe, onGenerate: (@Sendable () -> Void)? = nil) {
        supportsImageEditing = editing
        self.probe = probe
        self.onGenerate = onGenerate
    }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        probe.record(referenceImages.isEmpty ? "image" : "edit")
        onGenerate?()
        if let reference = referenceImages.first {
            reference[0, 0, 0, 0] = MLXArray(Float(0.25))
        }
        return .init(
            image: referenceImages.first ?? MLXArray.zeros([2, 2, 3]), seed: parameters.seed ?? 0,
            width: parameters.width, height: parameters.height, steps: parameters.steps ?? 4,
            guidance: parameters.guidance ?? 1)
    }
    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult { fatalError("unused") }
}

final class ModalityExecutionTests: XCTestCase {
    func testAudioCapabilityReturnsHostSamplesAndSharesTextAdmission() async throws {
        let probe = ModalityProbe()
        let directory = URL(fileURLWithPath: "/tmp/modality-fixture")
        let loaded = LoadedModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(), model: AudioFixture(probe),
            processor: ExecutionFixtureProcessor(), tokenizer: ExecutionFixtureTokenizer())
        let loadedModel = loaded.intoNativModel()
        let capabilities = await loadedModel.capabilities
        XCTAssertTrue(capabilities.contains(.audio))
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 3))
        let audio = loadedModel.generateAudio(input: UserInput(prompt: "hello"), options: options)
        let text = loadedModel.generate(input: UserInput(prompt: "hello"), options: options)
        let result = try await audio.result()
        for try await _ in text {}
        XCTAssertEqual(result.samples, [0.3, 0.3, 0.3])
        let waveform = await Task.detached { result.waveform }.value
        XCTAssertEqual(waveform.shape, [1, 1, 3])
        XCTAssertEqual(waveform.asArray(Float.self), result.samples)
        XCTAssertEqual(result.shape, waveform.shape)
        XCTAssertEqual(result.sampleRate, 24000)
        XCTAssertEqual(result.tokenIDs, [3, 3, 3])
        XCTAssertEqual(result.summary.generatedTokenCount, 3)
        let audioIndex = try XCTUnwrap(probe.log.firstIndex(of: "audio"))
        XCTAssertTrue(audioIndex == 4 || audioIndex == 8)
        let before = probe.log.count
        let invalid = loadedModel.generateAudio(
            input: UserInput(prompt: "hello"), options: options, audio: .init(chunkSize: 0))
        do {
            _ = try await invalid.result()
            XCTFail("Expected invalid audio options")
        } catch {}
        XCTAssertEqual(probe.log.count, before)
    }
    func testPreparedAudioInputRetainsOwnershipAndRejectsForeignExecutor() async throws {
        func loadedModel() -> NativModel {
            makeTestModel(
                context: .init(
                    metadata: .init(), model: AudioFixture(ModalityProbe()),
                    processor: ExecutionFixtureProcessor(), tokenizer: ExecutionFixtureTokenizer()))
        }
        let owner = loadedModel()
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 2))
        let prepared = try await owner.prepare(input: UserInput(prompt: "audio"))
        let tokens = prepared.inspect { $0.text.tokens.materialized() }
        XCTAssertEqual(tokens.asArray(Int.self), [1, 2])
        let audio = owner.generateAudio(prepared: prepared, options: options)
        let result = try await audio.result()
        XCTAssertEqual(result.samples, [0.3, 0.3])
        let foreign = try await owner.prepare(input: UserInput(prompt: "audio"))
        let other = loadedModel()
        let invalid = other.generateAudio(prepared: foreign, options: options)
        do {
            _ = try await invalid.result()
            XCTFail("Expected foreign input rejection")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .foreignPreparedInput) }
        let next = other.generateAudio(input: UserInput(prompt: "valid"), options: options)
        let nextResult = try await next.result()
        XCTAssertEqual(nextResult.samples, [0.3, 0.3])
    }

    func testImageEditingReturnsHostPixelsAndRejectsInvalidInputsBeforeModelWork() async throws {
        let probe = ModalityProbe()
        let loadedModel = NativImageModel(
            model: ImageFixture(editing: true, probe: probe),
            tokenizer: ExecutionFixtureTokenizer(),
            source: .directory(URL(fileURLWithPath: "/tmp")),
            modelDirectory: URL(fileURLWithPath: "/tmp"))
        let reference = try ModelImageData(
            pixels: Array(repeating: 0.5, count: 12), shape: [1, 3, 2, 2])
        let request = loadedModel.generate(
            prompt: "edit", referenceImages: [reference],
            parameters: .init(seed: 42, steps: 2, width: 2, height: 2))
        let result = try await request.result()
        XCTAssertEqual(result.image.pixels, [0.25] + Array(repeating: 0.5, count: 11))
        XCTAssertEqual(reference.tensor.asArray(Float.self), Array(repeating: 0.5, count: 12))
        let tensorReference = try ModelImageData(tensor: reference.tensor.asType(.float16))
        let edited = try await loadedModel.generate(
            prompt: "edit", referenceImages: [tensorReference]
        )
        .result()
        XCTAssertEqual(edited.image.tensor.dtype, .float32)
        XCTAssertEqual(edited.image.pixels, result.image.pixels)
        XCTAssertEqual(tensorReference.pixels, reference.pixels)
        XCTAssertEqual(result.image.shape, reference.shape)
        XCTAssertEqual(result.seed, 42)
        XCTAssertEqual(result.steps, 2)
        XCTAssertEqual(probe.log, ["edit", "edit"])
        let invalidImages = try [
            ModelImageData(pixels: [0], shape: [1]),
            ModelImageData(pixels: [.nan, 0, 0], shape: [1, 3, 1, 1]),
            ModelImageData(tensor: MLXArray([Float.nan, 0, 0], [1, 3, 1, 1])),
            ModelImageData(tensor: MLXArray([Float(2), 0, 0], [1, 3, 1, 1])),
        ]
        for image in invalidImages {
            do {
                _ = try await loadedModel.generate(prompt: "edit", referenceImages: [image])
                    .result()
                XCTFail("Expected invalid image")
            } catch {}
        }
        XCTAssertEqual(probe.log, ["edit", "edit"])
        let noEdit = NativImageModel(
            model: ImageFixture(editing: false, probe: probe),
            tokenizer: ExecutionFixtureTokenizer(),
            source: .directory(URL(fileURLWithPath: "/tmp")),
            modelDirectory: URL(fileURLWithPath: "/tmp"))
        let unsupported = noEdit.generate(prompt: "edit", referenceImages: [reference])
        do {
            _ = try await unsupported.result()
            XCTFail("Expected unsupported editing")
        } catch {}
        XCTAssertEqual(probe.log, ["edit", "edit"])
        XCTAssertThrowsError(try ModelImageData(pixels: [], shape: [Int.max, 3]))
    }
    func testImageCancellationWaitsForWorkerBeforeNextRequest() async throws {
        let probe = ModalityProbe()
        let reached = expectation(description: "worker reached")
        let gate = DispatchSemaphore(value: 0)
        let loadedModel = NativImageModel(
            model: ImageFixture(
                editing: false, probe: probe,
                onGenerate: {
                    if probe.log.count == 1 {
                        reached.fulfill()
                        _ = gate.wait(timeout: .now() + 5)
                    }
                }), tokenizer: ExecutionFixtureTokenizer(),
            source: .directory(URL(fileURLWithPath: "/tmp")),
            modelDirectory: URL(fileURLWithPath: "/tmp"))
        let first = loadedModel.generate(prompt: "first")
        await fulfillment(of: [reached], timeout: 3)
        first.cancel()
        let second = loadedModel.generate(prompt: "second")
        XCTAssertEqual(probe.log.count, 1)
        gate.signal()
        do {
            _ = try await first.result()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        _ = try await second.result()
        XCTAssertEqual(probe.log, ["image", "image"])
    }
}
