import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class DiffusionGenerationTests: XCTestCase {
    func testModelRoutesMaskedAndBlockDiffusion() async throws {
        for block in [false, true] {
            let model: any LanguageModel =
                block ? ToyBlockDiffusionModel() : ToyMaskedDiffusionModel()
            let directory = URL(fileURLWithPath: "/tmp")
            let loaded = LoadedModel(
                assets: .init(
                    source: .directory(directory), modelDirectory: directory,
                    tokenizerDirectory: directory),
                metadata: .init(), model: model,
                processor: ToyProcessor(), tokenizer: ToyTokenizer())
            let loadedModel = loaded.intoNativModel()
            let request = loadedModel.generate(
                input: UserInput(prompt: "fixture"),
                options: .init(
                    sampling: .init(temperature: 0), output: .init(maxTokens: 4),
                    diffusion: .init(
                        maxTokens: 4, blockLength: 2, steps: 2,
                        eosEarlyStop: false, skipSpecialTokens: false)))
            var chunks: [String] = []
            var completions = 0
            for try await event in request {
                switch event {
                case .chunk(let chunk): chunks.append(chunk)
                case .completion(let summary):
                    completions += 1
                    XCTAssertEqual(summary.promptTokenCount, 2)
                    XCTAssertEqual(summary.generatedTokenCount, 4)
                    XCTAssertEqual(summary.stopReason, .length)
                default: break
                }
            }
            XCTAssertEqual(chunks, block ? ["4 4", "4 4"] : ["3 3", "3 3"])
            XCTAssertEqual(completions, 1)
            if block {
                let counts = try await loadedModel.withModel { _, context in
                    let model = context.model as! ToyBlockDiffusionModel
                    return (model.prefillCount, model.updateTokenCount)
                }
                XCTAssertEqual(counts.0, 1)
                XCTAssertEqual(counts.1, 4)
            }
            let session = ModelSession(
                model: loadedModel,
                options: .init(
                    sampling: .init(temperature: 0), output: .init(maxTokens: 4),
                    diffusion: .init(
                        maxTokens: 4, blockLength: 2, steps: 2,
                        eosEarlyStop: false, skipSpecialTokens: false)))
            let turn = try await session.respond(to: "fixture")
            for try await _ in turn {}
            let result = try await turn.metrics()
            XCTAssertNil(result.cache)
            let transcript = try await session.withTranscript { $0.map(\.content) }
            XCTAssertEqual(transcript, ["fixture", chunks.joined()])
            await session.close()
        }
    }

    func testOwnedDiffusionEmitsBeforeCompletionAndCancellationJoinsWorker() async throws {
        let reached = expectation(description: "second block")
        let gate = DispatchSemaphore(value: 0)
        let model = ToyMaskedDiffusionModel(onLogits: { call in
            if call == 2 {
                reached.fulfill()
                _ = gate.wait(timeout: .now() + 5)
            }
        })
        let directory = URL(fileURLWithPath: "/tmp")
        let loaded = LoadedModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(), model: model, processor: ToyProcessor(),
            tokenizer: ToyTokenizer())
        let request = loaded.intoNativModel().generate(
            input: UserInput(prompt: "fixture"),
            options: .init(
                sampling: .init(temperature: 0),
                diffusion: .init(maxTokens: 6, blockLength: 2, steps: 1, eosEarlyStop: false)))
        await fulfillment(of: [reached], timeout: 3)
        var iterator = request.makeAsyncIterator()
        if case .chunk(let chunk) = try await iterator.next() {
            XCTAssertEqual(chunk, "3 3")
        } else {
            XCTFail("Expected the first committed block")
        }
        request.cancel()
        gate.signal()
        do {
            _ = try await request.metrics()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
    }

    func testDiffusionRejectsUnsupportedOptionsAndHonorsBoundedBuffer() async throws {
        for unsupported in [true, false] {
            let directory = URL(fileURLWithPath: "/tmp")
            let loaded = LoadedModel(
                assets: .init(
                    source: .directory(directory), modelDirectory: directory,
                    tokenizerDirectory: directory),
                metadata: .init(), model: ToyMaskedDiffusionModel(),
                processor: ToyProcessor(), tokenizer: ToyTokenizer())
            let request = loaded.intoNativModel().generate(
                input: UserInput(prompt: "fixture"),
                options: .init(
                    sampling: .init(temperature: 0, repetition: unsupported ? .init(1.1) : nil),
                    output: .init(bufferCapacity: 1),
                    diffusion: .init(maxTokens: 6, blockLength: 2, steps: 1, eosEarlyStop: false)))
            do {
                _ = try await request.metrics()
                XCTFail("Expected rejected options or overflow")
            } catch let error as ModelExecutionError {
                if unsupported {
                    if case .invalidOptions = error {} else { XCTFail("\(error)") }
                } else {
                    XCTAssertEqual(error, .outputBufferOverflow)
                }
            }
        }
        var invalid = DiffusionGenerateParameters()
        invalid.steps = 0
        XCTAssertThrowsError(try invalid.validate())
    }

}

private final class ToyMaskedDiffusionModel: Module, MaskedDiffusionLanguageModel {
    let onLogits: (@Sendable (Int) -> Void)?
    private var calls = 0
    init(onLogits: (@Sendable (Int) -> Void)? = nil) { self.onLogits = onLogits }

    let vocabularySize = 8
    let diffusionMaskTokenID = 7
    let diffusionEOSTokenIDs = [0]
    let diffusionPrefersDiffusionGeneration = true

    func prepare(
        _ input: LMInput, cache _: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache _: [KVCache]?) -> MLXArray {
        diffusionLogits(inputs)
    }

    func newCache(configuration _: KVCacheConfiguration?) -> [KVCache] {
        []
    }

    func diffusionLogits(_ tokens: MLXArray) -> MLXArray {
        calls += 1
        onLogits?(calls)
        let length = tokens.dim(1)
        var values = Array(repeating: Float(-10), count: length * vocabularySize)
        for position in 0 ..< length {
            values[position * vocabularySize + 3] = 10
        }
        return MLXArray(values, [1, length, vocabularySize])
    }
}

private final class ToyBlockDiffusionModel: Module, BlockDiffusionLanguageModel {
    let vocabularySize = 8
    let diffusionCanvasLength = 3
    let diffusionEOSTokenIDs = [0]
    let diffusionPrefersDiffusionGeneration = true
    let diffusionVocabularySize = 8
    var prefillCount = 0
    var updateTokenCount = 0

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        try diffusionPrefill(input: input, cache: cache)
        return .logits(
            LMOutput(
                logits: diffusionDecoderLogits(
                    canvas: MLXArray([1], [1, 1]),
                    cache: cache,
                    selfConditioning: nil,
                ),
            ),
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        diffusionDecoderLogits(canvas: inputs, cache: cache ?? [], selfConditioning: nil)
    }

    func newCache(configuration _: KVCacheConfiguration?) -> [KVCache] {
        []
    }

    func diffusionMakeCache(parameters _: DiffusionGenerateParameters?) -> [KVCache] {
        []
    }

    func diffusionPrefill(input _: LMInput, cache _: [KVCache]) throws {
        prefillCount += 1
    }

    func diffusionUpdateCache(tokens: MLXArray, cache _: [KVCache]) {
        updateTokenCount += tokens.dim(1)
    }

    func diffusionDecoderLogits(
        canvas: MLXArray,
        cache _: [KVCache],
        selfConditioning _: MLXArray?,
    )
        -> MLXArray
    {
        let length = canvas.dim(1)
        var values = Array(repeating: Float(-10), count: length * vocabularySize)
        for position in 0 ..< length {
            values[position * vocabularySize + 4] = 10
        }
        return MLXArray(values, [1, length, vocabularySize])
    }

    func diffusionSelfConditioning(logits: MLXArray) -> MLXArray? {
        logits
    }
}

private struct ToyTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] {
        []
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_: String) -> Int? {
        nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        throw TokenizerError.missingChatTemplate
    }
}

private struct ToyProcessor: UserInputProcessor {
    func prepare(input _: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([1, 2], [1, 2]))
    }
}
