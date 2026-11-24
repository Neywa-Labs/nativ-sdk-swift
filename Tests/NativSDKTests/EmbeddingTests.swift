import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct EmbeddingTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { Int($0 % 30) + 1 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}
private struct EmbeddingTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { EmbeddingTokenizer() }
}

final class EmbeddingTests: XCTestCase {
    private let json =
        #"{"model_type":"qwen3","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":32,"rms_norm_eps":0.000001,"max_position_embeddings":32,"tie_word_embeddings":true}"#

    private func model(pooling: EmbeddingPooling = .lasttoken) throws -> Qwen3EmbeddingModel {
        try Qwen3EmbeddingModel(
            JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8)), pooling: pooling)
    }
    private func checkpoint(quantized: Bool = false) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let model = try model()
        if quantized {
            quantize(model: model, groupSize: 32, bits: 4)
            config["quantization"] = ["bits": 4, "group_size": 32]
        }
        try JSONSerialization.data(withJSONObject: config).write(
            to: root.appendingPathComponent("config.json"))
        // Source checkpoints may omit the outer model prefix and retain an unused LM head.
        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                (String($0.0.dropFirst(6)), $0.1)
            })
        weights["lm_head.weight"] = MLXArray.zeros([32, 64])
        try save(arrays: weights, url: root.appendingPathComponent("model.safetensors"))
        return root
    }
    private func load(_ directory: URL) async throws -> NativEmbeddingModel {
        try await NativEmbeddingModel.loadModel(
            from: .directory(directory), using: EmbeddingTokenizerLoader())
    }
    private func vector(_ output: EmbeddingOutput) throws -> MLXArray {
        guard case .pooled(let array) = output else {
            throw ModelExecutionError.invalidOptions("Expected pooled embeddings")
        }
        return array
    }

    func testEmbeddingFactoryLoadsQuantizedWeightsAndUsesRawText() async throws {
        let root = try checkpoint(quantized: true)
        let loadedModel = try await load(root)
        let capabilities = await loadedModel.capabilities
        XCTAssertEqual(capabilities, [.embedding])
        let result = try await loadedModel.embed(input: .text("abc")).result()
        guard case .vector(let values) = result.embeddings else {
            return XCTFail("Expected one vector")
        }
        XCTAssertEqual(result.tensor.shape, [64])
        XCTAssertEqual(result.tensor.asArray(Float.self), values)
        XCTAssertEqual(values.count, 64)
        XCTAssertEqual(result.tokenCount, 3)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        XCTAssertEqual(values.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-5)
        let prepared = try await loadedModel.prepare(input: UserInput(prompt: .text("abc")))
        let repeated = try await loadedModel.embed(prepared: prepared).result()
        XCTAssertEqual(repeated.embeddings, result.embeddings)
        let other = try await loadedModel.embed(input: .text("xyz")).result()
        XCTAssertNotEqual(result.embeddings, other.embeddings)
    }

    func testDefaultLanguageLoadingAndUnsupportedOperations() async throws {
        let root = try checkpoint()
        // Save matching language weights so the default path is checked independently.
        let language = Qwen3Model(
            try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8)))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: language.parameters().flattened()),
            url: root.appendingPathComponent("model.safetensors"))
        let loadedModel = try await NativModel.loadModel(
            from: .directory(root), using: EmbeddingTokenizerLoader())
        let capabilities = await loadedModel.capabilities
        XCTAssertTrue(capabilities.contains(.autoregressive))
        XCTAssertFalse(capabilities.contains(.embedding))
        do {
            _ = try await loadedModel.withModel { _, context in
                try context.embeddingModel.maximumSequenceLength
            }
            XCTFail("Language resources must reject embedding access")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .unsupportedOperation("embed")) }
        let encoder = try await load(root)
        do {
            _ = try await encoder.withModel { _, context in
                _ = try context.languageModel
                return true
            }
            XCTFail("Embedding resources must reject generation access")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .unsupportedOperation("generate")) }
        // Failed operations release admission and leave encoding usable.
        _ = try await encoder.embed(input: .text("abc")).result()
    }

    func testPoolingIgnoresPaddingAndQwenAttentionPreservesValidTokens() throws {
        let hidden = MLXArray([Float(100), -100, 1, 2, 3, 4, -50, 90], [1, 4, 2])
        let mask = MLXArray([[0, 1, 1, 0]])
        for (pool, expected) in [
            (EmbeddingPooling.mean, [Float(2), 3]), (.cls, [1, 2]), (.lasttoken, [3, 4]),
            (.max, [3, 4]),
        ] {
            XCTAssertEqual(pool.apply(hidden, mask: mask).asArray(Float.self), expected)
            let model = try model(pooling: pool)
            let plain = try vector(model.embed(LMInput(tokens: MLXArray([[1, 2, 3]]))))
            for (ids, valid) in [
                ([0, 0, 1, 2, 3], [0, 0, 1, 1, 1]), ([1, 2, 3, 0, 0], [1, 1, 1, 0, 0]),
            ] {
                let padded = try vector(
                    model.embed(
                        LMInput(text: .init(tokens: MLXArray([ids]), mask: MLXArray([valid])))))
                // Different sequence lengths can select different Metal TF32 kernels.
                XCTAssertTrue(
                    allClose(plain, padded, rtol: 3e-3, atol: 5e-4).item(Bool.self),
                    "\(pool), ids \(ids), max error \(abs(plain - padded).max().item(Float.self))")
            }
        }
        let encoder = try model()
        for ids in [[], [-1], [32]] {
            XCTAssertThrowsError(try encoder.embed(LMInput(tokens: MLXArray(ids))))
        }
        XCTAssertThrowsError(
            try encoder.embed(
                LMInput(text: .init(tokens: MLXArray([[1, 2]]), mask: MLXArray([[0, 0]])))))
    }

    func testPoolingSidecarChangesEncodingAndRejectsUnsupportedModes() async throws {
        let root = try checkpoint()
        let defaultExecutor = try await load(root)
        let last = try await defaultExecutor.embed(input: .text("abc")).result()
        let folder = root.appendingPathComponent("1_Pooling")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appendingPathComponent("config.json")
        try Data(#"{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}"#.utf8).write(
            to: file)
        let meanExecutor = try await load(root)
        let mean = try await meanExecutor.embed(input: .text("abc")).result()
        XCTAssertNotEqual(last.embeddings, mean.embeddings)
        for text in [
            #"{"pooling_mode":"weightedmean"}"#,
            #"{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":true}"#,
            #"{"pooling_mode":"mean","include_prompt":false}"#,
        ] {
            try Data(text.utf8).write(to: file)
            do {
                _ = try await load(root)
                XCTFail("Unsupported pooling must fail at load time")
            } catch { XCTAssertNotNil(error as? ModelExecutionError) }
        }
    }

    func testInputLimitsAndForeignPreparedInputsRejectWithoutPoisoningExecutor() async throws {
        let root = try checkpoint()
        try Data(#"{"max_seq_length":3}"#.utf8).write(
            to: root.appendingPathComponent("sentence_bert_config.json"))
        let a = try await load(root)
        let b = try await NativEmbeddingModel.loadModel(
            assets: ResolvedModelAssets(
                source: .directory(root), modelDirectory: root, tokenizerDirectory: root),
            metadata: .init(stopStrings: ["caller-stop"]),
            tokenizerLoader: EmbeddingTokenizerLoader())
        XCTAssertEqual(b.metadata.stopStrings, ["caller-stop"])
        for text in ["", "    ", "abcd"] {
            do {
                _ = try await a.embed(input: .text(text)).result()
                XCTFail("Invalid input must fail")
            } catch { XCTAssertNotNil(error as? ModelExecutionError) }
        }
        do {
            _ = try await a.prepare(input: UserInput(chat: [.user("abc")]))
            XCTFail("Embeddings must not flatten a conversation")
        } catch { XCTAssertNotNil(error as? ModelExecutionError) }
        let prepared = try await a.prepare(input: UserInput(prompt: .text("abc")))
        do {
            _ = try await b.embed(prepared: prepared).result()
            XCTFail("Foreign input must fail")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .foreignPreparedInput) }
        _ = try await b.embed(input: .text("abc")).result()
    }

    func testUnsupportedEmbeddingArchitectureDoesNotFallBackToGeneration() async throws {
        let root = try checkpoint()
        let config = json.replacingOccurrences(of: "qwen3", with: "llama")
        try Data(config.utf8).write(to: root.appendingPathComponent("config.json"))
        do {
            _ = try await load(root)
            XCTFail("An unsupported embedding architecture must fail")
        } catch { XCTAssertNotNil(error as? ModelFactoryError) }
    }
}

private actor EmbeddingGate {
    private var continuation: CheckedContinuation<Void, Never>?
    var entered: Bool { continuation != nil }
    func wait() async { await withCheckedContinuation { continuation = $0 } }
    func release() {
        continuation?.resume()
        continuation = nil
    }
}

private final class TokenEmbeddingModel: Module, EmbeddingModel {
    let maximumSequenceLength = 32
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let ids = input.text.tokens.asType(.float32)
        return .tokens(stacked([ids, -ids], axis: -1))
    }
}
private struct MaskedEmbeddingProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        let single = if case .text("single") = input.prompt { true } else { false }
        let mask = single ? [0, 0, 0, 1] : [0, 1, 0, 1]
        return LMInput(
            text: .init(tokens: MLXArray([[9, 1, 9, 2]]), mask: MLXArray(mask, [1, 4])))
    }
}

extension EmbeddingTests {
    func testTokenEmbeddingResultsRemovePaddingAndOwnHostValues() async throws {
        let directory = FileManager.default.temporaryDirectory
        let assets = ResolvedModelAssets(
            source: .directory(directory), modelDirectory: directory, tokenizerDirectory: directory)
        let context = ModelContext(
            metadata: .init(), model: TokenEmbeddingModel(), processor: MaskedEmbeddingProcessor(),
            tokenizer: EmbeddingTokenizer())
        let loadedModel = NativEmbeddingModel(
            assets: assets, metadata: .init(), context: ExecutionTransfer(context), policy: .init())
        let result = try await loadedModel.embed(input: .text("ignored")).result()
        let tensor = await Task.detached { result.tensor }.value
        XCTAssertEqual(tensor.shape, [2, 2])
        XCTAssertEqual(tensor.asArray(Float.self), [1, -1, 2, -2])
        let single = try await loadedModel.embed(input: .text("single")).result()
        XCTAssertEqual(single.tensor.shape, [1, 2])
        XCTAssertEqual(single.embeddings, .tokenVectors([[2, -2]]))
        XCTAssertEqual(result.embeddings, .tokenVectors([[1, -1], [2, -2]]))
        XCTAssertEqual(result.tokenCount, 2)
    }

    func testCancellingQueuedEmbeddingReleasesAdmissionForFollowingRequest() async throws {
        let root = try checkpoint()
        let loadedModel = try await load(root)
        let gate = EmbeddingGate()
        let blocker = Task { try await loadedModel.withModel { _, _ in await gate.wait() } }
        for _ in 0 ..< 10_000 {
            if await gate.entered { break }
            await Task.yield()
        }
        let entered = await gate.entered
        XCTAssertTrue(entered)
        guard entered else {
            blocker.cancel()
            return
        }
        let cancelled = loadedModel.embed(input: .text("abc"))
        for _ in 0 ..< 10_000 {
            if await loadedModel.queuedRequestCount > 0 { break }
            await Task.yield()
        }
        let queued = await loadedModel.queuedRequestCount
        XCTAssertEqual(queued, 1)
        cancelled.cancel()
        do {
            _ = try await cancelled.result()
            XCTFail("Queued encoding must cancel")
        } catch { XCTAssertTrue(error is CancellationError) }
        await gate.release()
        try await blocker.value
        _ = try await loadedModel.embed(input: .text("abc")).result()
    }
}
