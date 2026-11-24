import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct LoadingTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.compactMap { $0["content"] as? String }.joined().utf8.map(Int.init)
    }
}

private struct LoadingTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { LoadingTokenizer() }
}

private struct LoadingMessageGenerator: MessageGenerator {
    let marker: String
    func generate(message: Chat.Message) -> Message {
        ["role": message.role.rawValue, "content": marker]
    }
}

private final class LoadingTextModel: Module, LLMModel {
    var loraLayers: [Module] { [] }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [] }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        LoadingMessageGenerator(marker: "model")
    }
}

final class ModelLoadingTests: XCTestCase {
    func testGemma3nRoutesFlatNestedAndMultimodalConfigurations() async throws {
        let data = try textConfigurationData()
        let nested = try ModelRegistry.default.createModel(
            configuration: data, modelType: "gemma3n")
        XCTAssertTrue(nested is Gemma3nTextModel)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let flat = try JSONSerialization.data(withJSONObject: XCTUnwrap(object["text_config"]))
        let text = try ModelRegistry.default.createModel(
            configuration: flat, modelType: "gemma3n_text")
        XCTAssertTrue(text is Gemma3nTextModel)
        let vision = try ModelRegistry.default.createModel(
            configuration: Self.multimodalConfigData, modelType: "gemma3n")
        XCTAssertTrue(vision is Gemma3n)
        var causal = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Self.multimodalConfigData) as? [String: Any])
        causal["architectures"] = ["Gemma3nForCausalLM"]
        let languageOnly = try ModelRegistry.default.createModel(
            configuration: JSONSerialization.data(withJSONObject: causal), modelType: "gemma3n")
        XCTAssertTrue(languageOnly is Gemma3nTextModel)
    }

    func testGenericLoadingSupportsTextWeightsWithoutProcessorMetadata() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try textConfigurationData()
        try data.write(to: directory.appendingPathComponent("config.json"))
        let original = try ModelRegistry.default.createModel(
            configuration: data, modelType: "gemma3n")
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))

        let loadedModel = try await NativModel.loadModel(
            from: .directory(directory), using: LoadingTokenizerLoader())
        let expected = original(
            MLXArray([2, 3], [1, 2]), cache: try original.newCache(configuration: nil)
        ).asArray(Float.self)
        try await loadedModel.withModel { _, context in
            XCTAssertTrue(context.model is Gemma3nTextModel)
            XCTAssertTrue(context.processor is TextProcessor)
            XCTAssertEqual(context.metadata.eosTokenIds, [1, 106])
            let actual = try context.languageModel(
                MLXArray([2, 3], [1, 2]),
                cache: try context.languageModel.newCache(configuration: nil))
            let expected = MLXArray(expected).reshaped(actual.shape)
            XCTAssertTrue(allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        }

        let configured = try await NativModel.loadModel(
            assets: loadedModel.assets,
            metadata: .init(messageGenerator: LoadingMessageGenerator(marker: "caller")),
            tokenizerLoader: LoadingTokenizerLoader())
        XCTAssertTrue(configured.metadata.messageGenerator is LoadingMessageGenerator)
        let prepared = try await configured.prepare(input: UserInput(prompt: "input"))
        let tokens = prepared.inspect { $0.text.tokens.asArray(Int.self) }
        XCTAssertEqual(tokens, "caller".utf8.map(Int.init))
    }

    func testDiffusionGemmaRetainsMediaProcessorAndTextFallback() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = LoadingTextModel()
        let fallback = try await ModelLoader().loadProcessor(
            from: directory, modelType: "diffusion_gemma", configData: Data("{}".utf8),
            model: model, tokenizer: LoadingTokenizer())
        XCTAssertTrue(fallback is TextProcessor)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("preprocessor_config.json"))
        let multimodal = try await ModelLoader().loadProcessor(
            from: directory, modelType: "diffusion_gemma", configData: Data("{}".utf8),
            model: model, tokenizer: LoadingTokenizer())
        XCTAssertTrue(multimodal is DiffusionGemmaProcessor)
    }

    func testTextProcessorUsesModelAndCallerMessageGeneratorsWithoutVisionMetadata() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Text-only models may be distributed alongside irrelevant vision processor files.
        try Data("{processor_class: 'UnavailableVisionProcessor'}".utf8)
            .write(to: directory.appendingPathComponent("preprocessor_config.json"))
        for modelType in ["mistral3", "mistral4", "gemma3n", "gemma3n_text", "cohere2_moe"] {
            let model = LoadingTextModel()
            let processor = try await ModelLoader().loadProcessor(
                from: directory, modelType: modelType, configData: Data("{}".utf8),
                model: model, tokenizer: LoadingTokenizer())
            let prepared = try await processor.prepare(input: UserInput(prompt: "input"))
            XCTAssertEqual(prepared.text.tokens.asArray(Int.self), "model".utf8.map(Int.init))
            let configured = try await ModelLoader().loadProcessor(
                from: directory, modelType: modelType, configData: Data("{}".utf8),
                model: model, tokenizer: LoadingTokenizer(),
                messageGenerator: LoadingMessageGenerator(marker: "caller"))
            let callerPrepared = try await configured.prepare(input: UserInput(prompt: "input"))
            XCTAssertEqual(
                callerPrepared.text.tokens.asArray(Int.self), "caller".utf8.map(Int.init))
        }
    }

    func testTextProcessorRejectsAudioAndChatOnlyMedia() async throws {
        let processor = TextProcessor(.init(), tokenizer: LoadingTokenizer())
        var chat = UserInput(chat: [.user("audio", audios: [.array(MLXArray([Float(0)]))])])
        chat.audios = []
        let audio = UserInput(prompt: "audio", audios: [.array(MLXArray([Float(0)]))])
        for input in [audio, chat] {
            do {
                _ = try await processor.prepare(input: input)
                XCTFail("Text processing should reject audio")
            } catch TextProcessorError.mediaInputsUnsupported {}
        }
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func textConfigurationData() throws -> Data {
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Self.multimodalConfigData) as? [String: Any])
        var text = try XCTUnwrap(root["text_config"] as? [String: Any])
        text["eos_token_id"] = [1, 106]
        return try JSONSerialization.data(withJSONObject: [
            "model_type": "gemma3n", "text_config": text,
        ])
    }

    private static let multimodalConfigData = Data(
        """
        {
          "model_type": "gemma3n",
          "image_token_id": 262145,
          "audio_token_id": 262273,
          "vision_soft_tokens_per_image": 256,
          "audio_soft_tokens_per_image": 188,
          "hidden_size": 16,
          "eos_token_id": [1, 106],
          "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": [32, 32],
            "num_attention_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "query_pre_attn_scalar": null,
            "vocab_size_per_layer_input": 256,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "activation_sparsity_pattern": [0.0, 0.0],
            "hidden_size_per_layer_input": 8,
            "altup_num_inputs": 2,
            "altup_coef_clip": null,
            "altup_correct_scale": false,
            "altup_active_idx": 0,
            "laurel_rank": 4,
            "rope_scaling": null,
            "sliding_window_pattern": null
          },
          "vision_config": {
            "model_type": "gemma3n_vision",
            "hidden_size": 2048,
            "image_size": null,
            "vocab_size": 128,
            "vocab_offset": 262144,
            "rms_norm_eps": 1e-6
          },
          "audio_config": {
            "hidden_size": 1536,
            "vocab_size": 128,
            "vocab_offset": 262272
          }
        }
        """.utf8)
}
