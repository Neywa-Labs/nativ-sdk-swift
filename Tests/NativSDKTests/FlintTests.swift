import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct FlintTestTokenizer: Tokenizer {
    var bosToken: String? { "<|begin_of_text|>" }
    var eosToken: String? { "<|end_of_text|>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { token == eosToken ? 120 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 120 ? eosToken : nil }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text =
            messages.map { message in
                var text = "<|\(message["role"] as? String ?? "user")|>"
                if let reasoning = message["reasoning_content"] as? String {
                    text += "<think>\(reasoning)</think>"
                }
                text += message["content"] as? String ?? ""
                for call in message["tool_calls"] as? [[String: any Sendable]] ?? [] {
                    let function = call["function"] as! [String: any Sendable]
                    let arguments = function["arguments"] as! [String: any Sendable]
                    text +=
                        "<tool_call>\(function["name"]!)<arg_key>query</arg_key><arg_value>\(arguments["query"]!)</arg_value></tool_call>"
                }
                return text
            }.joined() + "<|assistant|>"
        return encode(text: text, addSpecialTokens: false)
    }
}
private struct FlintTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { FlintTestTokenizer() }
}

final class FlintTests: XCTestCase {
    private func configuration(window: Int = 5) -> Data {
        Data(
            """
            {"model_type":"flint","hidden_size":64,"intermediate_size":128,"num_hidden_layers":3,
             "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":128,
             "layer_types":["full_attention","sliding_attention","full_attention"],
             "no_rope_layer_indices":[0,2],"mlp_only_layers":[0],"num_experts":4,
             "num_experts_per_tok":2,"moe_intermediate_size":64,"shared_expert_intermediate_size":64,
             "sliding_window":\(window),"eos_token_id":[120,121,122,123]}
            """.utf8)
    }
    private func model(window: Int = 5) throws -> Flint {
        MLXRandom.seed(47)
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(
                configuration: configuration(window: window), modelType: "flint") as? Flint)
        model.train(false)
        return model
    }
    private func directory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: directory) }
        return directory
    }
    private func sourceWeights(_ model: Flint) -> [String: MLXArray] {
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for index in [1, 2] {
            let prefix = "model.layers.\(index).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard
                        let value = weights.removeValue(
                            forKey: prefix + "switch_mlp.\(projection).\(suffix)")
                    else { continue }
                    for expert in 0 ..< 4 {
                        weights[prefix + "experts.\(expert).\(projection).\(suffix)"] =
                            value[expert]
                    }
                }
            }
            weights.removeValue(forKey: prefix + "expert_bias")
        }
        return weights
    }

    func testConfigurationAndHybridCacheRegistration() throws {
        let model = try model(window: 513)
        XCTAssertTrue(model.model.layers[0].mlp is Qwen3MLP)
        XCTAssertTrue(model.model.layers[1].mlp is FlintLayers.MoE)
        XCTAssertNil(model.model.layers[0].attention.rope)
        XCTAssertNotNil(model.model.layers[1].attention.rope)
        XCTAssertNil(model.model.layers[2].attention.rope)
        let cache = try model.newCache(configuration: nil)
        XCTAssertTrue(cache[0] is KVCacheSimple)
        XCTAssertTrue(cache[1] is RotatingKVCache)
        XCTAssertEqual(cache[1].maxSize, 513)
        XCTAssertEqual((cache[1] as? RotatingKVCache)?.keepCount, 0)
        var data = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        for (key, value) in [
            ("layer_types", ["bad"] as Any), ("num_experts_per_tok", 5), ("sliding_window", 0),
            ("hidden_act", "relu"),
        ] {
            var invalid = data
            invalid[key] = value
            XCTAssertThrowsError(
                try ModelRegistry.default.createModel(
                    configuration: JSONSerialization.data(withJSONObject: invalid),
                    modelType: "flint"))
        }
        data["use_sliding_window"] = false
        let noWindow = Flint(
            try JSONDecoder().decode(
                FlintConfiguration.self, from: JSONSerialization.data(withJSONObject: data)))
        XCTAssertTrue(try noWindow.newCache(configuration: nil).allSatisfy { $0 is KVCacheSimple })
        XCTAssertEqual(model.toolCallFormat, .glm4)
        XCTAssertEqual(model.reasoningConfig?.promptStrategy, ReasoningPromptStrategy.none)
        XCTAssertThrowsError(
            try model.newCache(configuration: .init(strategy: .turboQuant(.balanced))))
    }

    func testRouterBiasOnlySelectsExpertsAndWeightsUseUnbiasedSigmoid() throws {
        let model = try model()
        let moe = try XCTUnwrap(model.model.layers[1].mlp as? FlintLayers.MoE)
        let weights = MLXArray([-2.0 as Float, -1, 1, 2])[0..., .newAxis] * MLXArray.ones([4, 64])
        try moe.update(
            parameters: .unflattened([
                ("gate.weight", weights), ("expert_bias", MLXArray([10.0 as Float, 10, 0, 0])),
            ]), verify: .noUnusedKeys)
        let (indices, scores) = moe.route(MLXArray.ones([1, 1, 64]) / 64)
        XCTAssertEqual(Set(indices.asArray(Int.self)), [0, 1])
        let expected = sigmoid(MLXArray([-2.0 as Float, -1]))
        let normalized = expected / expected.sum()
        for (index, score) in zip(indices.asArray(Int.self), scores.asArray(Float.self)) {
            XCTAssertEqual(score, normalized[index].item(Float.self), accuracy: 1e-6)
        }
        XCTAssertEqual(scores.sum().item(Float.self), 1, accuracy: 1e-6)
    }

    func testLearnedSinksSuppressAttentionAndOutputProjectionIsFloat32() throws {
        let model = try model()
        let attention = model.model.layers[0].attention
        let x = MLXArray.ones([1, 4, 64])
        let baseline = attention(x, cache: nil)
        try attention.update(
            parameters: .unflattened([
                ("softmax_offset", MLXArray.full([4], values: MLXArray(Float(20))))
            ]), verify: .noUnusedKeys)
        let suppressed = attention(x, cache: nil)
        XCTAssertLessThan(
            abs(suppressed).max().item(Float.self), abs(baseline).max().item(Float.self) / 100)
        try model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }), verify: .all
        )
        XCTAssertEqual(model(MLXArray([1, 2, 3], [1, 3]), cache: nil).dtype, .float32)
    }

    func testSlidingWindowBoundaryAndCachedContinuation() throws {
        let model = try model(window: 513)
        let ids = ((MLXArray(0 ..< 545) * 7 + 3) % 128)[.newAxis]
        let expected = model(ids, cache: nil)[0, -1]
        let cache = try model.newCache(configuration: nil)
        var start = 0
        var actual: MLXArray?
        for count in [256, 256, 32, 1] {
            actual = model(ids[0..., start ..< start + count], cache: cache)
            start += count
            eval(actual!, cache)
        }
        XCTAssertEqual(cache.map(\.offset), [545, 545, 545])
        XCTAssertLessThan(abs(actual![0, -1] - expected).max().item(Float.self), 5e-3)
        let next = model(MLXArray([11], [1, 1]), cache: cache)
        let complete = model(concatenated([ids, MLXArray([11], [1, 1])], axis: 1), cache: nil)[
            0, -1]
        XCTAssertLessThan(abs(next[0, 0] - complete).max().item(Float.self), 5e-3)
        XCTAssertEqual(cache[1].maxSize, 513)
    }

    func testAffineCacheContinuationRetainsLearnedSinks() throws {
        var config = try JSONDecoder().decode(FlintConfiguration.self, from: configuration())
        config.headDim = 64
        MLXRandom.seed(47)
        let model = Flint(config)
        model.train(false)
        for layer in model.model.layers {
            try layer.attention.update(
                parameters: .unflattened([
                    ("softmax_offset", MLXArray([0.0 as Float, 1, 2, 3]))
                ]), verify: .noUnusedKeys)
        }
        let ids = MLXArray([1, 4, 7, 2, 9, 3, 8, 5, 6], [1, 9])
        let expected = model(ids, cache: nil)
        var cache = try model.newCache(configuration: nil)
        for index in [0, 2] { cache[index] = QuantizedKVCache(groupSize: 32, bits: 8) }
        var outputs = [model(ids[0..., ..<6], cache: cache)]
        for index in 6 ..< 9 {
            outputs.append(model(ids[0..., index ..< index + 1], cache: cache))
        }
        let actual = concatenated(outputs, axis: 1)
        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.02)
        XCTAssertEqual(cache.map(\.offset), [9, 9, 9])
    }

    func testIndividualAndStackedCheckpointWeightsReloadStrictly() throws {
        let reference = try model()
        let directory = try directory()
        let ids = MLXArray([1, 4, 7, 2, 9, 3, 8, 5, 6], [1, 9])
        let expected = reference(ids, cache: nil)
        let source = sourceWeights(reference)
        let native = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        for weights in [source, native] {
            let once = reference.sanitize(weights: weights)
            let twice = reference.sanitize(weights: once)
            XCTAssertEqual(Set(once.keys), Set(native.keys))
            for (key, value) in once {
                XCTAssertTrue(arrayEqual(twice[key]!, value).item(Bool.self), key)
            }
            try MLX.save(
                arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
            let loaded = try model()
            try CheckpointLoader.load(modelDirectory: directory, model: loaded)
            XCTAssertTrue(arrayEqual(loaded(ids, cache: nil), expected).item(Bool.self))
        }
        var partial = source
        partial.removeValue(forKey: "model.layers.1.mlp.experts.2.up_proj.weight")
        try MLX.save(arrays: partial, url: directory.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try CheckpointLoader.load(modelDirectory: directory, model: model()))
    }

    func testMixedEmbeddingExportLoadsWrappedAndBytePackedWeights() async throws {
        for wrapped in [false, true] {
            for bytePacked in [false, true] {
                let reference = try model()
                let directory = try directory()
                quantize(model: reference) { path, module in
                    guard module is Linear || module is Embedding || module is SwitchLinear else {
                        return nil
                    }
                    return (32, path == "model.embed_tokens" ? 8 : 4, .affine)
                }
                let original = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
                let arrays = Dictionary(
                    uniqueKeysWithValues: original.map { key, value in
                        let path = String(key.dropLast(7))
                        let value =
                            bytePacked && key.hasSuffix(".weight")
                                && original[path + ".scales"] != nil
                            ? value.view(dtype: .uint8) : value
                        return (wrapped ? "language_model." + key : key, value)
                    })
                var config = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
                config["quantization"] = ["group_size": 32, "bits": 4]
                config["mlx_export"] = ["embedding_bits": 8]
                try JSONSerialization.data(withJSONObject: config).write(
                    to: directory.appendingPathComponent("config.json"))
                try MLX.save(
                    arrays: arrays, metadata: ["format": "mlx"],
                    url: directory.appendingPathComponent("model.safetensors"))
                let expected = reference(MLXArray([1, 4, 2], [1, 3]), cache: nil).asArray(
                    Float.self)
                let loaded = try await NativModel.loadModel(
                    from: .directory(directory), using: FlintTestTokenizerLoader())
                try await loaded.withModel { _, context in
                    let model = try XCTUnwrap(context.model as? Flint)
                    XCTAssertEqual(
                        model(MLXArray([1, 4, 2], [1, 3]), cache: nil).asArray(Float.self), expected
                    )
                }
            }
        }
    }

    func testExplicitEmbeddingQuantizationOverridesExportHint() throws {
        var data = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        data["mlx_export"] = ["embedding_bits": 8]
        let model = Flint(
            try JSONDecoder().decode(
                FlintConfiguration.self, from: JSONSerialization.data(withJSONObject: data)))
        let original = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 32, bits: 4),
            perLayerQuantization: [
                "language_model.model.embed_tokens": .quantize(.init(groupSize: 32, bits: 6))
            ])
        let result = try XCTUnwrap(model.checkpointQuantization(original))
        XCTAssertEqual(result.quantization(layer: "model.embed_tokens")?.bits, 6)
    }

    func testConversionUsesEightBitRoutersAndPreservesPythonNamespace() throws {
        for wrapped in [false, true] {
            let reference = try model()
            let source = try directory()
            let destination = source.appendingPathComponent("quantized")
            try configuration().write(to: source.appendingPathComponent("config.json"))
            let weights = Dictionary(
                uniqueKeysWithValues: reference.parameters().flattened().map {
                    (wrapped ? "language_model." + $0.0 : $0.0, $0.1)
                })
            try MLX.save(
                arrays: weights, metadata: ["format": "mlx"],
                url: source.appendingPathComponent("model.safetensors"))
            let result = try ModelConversion.convert(
                source: source, destination: destination,
                quantization: .init(groupSize: 32, bits: 4))
            XCTAssertGreaterThan(result.quantizedMatrixCount, 0)
            let data = try Data(contentsOf: destination.appendingPathComponent("config.json"))
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let quantization = try XCTUnwrap(
                reference.checkpointQuantization(base.perLayerQuantization))
            XCTAssertEqual(quantization.quantization(layer: "model.layers.1.mlp.gate")?.bits, 8)
            let loaded = try model()
            try CheckpointLoader.load(
                modelDirectory: destination, model: loaded, perLayerQuantization: quantization)
            XCTAssertTrue(
                all(isFinite(loaded(MLXArray([1, 2], [1, 2]), cache: nil))).item(Bool.self))
        }
    }

    func testReasoningAndToolHistoryArePreservedWithoutMutatingInput() async throws {
        let model = try model()
        let calls: [[String: any Sendable]] = [
            ["function": ["name": "lookup", "arguments": "{\"query\":\"flint\"}"]]
        ]
        let input = UserInput(
            prompt: .messages([
                [
                    "role": "assistant", "content": "", "reasoning_content": "Look it up.",
                    "tool_calls": calls,
                ], ["role": "tool", "content": "Found."],
            ]))
        let processor = TextProcessor(
            .init(), tokenizer: FlintTestTokenizer(),
            messageGenerator: model.messageGenerator(tokenizer: FlintTestTokenizer()))
        let prepared = try await processor.prepare(input: input)
        let text = FlintTestTokenizer().decode(
            tokenIds: prepared.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        XCTAssertTrue(text.contains("<think>Look it up.</think><tool_call>lookup"))
        XCTAssertTrue(text.contains("<arg_key>query</arg_key><arg_value>flint</arg_value>"))
        XCTAssertTrue((calls[0]["function"] as? [String: any Sendable])?["arguments"] is String)
        let parsed = GLM4ToolCallParser().parse(
            content:
                "<tool_call>lookup<arg_key>query</arg_key><arg_value>flint</arg_value></tool_call>",
            tools: nil)
        XCTAssertEqual(parsed?.function.name, "lookup")
        XCTAssertEqual(parsed?.function.arguments["query"], .string("flint"))
    }

    func testRealCheckpointWhenRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["FLINT_MODEL"] else {
            throw XCTSkip("Set FLINT_MODEL for local full-checkpoint validation")
        }
        let ticket = WiredMemoryTicket(
            size: GPU.maxRecommendedWorkingSetBytes() ?? 0, policy: WiredMaxPolicy())
        let model = try await ticket.withWiredLimit {
            try await NativModel.loadModel(
                from: .directory(URL(fileURLWithPath: path)),
                policy: .init(wiredMemoryTicket: ticket))
        }
        let request = model.generate(
            input: UserInput(chat: [.user("What is 17 * 23? Give the numerical answer.")]),
            options: .init(
                prefill: .init(stepSize: 256), sampling: .init(temperature: 0),
                output: .init(maxTokens: 512)))
        let text = try await request.response()
        print("Flint answer: \(text)")
        XCTAssertTrue(text.contains("391"))
        let metrics = try await request.metrics()
        XCTAssertEqual(metrics.stopReason, .stop)
    }
}
