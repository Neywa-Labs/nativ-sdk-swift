import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class AdapterBlock: Module {
    @ModuleInfo var projection: Linear
    let trace = CompiledTrace<AdapterBlock> { owner, inputs in [owner.projection(inputs[0])] }
    init(quantized: Bool) {
        let weight = sin(arange(32 * 32).asType(.float32)).reshaped(32, 32) * 0.1
        let linear = Linear(weight: weight, bias: MLXArray.ones([32]) * 0.01)
        _projection.wrappedValue =
            quantized ? QuantizedLinear(linear, groupSize: 32, bits: 8) : linear
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { trace(self, x) }
}

private final class AdapterModel: Module, LLMModel, LoRAModel {
    let layers: [AdapterBlock]
    var failPreparation = false
    var loraLayers: [Module] { layers }
    var loraDefaultKeys: [String] { ["projection"] }
    init(quantized: Bool = false) {
        layers = [AdapterBlock(quantized: quantized), AdapterBlock(quantized: quantized)]
    }
    func prepare() throws {
        if failPreparation {
            failPreparation = false
            throw ModelAdapterError.invalidCheckpoint("Test preparation failure")
        }
    }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [KVCacheSimple()] }
    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = MLXArray.eye(32)[tokens]
        for layer in layers { h = layer(h) }
        if let cache {
            _ = cache[0].update(
                keys: h.expandedDimensions(axis: 1), values: h.expandedDimensions(axis: 1))
        }
        return h
    }
}

private struct AdapterTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) - 64 } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 + 64) }, encoding: .utf8)!
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: messages.compactMap { $0["content"] as? String }.joined(), addSpecialTokens: false
        )
    }
}

private actor AdapterBarrier {
    var entered = false
    var continuation: CheckedContinuation<Void, Never>?
    func hold() async {
        entered = true
        await withCheckedContinuation { continuation = $0 }
    }
    func release() {
        continuation?.resume()
        continuation = nil
    }
}

final class ModelAdapterTests: XCTestCase {
    private func loadedModel(quantized: Bool = false) -> NativModel {
        let tokenizer = AdapterTokenizer()
        return makeTestModel(
            context: .init(
                metadata: .init(), model: AdapterModel(quantized: quantized),
                processor: TextProcessor(.init(), tokenizer: tokenizer), tokenizer: tokenizer))
    }
    private func logits(_ loadedModel: NativModel) async throws -> [Float] {
        try await loadedModel.withModel { _, context in
            try context.languageModel(MLXArray([[1, 2, 3]]), cache: nil).asArray(Float.self)
        }
    }
    private func fixture(
        peft: Bool = false, dora: Bool = false,
        edit: (inout [String: Any], inout [String: MLXArray]) -> Void = { _, _ in }
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let a = MLXArray.ones([32, 2]) * 0.1
        let b = cos(arange(64).asType(.float32)).reshaped(2, 32) * 0.2
        var config: [String: Any] =
            peft
            ? [
                "peft_type": "LORA", "r": 2, "lora_alpha": 4, "lora_dropout": 0.7,
                "target_modules": ["projection"],
            ]
            : [
                "fine_tune_type": dora ? "dora" : "lora", "num_layers": 1,
                "lora_parameters": ["rank": 2, "scale": 2, "dropout": 0.7, "keys": ["projection"]],
            ]
        var weights: [String: MLXArray] =
            peft
            ? [
                "base_model.model.layers.1.projection.lora_A.weight": a.T,
                "base_model.model.layers.1.projection.lora_B.weight": b.T,
            ]
            : ["layers.1.projection.lora_a": a, "layers.1.projection.lora_b": b]
        if dora { weights["layers.1.projection.m"] = MLXArray.ones([32]) * 0.6 }
        edit(&config, &weights)
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("adapter_config.json"))
        try save(
            arrays: weights,
            url: directory.appendingPathComponent(
                peft ? "adapter_model.safetensors" : "adapters.safetensors"))
        return directory
    }
    private func close(
        _ a: [Float], _ b: [Float], tolerance: Float = 2e-5, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        XCTAssertLessThanOrEqual(
            zip(a, b).map { abs($0 - $1) }.max() ?? 0, tolerance, file: file, line: line)
    }

    func testNativeAndPEFTLifecycleRetracesAndPreservesBaseWeights() async throws {
        for quantized in [false, true] {
            var nativeLoRA: [Float]?
            for (peft, dora) in [(false, false), (false, true), (true, false)] {
                let directory = try fixture(peft: peft, dora: dora)
                defer { try? FileManager.default.removeItem(at: directory) }
                let loadedModel = loadedModel(quantized: quantized)
                let base = try await logits(loadedModel)  // Populate compiled traces before replacing modules.
                try await loadedModel.loadAdapter(from: .directory(directory))
                let adapted = try await logits(loadedModel)
                if !peft && !dora { nativeLoRA = adapted }
                if peft { close(adapted, try XCTUnwrap(nativeLoRA), tolerance: 0) }
                XCTAssertGreaterThan(zip(base, adapted).map { abs($0 - $1) }.max()!, 1e-4)
                close(adapted, try await logits(loadedModel))  // Dropout must remain disabled in inference.
                let state = try await loadedModel.withModel { _, context in
                    let model = context.model as! AdapterModel
                    return (
                        model.layers[0].projection is any LoRALayer,
                        model.layers[1].projection is any LoRALayer
                    )
                }
                try await loadedModel.withModel { _, context in context.model.setLoRAEnabled(false)
                }
                close(base, try await logits(loadedModel), tolerance: 0)
                try await loadedModel.withModel { _, context in context.model.setLoRAEnabled(true) }
                close(adapted, try await logits(loadedModel), tolerance: 0)
                XCTAssertFalse(state.0)
                XCTAssertTrue(state.1)  // Sparse replacement when layer zero is unchanged.
                do {
                    try await loadedModel.loadAdapter(from: .directory(directory))
                    XCTFail("Stacked adapters must be rejected")
                } catch ModelAdapterError.alreadyLoaded {}
                try await loadedModel.unloadAdapter()
                close(base, try await logits(loadedModel), tolerance: 0)
                try await loadedModel.loadAdapter(from: .directory(directory))
                try await loadedModel.fuseAdapter()
                close(adapted, try await logits(loadedModel), tolerance: quantized ? 3e-3 : 1e-4)
                do {
                    try await loadedModel.unloadAdapter()
                    XCTFail("Fusion must be permanent")
                } catch ModelAdapterError.notLoaded {}
            }
        }
    }

    func testLayerNumericsAndMixedDtypesMatchExplicitLoRAAndDoRA() throws {
        for quantized in [false, true] {
            for dora in [false, true] {
                let weight = sin(arange(32 * 8).asType(.float32)).reshaped(8, 32).asType(.bfloat16)
                let bias = MLXArray.ones([8], dtype: .bfloat16) * 0.1
                let base: Linear =
                    quantized
                    ? QuantizedLinear(weight: weight, bias: bias, groupSize: 32, bits: 8)
                    : Linear(weight: weight, bias: bias)
                let adapter =
                    dora
                    ? DoRALinear.from(linear: base, rank: 2, scale: 0.5)
                    : LoRALinear.from(linear: base, rank: 2, scale: 0.5)
                let a = (MLXArray.ones([32, 2]) * 0.1).asType(.float16)
                let b = cos(arange(16).asType(.float32)).reshaped(2, 8).asType(.float16)
                var parameters = ["lora_a": a, "lora_b": b]
                if dora { parameters["m"] = MLXArray.ones([8]) * 2 }
                try adapter.update(parameters: .unflattened(parameters), verify: .noUnusedKeys)
                adapter.train(false)
                let x = cos(arange(64).asType(.float32)).reshaped(2, 32).asType(.bfloat16)
                let w = (base as? QuantizedLinear)?.dequantizedWeight ?? weight
                let delta = 0.5 * matmul(matmul(x, a), b)
                var expected = matmul(x, w.T) + delta.asType(.bfloat16)
                if dora {
                    let adapted = w + matmul(0.5 * b.T, a.T)
                    expected *= (2 / sqrt(sum(square(adapted), axis: 1))).asType(.bfloat16)
                }
                expected += bias
                let actual = (adapter as! Linear)(x)
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 0.015, atol: 0.04).item(Bool.self),
                    "bf16 output differs from explicit projection: quantized=\(quantized), dora=\(dora)"
                )
                let fused = adapter.fused() as! Linear
                XCTAssertEqual(fused(x).dtype, .bfloat16)
                adapter.loraEnabled = false
                close(
                    (adapter as! Linear)(x).asArray(Float.self), base(x).asArray(Float.self),
                    tolerance: 0)
            }
        }
    }

    func testInvalidCheckpointsAndPreparationFailureLeaveModelUnchanged() async throws {
        let loadedModel = loadedModel()
        let base = try await logits(loadedModel)
        for failure in 0 ..< 4 {
            let directory = try fixture { _, weights in
                switch failure {
                case 0: weights.removeValue(forKey: "layers.1.projection.lora_b")
                case 1: weights["layers.1.projection.lora_a"] = MLXArray.ones([31, 2])
                case 2: weights["layers.0.projection.lora_a"] = MLXArray.ones([32, 2])
                default: break
                }
            }
            defer { try? FileManager.default.removeItem(at: directory) }
            if failure == 3 {
                try await loadedModel.withModel { _, context in
                    (context.model as! AdapterModel).failPreparation = true
                }
            }
            do {
                try await loadedModel.loadAdapter(from: .directory(directory))
                XCTFail("Expected rejection")
            } catch is ModelAdapterError {}
            close(base, try await logits(loadedModel), tolerance: 0)
        }
        let valid = try fixture()
        defer { try? FileManager.default.removeItem(at: valid) }
        try await loadedModel.loadAdapter(from: .directory(valid))
        try await loadedModel.unloadAdapter()
    }

    func testMutationWaitsForAdmissionAndQueuedCancellationDoesNotInstall() async throws {
        let loadedModel = loadedModel()
        let base = try await logits(loadedModel)
        let directory = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let barrier = AdapterBarrier()
        let holding = Task { try await loadedModel.withModel { _, _ in await barrier.hold() } }
        while !(await barrier.entered) { await Task.yield() }
        let loading = Task { try await loadedModel.loadAdapter(from: .directory(directory)) }
        while await loadedModel.queuedRequestCount == 0 { await Task.yield() }
        loading.cancel()
        await barrier.release()
        try await holding.value
        do {
            try await loading.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        close(base, try await logits(loadedModel), tolerance: 0)
    }

    func testQwen35RebuildsFusedProjectionsAndKeepsBF16WithFP16Adapter() async throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5","hidden_size":64,"num_hidden_layers":2,
                 "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
                 "head_dim":32,"linear_num_value_heads":4,"linear_num_key_heads":2,
                 "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
                 "vocab_size":32,"full_attention_interval":2,"num_experts":0,"num_experts_per_tok":0}
                """.utf8))
        let model = Qwen35TextModel(configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        try quantizeByParent(model: model, groupSize: 32, bits: 4) { path, _ in
            path.contains("linear_attn.in_proj_")
        }
        model.train(false)
        try model.prepare()
        let gdn = try XCTUnwrap(model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first)
        XCTAssertTrue(gdn.hasFusedInputProjection)
        let base = model(MLXArray([[1, 2, 3]]), cache: try model.newCache(configuration: nil))
            .asArray(Float.self)
        let tokenizer = AdapterTokenizer()
        let loadedModel = makeTestModel(
            context: .init(
                metadata: .init(), model: model,
                processor: TextProcessor(.init(), tokenizer: tokenizer), tokenizer: tokenizer))
        let directory = try fixture(peft: true) { config, weights in
            config["target_modules"] = ["linear_attn.in_proj_qkv"]
            weights = [
                "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight":
                    MLXArray.full([2, 64], values: MLXArray(0.03)).asType(.float16),
                "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight":
                    MLXArray.full([256, 2], values: MLXArray(0.02)).asType(.float16),
            ]
        }
        defer { try? FileManager.default.removeItem(at: directory) }
        try await loadedModel.loadAdapter(from: .directory(directory))
        let adapted = try await loadedModel.withModel { _, context in
            let model = try context.languageModel
            let gdn = model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first!
            XCTAssertFalse(gdn.hasFusedInputProjection)
            let cache = try model.newCache(configuration: nil)
            let prefill = model(MLXArray([[1, 2, 3]]), cache: cache)
            let decode = model(MLXArray([[4]]), cache: cache)
            eval(prefill, decode)
            XCTAssertEqual(prefill.dtype, .bfloat16)
            XCTAssertEqual(decode.dtype, .bfloat16)
            XCTAssertTrue(all(isFinite(decode)).item(Bool.self))
            return prefill.asArray(Float.self)
        }
        XCTAssertNotEqual(base, adapted)
        try await loadedModel.unloadAdapter()
        let restored = try await loadedModel.withModel { _, context in
            let model = try context.languageModel
            XCTAssertTrue(
                model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first!
                    .hasFusedInputProjection)
            return model(MLXArray([[1, 2, 3]]), cache: try model.newCache(configuration: nil))
                .asArray(Float.self)
        }
        close(base, restored, tolerance: 0)
    }

    func testNemotronDiscoversAdaptersAfterItsFirstGeneration() async throws {
        let config = try JSONDecoder().decode(
            NemotronLabsDiffusionConfiguration.self,
            from: Data(
                """
                {"model_type":"nemotron_labs_diffusion","vocab_size":32,"hidden_size":32,
                 "intermediate_size":64,"num_hidden_layers":1,"num_attention_heads":4,
                 "num_key_value_heads":2,"head_dim":8,"mask_token_id":31,"eos_token_id":0}
                """.utf8))
        let model = NemotronLabsDiffusion(config)
        _ = model.linearSpecGenerate(promptIds: [1, 2, 3], maxNewTokens: 1)
        let tokenizer = AdapterTokenizer()
        let loadedModel = makeTestModel(
            context: .init(
                metadata: .init(), model: model,
                processor: TextProcessor(.init(), tokenizer: tokenizer), tokenizer: tokenizer))
        let base = try await logits(loadedModel)
        let directory = try fixture { config, weights in
            config["lora_parameters"] = ["rank": 2, "scale": 2, "keys": ["self_attn.q_proj"]]
            weights = Dictionary(
                uniqueKeysWithValues: weights.map { key, value in
                    (
                        key.replacingOccurrences(
                            of: "layers.1.projection",
                            with: "language_model.model.layers.0.self_attn.q_proj"), value
                    )
                })
        }
        defer { try? FileManager.default.removeItem(at: directory) }
        try await loadedModel.loadAdapter(from: .directory(directory))
        try await loadedModel.withModel { _, context in
            let model = context.model as! NemotronLabsDiffusion
            _ = model.linearSpecGenerate(promptIds: [1, 2, 3], maxNewTokens: 1)
            let adapter = model.modules().compactMap { $0 as? LoRALinear }.first!
            XCTAssertFalse(
                adapter.loraEnabled, "The AR verifier must disable newly installed adapters")
        }
        close(base, try await logits(loadedModel), tolerance: 0)
        try await loadedModel.unloadAdapter()
    }

    func testSessionsRebuildCachesAfterLoadUnloadAndFusion() async throws {
        let loadedModel = loadedModel()
        let session = ModelSession(
            model: loadedModel,
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 1)))
        func turn() async throws -> ModelGenerationSummary {
            let operation = try await session.respond(to: "ABC")
            for try await _ in operation {}
            return try await operation.metrics()
        }
        _ = try await turn()
        let warm = try await turn()
        XCTAssertGreaterThan(warm.cache?.reusedTokenCount ?? 0, 0)
        let directory = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        for change in 0 ..< 4 {
            if change == 0 || change == 2 {
                try await loadedModel.loadAdapter(from: .directory(directory))
            } else if change == 1 {
                try await loadedModel.unloadAdapter()
            } else {
                try await loadedModel.fuseAdapter()
            }
            let cold = try await turn()
            XCTAssertEqual(cold.cache?.reusedTokenCount, 0)
            let warm = try await turn()
            XCTAssertGreaterThan(warm.cache?.reusedTokenCount ?? 0, 0)
        }
        let count = try await session.withTranscript { $0.count }
        XCTAssertEqual(count, 20)
        await session.close()
    }
}
