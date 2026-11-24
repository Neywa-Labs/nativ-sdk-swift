import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct MiniCPMV47TemplateTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text =
            messages.map { "\($0["role"] as? String ?? ""): \($0["content"] as? String ?? "")" }
            .joined(separator: "\n")
            + "\nthinking: \(additionalContext?["enable_thinking"] as? Bool ?? true)"
        return encode(text: text, addSpecialTokens: false)
    }
}

private final class MiniCPMVPrefillProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func append(_ completed: Int) { lock.withLock { values.append(completed) } }
    var completed: [Int] { lock.withLock { values } }
}

final class MiniCPMV47Tests: XCTestCase {
    private func configuration() throws -> MiniCPMVConfiguration {
        try JSONDecoder().decode(
            MiniCPMVConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "minicpmv4_7", "insert_layer_id": 0,
                  "image_token_id": 5, "image_start_id": 6, "image_end_id": 7,
                  "slice_start_id": 8, "slice_end_id": 9, "newline_id": 10,
                  "text_config": {
                    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                    "vocab_size": 32, "full_attention_interval": 2, "tie_word_embeddings": false,
                    "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                    "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                    "num_experts": 4, "num_experts_per_tok": 2,
                    "shared_expert_intermediate_size": 32, "moe_intermediate_size": 32,
                    "rope_parameters": {"partial_rotary_factor": 1.0, "mrope_section": [2,3,3]}
                  },
                  "vision_config": {
                    "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
                    "num_attention_heads": 2, "patch_size": 2, "image_size": 8
                  }
                }
                """.utf8))
    }

    func testOfficial35BA3BConfigurationAndRegistration() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "minicpmv4_7-config", withExtension: "json", subdirectory: "Fixtures"))
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(MiniCPMVConfiguration.self, from: data)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 40)
        XCTAssertEqual(config.textConfiguration.numExperts, 256)
        XCTAssertEqual(config.textConfiguration.numExpertsPerToken, 8)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 248_144)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertTrue(
            try ModelRegistry.default.createModel(configuration: data, modelType: "minicpmv4_7")
                is MiniCPMV)
        XCTAssertEqual(
            try ModelConventions.supplementalStopTokens(
                modelType: "minicpmv4_7", configuration: data), ["<|im_end|>"])
    }

    func test47ProcessorPreservesHistoryAndThinkingSetting() async throws {
        let config = try JSONDecoder().decode(
            MiniCPMVProcessorConfiguration.self,
            from: Data("{\"processor_class\":\"MiniCPMV4_7Processor\"}".utf8))
        let tokenizer = MiniCPMV47TemplateTokenizer()
        let processor = MiniCPMVProcessor(config, tokenizer: tokenizer)
        for thinking in [false, true] {
            let input = UserInput(
                chat: [
                    .system("Be concise."), .user("First"), .assistant("Answer"), .user("Second"),
                ],
                additionalContext: thinking ? ["enable_thinking": true] : nil)
            let result = try await processor.prepare(input: input)
            let text = tokenizer.decode(
                tokenIds: result.text.tokens.asArray(Int.self), skipSpecialTokens: false)
            XCTAssertEqual(
                text,
                "system: Be concise.\nuser: First\nassistant: Answer\nuser: Second\nthinking: \(thinking)"
            )
        }
    }

    func testProcessorComposesFlatSidecarWithModelGeometry() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"scale_resolution\":112,\"max_slice_nums\":1}".utf8)
            .write(to: directory.appendingPathComponent("preprocessor_config.json"))
        let (data, type) = try await ModelLoader().loadProcessorConfig(
            from: directory,
            modelType: "minicpmv4_7",
            configData: Data(
                "{\"downsample_mode\":\"4x\",\"vision_config\":{\"patch_size\":14}}".utf8))
        let config = try JSONDecoder().decode(MiniCPMVProcessorConfiguration.self, from: data)
        XCTAssertEqual(type, "MiniCPMV4_7Processor")
        XCTAssertEqual(config.downsampleMode, "4x")
        XCTAssertEqual(config.scaleResolution, 112)
        XCTAssertEqual(config.maxSliceNums, 1)
    }

    private func checkpoint(
        _ weights: [String: MLXArray], metadata: [String: String] = ["format": "mlx"]
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: metadata,
            url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func pair(
        mode: QuantizationMode? = nil, prefetch: Bool = true, batchSize: Int = 8,
        cacheSizeBytes: Int = 65_536, separateExperts: Bool = false, sharedBuffers: Bool = false
    ) throws -> (MiniCPMV, MiniCPMV) {
        MLXRandom.seed(47)
        let config = try configuration()
        let source = MiniCPMV(config)
        var quantization: BaseConfiguration.Quantization?
        if let mode {
            quantize(model: source) { path, _ in
                path.contains(".switch_mlp.") ? (32, 4, mode) : nil
            }
            quantization = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self,
                from: Data("{\"group_size\":32,\"bits\":4,\"mode\":\"\(mode.rawValue)\"}".utf8))
        }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        var storedWeights = weights
        if separateExperts {
            for (name, value) in weights where name.contains(".switch_mlp.") {
                storedWeights[name] = nil
                for expert in 0 ..< config.textConfiguration.numExperts {
                    storedWeights[
                        name.replacingOccurrences(of: ".switch_mlp.", with: ".experts.\(expert).")] =
                        value[expert]
                }
            }
        }
        let directory = try checkpoint(storedWeights)
        let streamed = MiniCPMV(config)
        try CheckpointLoader.load(
            modelDirectory: directory, model: streamed, quantization: quantization,
            expertStreaming: .init(
                cacheSizeBytes: cacheSizeBytes, prefetch: prefetch, expertBatchSize: batchSize,
                sharedExpertBuffers: sharedBuffers, deferVisionWeights: sharedBuffers))
        XCTAssertFalse(streamed.parameters().flattened().contains { $0.0.contains(".switch_mlp.") })
        for (key, value) in streamed.parameters().flattened() {
            if let original = weights[key], value.dtype.isFloatingPoint {
                XCTAssertEqual(abs(value - original).max().item(Float.self), 0, key)
            }
        }
        return (source, streamed)
    }

    func testStreamedPrefillAndDecodeMatchResidentModel() throws {
        // Batch 1 covers the one-fence fallback; 3 forces an incomplete final group;
        // 8 can cover all toy experts. The small cache independently restricts live weights.
        for mode: QuantizationMode? in [nil, .affine, .mxfp4] {
            for (batchSize, budget) in [(1, 65_536), (3, 65_536), (8, 65_536), (8, 16_384)] {
                let (resident, streamed) = try pair(
                    mode: mode, batchSize: batchSize, cacheSizeBytes: budget)
                let lhsCache = try resident.newCache(configuration: nil)
                let rhsCache = try streamed.newCache(configuration: nil)
                for tokens in [[1, 2, 3, 4, 5, 6], [7], [8], [9]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: lhsCache)
                    let rhs = streamed(input, cache: rhsCache)
                    eval(lhs, rhs)
                    try streamed.checkInferenceError()
                    XCTAssertLessThan(
                        abs(lhs - rhs).max().item(Float.self), 2e-4,
                        "mode=\(String(describing: mode)), batch=\(batchSize), cache=\(budget)")
                }
            }
        }
    }

    func testBatchedLongPrefillMatchesSingleExpertExecution() throws {
        for mode: QuantizationMode? in [nil, .affine, .mxfp4] {
            let (_, single) = try pair(mode: mode, prefetch: false, batchSize: 1)
            let (_, batched) = try pair(mode: mode, batchSize: 3)
            let singleCache = try single.newCache(configuration: nil)
            let batchCache = try batched.newCache(configuration: nil)
            for tokens in [Array((0 ..< 40).map { $0 % 30 + 1 }), [7], [8]] {
                let input = MLXArray(tokens, [1, tokens.count])
                let a = single(input, cache: singleCache)
                let b = batched(input, cache: batchCache)
                eval(a, b)
                try single.checkInferenceError()
                try batched.checkInferenceError()
                XCTAssertLessThan(abs(a - b).max().item(Float.self), 2e-4)
            }
        }
    }

    func testStreamedPrepareUsesRequestedChunkSizeAndPlatformDefault() throws {
        let (resident, streamed) = try pair(prefetch: false)
        let tokens = (0 ..< 257).map { $0 % 30 + 1 }
        let input = LMInput(text: .init(tokens: MLXArray(tokens, [1, tokens.count])))
        let referenceCache = try resident.newCache(configuration: nil)
        guard
            case .logits(let reference) = try resident.prepare(
                input, cache: referenceCache, state: nil, prefill: .init(chunking: .unchunked))
        else { return XCTFail("Expected reference logits") }
        let lastLogits = reference.logits[0..., -1, 0...]
        let next = LMInput.Text(tokens: MLXArray([3], [1, 1]))
        let continuation = resident(next, cache: referenceCache, state: reference.state)
        eval(lastLogits, continuation.logits)

        #if os(iOS)
            let defaultProgress = [256, 257]
        #else
            let defaultProgress = [257]
        #endif
        for (stepSize, chunking, expected): (Int?, ModelPrefillOptions.Chunking, [Int]) in [
            (nil, .remainder, defaultProgress),
            (64, .remainder, [64, 128, 192, 256, 257]),
            (256, .remainder, [256, 257]),
            (512, .remainder, [257]),
            (256, .balanced, [129, 257]),
            (64, .unchunked, [257]),
        ] {
            let cache = try streamed.newCache(configuration: nil)
            let progress = MiniCPMVPrefillProgress()
            guard
                case .logits(let output) = try streamed.prepare(
                    input, cache: cache, state: nil,
                    prefill: .init(
                        stepSize: stepSize, chunking: chunking,
                        progress: { completed, total in
                            XCTAssertEqual(total, tokens.count)
                            progress.append(completed)
                        }))
            else { return XCTFail("Expected streamed logits") }
            XCTAssertEqual(progress.completed, expected)
            XCTAssertEqual(cache.last?.offset, tokens.count)
            // Long float prompts accumulate rounding differences between resident GEMM
            // and streamed gather kernels, including at the original 64-token size.
            let actualLastLogits = output.logits[0..., -1, 0...]
            XCTAssertLessThan(
                abs(actualLastLogits - lastLogits).max().item(Float.self), 5e-3)
            XCTAssertEqual(
                argMax(actualLastLogits, axis: -1).item(Int.self),
                argMax(lastLogits, axis: -1).item(Int.self))
            let decoded = streamed(next, cache: cache, state: output.state)
            XCTAssertLessThan(
                abs(decoded.logits - continuation.logits).max().item(Float.self), 5e-3)
            XCTAssertEqual(
                argMax(decoded.logits, axis: -1).item(Int.self),
                argMax(continuation.logits, axis: -1).item(Int.self))
            try streamed.checkInferenceError()
        }
    }

    func testPrefillChunksAndImageContinuationPreservePositions() throws {
        let (resident, streamed) = try pair(prefetch: false)
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 6, 5, 7, 2], [1, 5])),
            image: .init(pixels: MLXArray.zeros([192]), frames: [THW(1, 4, 4)]),
            video: .init(pixels: MLXArray([2, 3], [1, 2])))
        let lhsCache = try resident.newCache(configuration: nil)
        let rhsCache = try streamed.newCache(configuration: nil)
        guard
            case .logits(let lhs) = try resident.prepare(
                input, cache: lhsCache, state: nil, prefill: .init(stepSize: 2)),
            case .logits(let rhs) = try streamed.prepare(
                input, cache: rhsCache, state: nil, prefill: .init(stepSize: 2))
        else { return XCTFail("Expected logits") }
        XCTAssertLessThan(
            abs(lhs.logits[0..., -1, 0...] - rhs.logits[0..., -1, 0...]).max().item(Float.self),
            2e-4)
        XCTAssertFalse(try XCTUnwrap(rhs.state).serializedArrays().isEmpty)
        let next = LMInput.Text(tokens: MLXArray([3], [1, 1]))
        let a = resident(next, cache: lhsCache, state: lhs.state)
        let b = streamed(next, cache: rhsCache, state: rhs.state)
        XCTAssertLessThan(abs(a.logits - b.logits).max().item(Float.self), 2e-4)
    }

    func testCanvasThumbnailSlicesAndAdjacentFrames() throws {
        let config = try configuration()
        // Thumbnail (2x2), two detail crops (2x2 each), then an adjacent video frame (1x1).
        let tokens = [1, 6, 5, 5, 5, 5, 7, 8, 5, 5, 5, 5, 9, 8, 5, 5, 5, 5, 9, 6, 5, 7, 2]
        let canvas = try MiniCPMVCanvasPositions.make(
            tokens: tokens,
            bounds: [2 ..< 6, 8 ..< 12, 14 ..< 18, 20 ..< 21],
            grids: [THW(1, 8, 8), THW(1, 8, 8), THW(1, 8, 8), THW(1, 4, 4)], configuration: config)
        let p = canvas.positions
        XCTAssertEqual(p[0, 0, 2 ..< 6].asArray(Int.self), [1, 1, 1, 1])
        XCTAssertEqual(p[1, 0, 2 ..< 6].asArray(Int.self), [1, 1, 2, 2])
        XCTAssertEqual(p[2, 0, 2 ..< 6].asArray(Int.self), [1, 4, 1, 4])
        XCTAssertEqual(p[2, 0, 14 ..< 18].asArray(Int.self), [3, 4, 3, 4])
        XCTAssertEqual(p[0, 0, 20].item(Int.self), 6)
        XCTAssertEqual(p[0, 0, 22].item(Int.self), 8)
        XCTAssertEqual(canvas.delta, -14)
    }

    func testFusedTransformersExpertLayoutAndBoundedReads() throws {
        let config = try configuration()
        let model = MiniCPMV(config)
        var raw: [String: MLXArray] = [:]
        for layer in 0 ..< 2 {
            let prefix = "model.language_model.layers.\(layer).mlp.experts."
            raw[prefix + "gate_up_proj"] = MLXRandom.normal([4, 64, 32]) / 20
            raw[prefix + "down_proj"] = MLXRandom.normal([4, 32, 32]) / 20
        }
        let mapped = model.sanitize(weights: raw)
        XCTAssertEqual(
            mapped["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [4, 32, 32])
        let dir = try checkpoint(raw, metadata: [:])
        let store = try ExpertWeightStore(
            urls: [dir.appendingPathComponent("model.safetensors")],
            configuration: config, options: .init(cacheSizeBytes: 12_288), quantization: nil,
            perLayerQuantization: nil)
        // Deliberately wrong predictions for the next demand.
        store.prefetch(layer: 1, experts: [0, 1])
        let x = MLXRandom.normal([2, 32])
        for expert in [3, 2, 1, 0, 3] {
            let gate = mapped["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]![
                expert]
            let up = mapped["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]![expert]
            let down = mapped["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]![
                expert]
            let expected = matmul(silu(matmul(x, gate.T)) * matmul(x, up.T), down.T)
            let actual = try store.apply(x, layer: 0, expert: expert)
            XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-5)
            XCTAssertLessThanOrEqual(store.residentBytes, 12_288)
        }
        XCTAssertEqual(store.excludedWeightNames.count, 4)
        let loaded = try loadWeightArrays(
            urls: [dir.appendingPathComponent("model.safetensors")],
            excluding: store.excludedWeightNames)
        XCTAssertTrue(loaded.weights.isEmpty)
        XCTAssertThrowsError(
            try ExpertWeightStore(
                urls: [dir.appendingPathComponent("model.safetensors")],
                configuration: config, options: .init(cacheSizeBytes: 1), quantization: nil,
                perLayerQuantization: nil))
    }

    func testBatchedFusedExpertsPreserveRowAssignmentsAndBoundWeights() throws {
        let config = try configuration()
        for dtype: DType in [.float32, .float16, .bfloat16] {
            var raw: [String: MLXArray] = [:]
            for layer in 0 ..< 2 {
                let prefix = "model.language_model.layers.\(layer).mlp.experts."
                raw[prefix + "gate_up_proj"] = (MLXRandom.normal([4, 64, 32]) / 20).asType(dtype)
                raw[prefix + "down_proj"] = (MLXRandom.normal([4, 32, 32]) / 20).asType(dtype)
            }
            let dir = try checkpoint(raw, metadata: [:])
            let expertBytes = 3 * 32 * 32 * dtype.size
            let store = try ExpertWeightStore(
                urls: [dir.appendingPathComponent("model.safetensors")],
                configuration: config,
                options: .init(cacheSizeBytes: 3 * expertBytes, expertBatchSize: 8),
                quantization: nil, perLayerQuantization: nil)
            XCTAssertEqual(store.batchSize(layer: 0), 3)
            store.prefetch(layer: 1, experts: [0, 1, 2, 3])
            let x = (MLXRandom.normal([5, 32]) / 2).asType(dtype)
            // Sparse checkpoint IDs and repeated, unsorted local assignments exercise gather.
            let experts = [3, 0, 2]
            let assignments = [2, 0, 2, 1, 0]
            var expected: [MLXArray] = []
            for (row, index) in assignments.enumerated() {
                let id = experts[index]
                let gateUp = raw["model.language_model.layers.0.mlp.experts.gate_up_proj"]![id]
                let down = raw["model.language_model.layers.0.mlp.experts.down_proj"]![id]
                let parts = matmul(x[row ..< row + 1], gateUp.T).split(parts: 2, axis: -1)
                expected.append(matmul(silu(parts[0]) * parts[1], down.T))
            }
            let actual = try store.apply(x, layer: 0, experts: experts, expertIndices: assignments)
            let tolerance: Float = dtype == .float32 ? 1e-5 : 1e-3
            XCTAssertLessThan(
                abs(concatenated(expected) - actual).max().item(Float.self), tolerance)
            XCTAssertLessThanOrEqual(store.residentBytes, 3 * expertBytes)
            XCTAssertThrowsError(
                try store.apply(x, layer: 0, experts: [0, 1, 2, 3], expertIndices: assignments))
            XCTAssertThrowsError(
                try ExpertWeightStore(
                    urls: [dir.appendingPathComponent("model.safetensors")],
                    configuration: config, options: .init(expertBatchSize: 0), quantization: nil,
                    perLayerQuantization: nil))
        }
    }

    func testMalformedExpertShapesAreRejectedDuringLoading() throws {
        let config = try configuration()
        let source = MiniCPMV(config)
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"] = MLXArray.zeros([
            4, 31, 32,
        ])
        let dir = try checkpoint(weights)
        XCTAssertThrowsError(
            try ExpertWeightStore(
                urls: [dir.appendingPathComponent("model.safetensors")],
                configuration: config, options: .init(), quantization: nil,
                perLayerQuantization: nil))
    }

    func testTruncatedExpertFileFailsInsteadOfDroppingExperts() throws {
        let config = try configuration()
        let source = MiniCPMV(config)
        let directory = try checkpoint(
            Dictionary(uniqueKeysWithValues: source.parameters().flattened()))
        let url = directory.appendingPathComponent("model.safetensors")
        let store = try ExpertWeightStore(
            urls: [url], configuration: config,
            options: .init(prefetch: false), quantization: nil, perLayerQuantization: nil)
        let file = try FileHandle(forWritingTo: url)
        try file.truncate(atOffset: 8)
        try file.close()
        XCTAssertThrowsError(try store.apply(MLXArray.ones([1, 32]), layer: 0, expert: 0))
        XCTAssertThrowsError(
            try ExpertWeightStore(
                urls: [url], configuration: config,
                options: .init(), quantization: nil, perLayerQuantization: nil))
    }
    func testCompiledRecurrentUpdatePreservesDtypeRounding() throws {
        for dtype: DType in [.float32, .float16, .bfloat16] {
            for width in [16, 128] {
                MLXRandom.seed(73)
                let q = (MLXRandom.normal([1, 4, width]) / Float(width)).asType(dtype)
                let k = (MLXRandom.normal([1, 4, width]) / Float(width)).asType(dtype)
                let v = MLXRandom.normal([1, 4, 32]).asType(dtype)
                let g = sigmoid(MLXRandom.normal([1, 4])).asType(dtype)
                let beta = sigmoid(MLXRandom.normal([1, 4])).asType(dtype)
                var actual = MLXRandom.normal([1, 4, 32, width]).asType(dtype)
                var expected = actual
                for _ in 0 ..< 20 {
                    let next = compiledMiniCPMVGatedDeltaStep([q, k, v, g, beta, actual])
                    let reference = miniCPMVGatedDeltaStepOps(
                        q: q, k: k, v: v, g: g, beta: beta, state: expected)
                    XCTAssertEqual(
                        next[0].asArray(Float.self), reference.0.asArray(Float.self),
                        "dtype=\(dtype), width=\(width)")
                    XCTAssertEqual(
                        next[1].asArray(Float.self), reference.1.asArray(Float.self),
                        "dtype=\(dtype), width=\(width)")
                    actual = next[1]
                    expected = reference.1
                }
            }
        }
    }

    func testSeparateExpertsMatchResidentPrefillAndDecode() throws {
        // Batch 1 covers the one-fence fallback; 3 forces an incomplete final group;
        // 8 can cover all toy experts. The small cache independently restricts live weights.
        for mode: QuantizationMode? in [nil, .affine, .mxfp4] {
            for (batchSize, budget) in [(1, 65_536), (3, 65_536), (8, 65_536), (8, 16_384)] {
                let (resident, streamed) = try pair(
                    mode: mode, batchSize: batchSize, cacheSizeBytes: budget, separateExperts: true)
                let lhsCache = try resident.newCache(configuration: nil)
                let rhsCache = try streamed.newCache(configuration: nil)
                for tokens in [[1, 2, 3, 4, 5, 6], [7], [8], [9]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: lhsCache)
                    let rhs = streamed(input, cache: rhsCache)
                    eval(lhs, rhs)
                    try streamed.checkInferenceError()
                    XCTAssertLessThan(
                        abs(lhs - rhs).max().item(Float.self), 2e-4,
                        "mode=\(String(describing: mode)), batch=\(batchSize), cache=\(budget)")
                }
            }
        }
    }

    func testSeparateExpertValidationRejectsMissingDuplicateAndInconsistentTensors() throws {
        let config = try configuration()
        var weights: [String: MLXArray] = [:]
        for layer in 0 ..< 2 {
            for expert in 0 ..< 4 {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    weights[
                        "language_model.model.layers.\(layer).mlp.experts.\(expert).\(projection).weight"
                    ] = MLXArray.ones([32, 32])
                }
            }
        }
        let name = "language_model.model.layers.0.mlp.experts.0.gate_proj.weight"
        for scenario in 0 ..< 4 {
            var malformed = weights
            switch scenario {
            case 0: malformed[name] = nil
            case 1:
                malformed["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"] =
                    MLXArray.ones([4, 32, 32])
            case 2: malformed[name] = MLXArray.ones([31, 32])
            default:
                malformed[name.replacingOccurrences(of: "experts.0.", with: "experts.4.")] =
                    MLXArray.ones([32, 32])
            }
            let directory = try checkpoint(malformed)
            XCTAssertThrowsError(
                try ExpertWeightStore(
                    urls: [directory.appendingPathComponent("model.safetensors")],
                    configuration: config,
                    options: .init(), quantization: nil, perLayerQuantization: nil),
                "scenario=\(scenario)")
        }
    }

    func testPromptCacheReuseMatchesColdWithBatchedMiniCPMInputs() throws {
        let (resident, streamed) = try pair(mode: .mxfp4, prefetch: false)
        for model in [resident, streamed] {
            let tokenizer = MiniCPMV47TemplateTokenizer()
            let processorConfig = try JSONDecoder().decode(
                MiniCPMVProcessorConfiguration.self,
                from: Data("{\"processor_class\":\"MiniCPMV4_7Processor\"}".utf8))
            let context = ModelContext(
                metadata: .init(), model: model,
                processor: MiniCPMVProcessor(processorConfig, tokenizer: tokenizer),
                tokenizer: tokenizer)
            let configuration = PromptCacheReuseConfiguration(
                maxCheckpoints: 16, checkpointInterval: 2)
            let options = ModelGenerationOptions(
                prefill: .init(stepSize: 2), sampling: .init(temperature: 0),
                output: .init(maxTokens: 3, format: .tokens))
            let vault = PromptCacheVault()
            func run(_ tokens: [Int], vault: PromptCacheVault, reuse: Bool) throws
                -> ([Int], ModelCacheStatistics)
            {
                var output: [Int] = []
                var settings = configuration
                settings.reuseEnabled = reuse
                let summary = try PromptCacheEngine.run(
                    context: context,
                    prepared: LMInput(tokens: MLXArray(tokens, [1, tokens.count])), tools: nil,
                    options: options, configuration: settings, vault: vault,
                    emit: { if case .token(let token) = $0 { output.append(token) } })
                return (output, try XCTUnwrap(summary.cache))
            }
            let first = [1, 2, 3, 4, 5, 6, 7]
            let (cold, coldStats) = try run(first, vault: vault, reuse: true)
            XCTAssertEqual(coldStats.kind, .cold)
            XCTAssertEqual(cold, try run(first, vault: PromptCacheVault(), reuse: false).0)

            let kept = vault.history.count
            let extensionPrompt = vault.history + [8, 9]
            let (extended, extensionStats) = try run(extensionPrompt, vault: vault, reuse: true)
            XCTAssertEqual(extensionStats.kind, .extend)
            XCTAssertEqual(extensionStats.reusedTokenCount, kept)
            XCTAssertEqual(extensionStats.prefilledTokenCount, 2)
            XCTAssertEqual(
                extended, try run(extensionPrompt, vault: PromptCacheVault(), reuse: false).0)

            let edited = [1, 2, 3, 4, 10, 11]
            let (restored, restoredStats) = try run(edited, vault: vault, reuse: true)
            XCTAssertEqual(restoredStats.kind, .diverge)
            XCTAssertTrue(restoredStats.usedCheckpoint)
            XCTAssertEqual(restoredStats.reusedTokenCount, 4)
            XCTAssertEqual(restored, try run(edited, vault: PromptCacheVault(), reuse: false).0)
        }
    }

    func testSharedBuffersMatchResidentPrefillAndDecode() throws {
        // Batch 1 covers the one-fence fallback; 3 forces an incomplete final group;
        // 8 can cover all toy experts. The small cache independently restricts live weights.
        for mode: QuantizationMode? in [nil, .affine, .mxfp4] {
            for (batchSize, budget) in [(1, 262_144), (3, 262_144), (8, 524_288)] {
                let (resident, streamed) = try pair(
                    mode: mode, batchSize: batchSize, cacheSizeBytes: budget, separateExperts: true,
                    sharedBuffers: true)
                let lhsCache = try resident.newCache(configuration: nil)
                let rhsCache = try streamed.newCache(configuration: nil)
                for tokens in [[1, 2, 3, 4, 5, 6], [7], [8], [9]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: lhsCache)
                    let rhs = streamed(input, cache: rhsCache)
                    eval(lhs, rhs)
                    try streamed.checkInferenceError()
                    XCTAssertLessThan(
                        abs(lhs - rhs).max().item(Float.self), 2e-4,
                        "mode=\(String(describing: mode)), batch=\(batchSize), cache=\(budget)")
                }
            }
        }
    }

}
