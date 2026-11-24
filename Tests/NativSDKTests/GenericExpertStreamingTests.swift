import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class ExpertProjectionFixture: Module, BaseLanguageModel {
    @ModuleInfo var experts: Module
    init(_ experts: Module) { self._experts.wrappedValue = experts }
}

final class GenericExpertStreamingTests: XCTestCase {
    override func setUp() {
        super.setUp()
        MLXRandom.seed(47)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func checkpoint(_ model: Module, at directory: URL) throws {
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            metadata: ["format": "mlx"], url: directory.appendingPathComponent("model.safetensors"))
    }

    /// Optional full-checkpoint check; never downloads a model during the unit suite.
    func testLocalPretrainedCheckpointTeacherForcedParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["NATIV_EXPERT_STREAMING_VALIDATION_MODEL"]
            ?? environment["NATIV_LES_VALIDATION_MODEL"]
        else {
            throw XCTSkip(
                "Set NATIV_EXPERT_STREAMING_VALIDATION_MODEL to a local checkpoint that supports expert streaming")
        }
        // Raw checkpoints can allocate a second expert copy while stacking resident
        // weights. Release those temporary buffers before comparing both models.
        let previousCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 64 * 1024 * 1024
        Memory.clearCache()
        defer { Memory.cacheLimit = previousCacheLimit }
        // The opt-in paired-model check can exceed MLX's default residency limit.
        // Reserve its working set to avoid paging during correctness validation.
        let ticket = WiredMemoryTicket(
            size: Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.84),
            policy: WiredSumPolicy())
        _ = await ticket.start()
        addTeardownBlock { _ = await ticket.end() }
        let directory = URL(fileURLWithPath: path)
        let tokenizer = try await DefaultTokenizerLoader().load(from: directory)
        let tokens = try tokenizer.applyChatTemplate(messages: [
            [
                "role": "user",
                "content":
                    "Write a Swift function that returns the first n Fibonacci numbers. Include an explanation of its time complexity.",
            ]
        ])
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        func load(_ options: ExpertStreamingOptions?) throws -> any LanguageModel {
            let model = try ModelRegistry.default.createModel(
                configuration: data, modelType: configuration.modelType)
            try CheckpointLoader.load(
                modelDirectory: directory, model: model,
                quantization: configuration.perLayerQuantization?.quantization,
                perLayerQuantization: configuration.perLayerQuantization,
                expertStreaming: options)
            model.train(false)
            return model
        }
        let resident = try load(nil)
        let streamed = try load(.init())
        let check = InferenceErrorCheck(streamed)
        let leftCache = try resident.newCache(configuration: nil)
        let rightCache = try streamed.newCache(configuration: nil)
        var leftState: LMOutput.State?
        var rightState: LMOutput.State?
        var input = MLXArray(tokens, [1, tokens.count])
        var maxError: Float = 0
        var maxRMSE: Float = 0
        var maxTV: Float = 0
        var totalTV: Float = 0
        var matches = 0
        for _ in 0 ..< 128 {
            try autoreleasepool {
                let lhs = resident(.init(tokens: input), cache: leftCache, state: leftState)
                let rhs = streamed(.init(tokens: input), cache: rightCache, state: rightState)
                leftState = lhs.state
                rightState = rhs.state
                let a = lhs.logits[0, -1].asType(.float32)
                let b = rhs.logits[0, -1].asType(.float32)
                eval(a, b)
                try check.check()
                let delta = a - b
                let tv = (abs(softmax(a) - softmax(b)).sum() * 0.5).item(Float.self)
                maxError = max(maxError, abs(delta).max().item(Float.self))
                maxRMSE = max(maxRMSE, sqrt((delta * delta).mean()).item(Float.self))
                maxTV = max(maxTV, tv)
                totalTV += tv
                let token = argMax(a).item(Int.self)
                if token == argMax(b).item(Int.self) { matches += 1 }
                // Both models see the same history, so errors cannot accumulate merely
                // because a near-tied greedy choice led to a different sentence.
                input = MLXArray([token], [1, 1])
            }
        }
        print(
            "Expert streaming pretrained: prompt=\(tokens.count) top1=\(matches)/128 maxLogitError=\(maxError) maxRMSE=\(maxRMSE) maxTV=\(maxTV) meanTV=\(totalTV / 128)"
        )
        XCTAssertLessThan(maxRMSE, 0.1)
        XCTAssertLessThan(maxTV, 0.05)
        XCTAssertLessThan(totalTV / 128, 0.01)
        XCTAssertEqual(leftCache.map(\.offset), rightCache.map(\.offset))
    }

    func testExpertMathBiasesQuantizationAndBroadcastingSurviveStreaming() throws {
        typealias Forward = (Module, MLXArray, MLXArray) -> MLXArray
        let variants: [(String, () -> Module, Forward)] = [
            (
                "silu", { SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true) },
                { ($0 as! SwitchGLU)($1, $2) }
            ),
            (
                "gelu",
                {
                    SwitchGLU(
                        inputDims: 64, hiddenDims: 64, numExperts: 4, activation: geluApproximate)
                },
                { ($0 as! SwitchGLU)($1, $2) }
            ),
            (
                "ungated", { SwitchMLP(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true) },
                { ($0 as! SwitchMLP)($1, $2) }
            ),
            (
                "fused",
                { FusedGateUpSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true) },
                { ($0 as! FusedGateUpSwitchGLU)($1, $2) }
            ),
            (
                "gpt-oss",
                { SwiGLUSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true) },
                { ($0 as! SwiGLUSwitchGLU)($1, $2) }
            ),
            (
                "inkling",
                {
                    let module = InklingSwitchGLU(
                        inputDimensions: 64, hiddenDimensions: 64, experts: 4)
                    module.update(
                        parameters: .unflattened([
                            "gate_scale": MLXArray([Float(0.7), 1.1, 0.8, 1.4]),
                            "out_scale": MLXArray([Float(1.2), 0.9, 1.3, 0.6]),
                        ]))
                    return module
                }, { ($0 as! InklingSwitchGLU)($1, $2) }
            ),
        ]
        for (name, make, forward) in variants {
            for mode: QuantizationMode? in [nil, .affine, .mxfp4, .mxfp8, .nvfp4] {
                let resident = ExpertProjectionFixture(make())
                let group = mode == .nvfp4 ? 16 : 32
                let bits = mode == .mxfp8 ? 8 : 4
                if let mode {
                    try quantizeByParent(model: resident) { _, module in
                        module is SwitchLinear ? (group, bits, mode) : nil
                    }
                }
                let url = try directory()
                try checkpoint(resident, at: url)
                let streamed = ExpertProjectionFixture(make())
                try CheckpointLoader.load(
                    modelDirectory: url, model: streamed,
                    quantization: mode.map { .init(groupSize: group, bits: bits, mode: $0) },
                    expertStreaming: .init(cacheSizeBytes: 40_000, expertBatchSize: 2))
                let check = InferenceErrorCheck(streamed)
                XCTAssertFalse(
                    streamed.parameters().flattened().contains { $0.0.hasSuffix(".weight") })
                let evaluation = try XCTUnwrap(
                    streamed.modules().compactMap { $0 as? StreamingSwitchLinear }.first?.schedule
                        .evaluation)
                for (batch, count) in [(1, 1), (2, 37), (2, 1), (1, 1)] {
                    let before = evaluation.evaluationCount
                    let input = MLXRandom.normal([batch, count, 64])
                    let indices = MLXArray(
                        (0 ..< batch * count * 3).map { UInt32($0 % 4) }, [batch, count, 3])
                    let lhs = forward(resident.experts, input, indices)
                    let rhs = forward(streamed.experts, input, indices)
                    eval(lhs, rhs)
                    try check.check()
                    XCTAssertEqual(lhs.shape, rhs.shape)
                    XCTAssertEqual(evaluation.pendingWeightBytes, 0)
                    if batch == 1 && count == 1 {
                        XCTAssertGreaterThan(evaluation.evaluationCount, before, name)
                        // Quantized fixture groups fit the cap despite individual
                        // projections being split into multiple expert batches.
                        if mode != nil {
                            let bytes = streamed.modules().compactMap {
                                $0 as? StreamingSwitchLinear
                            }
                            .reduce(0) { total, projection in
                                total + projection.schedule.store.layers[projection.projection]
                                    .tensors.values.reduce(0) { $0 + $1.bytesPerExpert } * 3
                            }
                            XCTAssertEqual(
                                evaluation.evaluationCount - before, bytes <= 40_000 ? 1 : 2, name)
                        }
                    } else {
                        XCTAssertEqual(evaluation.evaluationCount, before, name)
                    }
                    XCTAssertTrue(
                        allClose(lhs, rhs, rtol: 2e-3, atol: 2e-4).item(Bool.self),
                        "\(name) \(String(describing: mode)) count=\(count) error=\(abs(lhs-rhs).max().item(Float.self))"
                    )
                }
                let projections = streamed.modules().compactMap { $0 as? StreamingSwitchLinear }
                XCTAssertFalse(projections.isEmpty)
                XCTAssertLessThanOrEqual(projections[0].schedule.store.residentBytes, 40_000)
            }
        }
    }

    func testSharedProjectionBuffersPreservePrefillDecodeAndEviction() throws {
        for mode: QuantizationMode? in [nil, .affine, .mxfp4, .nvfp4] {
            func make() -> ExpertProjectionFixture {
                ExpertProjectionFixture(
                    SwitchGLU(
                        inputDims: 256, hiddenDims: 64,
                        numExperts: 16, bias: true))
            }
            let resident = make()
            resident.update(parameters: resident.parameters().mapValues { $0.asType(.bfloat16) })
            let group = mode == .nvfp4 ? 16 : 32
            if let mode {
                try quantizeByParent(model: resident) { _, module in
                    module is SwitchLinear ? (group, 4, mode) : nil
                }
            }
            let directory = try directory()
            try checkpoint(resident, at: directory)
            let streamed = make()
            let budget = 1024 * 1024
            try CheckpointLoader.load(
                modelDirectory: directory, model: streamed,
                quantization: mode.map { .init(groupSize: group, bits: 4, mode: $0) },
                expertStreaming: .init(
                    cacheSizeBytes: budget, prefetch: true,
                    expertBatchSize: 2, sharedExpertBuffers: true))
            let projection = try XCTUnwrap(
                streamed.modules().compactMap { $0 as? StreamingSwitchLinear }.first)
            // More selected experts than the batch size exercise pool reuse while
            // sibling projection results are still lazy. Alternating routes evicts
            // old slots while speculative reads use the same cache.
            for (length, first) in [(32, 0), (1, 0), (1, 8), (1, 0), (64, 8), (1, 4)] {
                let input = MLXRandom.normal([1, length, 256]).asType(.bfloat16)
                let ids = MLXArray(
                    (0 ..< length * 6).map { UInt32((first + $0) % 16) }, [1, length, 6])
                let expected = (resident.experts as! SwitchGLU)(input, ids)
                let actual = (streamed.experts as! SwitchGLU)(input, ids)
                eval(expected, actual)
                try InferenceErrorCheck(streamed).check()
                XCTAssertTrue(
                    arrayEqual(expected, actual).item(Bool.self),
                    "mode=\(String(describing: mode)), length=\(length)")
                XCTAssertLessThanOrEqual(projection.schedule.store.residentBytes, budget)
                projection.schedule.evaluation.flush()
                XCTAssertEqual(projection.schedule.evaluation.pendingWeightBytes, 0)
            }
        }
    }

    func testPackedQuantizedPrefillPreservesMatrixAndVectorKernelRounding() throws {
        func make() -> ExpertProjectionFixture {
            ExpertProjectionFixture(
                SwitchLinear(inputDims: 256, outputDims: 64, numExperts: 128, bias: false))
        }
        let resident = make()
        resident.update(parameters: resident.parameters().mapValues { $0.asType(.bfloat16) })
        try quantizeByParent(model: resident) { _, module in
            module is SwitchLinear ? (32, 5, .affine) : nil
        }
        let directory = try directory()
        try checkpoint(resident, at: directory)
        let streamed = make()
        try CheckpointLoader.load(
            modelDirectory: directory, model: streamed,
            quantization: .init(groupSize: 32, bits: 5),
            expertStreaming: .init(cacheSizeBytes: 64_000, expertBatchSize: 2))
        for count in [64, 512] {
            // One very popular expert and many rare experts exercise both dispatch
            // thresholds: B/E < 4 and tiny batches from an originally large prefill.
            let tail = min(count / 2, 127)
            let ids = MLXArray(Array(repeating: 0, count: count - tail) + Array(1 ... tail))
            let input = MLXRandom.normal([count, 1, 256]).asType(.bfloat16)
            let lhs = (resident.experts as! SwitchLinear)(input, ids, sortedIndices: true)
            let rhs = (streamed.experts as! SwitchLinear)(input, ids, sortedIndices: true)
            eval(lhs, rhs)
            try InferenceErrorCheck(streamed).check()
            XCTAssertTrue(arrayEqual(lhs, rhs).item(Bool.self), "count=\(count)")
        }
    }

    func testNativeMoEModelsKeepRoutingSharedExpertsAndCacheContinuation() throws {
        let standard = [
            "mixtral", "phimoe", "qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_5_moe",
            "deepseek_v2", "deepseek_v3", "granitemoehybrid", "mimo_v2_flash", "minimax",
            "glm4_moe", "glm4_moe_lite", "gpt_oss", "olmoe", "bailing_moe",
            "lfm2_moe", "afmoe", "jamba",
        ]
        let custom = [
            "mellum", "exaone_moe", "hunyuan", "bailing_moe_linear", "deepseek_v32",
            "glm_moe_dsa", "inkling", "kimi_linear", "longcat_flash", "step3p5",
        ]
        for (name, data) in try standard.map({
            ($0, try ModelTestConfigurations.textConfiguration($0))
        })
            + custom.map({ ($0, try ModelTestConfigurations.configuration($0)) })
            + [
                "streaming_granitemoe", "streaming_dbrx", "streaming_deepseek", "streaming_dots1", "streaming_phixtral",
                "streaming_klear", "streaming_cohere2_moe", "streaming_mistral4", "streaming_laguna", "streaming_deepseek_v4",
                "streaming_llada2_moe", "streaming_diffusion_gemma", "streaming_step3p7", "streaming_llama4",
                "streaming_youtu_vl", "streaming_glm4v_moe", "streaming_ernie4_5_moe_vl", "streaming_zaya1_vl",
                "streaming_qwen3_omni_moe", "streaming_kimi_vl", "streaming_deepseek_vl_v2", "streaming_moondream3",
                "streaming_kimi_k3", "streaming_minimax_m3_vl", "streaming_deepseek_v41", "streaming_qwen4_exp",
                "streaming_deepseek_ocr", "streaming_nemotron_omni", "streaming_qwen3_vl_moe", "streaming_ernie4_5_moe",
                "streaming_nemotron_h", "streaming_gemma4_text", "streaming_qwen35_vl",
            ].map({ ($0, try ModelTestConfigurations.configuration($0)) })
        {
            for (packed, shared) in [(false, false), (true, false), (false, true), (true, true)] {
                FileHandle.standardOutput.write(
                    Data("Expert streaming parity \(name) quantized=\(packed) shared=\(shared)\n".utf8))
                do {
                    let config = try JSONDecoder().decode(BaseConfiguration.self, from: data)
                    let resident = try ModelRegistry.default.createModel(
                        configuration: data, modelType: config.modelType)
                    XCTAssertTrue(resident.modules().contains { $0 is SwitchLinear }, name)
                    resident.train(false)
                    resident.update(
                        parameters: resident.parameters().mapValues {
                            $0.ndim > 1 && $0.dtype == .float32 ? $0 * 0.1 : $0
                        })
                    if packed {
                        try quantizeByParent(model: resident) { _, module in
                            guard let projection = module as? SwitchLinear,
                                projection.inputDims.isMultiple(of: 32)
                            else { return nil }
                            return (32, 4, .affine)
                        }
                    }
                    let url = try directory()
                    try checkpoint(resident, at: url)
                    let streamed = try ModelRegistry.default.createModel(
                        configuration: data, modelType: config.modelType)
                    try CheckpointLoader.load(
                        modelDirectory: url, model: streamed,
                        quantization: packed ? .init(groupSize: 32, bits: 4) : nil,
                        expertStreaming: .init(
                            cacheSizeBytes: shared ? 2 * 1024 * 1024 : 100_000, expertBatchSize: 2,
                            sharedExpertBuffers: shared))
                    let check = InferenceErrorCheck(streamed)
                    if ["streaming_qwen3_omni_moe", "streaming_nemotron_omni"].contains(name) {
                        let original = Dictionary(
                            uniqueKeysWithValues: resident.parameters().flattened())
                        for (path, value) in streamed.parameters().flattened() {
                            XCTAssertTrue(
                                arrayEqual(value, try XCTUnwrap(original[path])).item(Bool.self),
                                path)
                        }
                    }
                    let originalProjections = Dictionary(
                        uniqueKeysWithValues: resident.namedModules().compactMap {
                            path, module -> (String, SwitchLinear)? in
                            (module as? SwitchLinear).map { (path, $0) }
                        })
                    for (path, module) in streamed.namedModules() {
                        guard let projection = module as? StreamingSwitchLinear else { continue }
                        let original = try XCTUnwrap(originalProjections[path])
                        let probe = MLXRandom.normal([2, 1, projection.inputDims])
                        let ids = MLXArray([0, projection.numExperts - 1])
                        let lhs = original(probe, ids)
                        let rhs = projection(probe, ids)
                        eval(lhs, rhs)
                        projection.schedule.evaluation.flush()
                        try check.check()
                        XCTAssertTrue(
                            allClose(lhs, rhs, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                            "\(name): \(path)")
                    }
                    XCTAssertEqual(
                        originalProjections.count,
                        streamed.modules().filter { $0 is StreamingSwitchLinear }.count)
                    let evaluation = try XCTUnwrap(
                        streamed.modules().compactMap { $0 as? StreamingSwitchLinear }
                            .first?.schedule.evaluation)
                    let leftCache = try resident.newCache(configuration: nil)
                    let rightCache = try streamed.newCache(configuration: nil)
                    var leftState: LMOutput.State?
                    var rightState: LMOutput.State?
                    for tokens in [[1, 3, 5, 7, 9], [11], [13]] {
                        let before = evaluation.evaluationCount
                        let input = MLXArray(tokens, [1, tokens.count])
                        let leftOutput = resident(
                            LMInput.Text(tokens: input), cache: leftCache, state: leftState)
                        let rightOutput = streamed(
                            LMInput.Text(tokens: input), cache: rightCache, state: rightState)
                        leftState = leftOutput.state
                        rightState = rightOutput.state
                        let lhs = leftOutput.logits
                        let rhs = rightOutput.logits
                        eval(lhs, rhs)
                        evaluation.flush()
                        try check.check()
                        XCTAssertTrue(
                            allClose(lhs, rhs, rtol: 2e-3, atol: 2e-4).item(Bool.self),
                            "\(name) max error \(abs(lhs-rhs).max().item(Float.self))")
                        XCTAssertEqual(
                            argMax(lhs, axis: -1).asArray(Int.self),
                            argMax(rhs, axis: -1).asArray(Int.self), name)
                        XCTAssertEqual(leftCache.map(\.offset), rightCache.map(\.offset), name)
                        if tokens.count == 1 {
                            XCTAssertGreaterThan(evaluation.evaluationCount, before, name)
                        }
                    }
                } catch { XCTFail("\(name) quantized=\(packed): \(error)") }
            }
        }
    }

    func testLargeExpertBatchesPreserveSlotsThroughEvictionAndPrefetch() throws {
        let make = {
            ExpertProjectionFixture(
                SwitchGLU(inputDims: 512, hiddenDims: 512, numExperts: 6, bias: true))
        }
        let url = try directory()
        let resident = make()
        try checkpoint(resident, at: url)
        // Four 1 MiB projections fit a batch, but the three sibling projections
        // cannot coexist in the cache. Exercise concurrent reads plus eviction.
        let budget = 5 * 1024 * 1024
        for prefetch in [false, true] {
            let streamed = make()
            try CheckpointLoader.load(
                modelDirectory: url, model: streamed,
                expertStreaming: .init(
                    cacheSizeBytes: budget, prefetch: prefetch, expertBatchSize: 4))
            let projection = try XCTUnwrap(
                streamed.modules().compactMap { $0 as? StreamingSwitchLinear }.first)
            for ids in [[5, 1, 5, 3, 0], [2, 4, 2, 0, 1], [5, 1, 5, 3, 0]] {
                let input = MLXRandom.normal([1, 1, 512])
                let indices = MLXArray(ids.map(UInt32.init), [1, 1, ids.count])
                let expected = (resident.experts as! SwitchGLU)(input, indices)
                let actual = (streamed.experts as! SwitchGLU)(input, indices)
                eval(expected, actual)
                try InferenceErrorCheck(streamed).check()
                XCTAssertTrue(allClose(expected, actual, rtol: 1e-5, atol: 1e-5).item(Bool.self))
                XCTAssertLessThanOrEqual(projection.schedule.store.residentBytes, budget)
                XCTAssertLessThanOrEqual(
                    projection.schedule.evaluation.peakPendingWeightBytes, budget)
                XCTAssertEqual(projection.schedule.evaluation.pendingWeightBytes, 0)
            }
        }
    }

    func testMissingAndTruncatedWeightsFailRatherThanGeneratingWithRandomExperts() throws {
        let make = {
            ExpertProjectionFixture(SwitchGLU(inputDims: 512, hiddenDims: 512, numExperts: 4))
        }
        let url = try directory()
        let resident = make()
        try checkpoint(resident, at: url)
        let streamed = make()
        try CheckpointLoader.load(
            modelDirectory: url, model: streamed, expertStreaming: .init(prefetch: false))
        let handle = try FileHandle(forWritingTo: url.appendingPathComponent("model.safetensors"))
        try handle.truncate(atOffset: 0)
        try handle.close()
        eval((streamed.experts as! SwitchGLU)(MLXArray.ones([1, 512]), MLXArray([0, 1], [1, 2])))
        XCTAssertThrowsError(try InferenceErrorCheck(streamed).check())
        XCTAssertThrowsError(
            try CheckpointLoader.load(
                modelDirectory: url, model: make(), expertStreaming: .init()))
        try checkpoint(resident, at: url)
        XCTAssertThrowsError(
            try CheckpointLoader.load(
                modelDirectory: url, model: make(),
                expertStreaming: .init(cacheSizeBytes: 16)))
        var missing = Dictionary(uniqueKeysWithValues: resident.parameters().flattened())
        missing.removeValue(forKey: "experts.gate_proj.weight")
        try save(arrays: missing, url: url.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(
            try CheckpointLoader.load(
                modelDirectory: url, model: make(), expertStreaming: .init()))
    }

    func testMixedPrecisionPromotionAndPerProjectionOverrides() throws {
        let make = {
            ExpertProjectionFixture(
                SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true))
        }
        let policy = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 32, bits: 4),
            perLayerQuantization: [
                "experts.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                "experts.down_proj": .skip,
            ])
        for dtype in [DType.float16, .bfloat16] {
            let resident = make()
            resident.update(parameters: resident.parameters().mapValues { $0.asType(dtype) })
            try quantizeByParent(model: resident) { path, module in
                module is SwitchLinear ? policy.quantization(layer: path)?.asTuple : nil
            }
            let url = try directory()
            try checkpoint(resident, at: url)
            let streamed = make()
            try CheckpointLoader.load(
                modelDirectory: url, model: streamed, perLayerQuantization: policy,
                expertStreaming: .init(
                    cacheSizeBytes: 40_000, prefetch: false, expertBatchSize: 1))
            for inputType in [dtype, .float32] {
                let input = MLXRandom.normal([1, 3, 64]).asType(inputType)
                let ids = MLXArray([0, 3, 2, 1, 1, 0], [1, 3, 2])
                let lhs = (resident.experts as! SwitchGLU)(input, ids)
                let rhs = (streamed.experts as! SwitchGLU)(input, ids)
                eval(lhs, rhs)
                try InferenceErrorCheck(streamed).check()
                XCTAssertEqual(lhs.dtype, rhs.dtype)
                XCTAssertTrue(allClose(lhs, rhs, rtol: 0.02, atol: 0.003).item(Bool.self))
            }
        }
    }

    func testDiffusionParityAndReadErrorsBeforeEmission() throws {
        for name in ["streaming_llada2_moe", "streaming_diffusion_gemma"] {
            let data = try ModelTestConfigurations.configuration(
                name, overrides: ["mask_token_id": 31])
            let config = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            let make = {
                try ModelRegistry.default.createModel(
                    configuration: data, modelType: config.modelType)
            }
            let resident = try make()
            let url = try directory()
            try checkpoint(resident, at: url)
            func load() throws -> any LanguageModel {
                let model = try make()
                try CheckpointLoader.load(
                    modelDirectory: url, model: model,
                    expertStreaming: .init(cacheSizeBytes: 100_000, prefetch: false))
                return model
            }
            func run(_ model: any LanguageModel, emit: ([Int]) throws -> Void) throws {
                let context = ModelContext(
                    metadata: .init(), model: model,
                    processor: ExecutionFixtureProcessor(), tokenizer: ExecutionFixtureTokenizer())
                _ = try runDiffusion(
                    input: LMInput(tokens: MLXArray([1, 2, 3], [1, 3])),
                    options: .init(sampling: .init(temperature: 0, seed: 42)),
                    diffusionParameters: .init(
                        maxTokens: 4, blockLength: 2, steps: 2, eosEarlyStop: false),
                    context: context, emit: emit)
            }
            let streamed = try load()
            var expected: [Int] = []
            var actual: [Int] = []
            try run(resident) { expected += $0 }
            try run(streamed) { actual += $0 }
            XCTAssertEqual(actual, expected, name)
            XCTAssertEqual(actual.count, 4)
            let broken = try load()
            let handle = try FileHandle(
                forWritingTo: url.appendingPathComponent("model.safetensors"))
            try handle.truncate(atOffset: 0)
            try handle.close()
            var emitted = false
            XCTAssertThrowsError(try run(broken) { _ in emitted = true })
            XCTAssertFalse(emitted, name)
        }
    }

    func testMultimodalPrefillPreservesImageFeaturesAndContinuation() throws {
        let data = try ModelTestConfigurations.configuration("streaming_qwen3_vl_moe")
        let make = {
            try ModelRegistry.default.createModel(configuration: data, modelType: "qwen3_vl_moe")
        }
        let resident = try make()
        let url = try directory()
        try checkpoint(resident, at: url)
        let streamed = try make()
        try CheckpointLoader.load(
            modelDirectory: url, model: streamed,
            expertStreaming: .init(cacheSizeBytes: 100_000))
        let pixels = MLXRandom.normal([16, 12])
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 92, 90, 90, 90, 90, 93, 2], [1, 8])),
            image: .init(pixels: pixels, frames: [THW(1, 4, 4)]))
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 4), sampling: .init(temperature: 0),
            output: .init(maxTokens: 8))
        let lhs = try AutoregressiveIterator(input: input, model: resident, options: options)
        let rhs = try AutoregressiveIterator(input: input, model: streamed, options: options)
        XCTAssertEqual(Array(lhs), Array(rhs))
        XCTAssertNil(rhs.failure)
        XCTAssertEqual(lhs.cache.map(\.offset), rhs.cache.map(\.offset))
    }

    func testChunkedPrefillAndGenerationReportStorageFailures() throws {
        let data = try ModelTestConfigurations.textConfiguration("qwen3_moe")
        let make = {
            try ModelRegistry.default.createModel(configuration: data, modelType: "qwen3_moe")
        }
        let resident = try make()
        let url = try directory()
        try checkpoint(resident, at: url)
        let streamed = try make()
        try CheckpointLoader.load(
            modelDirectory: url, model: streamed,
            expertStreaming: .init(cacheSizeBytes: 100_000, prefetch: false))
        let input = LMInput(tokens: MLXArray((0 ..< 520).map { $0 % 32 }))
        for size in [256, 512] {
            let options = ModelGenerationOptions(
                prefill: .init(stepSize: size),
                sampling: .init(temperature: 0), output: .init(maxTokens: 8))
            let lhs = try AutoregressiveIterator(input: input, model: resident, options: options)
            let rhs = try AutoregressiveIterator(input: input, model: streamed, options: options)
            XCTAssertEqual(Array(lhs), Array(rhs))
            XCTAssertNil(rhs.failure)
            XCTAssertEqual(lhs.cache.map(\.offset), rhs.cache.map(\.offset))
        }
        // A fresh store has no cached bytes that could mask the truncated source.
        let broken = try make()
        try CheckpointLoader.load(
            modelDirectory: url, model: broken,
            expertStreaming: .init(prefetch: false))
        let handle = try FileHandle(forWritingTo: url.appendingPathComponent("model.safetensors"))
        try handle.truncate(atOffset: 0)
        try handle.close()
        XCTAssertThrowsError(
            try AutoregressiveIterator(
                input: input, model: broken,
                options: .init(prefill: .init(stepSize: 256))))
    }
}
