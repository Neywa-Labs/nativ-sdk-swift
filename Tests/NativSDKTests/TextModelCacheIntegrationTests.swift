import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class TextModelCacheIntegrationTests: XCTestCase {
    func testAFMReuseTracksLogicalPositionsAfterEviction() throws {
        let loaded = try ModelTestConfigurations.model("afm7")
        let model = try XCTUnwrap(loaded as? AFM7Model)
        let bounded = try model.newCache(
            configuration: .init(
                capacity: try .init(maxTokens: 6, preservedPrefixTokens: 0)))
        let complete = try model.newCache(configuration: nil)
        // Keep the full history in the baseline and mask it explicitly. Its RoPE
        // positions remain independent of the bounded cache's physical row count.
        for position in 0 ..< 20 {
            let tokens = MLXArray([position % 17 + 1], [1, 1])
            let actual = model(tokens, cache: bounded)
            var h = model.model.embedding(tokens)
            let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
                createCausalMask(n: 1, offset: position, windowSize: 6))
            for (index, layer) in model.model.layers.enumerated() {
                h = layer(h, mask: mask, cache: complete[index])
            }
            let shared = try XCTUnwrap(complete.last).state
            for layer in model.model.kvReuseLayers {
                h = layer(h, keys: shared[0], values: shared[1], offset: position, mask: mask)
            }
            let expected = model.model.embedding.asLinear(model.model.outputNorm(h))
            XCTAssertTrue(
                allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                "position \(position)")
        }
    }

    func testSparseAttentionPatternAcrossPartialBlocksAndCapacityRejection() throws {
        let loaded = try ModelTestConfigurations.model("phi3small")
        let model = try XCTUnwrap(loaded as? Phi3SmallModel)
        let attention = try XCTUnwrap(model.model.layers.first).selfAttention
        for queryLength in [1, 11, 37, 75] {
            let actual = attention.blockSparseMask(queryLength: queryLength, keyLength: 75).asArray(
                Bool.self)
            var expected = [Bool]()
            for head in 0 ..< 4 {
                for query in (75 - queryLength) ..< 75 {
                    for key in 0 ..< 75 {
                        let qBlock = query / 32
                        let kBlock = key / 32
                        expected.append(
                            qBlock >= kBlock
                                && (qBlock - kBlock < 1 || (kBlock + head + 1) % 2 == 0))
                    }
                }
            }
            XCTAssertEqual(actual, expected)
        }
        XCTAssertThrowsError(
            try model.newCache(configuration: .init(capacity: try .init(maxTokens: 64)))
        ) {
            guard case KVCacheConfigurationError.incompatibleCapacity(expected: 64, count: 2) = $0
            else {
                return XCTFail("Unexpected error: \($0)")
            }
        }
    }

    func testMambaSingleTapConvolutionDoesNotRetainEarlierInputs() throws {
        let data = try ModelTestConfigurations.configuration("mamba")
        var config = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        config["conv_kernel"] = 1
        let model = try ModelRegistry.default.createModel(
            configuration: JSONSerialization.data(withJSONObject: config), modelType: "mamba")
        try assertTextModelCacheEquivalence(model)
    }

    func testSnapshotsResumeIndependentlyAfterWindowWrap() throws {
        for name in ModelTestConfigurations.textModels {
            let model = try ModelTestConfigurations.model(name)
            for bounded in [false, true] {
                // Block-sparse attention addresses absolute positions in an unbounded history.
                if bounded && name == "phi3small" { continue }
                let configuration =
                    bounded
                    ? KVCacheConfiguration(
                        capacity: try .init(maxTokens: 6, preservedPrefixTokens: 0))
                    : nil
                let cache = try model.newCache(configuration: configuration)
                if let configuration {
                    try validateKVCacheCompatibility(cache, configuration: configuration)
                }
                let prefix = (0 ..< 13).map { $0 % 17 + 1 }
                for token in prefix { eval(model(MLXArray([token], [1, 1]), cache: cache), cache) }
                let scope = ContinuationScope(
                    model: model, processorRevision: "fixture", configurationRevision: "fixture",
                    cacheConfiguration: configuration)
                let snapshot = try ContinuationSnapshot(
                    model: model, caches: cache, state: nil, tokens: prefix, scope: scope)
                let continuation = MLXArray([14, 15, 16], [1, 3])
                let expected = model(continuation, cache: cache)
                eval(expected, cache)
                for _ in 0 ..< 2 {
                    let restored = try snapshot.restore(model: model, scope: scope)
                    let actual = model(continuation, cache: restored.caches)
                    eval(actual, restored.caches)
                    XCTAssertTrue(
                        allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                        "\(name), bounded: \(bounded)")
                }
                if bounded && name != "mamba" {
                    XCTAssertEqual(
                        KVCacheStatus(cache: cache, configuration: configuration)
                            .capacityDisposition, .fullyApplied, name)
                }
            }
        }
    }

    func testMaterializedHistoriesRejectCompressionWithoutMutation() throws {
        let strategies: [KVCacheConfiguration.Strategy] = [
            .affine(try .init(bits: 4, groupSize: 32)),
            .turboQuant(try .init(keyPrecision: .threeBit, valuePrecision: .threeBit)),
            .varianceNormalized(.balanced),
        ]
        for name in ["afm7", "iquestloopcoder"] {
            let model = try ModelTestConfigurations.model(name)
            for strategy in strategies {
                var caches = try model.newCache(configuration: nil)
                eval(model(MLXArray([1, 2, 3], [1, 3]), cache: caches), caches)
                let copies = caches.map { $0.copy() }
                let strict = KVCacheConfiguration(
                    strategy: strategy, compatibility: .requireAllLayers)
                XCTAssertThrowsError(
                    try applyKVCacheConfiguration(cache: &caches, configuration: strict))
                let partial = KVCacheConfiguration(strategy: strategy, compatibility: .allowPartial)
                let result = try applyKVCacheConfiguration(cache: &caches, configuration: partial)
                XCTAssertEqual(result.convertedLayerCount, 0)
                XCTAssertTrue(result.skipped.contains { $0.reason == .modelRequirement })
                let actual = model(MLXArray([4, 5], [1, 2]), cache: caches)
                let expected = model(MLXArray([4, 5], [1, 2]), cache: copies)
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 1e-5, atol: 1e-6).item(Bool.self), name)
            }
        }
    }

    func testHybridGlobalAttentionCanCompressWhileNativeWindowsStayIntact() throws {
        for name in ["cohere2", "mellum", "spark2_5", "exaone_moe", "ministral3", "phi3small"] {
            let data = try ModelTestConfigurations.configuration(name)
            var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            json["hidden_size"] = 128
            json["head_dim"] = 32
            let model = try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: json), modelType: name)
            model.train(false)
            var caches = try model.newCache(configuration: nil)
            eval(model(MLXArray([1, 2, 3], [1, 3]), cache: caches), caches)
            let expectedCache = caches.map { $0.copy() }
            let configuration = KVCacheConfiguration(
                strategy: .affine(try .init(bits: 8, groupSize: 32)),
                compatibility: .requireAtLeastOneLayer)
            let result = try applyKVCacheConfiguration(cache: &caches, configuration: configuration)
            XCTAssertGreaterThan(result.convertedLayerCount, 0, name)
            XCTAssertGreaterThan(result.skipped.count, 0, name)
            let actual = model(MLXArray([4], [1, 1]), cache: caches)
            let expected = model(MLXArray([4], [1, 1]), cache: expectedCache)
            eval(actual, expected)
            XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite), name)
            XCTAssertTrue(allClose(actual, expected, rtol: 0.05, atol: 0.08).item(Bool.self), name)
        }
    }

    func testQuantizedWeightsLoadAcrossDenseMoEAndRecurrentModels() async throws {
        for name in ["cohere2", "hunyuan", "mamba", "ministral3"] {
            let data = try ModelTestConfigurations.configuration(name)
            try await assertTextCheckpoint(configuration: data, modelType: name, quantized: true)
        }
    }

    func testPrunedAttentionCanStartWithLinearOrHaveNoAttention() throws {
        let data = try ModelTestConfigurations.configuration("nemotron-nas")
        var config = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases: [[String: Any]] = [
            ["attention": ["replace_with_linear": true], "ffn": ["ffn_mult": 2.625]],
            ["attention": ["no_op": true], "ffn": ["ffn_mult": 2.625]],
        ]
        for allPruned in [false, true] {
            config["block_configs"] = [
                cases[0], cases[1], cases[0],
                allPruned
                    ? cases[1] : ["attention": ["n_heads_in_group": 2], "ffn": ["no_op": true]],
            ]
            let model = try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: config),
                modelType: "nemotron_nas")
            try assertTextModelCacheEquivalence(model)
            let cache = try model.newCache(configuration: nil)
            XCTAssertEqual(cache.count, allPruned ? 0 : 1)
        }
    }
}
