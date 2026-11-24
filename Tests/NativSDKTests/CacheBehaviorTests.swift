import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class ContinuationFixture: Module, LLMModel, ContinuationStateCopying {
    static let position = LMOutput.Key<MLXArray>("position")
    var loraLayers: [Module] { [] }
    func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        [try makeAttentionKVCache(configuration: configuration)]
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let count = input.tokens.dim(-1)
        let position = state?[Self.position] ?? MLXArray([0])
        let values = broadcast(
            input.tokens.asType(.float32).reshaped([1, 1, count, 1]), to: [1, 1, count, 64])
        let output = attentionWithCacheUpdate(
            queries: values / 100, keys: values / 100, values: values,
            cache: cache?.first, scale: 0.125, mask: .causal)
        var next = LMOutput.State()
        next[Self.position] = position + count
        return LMOutput(logits: output + position, state: next)
    }
    func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var copy = LMOutput.State()
        copy[Self.position] = state[Self.position]?[.ellipsis]
        if let position = copy[Self.position] { eval(position) }
        return copy
    }
}

final class CacheBehaviorTests: XCTestCase {
    private func tensor(_ count: Int, heads: Int = 1, phase: Float = 0) -> MLXArray {
        sin(MLXArray(0 ..< count * heads * 64).asType(.float32) * 0.017 + phase)
            .reshaped([1, heads, count, 64])
    }

    private func attend(_ cache: KVCache, count: Int, phase: Float) -> MLXArray {
        let q = tensor(count, heads: 2, phase: phase)
        let k = tensor(count, phase: phase + 1)
        let v = tensor(count, phase: phase + 2)
        let mask = cache.makeMask(n: count, windowSize: nil, returnArray: false)
        return attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: 0.125, mask: mask)
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, tolerance: Double = 1e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        XCTAssertTrue(
            allClose(a, b, rtol: tolerance, atol: tolerance).item(Bool.self),
            "max absolute error: \(abs(a - b).max().item(Float.self))", file: file, line: line)
    }

    func testOrdinaryRotatingAndAffineAttentionMatchPinnedFixtures() throws {
        let caches: [(KVCache, String)] = [
            (KVCacheSimple(), "KVCacheSimple"),
            (RotatingKVCache(maxSize: 8, keep: 2), "RotatingKVCache"),
            (QuantizedKVCache(groupSize: 32, bits: 4), "QuantizedKVCache"),
        ]
        for (owned, name) in caches {
            var offset = 0
            for (index, count) in [5, 3, 1, 4, 1].enumerated() {
                assertClose(
                    attend(owned, count: count, phase: Float(index)),
                    try RuntimeReferenceFixtures.tensor(
                        "cache-\(name)-\(index)", shape: [1, 2, count, 64]))
                offset += count
                XCTAssertEqual(owned.offset, offset)
            }
            let copy = owned.copy()
            eval(copy.innerState())
            let expected = attend(copy, count: 2, phase: 7)
            _ = attend(owned, count: 2, phase: 8)
            let copy2 = copy.copy()
            eval(copy2.innerState())
            _ = attend(copy, count: 1, phase: 9)
            XCTAssertEqual(copy2.offset, 16)
            eval(expected)
        }
    }

    func testTurboQuantAttentionAndRepeatedSnapshotsAcrossCompressionModes() {
        for keyBits in [0, 3, 8] {
            let owned = TurboQuantKVCache(keyBits: keyBits, valueBits: 3)
            var keys: [MLXArray] = []
            var values: [MLXArray] = []
            for (index, count) in [5, 1, 2].enumerated() {
                keys.append(tensor(count, phase: Float(index) + 1))
                values.append(tensor(count, phase: Float(index) + 2))
                let allKeys = concatenated(keys, axis: 2)
                let allValues = concatenated(values, axis: 2)
                let q = tensor(count, heads: 2, phase: Float(index))
                // The pinned WHT encoder has a cross-SIMD shared-memory race.
                // Use independent MLX quantize/dequantize + SDPA as the oracle.
                let expected: MLXArray
                if index == 0 {
                    expected = MLXFast.scaledDotProductAttention(
                        queries: q, keys: allKeys,
                        values: allValues, scale: 0.125, mask: .causal)
                } else {
                    let key: MLXArray
                    if keyBits == 0 {
                        key = allKeys
                    } else if keyBits == 8 {
                        let quant = quantized(allKeys, groupSize: 64, bits: 8)
                        key = dequantized(
                            quant.wq, scales: quant.scales, biases: quant.biases, groupSize: 64,
                            bits: 8)
                    } else {
                        let codec = MSECodec(dim: 64, bits: keyBits, seed: 42)
                        key = codec.decode(codec.encode(allKeys))
                    }
                    let codec = MSECodec(dim: 64, bits: 3, seed: 43)
                    let value = codec.decode(codec.encode(allValues))
                    expected = MLXFast.scaledDotProductAttention(
                        queries: q, keys: key,
                        values: value, scale: 0.125, mask: .causal)
                }
                // Fused versus unfused attention uses the same 1e-3 tolerance
                // as the affine/chunked comparisons; snapshot checks stay 2e-5.
                assertClose(
                    attend(owned, count: count, phase: Float(index)), expected, tolerance: 1e-3)
            }
            XCTAssertTrue(owned.isCompressed)
            let snapshot = owned.copy()
            eval(snapshot.innerState())
            let expected = attend(owned, count: 2, phase: 4)
            let restored = snapshot.copy()
            assertClose(attend(restored, count: 2, phase: 4), expected)
            _ = attend(restored, count: 1, phase: 9)
            assertClose(attend(snapshot.copy(), count: 2, phase: 4), expected)
            XCTAssertEqual(snapshot.offset, 8)
        }
        // Copy before compression, including the empty-cache case.
        let raw = TurboQuantKVCache()
        XCTAssertEqual(raw.copy().offset, 0)
        _ = attend(raw, count: 4, phase: 0)
        let copy = raw.copy()
        assertClose(attend(raw, count: 1, phase: 1), attend(copy, count: 1, phase: 1))
    }

    func testWHTEncoderMatchesIndependentRotationAcrossSIMDGroups() {
        for dim in [64, 128, 256] {
            let codec = MSECodec(dim: dim, bits: 3, seed: 43)
            let signs = codec.whtSigns!.asArray(Float.self).map(Double.init)
            let boundaries = codec.boundaries.asArray(Float.self).map(Double.init)
            for seed: UInt64 in 0 ..< 8 {
                let values = MLXRandom.normal([17, dim], key: MLXRandom.key(seed))
                let input = values.asArray(Float.self).map(Double.init)
                var packed: [UInt32] = []
                var norms: [Float] = []
                // Scalar Double butterfly is independent of GPU scheduling,
                // fused reductions and the reduced-precision matmul oracle.
                for row in 0 ..< 17 {
                    let x = Array(input[(row * dim) ..< ((row + 1) * dim)])
                    let norm = sqrt(x.reduce(0) { $0 + $1 * $1 })
                    norms.append(Float(norm))
                    var rotated = zip(x, signs).map { $0 * $1 / norm }
                    var half = 1
                    while half < dim {
                        for block in stride(from: 0, to: dim, by: half * 2) {
                            for j in 0 ..< half {
                                let a = rotated[block + j]
                                let b = rotated[block + j + half]
                                rotated[block + j] = a + b
                                rotated[block + j + half] = a - b
                            }
                        }
                        half *= 2
                    }
                    var words = Array(repeating: UInt32(0), count: (dim * 3 + 31) / 32)
                    for j in 0 ..< dim {
                        let value = rotated[j] / sqrt(Double(dim))
                        let bin = UInt32(boundaries.filter { value > $0 }.count)
                        let bit = j * 3
                        let word = bit / 32
                        let shift = bit % 32
                        words[word] |= bin << shift
                        if shift > 29 { words[word + 1] |= bin >> (32 - shift) }
                    }
                    packed += words
                }
                let actual = TurboQuantKernelOps.fusedEncodeWHT(
                    input: values,
                    whtSigns: codec.whtSigns!, boundaries: codec.boundaries,
                    codebook: codec.codebook,
                    bits: 3, dim: dim)
                XCTAssertTrue(
                    actual.packed.asArray(UInt32.self) == packed, "dim=\(dim), seed=\(seed)")
                assertClose(actual.norms, MLXArray(norms))
            }
        }
    }

    func testVarianceNormalizedTilesMatchPinnedFixturesAndRestoreIndependently() throws {
        let owned = VarianceNormalizedKVCache(tileSize: 32, keyBits: 4, valueBits: 4)
        for (index, count) in [35, 1, 2].enumerated() {
            assertClose(
                attend(owned, count: count, phase: Float(index)),
                try RuntimeReferenceFixtures.tensor(
                    "cache-VarianceNormalizedKVCache-\(index)", shape: [1, 2, count, 64]))
        }
        let snapshot = owned.copy()
        eval(snapshot.innerState())
        let expected = attend(owned, count: 2, phase: 5)
        assertClose(attend(snapshot.copy(), count: 2, phase: 5), expected)
        XCTAssertEqual(snapshot.offset, 38)
    }

    func testAffineAccuracyAndChunkedPrefill() {
        let full = KVCacheSimple()
        let quantized = QuantizedKVCache(groupSize: 32, bits: 8)
        let expected = attend(full, count: 12, phase: 0)
        let compressed = attend(quantized, count: 12, phase: 0)
        assertClose(compressed, expected, tolerance: 0.02)
        let q = tensor(12, heads: 2)
        let k = tensor(12, phase: 1)
        let v = tensor(12, phase: 2)
        let chunked = KVCacheSimple()
        var outputs: [MLXArray] = []
        for range in [0 ..< 5, 5 ..< 9, 9 ..< 12] {
            outputs.append(
                attentionWithCacheUpdate(
                    queries: q[.ellipsis, range, 0...], keys: k[.ellipsis, range, 0...],
                    values: v[.ellipsis, range, 0...], cache: chunked, scale: 0.125, mask: .causal))
        }
        // SDPA selects different Metal kernels for these query lengths.
        assertClose(concatenated(outputs, axis: 2), expected, tolerance: 1e-3)
    }

    func testRecurrentCompositeCapabilitiesAndMissingSlots() throws {
        let recurrent = MambaCache(leftPadding: [2, 0])
        recurrent[1] = MLXArray.ones([2, 3])
        let composite = CacheList(KVCacheSimple(), recurrent)
        XCTAssertTrue(composite.capabilities.contains([.attention, .recurrent, .snapshot]))
        XCTAssertFalse(composite.capabilities.contains(.fullRewind))
        let copy = composite.copy() as! CacheList
        let restored = copy[1] as! MambaCache
        XCTAssertNil(restored[0])
        recurrent[1] = MLXArray.zeros([2, 3])
        XCTAssertEqual(restored[1]!.sum().item(Float.self), 6)
        XCTAssertFalse(PromptCacheOps.trimAll([composite], by: 1))
        let unsupported = CacheList(KVCacheSimple(), BaseKVCache())
        XCTAssertFalse(PromptCacheOps.isCheckpointSafe(unsupported))
        let model = ContinuationFixture()
        XCTAssertThrowsError(
            try ContinuationSnapshot(
                model: model, caches: [unsupported],
                state: nil, tokens: [], scope: scope(model))
        ) {
            XCTAssertEqual($0 as? ContinuationError, .unsupportedCache(0))
        }
    }

    func testMasksAndPreparedRecurrentLengths() {
        for window: Int? in [nil, 3] {
            let expected = (0 ..< 4).flatMap { query in
                (0 ..< 7).map { key in
                    key <= query + 3 && (window == nil || key > query + 3 - window!)
                }
            }
            XCTAssertEqual(
                createCausalMask(n: 4, offset: 3, windowSize: window).asArray(Bool.self),
                expected)
        }
        let cache = MambaCache(leftPadding: [2, 0])
        let hidden = MLXArray.zeros([2, 4, 8])
        cache.prepare(lengths: [3, 4])
        XCTAssertEqual(
            createSSMMask(h: hidden, cache: cache)!.asArray(Bool.self),
            [false, false, true, true, true, true, true, true])
        cache.finalize()
    }

    func testNestedCompressionPolicyPreservesRecurrentAndBoundaryLayers() throws {
        let attention = (0 ..< 6).map { _ in KVCacheSimple() }
        for cache in attention { _ = attend(cache, count: 2, phase: 0) }
        let recurrent = MambaCache()
        var caches: [KVCache] = [CacheList(caches: attention + [recurrent])]
        let configuration = KVCacheConfiguration(
            strategy: .turboQuant(
                try .init(keyPrecision: .threeBit, valuePrecision: .threeBit)))
        let applied = try applyKVCacheConfiguration(
            cache: &caches, configuration: configuration)
        XCTAssertEqual(applied.convertedLayerCount, 6)
        let leaves = KVCacheTree.leaves(in: caches)
        XCTAssertEqual(leaves.filter { $0.cache is QuantizedKVCache }.count, 4)
        XCTAssertEqual(leaves.filter { $0.cache is TurboQuantKVCache }.count, 2)
        XCTAssertTrue((leaves.last!.cache as? MambaCache) === recurrent)
        // TurboQuant defers packing until the first decode token.
        XCTAssertEqual(
            KVCacheStatus(cache: caches, configuration: configuration).compressedLayerCount, 4)
        for leaf in leaves where leaf.cache is TurboQuantKVCache {
            _ = attend(leaf.cache, count: 1, phase: 1)
        }
        XCTAssertEqual(
            KVCacheStatus(cache: caches, configuration: configuration).compressedLayerCount, 6)
    }

    private func scope(
        _ model: any LanguageModel, processor: String = "processor-v1",
        configuration: String = "weights-v1"
    ) -> ContinuationScope {
        .init(model: model, processorRevision: processor, configurationRevision: configuration)
    }

    func testContinuationRestoresPositionAndRejectsIncompatibleIdentityOrMedia() throws {
        let model = ContinuationFixture()
        let cache = try model.newCache(configuration: nil)
        let prefix = model(.init(tokens: MLXArray([1, 2, 3])[.newAxis]), cache: cache, state: nil)
        let snapshot = try ContinuationSnapshot(
            model: model, caches: cache, state: prefix.state,
            tokens: [1, 2, 3], scope: scope(model))
        let expected = model(
            .init(tokens: MLXArray([4, 5])[.newAxis]), cache: cache, state: prefix.state
        ).logits
        eval(expected)
        prefix.state?[ContinuationFixture.position]?[0] = MLXArray(999)
        for _ in 0 ..< 2 {
            let restored = try snapshot.restore(model: model, scope: scope(model))
            XCTAssertEqual(restored.state?[ContinuationFixture.position]?.item(Int.self), 3)
            assertClose(
                model(
                    .init(tokens: MLXArray([4, 5])[.newAxis]), cache: restored.caches,
                    state: restored.state
                ).logits, expected)
        }
        XCTAssertThrowsError(
            try snapshot.restore(model: model, scope: scope(model, processor: "processor-v2")))
        XCTAssertThrowsError(
            try snapshot.restore(model: model, scope: scope(model, configuration: "weights-v2")))
        let other = ContinuationFixture()
        XCTAssertThrowsError(try snapshot.restore(model: other, scope: scope(other)))
        XCTAssertThrowsError(
            try ContinuationSnapshot(
                model: model, caches: [], state: nil,
                tokens: [], scope: scope(model), mediaIdentities: ["image-v1"]))
        XCTAssertThrowsError(
            try ContinuationSnapshot(
                model: model, caches: cache, state: nil,
                tokens: [], scope: scope(model)))
    }

    func testOwnedPolicyConvertsLocallyAndHonorsCapacityAndConflicts() throws {
        let model = ContinuationFixture()
        let configuration = KVCacheConfiguration(
            strategy: .affine(try .init(bits: 4, groupSize: 32)))
        let iterator = try AutoregressiveIterator(
            input: .init(tokens: MLXArray([1, 2])), model: model,
            options: .init(cache: configuration, output: .init(maxTokens: 1)))
        XCTAssertTrue(iterator.cache[0] is QuantizedKVCache)
        let capacity = KVCacheConfiguration(
            capacity: try .init(maxTokens: 8, preservedPrefixTokens: 2))
        let bounded = try AutoregressiveIterator(
            input: .init(tokens: MLXArray([1, 2])), model: model,
            options: .init(cache: capacity, output: .init(maxTokens: 1)))
        XCTAssertEqual((bounded.cache[0] as? RotatingKVCache)?.keepCount, 2)
        XCTAssertEqual(bounded.cache[0].maxSize, 8)
        var cache: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 8)]
        let strict = KVCacheConfiguration(
            strategy: configuration.strategy, compatibility: .requireAllLayers)
        XCTAssertThrowsError(
            try applyKVCacheConfiguration(cache: &cache, configuration: strict))
        XCTAssertTrue(cache[0] is KVCacheSimple, "validation must be transactional")

    }
}
