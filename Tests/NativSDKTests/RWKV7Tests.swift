import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class RWKV7Tests: XCTestCase {

    private static func config() throws -> RWKV7Configuration {
        try JSONDecoder.json5().decode(
            RWKV7Configuration.self, from: ModelTestConfigurations.configuration("rwkv7"))
    }

    /// Replaces every all-zero parameter with random values.
    ///
    /// `g_norm`'s scale and shift and `r_k` are all initialized to zeros, and the
    /// time-mixing output is `g_norm(wkv) + (r * k * r_k).sum(-1) * v` -- so a
    /// default-initialized block returns exactly zero and every probe through it is blind.
    private static func nondegenerate<M: Module>(_ module: M) -> M {
        var updates: [(String, MLXArray)] = []
        for (index, (key, value)) in module.parameters().flattened().enumerated() {
            eval(value)
            if abs(value).max().item(Float.self) == 0 {
                updates.append(
                    (
                        key,
                        MLXRandom.normal(value.shape, key: MLXRandom.key(300 + UInt64(index)))
                            * 0.3
                    ))
            }
        }
        module.update(parameters: ModuleParameters.unflattened(updates))
        return module
    }

    /// The Metal kernel and the MLX step loop have to agree. The kernel is the path every
    /// real run takes, and nothing else in the model would notice if it were wrong.
    func testKernelMatchesTheReferenceRecurrence() {
        let key = MLXRandom.key(41)
        let shape = [1, 5, 2, 64]
        let r = MLXRandom.normal(shape, key: key) * 0.5
        let k = MLXRandom.normal(shape, key: MLXRandom.key(42)) * 0.5
        let v = MLXRandom.normal(shape, key: MLXRandom.key(43)) * 0.5
        let a = MLXRandom.normal(shape, key: MLXRandom.key(44)) * 0.1
        let b = MLXRandom.normal(shape, key: MLXRandom.key(45)) * 0.1
        let w = sigmoid(MLXRandom.normal(shape, key: MLXRandom.key(46)))
        let state = MLXRandom.normal([1, 2, 64, 64], key: MLXRandom.key(47)) * 0.1

        let (kernelOut, kernelState) = rwkv7WKV(
            r: r, w: w, k: k, v: v, a: a, b: b, state: state)
        let (referenceOut, referenceState) = rwkv7WKV(
            r: r, w: w, k: k, v: v, a: a, b: b, state: state, forceReference: true)
        eval(kernelOut, kernelState, referenceOut, referenceState)

        XCTAssertEqual(kernelOut.shape, referenceOut.shape)
        XCTAssertEqual(kernelState.shape, referenceState.shape)
        XCTAssertLessThan(
            abs(kernelOut - referenceOut).max().item(Float.self), 1e-5,
            "the Metal kernel disagrees with the recurrence it implements")
        XCTAssertLessThan(
            abs(kernelState - referenceState).max().item(Float.self), 1e-5,
            "the kernel's carried state disagrees with the reference")
        XCTAssertGreaterThan(
            abs(referenceOut).max().item(Float.self), 1e-2,
            "the probe is near zero, so agreement means nothing")
    }

    /// One step of the reference, written out: `state <- state * w + v k^T`, with `a` and
    /// `b` zeroed so the low-rank correction drops out, then `y = state r`.
    func testReferenceStepMatchesTheWrittenRecurrence() {
        let r = MLXArray(converting: [1.0, 0.0]).reshaped(1, 1, 2)
        let w = MLXArray(converting: [0.5, 0.5]).reshaped(1, 1, 2)
        let k = MLXArray(converting: [1.0, 2.0]).reshaped(1, 1, 2)
        let v = MLXArray(converting: [3.0, 4.0]).reshaped(1, 1, 2)
        let zero = MLXArray.zeros([1, 1, 2])
        let state = MLXArray.ones([1, 1, 2, 2])

        let (y, next) = rwkv7Step(r: r, w: w, k: k, v: v, a: zero, b: zero, state: state)
        eval(y, next)

        XCTAssertEqual(next.reshaped(4).asArray(Float.self), [3.5, 6.5, 4.5, 8.5])
        XCTAssertEqual(y.reshaped(2).asArray(Float.self), [3.5, 4.5])
    }

    /// A single position takes the carried state directly; a sequence is shifted right by
    /// one; an absent state is zeros.
    func testTokenShift() {
        let x = MLXArray(converting: [1.0, 2.0, 3.0]).reshaped(1, 3, 1)
        let fresh = rwkv7TokenShift(x, state: nil)
        eval(fresh)
        XCTAssertEqual(fresh.reshaped(3).asArray(Float.self), [0, 1, 2])

        let carried = MLXArray(converting: [9.0]).reshaped(1, 1, 1)
        let resumed = rwkv7TokenShift(x, state: carried)
        eval(resumed)
        XCTAssertEqual(resumed.reshaped(3).asArray(Float.self), [9, 1, 2])

        let single = rwkv7TokenShift(x[0..., ..<1], state: carried)
        eval(single)
        XCTAssertEqual(single.reshaped(1).asArray(Float.self), [9])
    }

    /// Layer 0 is shaped differently twice: it alone has a `pre_norm`, and it alone has no
    /// `v_lora` because it produces `v_first` rather than consuming one.
    func testLayerZeroIsShapedDifferently() throws {
        let model = RWKV7Model(try Self.config())
        XCTAssertNotNil(model.model.layers[0].preNorm)
        XCTAssertNil(model.model.layers[1].preNorm)
        XCTAssertNil(model.model.layers[0].timeMixing.vLoRA)
        XCTAssertNotNil(model.model.layers[1].timeMixing.vLoRA)

        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.0.pre_norm.weight"))
        XCTAssertFalse(keys.contains("model.layers.1.pre_norm.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.attn.v_lora.lora.0.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.attn.v_lora.lora.0.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.attn.v_lora.lora.2.bias"))
    }

    /// The two projections keep the checkpoint's sparse Sequential names and support
    /// strict loading even though no parameter-bearing module exists at index 1.
    func testLoRAKeepsTheGapAtIndexOne() throws {
        let lora = RWKV7LoRA(
            inputDimensions: 8, outputDimensions: 8, lowRankDimensions: 4,
            activation: "tanh")
        let keys = Set(lora.parameters().flattened().map(\.0))
        XCTAssertEqual(keys, ["lora.0.weight", "lora.2.weight", "lora.2.bias"])
        try lora.update(parameters: lora.parameters(), verify: .all)
    }

    func testQuantizedLoRAWeightsStrictlyLoadWithSparseNumericKeys() throws {
        let source = RWKV7LoRA(
            inputDimensions: 64, outputDimensions: 64, lowRankDimensions: 32, activation: "tanh")
        try quantizeByParent(model: source, groupSize: 32, bits: 4) { path, _ in path == "lora.0" }
        let target = RWKV7LoRA(
            inputDimensions: 64, outputDimensions: 64, lowRankDimensions: 32, activation: "tanh")
        try quantizeByParent(model: target, groupSize: 32, bits: 4) { path, _ in path == "lora.0" }
        XCTAssertEqual(target.lora.count, 3)
        try target.update(parameters: source.parameters(), verify: .all)
        let x = MLXArray.ones([1, 2, 64])
        let expected = source(x)
        let actual = target(x)
        eval(expected, actual)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-6)
    }

    /// `v_first` comes from layer 0 and is threaded through every later layer, so removing
    /// layer 0's contribution has to change what a later layer produces.
    func testValueFirstIsThreadedFromLayerZero() throws {
        let model = Self.nondegenerate(RWKV7Model(try Self.config()))
        let x = MLXRandom.normal([1, 4, 128], key: MLXRandom.key(51))

        let layerOne = model.model.layers[1].timeMixing
        let (withFirst, _) = layerOne(
            x, valueFirst: MLXArray.ones([1, 4, 2, 64]), cache: nil)
        let (withOther, _) = layerOne(
            x, valueFirst: MLXArray.ones([1, 4, 2, 64]) * -3, cache: nil)
        eval(withFirst, withOther)

        XCTAssertGreaterThan(
            abs(withFirst - withOther).max().item(Float.self), 1e-4,
            "a later layer ignores v_first, so layer 0's value never reaches it")

        let layerZero = model.model.layers[0].timeMixing
        let (_, produced) = layerZero(x, valueFirst: nil, cache: nil)
        eval(produced)
        XCTAssertEqual(produced.shape, [1, 4, 2, 64], "layer 0 must emit a v_first")
    }

    /// Three unrelated slots per layer, and a forward has to write all of them.
    func testCacheHasThreeSlotsAndAllAreWritten() throws {
        let model = RWKV7Model(try Self.config())
        let cache = try model.newCache(configuration: nil)
        XCTAssertEqual(cache.count, 3)

        let slots = try XCTUnwrap(cache[0] as? ArraysCache)
        XCTAssertNil(slots[0])
        XCTAssertNil(slots[1])
        XCTAssertNil(slots[2])

        _ = model(MLXArray(0 ..< 5).reshaped(1, 5), cache: cache)
        eval(cache[0].innerState())

        XCTAssertEqual(slots[0]?.shape, [1, 1, 128], "the time-mixing shift was not stored")
        XCTAssertEqual(slots[1]?.shape, [1, 2, 64, 64], "the recurrent state was not stored")
        XCTAssertEqual(
            slots[2]?.shape, [1, 1, 128], "the channel-mixing shift was not stored")
    }

    /// All three slots at once: two chunks through one cache must reproduce one forward.
    func testChunkedPrefillMatchesOnePass() throws {
        let config = try Self.config()
        let tokens = MLXArray(0 ..< 9).reshaped(1, 9)

        let whole = Self.nondegenerate(RWKV7Model(config))
        let onePass = whole(tokens, cache: try whole.newCache(configuration: nil))

        let split = RWKV7Model(config)
        split.update(
            parameters: ModuleParameters.unflattened(whole.parameters().flattened()))
        let cache = try split.newCache(configuration: nil)
        _ = split(tokens[0..., ..<6], cache: cache)
        let tail = split(tokens[0..., 6...], cache: cache)
        eval(onePass, tail)

        XCTAssertGreaterThan(
            abs(onePass).max().item(Float.self), 1e-3,
            "the logits are zero, so agreement means nothing")
        XCTAssertLessThan(
            abs(onePass[0, 8] - tail[0, 2]).max().item(Float.self), 2e-3,
            "chunking changed the answer, so one of the three cache slots is mishandled")
    }

    func testPerHeadNormUsesItsOwnEpsilon() throws {
        let model = RWKV7Model(try Self.config())
        XCTAssertEqual(model.configuration.normEps, 1e-5)
        XCTAssertEqual(model.model.layers[0].timeMixing.gNorm.eps, 64e-5)
        XCTAssertEqual(model.model.layers[0].attentionNorm.eps, 1e-5)
        XCTAssertEqual(model.model.layers[0].timeMixing.gNorm.weight.shape, [2, 64])
    }

    func testSanitizeReshapesFlatPerHeadParameters() throws {
        let model = RWKV7Model(try Self.config())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.attn.k_k": MLXArray.zeros([128]),
            "model.layers.0.attn.k_a": MLXArray.zeros([128]),
            "model.layers.0.attn.g_norm.weight": MLXArray.zeros([128]),
            "model.layers.0.attn.g_norm.bias": MLXArray.zeros([128]),
            "model.layers.0.attn.r_k": MLXArray.zeros([2, 64]),
            "model.embeddings.weight": MLXArray.zeros([48, 128]),
        ])

        XCTAssertEqual(sanitized["model.layers.0.attn.k_k"]?.shape, [2, 64])
        XCTAssertEqual(sanitized["model.layers.0.attn.g_norm.bias"]?.shape, [2, 64])
        XCTAssertEqual(sanitized["model.embeddings.weight"]?.shape, [48, 128])
    }

}
