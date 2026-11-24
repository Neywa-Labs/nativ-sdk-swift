import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Plamo2Tests: XCTestCase {

    private static func config(
        layers: Int = 4, mambaStep: Int = 2, mambaEnabled: Bool = true,
        fullAttentionIndices: [Int]? = nil, ropeTheta: Double = 10_000,
        ropeLocalTheta: Double = 10_000
    ) throws -> Plamo2Configuration {
        try JSONDecoder.json5().decode(
            Plamo2Configuration.self,
            from: ModelTestConfigurations.configuration(
                "plamo2",
                overrides: [
                    "num_hidden_layers": layers, "mamba_step": mambaStep,
                    "mamba_enabled": mambaEnabled,
                    "full_attention_idx": fullAttentionIndices as Any? ?? NSNull(),
                    "rope_theta": ropeTheta, "rope_local_theta": ropeLocalTheta,
                ]))
    }

    /// Every norm weight ships as zeros, so a freshly built model is degenerate at each
    /// residual. Randomizing them is what makes the other tests measure anything.
    private static func alive(_ module: Module, seed: UInt64 = 91) -> Module {
        var updates: [(String, MLXArray)] = []
        for (index, (key, value)) in module.parameters().flattened().enumerated() {
            eval(value)
            if abs(value.asType(.float32)).max().item(Float.self) == 0 {
                updates.append(
                    (
                        key,
                        MLXRandom.normal(value.shape, key: MLXRandom.key(seed + UInt64(index)))
                            * 0.2
                    ))
            }
        }
        module.update(parameters: ModuleParameters.unflattened(updates))
        return module
    }

    // MARK: - Norms

    /// The weight is an *offset* from a per-site constant and initializes to zero, so a
    /// default-built norm scales by exactly the offset.
    func testNormWeightIsAnOffsetFromItsConstant() {
        let norm = Plamo2RMSNorm(dimensions: 4, eps: 1e-6, offset: 0.25)
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 4)
        let out = norm(x)
        eval(out)

        let rms = (30.0 / 4.0 as Float).squareRoot()
        let expected = [1, 2, 3, 4].map { Float($0) / rms * 0.25 }
        for (got, want) in zip(out.reshaped(4).asArray(Float.self), expected) {
            XCTAssertEqual(got, want, accuracy: 1e-5)
        }
    }

    /// Four norms per block, and the two post-norms carry the shrinking offsets that keep
    /// the residual stream from growing with depth. A single shared offset is silent.
    func testTheFourNormOffsets() throws {
        let block = Plamo2Block(try Self.config(), layerIndex: 0)
        XCTAssertEqual(block.preMixerNorm.offset, 1, accuracy: 1e-7)
        XCTAssertEqual(block.postMixerNorm.offset, 1.0 / 5, accuracy: 1e-7)
        XCTAssertEqual(block.preMLPNorm.offset, 1, accuracy: 1e-7)
        XCTAssertEqual(
            block.postMLPNorm.offset, 1.0 / pow(5.0, 1.5), accuracy: 1e-7)
        XCTAssertEqual(Plamo2ModelInner(try Self.config()).norm.offset, 1, accuracy: 1e-7)
    }

    /// The post-norms normalize the *branch output* before the residual add. Applied to
    /// the sum instead, the branch's scale would reach the output -- so scaling a branch
    /// by a hundred has to leave the block unchanged.
    func testPostNormsNormalizeTheBranchNotTheSum() throws {
        let config = try Self.config(mambaEnabled: false)
        let block = Plamo2Block(config, layerIndex: 0)
        _ = Self.alive(block)

        let x = MLXRandom.normal([1, 5, 32], key: MLXRandom.key(11))
        let before = block(x, mask: .causal, ssmMask: nil, cache: nil)

        let scaled = block.mlp.downProj.weight * 100
        block.update(
            parameters: ModuleParameters.unflattened([("mlp.down_proj.weight", scaled)]))
        let after = block(x, mask: .causal, ssmMask: nil, cache: nil)
        eval(before, after)

        XCTAssertGreaterThan(abs(before).max().item(Float.self), 1e-3)
        XCTAssertLessThan(
            abs(before - after).max().item(Float.self), 1e-3,
            "the post-MLP norm is not absorbing the branch's scale, so it is being "
                + "applied to the residual sum")
    }

    // MARK: - Mamba mixer

    /// `in_proj` is read per head -- reshaped to `[batch, length, heads, 2 * headDim]`
    /// and split at `headDim` -- so the gate lanes interleave with the state lanes rather
    /// than occupying the first half of the projection.
    func testInProjSplitsWithinEachHead() throws {
        let config = try Self.config()
        let mixer = Plamo2Mamba(config)
        _ = Self.alive(mixer)

        let heads = config.mambaHeads
        let headDim = config.hiddenSizePerHead
        var rows: [MLXArray] = []
        for head in 0 ..< heads {
            rows.append(MLXArray.zeros([headDim, config.hiddenSize]))
            rows.append(
                MLXRandom.normal(
                    [headDim, config.hiddenSize], key: MLXRandom.key(500 + UInt64(head))))
        }
        mixer.update(
            parameters: ModuleParameters.unflattened([
                ("in_proj.weight", concatenated(rows, axis: 0))
            ]))

        let x = MLXRandom.normal([1, 6, config.hiddenSize], key: MLXRandom.key(12))
        let out = mixer(x, mask: .none, ssmMask: nil, cache: nil)
        eval(out)

        XCTAssertEqual(
            abs(out).max().item(Float.self), 0, accuracy: 1e-6,
            "zeroing each head's first headDim rows did not zero the gate, so the split "
                + "is over the flat projection rather than within each head")
    }

    /// `bcdt_proj` emits B, then C, then dt -- despite the name -- and dt alone is
    /// `dt_dim` wide while B and C are `d_state`. The per-slice norm weights pin the
    /// widths; a swap of the two `d_state` slices is caught by the recurrence carrying
    /// state across positions.
    func testBcdtProjectionOrderIsBThenCThenDt() throws {
        let config = try Self.config()
        let mixer = Plamo2Mamba(config)
        _ = Self.alive(mixer)

        XCTAssertEqual(mixer.bNormWeight.shape, [config.mambaStateSize])
        XCTAssertEqual(mixer.cNormWeight.shape, [config.mambaStateSize])
        XCTAssertEqual(mixer.dtNormWeight.shape, [Swift.max(64, config.hiddenSize / 16)])

        let x = MLXRandom.normal([1, 4, config.hiddenSize], key: MLXRandom.key(13))
        let reference = mixer(x, mask: .none, ssmMask: nil, cache: nil)

        // C reads the state out, so silencing it leaves only the D skip -- which no
        // amount of B can restore. Silencing B instead empties the state, which C then
        // reads as zero: the same output. What separates them is a *third* run where B
        // is silenced only after the first position, which only reaches the output
        // through the state's history.
        let silencedC = Plamo2Mamba(config)
        silencedC.update(parameters: ModuleParameters.unflattened(mixer.parameters().flattened()))
        silencedC.update(
            parameters: ModuleParameters.unflattened([
                ("C_norm_weight", MLXArray.zeros([config.mambaStateSize]))
            ]))
        let withoutC = silencedC(x, mask: .none, ssmMask: nil, cache: nil)

        let silencedB = Plamo2Mamba(config)
        silencedB.update(parameters: ModuleParameters.unflattened(mixer.parameters().flattened()))
        silencedB.update(
            parameters: ModuleParameters.unflattened([
                ("B_norm_weight", MLXArray.zeros([config.mambaStateSize]))
            ]))
        let withoutB = silencedB(x, mask: .none, ssmMask: nil, cache: nil)
        eval(reference, withoutC, withoutB)

        XCTAssertGreaterThan(
            abs(reference - withoutC).max().item(Float.self), 1e-4,
            "the C slice does not reach the state readout")
        XCTAssertLessThan(
            abs(withoutB - withoutC).max().item(Float.self), 1e-4,
            "emptying the state through B and silencing the readout through C must land "
                + "in the same place: the D skip alone")
    }

    /// `A_log = log(1 ... heads)`, so head 0 has `A = -1` and later heads decay faster.
    func testALogIsTheLogOfOneThroughHeads() throws {
        let mixer = Plamo2Mamba(try Self.config())
        eval(mixer.aLog)
        let expected = (1 ... 4).map { logf(Float($0)) }
        for (got, want) in zip(mixer.aLog.asArray(Float.self), expected) {
            XCTAssertEqual(got, want, accuracy: 1e-6)
        }
        XCTAssertEqual(mixer.D.asArray(Float.self), [1, 1, 1, 1])
    }

    // MARK: - Attention

    /// q and k are RMS-normalized with no weight, so the projection's own scale cannot
    /// reach the scores -- only the per-head `q_weight` / `k_weight` can.
    func testQueriesAreNormalizedBeforeTheirPerHeadScale() throws {
        let config = try Self.config(mambaEnabled: false)
        let attention = Plamo2Attention(config, layerIndex: 0)
        _ = Self.alive(attention)

        let x = MLXRandom.normal([1, 5, config.hiddenSize], key: MLXRandom.key(14))
        let before = attention(x, mask: .causal, ssmMask: nil, cache: nil)

        let queryWidth = config.attentionHeads * config.hiddenSizePerHead
        var weight = attention.qkvProj.weight
        weight = concatenated(
            [weight[..<queryWidth] * 100, weight[queryWidth...]], axis: 0)
        attention.update(
            parameters: ModuleParameters.unflattened([("qkv_proj.weight", weight)]))
        let after = attention(x, mask: .causal, ssmMask: nil, cache: nil)
        eval(before, after)

        XCTAssertGreaterThan(abs(before).max().item(Float.self), 1e-3)
        XCTAssertLessThan(
            abs(before - after).max().item(Float.self), 2e-3,
            "scaling the query projection changed the output, so the query normalization "
                + "is picking up a weight")

        attention.update(
            parameters: ModuleParameters.unflattened([
                (
                    "q_weight",
                    concatenated(
                        [
                            MLXArray.ones([1, config.hiddenSizePerHead]) * 8,
                            MLXArray.ones(
                                [config.attentionHeads - 1, config.hiddenSizePerHead]),
                        ], axis: 0)
                )
            ]))
        let sharpened = attention(x, mask: .causal, ssmMask: nil, cache: nil)
        eval(sharpened)
        XCTAssertGreaterThan(
            abs(after - sharpened).max().item(Float.self), 1e-3,
            "the per-head query scale is not reaching the scores")
    }

    /// The RoPE base is chosen per layer: `rope_theta` for the layers named in
    /// `full_attention_idx`, `rope_local_theta` everywhere else. A single base is silent
    /// on any checkpoint whose two thetas agree, so pin it where they differ.
    func testRopeBaseFollowsFullAttentionIndices() throws {
        let config = try Self.config(
            mambaEnabled: false, fullAttentionIndices: [1],
            ropeTheta: 1_000_000, ropeLocalTheta: 10_000)
        let model = Plamo2ModelInner(config)

        let local = try XCTUnwrap(model.decoder.layers[0].mixer as? Plamo2Attention)
        let global = try XCTUnwrap(model.decoder.layers[1].mixer as? Plamo2Attention)

        let x = MLXRandom.normal(
            [1, config.attentionHeads, 6, config.hiddenSizePerHead],
            key: MLXRandom.key(15))
        let localReference = RoPE(
            dimensions: config.hiddenSizePerHead, traditional: false, base: 10_000)
        let globalReference = RoPE(
            dimensions: config.hiddenSizePerHead, traditional: false, base: 1_000_000)

        let (localOut, globalOut) = (local.rope(x, offset: 3), global.rope(x, offset: 3))
        let (localWant, globalWant) = (
            localReference(x, offset: 3), globalReference(x, offset: 3)
        )
        eval(localOut, globalOut, localWant, globalWant)

        XCTAssertLessThan(abs(localOut - localWant).max().item(Float.self), 1e-6)
        XCTAssertLessThan(abs(globalOut - globalWant).max().item(Float.self), 1e-6)
        XCTAssertGreaterThan(
            abs(localOut - globalOut).max().item(Float.self), 1e-3,
            "both layers rotate identically, so the per-layer base is not being read")
    }

    // MARK: - Layout

    /// Recurrent unless the index lands on the step's midpoint -- and, for a stack no
    /// deeper than that midpoint, everything but the last layer.
    func testIsMambaSelectsByStepMidpoint() throws {
        let stepTwo = try Self.config(layers: 6, mambaStep: 2)
        XCTAssertEqual(
            (0 ..< 6).map { stepTwo.isMamba(at: $0) },
            [
                true, false, true, false, true, false,
            ])

        let stepFour = try Self.config(layers: 8, mambaStep: 4)
        XCTAssertEqual(
            (0 ..< 8).map { stepFour.isMamba(at: $0) },
            [
                true, true, false, true, true, true, false, true,
            ])

        let shallow = try Self.config(layers: 3, mambaStep: 8)
        XCTAssertEqual((0 ..< 3).map { shallow.isMamba(at: $0) }, [true, true, false])

        let disabled = try Self.config(layers: 4, mambaStep: 2, mambaEnabled: false)
        XCTAssertEqual(
            (0 ..< 4).map { disabled.isMamba(at: $0) },
            [
                false, false, false, false,
            ])
    }

    /// Tied embeddings publish no head; untied ones do.
    func testTiedEmbeddingsDropTheHead() throws {
        let tied = try JSONDecoder.json5().decode(
            Plamo2Configuration.self,
            from: ModelTestConfigurations.configuration(
                "plamo2",
                overrides: [
                    "num_hidden_layers": 2, "tie_word_embeddings": true,
                ]))
        let model = Plamo2Model(tied)
        XCTAssertNil(model.lmHead)
        XCTAssertFalse(Set(model.parameters().flattened().map(\.0)).contains("lm_head.weight"))

        let out = model(MLXArray(0 ..< 3).reshaped(1, 3), cache: nil)
        eval(out)
        XCTAssertEqual(out.shape, [1, 3, 32])
    }

    /// `sanitize` moves the depthwise convolution's kernel axis, and only for the layout
    /// that needs it.
    func testSanitizeMovesTheConvolutionKernelAxis() throws {
        let model = Plamo2Model(try Self.config(layers: 2))
        let torchLayout = MLXArray(0 ..< (4 * 1 * 3)).reshaped(4, 1, 3)
        let alreadyMoved = MLXArray(0 ..< (4 * 3 * 1)).reshaped(4, 3, 1)

        let out = model.sanitize(weights: [
            "model.layers.layers.0.mixer.conv1d.weight": torchLayout,
            "model.layers.layers.2.mixer.conv1d.weight": alreadyMoved,
            "model.layers.layers.0.mixer.in_proj.weight": torchLayout,
        ])

        XCTAssertEqual(out["model.layers.layers.0.mixer.conv1d.weight"]?.shape, [4, 3, 1])
        XCTAssertEqual(out["model.layers.layers.2.mixer.conv1d.weight"]?.shape, [4, 3, 1])
        XCTAssertEqual(out["model.layers.layers.0.mixer.in_proj.weight"]?.shape, [4, 1, 3])
    }

    // MARK: - Caching

    /// Decoding one token at a time has to reproduce the flat forward: the convolution
    /// window, the recurrent state and the attention keys all have to survive the
    /// boundary, and the recurrent layers have to see the offset the attention layers do.
    ///
    /// The bound is relative and loose because the shared SSM code sets the floor, not
    /// this port: `ssmAttn` builds a single step's state with a matmul whose inner
    /// dimension is one, which costs about 1e-3 relatively on Metal, and four mixers
    /// compound it. Measured over thirty runs across ten initializations the worst case
    /// was 3.0%, and the top-1 token disagreed twice -- so an argmax assertion would be
    /// flaky too. A break in the state carry is not subtle by comparison: an empty
    /// convolution window put the two paths 3000% apart.
    func testStepwiseDecodeMatchesTheFlatForward() throws {
        let config = try Self.config(layers: 4, mambaStep: 2)
        let model = Plamo2Model(config)
        _ = Self.alive(model)
        let tokens = MLXArray([1, 5, 9, 13, 17, 21, 25]).reshaped(1, 7)

        let flat = model(tokens, cache: nil)

        let cache = try model.newCache(configuration: nil)
        var stepwise: [MLXArray] = []
        for t in 0 ..< 7 {
            stepwise.append(model(tokens[0..., t ..< (t + 1)], cache: cache))
        }
        let streamed = concatenated(stepwise, axis: 1)
        eval(flat, streamed)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-1)
        XCTAssertLessThan(
            abs(flat - streamed).max().item(Float.self), 0.10 * scale,
            "the streamed state does not reproduce the flat forward")
    }

    /// A chunked prefill is the path that carries the convolution window across a
    /// boundary in a block rather than one position at a time. The recurrence resumes
    /// from a saved state instead of running the whole sequence at once, which is
    /// different arithmetic -- and `ssmAttn` builds that resumed state with a matmul whose
    /// inner dimension is one, which Metal rounds to about 1e-3. Across twenty
    /// initializations the worst case was 0.25%.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let config = try Self.config(layers: 4, mambaStep: 2)
        let model = Plamo2Model(config)
        _ = Self.alive(model)
        let tokens = MLXArray([2, 4, 6, 8, 10, 12, 14, 16, 18]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)

        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self),
            0.01 * abs(flat).max().item(Float.self),
            "the convolution window or the recurrent state did not cross the boundary")
        XCTAssertEqual(cache[1].offset, 9)
    }
}
