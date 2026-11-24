import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// A dense decoder with several mechanisms particular to it: learned scalars folded into
/// the projections, value embeddings gated into the attention values, a per-head gate on
/// the attention output, an optional subtraction along each head's value direction, depth
/// taps combined by a learned mixer, and a bounded sigmoid head.
final class LimiteTests: XCTestCase {
    private func configuration(muddMLP: Bool = true) throws -> LimiteConfiguration {
        try JSONDecoder().decode(
            LimiteConfiguration.self,
            from: ModelTestConfigurations.configuration("limite", overrides: ["mudd_mlp": muddMLP]))
    }

    /// Per-layer flags decide which optional parameters exist at all.
    func testPerLayerParametersFollowTheLayerLists() throws {
        let model = LimiteModel(try configuration())
        let keys = Set(model.parameters().flattened().map(\.0))
        // ve_layers [0, 1]
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.ve_gate"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.ve_gate"))
        XCTAssertFalse(keys.contains("model.layers.2.self_attn.ve_gate"))
        // attn_gate_channels is non-zero, so every layer has the output gate
        for layer in 0 ..< 4 {
            XCTAssertTrue(keys.contains("model.layers.\(layer).self_attn.attn_gate"))
        }
        // the four learned residual blends, and the folded projection scalars
        XCTAssertTrue(keys.contains("model.layers.0.resid_lambda_attn"))
        XCTAssertTrue(keys.contains("model.layers.0.post_lambda_mlp"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.qkv_scale"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.o_scale"))
    }

    /// The learned scalars are rank 0. A shape-(1,) scalar would promote the weight fold
    /// to float32 and diverge from the reference.
    func testFoldedScalarsAreRankZero() throws {
        let model = LimiteModel(try configuration())
        let published = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(published["model.layers.0.self_attn.qkv_scale"]?.shape, [])
        XCTAssertEqual(published["model.layers.0.resid_lambda_attn"]?.shape, [])
    }

    /// The fold rounds the scalar to the weight's dtype before multiplying, which is what
    /// the reference does by promoting a 0-dim tensor as a wrapped scalar.
    func testFoldRoundsTheScalarToTheWeightDtype() {
        let weight = MLXArray.ones([2, 2]).asType(.bfloat16)
        let scale = MLXArray(Float(1.0 / 3.0))
        let folded = limiteFold(scale, weight)
        XCTAssertEqual(folded.dtype, .bfloat16)
        XCTAssertEqual(
            folded[0, 0].item(Float.self), scale.asType(.bfloat16).item(Float.self),
            accuracy: 0)
    }

    /// The mudd mixer's tables are sized by layer count, taps and the inner width, and the
    /// r-way pair exists only when mudd_mlp is set.
    func testMixerTablesAndRWay() throws {
        let model = LimiteModel(try configuration())
        let published = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(published["model.mudd.dense1"]?.shape, [8, 32])
        XCTAssertEqual(published["model.mudd.dense2"]?.shape, [4, 3, 8])
        XCTAssertEqual(published["model.mudd.bias"]?.shape, [4, 3])
        XCTAssertEqual(published["model.mudd.dense2_mlp"]?.shape, [4, 3, 8])

        let withoutRWay = LimiteModel(try configuration(muddMLP: false))
        XCTAssertFalse(
            Set(withoutRWay.parameters().flattened().map(\.0)).contains("model.mudd.dense2_mlp"))
    }

    /// A sliding window keeps `window + 1` keys inclusive of the query; a global layer
    /// declaring -1 keeps everything causal.
    func testCausalWindowMaskBounds() {
        let windowed = limiteCausalWindowMask(
            queryLength: 1, offset: 5, window: 2, dtype: .float32)
        let row: [Float] = windowed[0].asArray(Float.self)
        XCTAssertEqual(row.count, 6)
        XCTAssertEqual(row[5], 0, "the query itself is kept")
        XCTAssertEqual(row[3], 0, "q - window is kept")
        XCTAssertTrue(row[2] < 0, "beyond the window is masked")

        let unbounded = limiteCausalWindowMask(
            queryLength: 2, offset: 0, window: nil, dtype: .float32)
        XCTAssertEqual(unbounded[1].asArray(Float.self), [0, 0])
        XCTAssertTrue(unbounded[0].asArray(Float.self)[1] < 0, "still causal")
    }

    /// The head is bounded in (0, a), not an unbounded logit.
    func testHeadIsABoundedSigmoid() throws {
        let model = LimiteModel(try configuration())
        let logits = model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        let values: [Float] = logits.reshaped(-1).asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0 > 0 && $0 < 30 }, "bounded by softcap a = 30")
    }

    /// The backbone is written at bare `model.`; a wrapper-prefixed key is accepted too.
    func testWrapperPrefixIsStripped() throws {
        let model = LimiteModel(try configuration())
        let out = model.sanitize(weights: [
            "language_model.model.mudd.bias": MLXArray.zeros([4, 3]),
            "model.layers.0.self_attn.qkv_scale": MLXArray(Float(0)),
        ])
        XCTAssertNotNil(out["model.mudd.bias"])
        XCTAssertNotNil(out["model.layers.0.self_attn.qkv_scale"])
        XCTAssertNil(out["language_model.model.mudd.bias"])
    }
}
