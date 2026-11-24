import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Ernie45MoETests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "ernie4_5_moe")
    }

    private static func json(
        gate: String = "softmax", start: String = "1", end: String? = nil, interval: Int = 1
    ) -> String {
        let endLine = end.map { ",\n  \"moe_layer_end_index\": \($0)" } ?? ""
        return """
            {
              "model_type": "ernie4_5_moe",
              "hidden_size": 32,
              "intermediate_size": 64,
              "max_position_embeddings": 256,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "num_hidden_layers": 6,
              "rms_norm_eps": 0.000001,
              "vocab_size": 48,
              "rope_theta": 10000.0,
              "use_bias": false,
              "tie_word_embeddings": false,
              "moe_num_experts": 4,
              "moe_layer_start_index": \(start),
              "moe_intermediate_size": 16,
              "moe_k": 2,
              "moe_layer_interval": \(interval),
              "moe_num_shared_experts": 1,
              "moe_gate_act": "\(gate)"\(endLine)
            }
            """
    }

    private static func config(
        gate: String = "softmax", start: String = "1", end: String? = nil, interval: Int = 1
    ) throws -> Ernie45MoEConfiguration {
        try JSONDecoder.json5().decode(
            Ernie45MoEConfiguration.self,
            from: XCTUnwrap(
                json(gate: gate, start: start, end: end, interval: interval)
                    .data(using: .utf8)))
    }

    /// The gate activation is a config choice here, and anything this does not implement is
    /// refused rather than defaulted -- a fallback would route by whichever activation
    /// happened to be the default.
    func testGateActivationIsAChoiceAndUnknownOnesAreRefused() throws {
        XCTAssertEqual(try Self.config(gate: "softmax").moeGateActivation, "softmax")
        XCTAssertEqual(try Self.config(gate: "sigmoid").moeGateActivation, "sigmoid")
        XCTAssertThrowsError(try Self.config(gate: "relu"))

        let softmaxed = try XCTUnwrap(
            Ernie45MoEModel(try Self.config(gate: "softmax")).model.layers[1].mlp
                as? Ernie45MoEBlockSparse)
        let sigmoidal = try XCTUnwrap(
            Ernie45MoEModel(try Self.config(gate: "sigmoid")).model.layers[1].mlp
                as? Ernie45MoEBlockSparse)
        XCTAssertFalse(softmaxed.usesSigmoidGate)
        XCTAssertTrue(sigmoidal.usesSigmoidGate)

        // Same weights in both, so only the activation differs -- and it changes the answer.
        sigmoidal.update(
            parameters: ModuleParameters.unflattened(softmaxed.parameters().flattened()))
        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let a = softmaxed(x)
        let b = sigmoidal(x)
        eval(a, b)
        XCTAssertGreaterThan(abs(a - b).max().item(Float.self), 1e-5)
    }

    /// The mixture layers are a windowed range *and* a stride: at or past the start, at or
    /// before the end, and on the stride counted one-based.
    func testMixtureLayersAreAWindowedRangeWithAStride() throws {
        let windowed = try Self.config(start: "1", end: "3")
        XCTAssertFalse(windowed.isMixtureLayer(0))  // before the start
        XCTAssertTrue(windowed.isMixtureLayer(1))
        XCTAssertTrue(windowed.isMixtureLayer(3))
        XCTAssertFalse(windowed.isMixtureLayer(4))  // past the end
        XCTAssertFalse(windowed.isMixtureLayer(5))

        // The end defaults to the last layer.
        let open = try Self.config(start: "1")
        XCTAssertTrue(open.isMixtureLayer(5))

        // And the stride thins within the window, one-based.
        let strided = try Self.config(start: "0", interval: 2)
        XCTAssertFalse(strided.isMixtureLayer(0))
        XCTAssertTrue(strided.isMixtureLayer(1))
        XCTAssertFalse(strided.isMixtureLayer(2))
    }

    /// Each bound may arrive as a scalar or as a list; a list collapses to its extreme.
    func testWindowBoundsAcceptScalarsOrLists() throws {
        XCTAssertEqual(try Self.config(start: "2").moeLayerStart, 2)
        // A list of starts takes the minimum, and of ends the maximum.
        XCTAssertEqual(try Self.config(start: "[3, 1, 2]").moeLayerStart, 1)
        XCTAssertEqual(try Self.config(start: "0", end: "[2, 4, 3]").moeLayerEnd, 4)
    }

    /// The renormalization floor is not decoration: under a sigmoid gate the selected
    /// weights can sum to effectively nothing, where a softmax's selection cannot, and
    /// dividing by that would be a division by zero.
    func testRenormalisationHasAFloorForTheSigmoidGate() throws {
        let moe = try XCTUnwrap(
            Ernie45MoEModel(try Self.config(gate: "sigmoid")).model.layers[1].mlp
                as? Ernie45MoEBlockSparse)

        // A gate driven far negative: every sigmoid weight underflows toward zero.
        moe.gate.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray.ones([4, 32]) * -1e3)
            ]))
        let x = MLXArray.ones([1, 1, 32])
        let out = moe(x)
        eval(out)

        // Finite rather than NaN, which is what dividing by the unfloored sum would give.
        XCTAssertFalse(out.max().item(Float.self).isNaN)
        XCTAssertTrue(out.max().item(Float.self).isFinite)
    }

    func testSanitizerStacksOnlyTheMixtureLayers() throws {
        let config = try Self.config(start: "1", end: "3")
        let model = Ernie45MoEModel(config)
        var weights: [String: MLXArray] = ["model.embed_tokens.weight": MLXArray.zeros([48, 32])]
        for layer in 1 ... 3 {
            for expert in 0 ..< config.moeExperts {
                for projection in ["gate_proj", "up_proj"] {
                    weights["model.layers.\(layer).mlp.experts.\(expert).\(projection).weight"] =
                        MLXArray.zeros([16, 32]) + Float(expert)
                }
                weights["model.layers.\(layer).mlp.experts.\(expert).down_proj.weight"] =
                    MLXArray.zeros([32, 16])
            }
        }
        weights["model.layers.0.mlp.gate_proj.weight"] = MLXArray.zeros([64, 32])

        let sanitized = model.sanitize(weights: weights)
        let stacked = try XCTUnwrap(sanitized["model.layers.2.mlp.switch_mlp.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [config.moeExperts, 16, 32])
        XCTAssertEqual(stacked[2, 0, 0].item(Float.self), 2.0)
        // The dense layer outside the window keeps its own MLP.
        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight"])
    }

}
