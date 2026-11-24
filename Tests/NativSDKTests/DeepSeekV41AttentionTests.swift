import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's DeepseekV41Attention,
/// for one layer of each mode driven through a prefill and two decode steps.
final class DeepSeekV41AttentionTests: XCTestCase {
    private func configuration(indexTopK: Int = 2, ropeHeadDim: Int = 8) throws
        -> DeepSeekV41Configuration
    {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32, "head_dim": 32, "qk_rope_head_dim": \(ropeHeadDim),
                  "q_lora_rank": 32, "o_lora_rank": 32, "o_groups": 2,
                  "num_attention_heads": 4, "num_hidden_layers": 4,
                  "compress_ratios": [0, 2, 2, 2], "kv_source_layer_ids": [1],
                  "index_source_layer_ids": [1, 2], "candidate_source_layer_id": 1,
                  "candidate_topk_blocks": 2, "candidate_block_size": 2,
                  "index_n_heads": 2, "index_head_dim": 32, "index_topk": \(indexTopK),
                  "sliding_window": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                  "compress_rope_theta": 160000.0, "max_position_embeddings": 4096,
                  "vocab_size": 16, "moe_intermediate_size": 32,
                  "n_routed_experts": 4, "num_experts_per_tok": 2
                }
                """.utf8))
    }

    /// `(arange % 13 - 6) / 11 * scale`, the fill the reference run used.
    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    private func layers(
        indexTopK: Int = 2, indexQueryScale: Float = 0.35, ropeHeadDim: Int = 8
    ) throws -> [DeepSeekV41Language.Attention] {
        let config = try configuration(indexTopK: indexTopK, ropeHeadDim: ropeHeadDim)
        return (0 ..< 4).map { index in
            let attention = DeepSeekV41Language.Attention(config, layerIndex: index)
            var updates: [(String, MLXArray)] = [
                ("wq_a.weight", ramp([32, 32], scale: 0.5)),
                ("q_norm.weight", ramp([32]) + 1),
                ("wq_b.weight", ramp([128, 32], scale: -0.5 - 0.1 * Float(index))),
                ("wkv.weight", ramp([32, 32], scale: 0.25)),
                ("kv_norm.weight", ramp([32]) + 1),
                ("wo_a.weight", ramp([2, 32, 64], scale: 0.3)),
                ("wo_b.weight", ramp([32, 64], scale: 0.2)),
                ("attn_sink", ramp([4], scale: 2)),
            ]
            if attention.compressor != nil {
                updates += [
                    ("compressor.wkv.weight", ramp([32, 32], scale: 0.4)),
                    ("compressor.norm.weight", ramp([32]) + 1),
                    ("compressor.wgate.weight", ramp([32, 32], scale: -0.3)),
                ]
            }
            if let indexer = attention.indexer {
                updates += [
                    ("indexer.wq_b.weight", ramp([64, 32], scale: indexQueryScale)),
                    ("indexer.weights_proj.weight", ramp([2, 32], scale: 0.45)),
                ]
                if indexer.ownsKeys {
                    updates += [
                        ("indexer.wk.weight", ramp([32, 32], scale: 0.55)),
                        ("indexer.k_norm.weight", ramp([32]) + 1),
                    ]
                }
            }
            attention.update(parameters: ModuleParameters.unflattened(updates))
            return attention
        }
    }

    /// The mode follows from the layer's role, and decides which submodules exist.
    func testModesFollowFromLayerRole() throws {
        let layers = try layers()
        XCTAssertEqual(layers.map(\.mode), [.local, .full, .reindex, .reuse])
        XCTAssertNil(layers[0].compressor)
        XCTAssertNil(layers[0].indexer)
        XCTAssertNotNil(layers[1].compressor, "a kv source owns its compressor")
        XCTAssertNotNil(layers[1].indexer)
        XCTAssertTrue(try XCTUnwrap(layers[1].indexer).ownsKeys)
        XCTAssertNil(layers[2].compressor, "a reindex layer shares the KV")
        XCTAssertNotNil(layers[2].indexer, "but rescores for itself")
        XCTAssertFalse(try XCTUnwrap(layers[2].indexer).ownsKeys)
        XCTAssertTrue(try XCTUnwrap(layers[2].indexer).usesCandidates)
        XCTAssertNil(layers[3].indexer, "a reuse layer takes the shared selection")
    }

    func testPrefillMatchesUpstream() throws {
        let layers = try layers()
        let cache = DeepSeekV41Cache(layers: 4)
        let x = ramp([1, 10, 32])
        let outputs = layers.map { $0(x, cache: cache) }
        cache.advance(10)

        XCTAssertEqual(outputs.map(\.shape), Array(repeating: [1, 10, 32], count: 4))
        let firsts: [[Float]] = [
            [0.279_46, -0.279_013, 0.055_912, -0.421_277, 0.098_251],
            [0.273_64, -0.273_336, 0.043_562, -0.403_672, 0.097_824],
            [0.269_38, -0.261_907, 0.044_839, -0.379_318, 0.108_331],
            [0.263_186, -0.269_389, 0.021_303, -0.389_376, 0.119_143],
        ]
        let lasts: [[Float]] = [
            [-0.404_985, -0.290_418, -0.323_663, 0.056_741, 0.374_319],
            [-0.609_166, -0.541_831, -0.548_446, -0.038_037, 0.368_919],
            [-0.439_767, -0.496_21, -0.626_618, -0.225_226, 0.139_167],
            [-0.435_22, -0.468_2, -0.597_297, -0.179_823, 0.185_884],
        ]
        for (index, out) in outputs.enumerated() {
            assertClose(out[0, 0, 0 ..< 5], firsts[index], label: "layer \(index) first")
            assertClose(out[0, 9, 0 ..< 5], lasts[index], label: "layer \(index) last")
        }

        XCTAssertEqual(
            cache.topKIndices?.flattened().asArray(Int32.self),
            [-1, -1, 0, -1, 0, -1, 0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 2, 3, 1, 4])

        XCTAssertEqual(cache.compress[1]?.dim(1), 5)
        XCTAssertNil(cache.compress[0])
        XCTAssertNil(cache.compress[2])
        XCTAssertEqual(cache.keys[1]?.dim(1), 5)
    }

    /// With top-k of one the selection is an argmax, which is the most score-sensitive
    /// form it takes: a perturbation anywhere in the scoring path has to move it. At
    /// top-k of two this fixture's choices turn out to be robust to the query
    /// quantization and the head scaling, so neither would otherwise be covered.
    func testArgmaxSelectionMatchesUpstream() throws {
        let layers = try layers(indexTopK: 1, indexQueryScale: 0.8)
        let cache = DeepSeekV41Cache(layers: 4)
        _ = layers.map { $0(ramp([1, 10, 32]), cache: cache) }
        XCTAssertEqual(
            cache.topKIndices?.flattened().asArray(Int32.self),
            [-1, 0, 0, 1, 0, 2, 1, 1, 2, 1])
    }

    /// The rope covering the whole index head, which is where the query rotation has the
    /// most say in the selection.
    ///
    /// It still has none: dropping the query rotation entirely leaves this argmax, and
    /// every other fixture tried, unchanged — confirmed against upstream, which behaves
    /// the same way. The score landscape here is peaked enough that one position wins
    /// either way. The rotation itself is verified exactly in the helper tests, and the
    /// *positions* it is given are covered by the decode test, but its presence at this
    /// call site is not something these fixtures can see.
    func testArgmaxWithFullWidthRopeMatchesUpstream() throws {
        let layers = try layers(indexTopK: 1, indexQueryScale: 0.8, ropeHeadDim: 32)
        let cache = DeepSeekV41Cache(layers: 4)
        _ = layers.map { $0(ramp([1, 10, 32]), cache: cache) }
        XCTAssertEqual(
            cache.topKIndices?.flattened().asArray(Int32.self),
            [-1, 0, 0, 1, 0, 2, 2, 0, 2, 1])
    }

    /// The published index keys and pooled latents, which the selection is computed
    /// from. Asserting them covers the rotation, the strided positions and the two
    /// quantization round-trips directly — a scoring change that happens not to flip a
    /// choice is invisible in the attention output.
    func testPublishedKeysAndLatentsMatchUpstream() throws {
        let layers = try layers()
        let cache = DeepSeekV41Cache(layers: 4)
        _ = layers.map { $0(ramp([1, 10, 32]), cache: cache) }

        let keys = try XCTUnwrap(cache.keys[1])
        XCTAssertEqual(keys.shape, [1, 5, 32])
        assertClose(
            keys[0, 0, 0 ..< 6], [-0.5, -0.25, -0.5, 1, -0.5, 0.75], label: "key 0")
        assertClose(
            keys[0, 4, 0 ..< 6], [-0.75, 0.75, -0.75, 0.75, -0.5, 0.25], label: "key 4")
        assertClose(
            keys[0, 2, 24 ..< 32], [-1.5, 0.5, -0.25, -0.75, 0.5, -0.75, 1, -1],
            label: "key 2 tail")
        assertClose(
            keys[0, 4, 24 ..< 32], [-1, -2, -1, 0.25, -0.75, 0.75, -0.5, 0.25],
            label: "key 4 tail")

        let pooled = try XCTUnwrap(cache.compress[1])
        XCTAssertEqual(pooled.shape, [1, 5, 32])
        assertClose(
            pooled[0, 4, 0 ..< 6],
            [-0.421_875, -0.562_5, -0.281_25, -0.562_5, -0.421_875, -0.843_75],
            label: "latent 4")
        assertClose(
            pooled[0, 4, 24 ..< 32],
            [
                1.687_5, -1.125, -0.140_625, -0.562_5, -0.281_25, -0.562_5, -0.421_875,
                -0.843_75,
            ], label: "latent 4 tail")
        assertClose(
            pooled[0, 2, 24 ..< 32],
            [
                -0.937_5, 1.406_25, 0, -0.703_125, -0.468_75, -0.937_5, -0.937_5,
                -0.937_5,
            ], label: "latent 2 tail")
    }

    /// Two decode steps: the first does not complete a compression group, the second
    /// does, which is where the compressor's held remainder and the ring window meet.
    func testDecodeStepsMatchUpstream() throws {
        let layers = try layers()
        let cache = DeepSeekV41Cache(layers: 4)
        _ = layers.map { $0(ramp([1, 10, 32]), cache: cache) }
        cache.advance(10)

        let first = layers.map { $0(ramp([1, 1, 32], scale: -1), cache: cache) }
        cache.advance(1)
        let firstExpected: [[Float]] = [
            [-0.328_551, -0.328_018, 0.160_014, 0.142_543, 0.807_848],
            [-0.172_159, -0.069_744, 0.401_989, 0.340_057, 0.771_165],
            [-0.165_341, -0.066_761, 0.467_614, 0.385_227, 0.845_739],
            [-0.187_5, -0.073_295, 0.476_705, 0.384_091, 0.882_386],
        ]
        for (index, out) in first.enumerated() {
            assertClose(out[0, 0, 0 ..< 5], firstExpected[index], label: "step 1 layer \(index)")
        }

        let second = layers.map { $0(ramp([1, 1, 32], scale: 0.7), cache: cache) }
        cache.advance(1)
        let secondExpected: [[Float]] = [
            [0.549_893, 0.496_307, 0.337_464, 0.477_77, 0.218_75],
            [0.662_216, 0.575_071, 0.450_071, 0.541_122, 0.258_239],
            [0.721_183, 0.688_565, 0.575_16, 0.661_648, 0.326_19],
            [0.775_497, 0.774_219, 0.671_84, 0.753_658, 0.390_447],
        ]
        for (index, out) in second.enumerated() {
            assertClose(out[0, 0, 0 ..< 5], secondExpected[index], label: "step 2 layer \(index)")
        }

        XCTAssertEqual(cache.compress[1]?.dim(1), 6)
        XCTAssertEqual(cache.keys[1]?.dim(1), 6)
        XCTAssertEqual(cache.topKIndices?.asArray(Int32.self), [2, 3])
        XCTAssertEqual(
            cache.candidates?.flattened().asArray(Bool.self),
            [false, false, true, true, true, true])
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], label: String, accuracy: Float = 3e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, label, file: file, line: line)
        for (index, pair) in zip(values, expected).enumerated() {
            XCTAssertEqual(
                pair.0, pair.1, accuracy: accuracy, "\(label) index \(index)", file: file,
                line: line)
        }
    }
}
