import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// The whole text stack against upstream: embed, expand to the hc copies, three blocks
/// of hyper-connected attention and MoE, collapse, project. Expectations captured from
/// upstream mlx-vlm.
final class DeepSeekV41LanguageModelTests: XCTestCase {
    private func configuration(engramLayers: [Int] = []) throws -> DeepSeekV41Configuration {
        let tables = engramLayers.map { _ in "420" }.joined(separator: ", ")
        return try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32, "head_dim": 32, "qk_rope_head_dim": 8,
                  "q_lora_rank": 32, "o_lora_rank": 32, "o_groups": 2,
                  "num_attention_heads": 4, "num_hidden_layers": 3,
                  "compress_ratios": [0, 2, 2], "kv_source_layer_ids": [1],
                  "index_source_layer_ids": [1], "candidate_source_layer_id": 1,
                  "candidate_topk_blocks": 2, "candidate_block_size": 2,
                  "index_n_heads": 2, "index_head_dim": 32, "index_topk": 2,
                  "sliding_window": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                  "compress_rope_theta": 160000.0, "max_position_embeddings": 4096,
                  "vocab_size": 16, "moe_intermediate_size": 32,
                  "n_routed_experts": 4, "num_experts_per_tok": 2,
                  "hc_mult": 2, "hc_sinkhorn_iters": 3, "hc_eps": 1e-6,
                  "engram_layer_ids": [\(engramLayers.map(String.init).joined(separator: ", "))],
                  "engram_num_embeddings": [\(tables)],
                  "engram_max_ngram_size": 3, "engram_n_heads": 2,
                  "engram_vocab_size": 100, "engram_head_dim": 8,
                  "engram_pad_token_id": 2, "engram_compressed_vocab_size": 8,
                  "image_token_id": 9
                }
                """.utf8))
    }

    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    private func model(engramLayers: [Int] = []) throws -> DeepSeekV41Language.LanguageModel {
        let config = try configuration(engramLayers: engramLayers)
        let model = DeepSeekV41Language.LanguageModel(config)
        var updates: [(String, MLXArray)] = [
            ("embed_tokens.weight", ramp([16, 32], scale: 0.6)),
            ("norm.weight", ramp([32]) + 1),
            ("head.weight", ramp([16, 32], scale: 0.4)),
        ]
        for (index, block) in model.layers.enumerated() {
            let p = "layers.\(index)"
            updates += [
                ("\(p).attn.wq_a.weight", ramp([32, 32], scale: 0.5)),
                ("\(p).attn.q_norm.weight", ramp([32]) + 1),
                ("\(p).attn.wq_b.weight", ramp([128, 32], scale: -0.5 - 0.1 * Float(index))),
                ("\(p).attn.wkv.weight", ramp([32, 32], scale: 0.25)),
                ("\(p).attn.kv_norm.weight", ramp([32]) + 1),
                ("\(p).attn.wo_a.weight", ramp([2, 32, 64], scale: 0.3)),
                ("\(p).attn.wo_b.weight", ramp([32, 64], scale: 0.2)),
                ("\(p).attn.attn_sink", ramp([4], scale: 2)),
                ("\(p).attn_norm.weight", ramp([32]) + 1),
                ("\(p).ffn_norm.weight", ramp([32], scale: 0.5) + 1),
                ("\(p).ffn.gate.weight", ramp([4, 32], scale: 0.3)),
                ("\(p).ffn.gate.bias", MLXArray([Float(0), 0, 0, 0])),
                ("\(p).ffn.gate.bias_vl", MLXArray([Float(0), -10, 0, 10])),
                ("\(p).ffn.switch_mlp.gate_proj.weight", ramp([4, 32, 32], scale: 0.2)),
                ("\(p).ffn.switch_mlp.up_proj.weight", ramp([4, 32, 32], scale: -0.2)),
                ("\(p).ffn.switch_mlp.down_proj.weight", ramp([4, 32, 32], scale: 0.1)),
                ("\(p).ffn.shared_experts.gate_proj.weight", ramp([32, 32], scale: 0.2)),
                ("\(p).ffn.shared_experts.up_proj.weight", ramp([32, 32], scale: -0.2)),
                ("\(p).ffn.shared_experts.down_proj.weight", ramp([32, 32], scale: 0.1)),
                ("\(p).hc_attn_fn", ramp([8, 64], scale: 0.5)),
                ("\(p).hc_ffn_fn", ramp([8, 64], scale: -0.4)),
                ("\(p).hc_attn_base", ramp([8], scale: 0.3)),
                ("\(p).hc_ffn_base", ramp([8], scale: -0.2)),
                ("\(p).hc_attn_scale", MLXArray([Float(1), 0.75, 1.25])),
                ("\(p).hc_ffn_scale", MLXArray([Float(0.8), 1.1, 0.9])),
            ]
            if block.attention.compressor != nil {
                updates += [
                    ("\(p).attn.compressor.wkv.weight", ramp([32, 32], scale: 0.4)),
                    ("\(p).attn.compressor.norm.weight", ramp([32]) + 1),
                    ("\(p).attn.compressor.wgate.weight", ramp([32, 32], scale: -0.3)),
                ]
            }
            if let indexer = block.attention.indexer {
                updates += [
                    ("\(p).attn.indexer.wq_b.weight", ramp([64, 32], scale: 0.35)),
                    ("\(p).attn.indexer.weights_proj.weight", ramp([2, 32], scale: 0.45)),
                ]
                if indexer.ownsKeys {
                    updates += [
                        ("\(p).attn.indexer.wk.weight", ramp([32, 32], scale: 0.55)),
                        ("\(p).attn.indexer.k_norm.weight", ramp([32]) + 1),
                    ]
                }
            }
            if block.engram != nil {
                updates += [
                    (
                        "\(p).engram.embed.weight",
                        (MLXArray(0 ..< 420 * 8).asType(.float32) % 13 - 6)
                            .reshaped(420, 8) / 13
                    ),
                    (
                        "\(p).engram.wkv.weight",
                        (MLXArray(0 ..< 96 * 32).asType(.float32) % 5 - 2)
                            .reshaped(96, 32) / 8
                    ),
                    (
                        "\(p).engram.q_weight",
                        MLXArray.full([2, 32], values: MLXArray(Float(0.75)))
                    ),
                    (
                        "\(p).engram.k_weight",
                        MLXArray.full([2, 32], values: MLXArray(Float(1.25)))
                    ),
                ]
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        if !engramLayers.isEmpty {
            model.engramHash = try DeepSeekV41NgramHashState(
                layout: try XCTUnwrap(model.layout),
                tokenMap: [0, 1, 2, 3, 4, 5, 6, 7, 3, 1, 0, 1, 2, 3, 4, 5],
                padTokenId: 2, compressedVocabularySize: 8)
        }
        return model
    }

    /// Two of the eight ids are the image token, so the MoE gate's image bias is live.
    private var tokens: MLXArray {
        MLXArray([Int32(1), 4, 9, 9, 2, 7, 3, 5]).reshaped(1, 8)
    }

    func testPrefillAndDecodeMatchUpstream() throws {
        let model = try model()
        let cache = try XCTUnwrap(model.newCache().first as? DeepSeekV41Cache)
        let logits = model(tokens, cache: cache)

        XCTAssertEqual(logits.shape, [1, 8, 16])
        XCTAssertEqual(logits.dtype, .float32, "the head keeps the logits in FP32")
        assertClose(
            logits[0, 0, 0 ..< 6],
            [0.688_087, -0.785_473, 1.649_577, -1.473_91, 1.135_035, -2.560_706],
            label: "prefill first")
        assertClose(
            logits[0, 7, 0 ..< 6],
            [-2.807_595, 2.476_95, -2.514_142, 2.562_991, -2.916_153, 1.701_272],
            label: "prefill last")
        XCTAssertEqual(cache.offset, 8, "the model advances the cache by the chunk width")

        let step = model(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
        assertClose(
            step[0, 0, 0 ..< 6],
            [-1.787_305, 1.769_635, -1.848_762, 2.202_757, -1.334_559, -0.273_975],
            label: "decode")
        XCTAssertEqual(cache.offset, 9)
    }

    /// An Engram layer writes its n-gram lookup into the residual, so the logits move.
    func testEngramLayerChangesTheLogits() throws {
        let model = try model(engramLayers: [1])
        XCTAssertNotNil(model.layout)
        XCTAssertNotNil(model.layers[1].engram)
        XCTAssertNil(model.layers[0].engram)

        let cache = try XCTUnwrap(model.newCache().first as? DeepSeekV41Cache)
        let logits = model(tokens, cache: cache)
        assertClose(
            logits[0, 0, 0 ..< 6],
            [0.647_444, -0.721_565, 1.576_106, -1.563_57, 1.193_477, -2.519_64],
            label: "prefill first")
        assertClose(
            logits[0, 7, 0 ..< 6],
            [-2.890_241, 2.574_681, -2.571_726, 2.524_517, -2.930_295, 1.774_926],
            label: "prefill last")

        XCTAssertNotNil(cache.engram)
        let step = model(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
        assertClose(
            step[0, 0, 0 ..< 6],
            [-1.867_168, 1.833_042, -1.921_345, 2.253_963, -1.382_709, -0.220_175],
            label: "decode")
    }

    /// No engram layers means no layout, no hash state and no per-block module.
    func testEngramCanBeAbsent() throws {
        let model = try model()
        XCTAssertNil(model.layout)
        XCTAssertNil(model.engramHash)
        XCTAssertTrue(model.layers.allSatisfy { $0.engram == nil })
        XCTAssertFalse(
            model.parameters().flattened().contains { $0.0.contains("engram") })
    }

    /// The hash ids are bounded by the primes, not by the declared table size, and an
    /// out-of-bounds gather in MLX comes back as zeros rather than an error. A table
    /// that does not cover its hash space would run and contribute nothing, so it is
    /// refused at decode.
    func testUndersizedEngramTableIsRefused() throws {
        let layout = try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: [1], maxNgramSize: 3, heads: 2, headDim: 8,
                vocabularySize: 100, numEmbeddings: [420]))
        XCTAssertEqual(layout.requiredEmbeddings(layer: 0), 420)
        XCTAssertNoThrow(try layout.validate())

        let small = try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: [1], maxNgramSize: 3, heads: 2, headDim: 8,
                vocabularySize: 100, numEmbeddings: [419]))
        XCTAssertThrowsError(try small.validate())

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                DeepSeekV41Configuration.self,
                from: Data(
                    #"""
                    {
                      "engram_layer_ids": [1], "engram_num_embeddings": [64],
                      "engram_max_ngram_size": 3, "engram_n_heads": 2,
                      "engram_vocab_size": 100, "engram_head_dim": 8
                    }
                    """#.utf8)
            ).validateModelConfiguration())

        let released = try JSONDecoder().decode(
            DeepSeekV41Configuration.self, from: Data("{}".utf8))
        XCTAssertNoThrow(try released.validateModelConfiguration())
        let releasedLayout = try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: released.engramLayerIds,
                maxNgramSize: released.engramMaxNgramSize, heads: released.engramHeads,
                headDim: released.engramHeadDim,
                vocabularySize: released.engramVocabularySize,
                numEmbeddings: released.engramNumEmbeddings))
        XCTAssertEqual(
            (0 ..< 2).map { releasedLayout.requiredEmbeddings(layer: $0) },
            [384_006_168, 384_016_682])
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], label: String, accuracy: Float = 5e-5,
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
