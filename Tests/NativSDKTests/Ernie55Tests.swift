import Foundation
import MLX
import XCTest

@testable import NativSDK

final class Ernie55Tests: XCTestCase {
    func testPartialCompressionWindowKeepsDistinctTokensAndSurvivesCacheCopy() {
        let original = DeepSeekV4PoolingCache(ratio: 5)
        func append(
            _ tokens: [Float], at offset: Int, to cache: DeepSeekV4PoolingCache
        ) -> (MLXArray, MLXArray, Int) {
            let value = MLXArray(tokens, [1, tokens.count, 1])
            return cache.accumulateWindows(kv: value, gate: value * 10, offset: offset)
        }
        XCTAssertEqual(append([1], at: 0, to: original).0.dim(1), 0)
        XCTAssertEqual(append([2, 3], at: 1, to: original).0.dim(1), 0)
        let restored = original.copy() as! DeepSeekV4PoolingCache
        let (kv, gate, position) = append([4, 5], at: 3, to: restored)
        XCTAssertEqual(kv.asArray(Float.self), [1, 2, 3, 4, 5])
        XCTAssertEqual(gate.asArray(Float.self), [10, 20, 30, 40, 50])
        XCTAssertEqual(position, 0)
        XCTAssertEqual(append([8, 9], at: 3, to: original).0.asArray(Float.self), [1, 2, 3, 8, 9])
    }

    func testSourceBlockFP8PreservesBytesAndExpandsScalesAcrossPartialBlocks() throws {
        let config = try JSONDecoder().decode(Ernie55Configuration.self, from: Data("{}".utf8))
        let path = "model.layers.0.mlp.up_proj"
        let raw = full([129, 160], values: MLXArray(UInt8(0x38)), dtype: .uint8)
        let scale = MLXArray([Float(0.5), 2, 0.125, 4], [2, 2])
        var weights = [path + ".weight": raw, path + ".weight_scale_inv": scale]
        XCTAssertTrue(try Ernie55Checkpoint.prepareSource(&weights, config: config))
        let packed = try XCTUnwrap(weights[path + ".weight"])
        let expanded = try XCTUnwrap(weights[path + ".scales"])
        XCTAssertNil(weights[path + ".weight_scale_inv"])
        XCTAssertEqual(packed.dtype, .uint32)
        XCTAssertTrue(all(packed.view(dtype: .uint8) .== raw).item(Bool.self))
        XCTAssertEqual(expanded.shape, [129, 5])
        XCTAssertEqual(expanded[0].asArray(UInt8.self), [126, 126, 126, 126, 128])
        XCTAssertEqual(expanded[128].asArray(UInt8.self), [124, 124, 124, 124, 129])
        let decoded = dequantized(
            packed, scales: expanded, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8,
            dtype: .bfloat16)
        XCTAssertTrue(all(decoded .== MiMoV2.dequantizeFP8Block(raw, scale: scale)).item(Bool.self))
    }

    func testHybridLayersMatchCachedDecodingPastCompressionAndSparseThresholds() throws {
        let config = try JSONDecoder().decode(
            Ernie55Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32, "vocab_size": 96, "intermediate_size": 32,
                  "num_hidden_layers": 2, "compress_ratios": [3, -2, -2],
                  "num_attention_heads": 2, "head_dim": 32, "q_lora_rank": 32,
                  "qk_rope_head_dim": 8, "o_groups": 2, "o_lora_rank": 16, "sliding_window": 4,
                  "hybrid_mla_num_attention_heads": 2, "hybrid_mla_q_lora_rank": 32,
                  "hybrid_mla_kv_lora_rank": 32, "hybrid_mla_qk_nope_head_dim": 24,
                  "hybrid_mla_qk_rope_head_dim": 8, "hybrid_mla_v_head_dim": 32,
                  "index_n_heads": 2, "index_head_dim": 16, "index_topk": 3,
                  "hc_mult": 2, "hc_sinkhorn_iters": 3, "vha_rank": 2,
                  "moe_intermediate_size": 32, "moe_latent_size": 32, "n_routed_experts": 4,
                  "num_experts_per_tok": 2
                }
                """.utf8))
        XCTAssertEqual(config.compression, [3, -2])
        let model = withRandomState(MLXRandom.RandomState(seed: 529)) { Ernie55Model(config) }
        let tokens = MLXArray([4, 9, 11, 82, 13, 3, 53, 23, 8, 27, 41, 15], [1, 12])
        let full = model(tokens, cache: nil)
        let caches = try model.newCache(configuration: nil)
        let ranges = [0 ..< 1, 1 ..< 3, 3 ..< 4, 4 ..< 9, 9 ..< 10, 10 ..< 12]
        var chunks: [MLXArray] = []
        for range in ranges {
            chunks.append(model(tokens[0..., range], cache: caches))
            eval(caches, chunks.last!)
        }
        let chunked = concatenated(chunks, axis: 1)
        XCTAssertEqual(chunked.shape, [1, 12, 96])
        XCTAssertTrue(all(isFinite(chunked)).item(Bool.self))
        XCTAssertLessThan((full - chunked).abs().max().item(Float.self), 2e-2)
        let hca = try XCTUnwrap(caches[0] as? CacheList)
        let mla = try XCTUnwrap(caches[1] as? CacheList)
        XCTAssertEqual(hca[0].offset, 12)
        XCTAssertEqual(hca[1].offset, 4)
        XCTAssertEqual(mla[0].offset, 12)
        XCTAssertEqual(mla[1].offset, 12)
    }
}
