// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import NativSDK
import XCTest

final class QwenHybridTests: XCTestCase {
    func testQwenHybridCacheHonorsCapacity() throws {
        let model = try makeTinyModel()
        let configuration = KVCacheConfiguration(
            capacity: try .init(maxTokens: 16, preservedPrefixTokens: 2))
        let cache = try model.newCache(configuration: configuration)
        XCTAssertTrue(cache[0] is MambaCache)
        XCTAssertTrue(cache[1] is RotatingKVCache)
        XCTAssertEqual(cache[1].maxSize, 16)

    }

    private func makeTinyModel() throws -> Qwen35 {
        let json = """
            {
                "model_type": "qwen3_5_vl",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "vocab_size": 512,
                    "full_attention_interval": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32,
                    "linear_value_head_dim": 32,
                    "linear_conv_kernel_dim": 4,
                    "max_position_embeddings": 4096,
                    "rope_parameters": {
                        "type": "default",
                        "mrope_section": [8, 4, 4],
                        "rope_theta": 100000.0,
                        "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // parallel tests must not share (or perturb) the global random stream.
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen35(config) }
    }
}
