import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// The multimodal wrapper: the image span layout, the weight transforms, and the text
/// path. Merge expectations captured from upstream mlx-vlm.
final class DeepSeekV41ModelTests: XCTestCase {
    private func configuration() throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {
                  "hidden_size": 16, "num_hidden_layers": 2,
                  "compress_ratios": [0, 1], "vocab_size": 8, "head_dim": 16,
                  "q_lora_rank": 16, "o_lora_rank": 16, "o_groups": 2,
                  "num_attention_heads": 2, "moe_intermediate_size": 16,
                  "n_routed_experts": 4, "num_experts_per_tok": 2,
                  "index_head_dim": 32, "engram_layer_ids": [],
                  "engram_num_embeddings": [], "hc_mult": 2,
                  "vision_config": {
                    "hidden_size": 8, "num_heads": 2, "intermediate_size": 8,
                    "patch_size": 2, "num_layers": 1, "rope_theta": 10000.0,
                    "downsample_ratio": 2
                  }
                }
                """#.utf8))
    }

    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    /// The vision half needs no particular width; the text half does, so the two tests
    /// use different fixtures.
    private func textConfiguration() throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {
                  "hidden_size": 32, "num_hidden_layers": 2,
                  "compress_ratios": [0, 1], "vocab_size": 8, "head_dim": 32,
                  "q_lora_rank": 32, "o_lora_rank": 32, "o_groups": 2,
                  "num_attention_heads": 2, "moe_intermediate_size": 32,
                  "n_routed_experts": 4, "num_experts_per_tok": 2,
                  "index_head_dim": 32, "engram_layer_ids": [],
                  "engram_num_embeddings": [], "hc_mult": 2, "sliding_window": 4,
                  "qk_rope_head_dim": 8
                }
                """#.utf8))
    }

    private func model() throws -> DeepSeekV41 {
        let model = DeepSeekV41(try configuration())
        model.update(
            parameters: ModuleParameters.unflattened([
                ("image_start", MLXArray.full([16], values: MLXArray(Float(0.25)))),
                ("image_end", MLXArray.full([16], values: MLXArray(Float(-0.25)))),
                ("image_newline", MLXArray.full([16], values: MLXArray(Float(0.5)))),
                ("vision.patch_embed.proj.weight", ramp([8, 12], scale: 0.4)),
                ("vision.patch_embed.proj.bias", ramp([8], scale: 0.1)),
                ("vision.blocks.0.norm1.weight", ramp([8]) + 1),
                ("vision.blocks.0.norm2.weight", ramp([8], scale: 0.5) + 1),
                ("vision.blocks.0.attn.wqkv.weight", ramp([24, 8], scale: 0.3)),
                ("vision.blocks.0.attn.wqkv.bias", ramp([24], scale: 0.05)),
                ("vision.blocks.0.attn.wo.weight", ramp([8, 8], scale: 0.25)),
                ("vision.blocks.0.attn.wo.bias", ramp([8], scale: 0.02)),
                ("vision.blocks.0.mlp.w1.weight", ramp([16, 8], scale: 0.2)),
                ("vision.blocks.0.mlp.w2.weight", ramp([8, 8], scale: 0.15)),
                ("vision.norm.weight", ramp([8]) + 1),
                ("aligner.w1.weight", ramp([16, 32], scale: 0.3)),
                ("aligner.w1.bias", ramp([16], scale: 0.05)),
                ("aligner.w2.weight", ramp([16, 16], scale: 0.2)),
                ("aligner.w2.bias", ramp([16], scale: 0.03)),
            ]))
        return model
    }

    /// The span runs start, then the aligner rows in reading order with a newline at the
    /// end of each row, then end. Delimiters take the learned embeddings and everything
    /// outside the span is untouched.
    func testImageMergeMatchesUpstream() throws {
        let span = DeepSeekV41ImageSpan(
            start: 1, patches: ramp([8, 3, 2, 2], scale: 0.8), height: 4, width: 2,
            types: [.start, .image, .newline, .image, .newline, .end])
        let h = ramp([1, 9, 16], scale: 0.9)
        let out = try model().mergeImageEmbeddings([[span]], into: h)

        XCTAssertEqual(out.shape, [1, 9, 16])
        assertClose(
            out[0, 0, 0 ..< 4],
            [-0.490_909, -0.409_091, -0.327_273, -0.245_455], label: "before the span")
        assertClose(out[0, 1, 0 ..< 4], [0.25, 0.25, 0.25, 0.25], label: "start marker")
        assertClose(
            out[0, 2, 0 ..< 4],
            [-0.000_267, 0.031_023, 0.003_946, -0.032_594], label: "first aligner row")
        assertClose(out[0, 3, 0 ..< 4], [0.5, 0.5, 0.5, 0.5], label: "newline")
        assertClose(
            out[0, 4, 0 ..< 4],
            [0.011_949, -0.035_607, -0.026_772, 0.001_573], label: "second aligner row")
        assertClose(out[0, 5, 0 ..< 4], [0.5, 0.5, 0.5, 0.5], label: "second newline")
        assertClose(out[0, 6, 0 ..< 4], [-0.25, -0.25, -0.25, -0.25], label: "end marker")
        assertClose(
            out[0, 8, 0 ..< 4],
            [0.409_091, 0.490_909, -0.490_909, -0.409_091], label: "after the span")
    }

    /// A decode step carries no image, so the merge is skipped on width one.
    func testDecodeStepSkipsTheMerge() throws {
        let model = try model()
        let span = DeepSeekV41ImageSpan(
            start: 0, patches: ramp([8, 3, 2, 2], scale: 0.8), height: 4, width: 2,
            types: [.image])
        let ids = MLXArray([Int32(3)]).reshaped(1, 1)
        let out = model.inputEmbeddings(inputIds: ids, images: [[span]])
        XCTAssertTrue(
            allClose(out, model.languageModel.embedTokens(ids), atol: 0).item(),
            "a one-token step must not re-merge the image")
    }

    func testImagePrepareMatchesExplicitSpansAcrossPrefillChunks() throws {
        let configuration = try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {"model_type":"deepseek_v41","hidden_size":32,"num_hidden_layers":2,
                 "compress_ratios":[0,1],"vocab_size":16,"head_dim":32,"q_lora_rank":32,
                 "o_lora_rank":32,"o_groups":2,"num_attention_heads":2,"moe_intermediate_size":32,
                 "n_routed_experts":4,"num_experts_per_tok":2,"index_head_dim":32,
                 "engram_layer_ids":[],"engram_num_embeddings":[],"hc_mult":2,"sliding_window":4,
                 "qk_rope_head_dim":8,"image_token_id":7,
                 "vision_config":{"hidden_size":8,"num_heads":2,"intermediate_size":8,
                  "patch_size":2,"num_layers":1,"rope_theta":10000,"downsample_ratio":2}}
                """#.utf8))
        let model = DeepSeekV41(configuration)
        let types = DeepSeekV41Processor.tokenTypes(height: 2, width: 1)
        let patches = ramp([8, 3, 2, 2])
        let tokens = MLXArray([1] + Array(repeating: 7, count: types.count) + [2, 3])[.newAxis]
        let span = DeepSeekV41ImageSpan(
            start: 1, patches: patches, height: 4, width: 2, types: types)
        let embeddings = model.inputEmbeddings(inputIds: tokens, images: [[span]])
        let expectedCache = try model.newCache(configuration: nil)
        let expected = model.languageModel(
            tokens, inputsEmbeds: embeddings, cache: expectedCache.first as? DeepSeekV41Cache)
        eval(expected)
        for step in [2, 32] {
            let cache = try model.newCache(configuration: nil)
            let input = LMInput(
                text: .init(tokens: tokens), image: .init(pixels: patches, frames: [THW(1, 4, 2)]))
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: cache, state: nil, prefill: .init(stepSize: step))
            else {
                XCTFail("Expected image prefill logits")
                return
            }
            XCTAssertLessThan(
                abs(output.logits[0, -1] - expected[0, -1]).max().item(Float.self), 1e-3)
        }
        let malformed = LMInput(
            text: .init(tokens: MLXArray([1, 7, 2])[.newAxis]),
            image: .init(pixels: patches, frames: [THW(1, 4, 2)]))
        XCTAssertThrowsError(
            try model.prepare(
                malformed, cache: model.newCache(configuration: nil), state: nil, prefill: .init()))
    }

    // MARK: Weights

    /// A released checkpoint keys the text stack at the top level; a converted one is
    /// already prefixed.
    func testWeightKeysAreRoutedToTheLanguageModel() throws {
        let out = try model().sanitize(weights: [
            "embed.weight": MLXArray.ones([8, 16]),
            "norm.weight": MLXArray.ones([16]),
            "head.weight": MLXArray.ones([8, 16]).asType(.bfloat16),
            "layers.0.attn_norm.weight": MLXArray.ones([16]),
            "model.foo.weight": MLXArray.ones([2]),
            "vision.norm.weight": MLXArray.ones([8]),
            "mtp.0.weight": MLXArray.ones([2]),
        ])
        XCTAssertNotNil(out["language_model.embed_tokens.weight"])
        XCTAssertNotNil(out["language_model.norm.weight"])
        XCTAssertNotNil(out["language_model.head.weight"])
        XCTAssertNotNil(out["language_model.layers.0.attn_norm.weight"])
        XCTAssertNotNil(out["language_model.model.foo.weight"])
        XCTAssertNotNil(out["vision.norm.weight"], "the tower keeps its own prefix")
        XCTAssertNil(out["mtp.0.weight"], "prediction layers are dropped")
        XCTAssertEqual(
            out["language_model.head.weight"]?.dtype, .float32,
            "the head is promoted once rather than per token")
    }

    /// The output projection is one weight per group, and a converter that stored it
    /// flat has to be reshaped back.
    func testOutputProjectionIsRegrouped() throws {
        let out = try model().sanitize(weights: [
            "language_model.layers.0.attn.wo_a.weight": MLXArray.ones([32, 16])
        ])
        XCTAssertEqual(
            out["language_model.layers.0.attn.wo_a.weight"]?.shape, [2, 16, 16])
    }

    /// Native FP8 bytes come with an E8M0 scale per 32-column block, and the scale rows
    /// may be one per 32 rows rather than one per row.
    func testSourceQuantizedWeightsAreUnpacked() throws {
        let bytes = MLXArray([UInt8](repeating: 0x38, count: 64 * 64)).reshaped(64, 64)
        let perRow = MLXArray([UInt8](repeating: 127, count: 64 * 2)).reshaped(64, 2)
        let (packed, scales, mode) = try DeepSeekV41.packSourceWeight(
            bytes, scales: perRow)
        XCTAssertEqual(mode, .mxfp8)
        XCTAssertEqual(packed.dtype, .uint32)
        XCTAssertEqual(packed.shape, [64, 16], "four bytes to a word")
        XCTAssertEqual(scales.shape, [64, 2])

        let blocked = MLXArray([UInt8](repeating: 127, count: 2 * 2)).reshaped(2, 2)
        XCTAssertEqual(
            try DeepSeekV41.packSourceWeight(bytes, scales: blocked).scales.shape,
            [64, 2])

        let fp4 = MLXArray([UInt8](repeating: 127, count: 64 * 4)).reshaped(64, 4)
        XCTAssertEqual(
            try DeepSeekV41.packSourceWeight(bytes, scales: fp4).mode, .mxfp4)

        XCTAssertThrowsError(
            try DeepSeekV41.packSourceWeight(
                bytes, scales: MLXArray([UInt8](repeating: 1, count: 64 * 3)).reshaped(64, 3)))
        XCTAssertThrowsError(
            try DeepSeekV41.packSourceWeight(
                bytes.asType(.float32), scales: perRow),
            "only byte weights carry a separate E8M0 scale")
    }

    /// Every activation the QAT rounds is blocked 32 wide, so a width the block does
    /// not divide is refused at decode rather than deep inside the fake-quant.
    func testUnalignedWidthsAreRefused() throws {
        func decode(_ overrides: [String: Int]) throws -> DeepSeekV41Configuration {
            var fields = [
                "hidden_size": 32, "q_lora_rank": 32, "head_dim": 32,
                "index_head_dim": 32, "moe_intermediate_size": 32, "o_lora_rank": 32,
                "o_groups": 2, "qk_rope_head_dim": 8, "num_hidden_layers": 2,
            ]
            fields.merge(overrides) { _, new in new }
            let body = fields.map { "\"\($0.key)\": \($0.value)" }.joined(separator: ", ")
            let json = """
                {
                  \(body), "compress_ratios": [0, 1],
                  "engram_layer_ids": [], "engram_num_embeddings": []
                }
                """
            return try JSONDecoder().decode(
                DeepSeekV41Configuration.self, from: Data(json.utf8))
        }

        XCTAssertNoThrow(try decode([:]).validateModelConfiguration())
        for field in [
            "hidden_size", "q_lora_rank", "head_dim", "index_head_dim",
            "moe_intermediate_size",
        ] {
            XCTAssertThrowsError(
                try decode([field: 48]).validateModelConfiguration(), "accepted \(field)")
        }
        XCTAssertThrowsError(try decode(["o_lora_rank": 17]).validateModelConfiguration())
        XCTAssertNoThrow(try decode(["o_lora_rank": 16]).validateModelConfiguration())

        XCTAssertThrowsError(
            try decode(["qk_rope_head_dim": 64]).validateModelConfiguration())

        XCTAssertNoThrow(try textConfiguration().validateModelConfiguration())
        XCTAssertNoThrow(
            try JSONDecoder().decode(
                DeepSeekV41Configuration.self, from: Data("{}".utf8)
            ).validateModelConfiguration(), "the released config must pass")
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], label: String, accuracy: Float = 2e-5,
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
