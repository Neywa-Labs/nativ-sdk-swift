import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class EmbeddingModelTests: XCTestCase {
    static let families = [
        "bert", "modernbert", "siglip", "llama_bidirec", "colidefics3", "gemma3_embedding",
        "lfm2_embedding", "lfm2_colbert", "ministral3_embedding",
        "xlm_roberta", "qwen3_vl_embedding", "colqwen2_5", "llama_nemotron_vl_embedding",
    ]

    static func configuration(_ family: String) -> [String: Any] {
        var text: [String: Any] = [
            "model_type": family, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "embedding_dim": 8,
        ]
        switch family {
        case "siglip":
            return ["model_type": family, "text_config": text, "vision_config": [:]]
        case "colidefics3":
            text["rope_theta"] = 10000
            return [
                "model_type": family, "text_config": text, "embedding_dim": 8,
                "image_token_id": 14, "vocab_size": 64, "scale_factor": 2,
                "vision_config": [
                    "model_type": "idefics3", "hidden_size": 8, "num_hidden_layers": 1,
                    "intermediate_size": 16, "num_attention_heads": 2, "image_size": 4,
                    "patch_size": 2,
                ],
            ]
        case "modernbert":
            text["local_attention"] = 2
        case "gemma3_embedding":
            text.merge(["sliding_window_pattern": 2, "query_pre_attn_scalar": 8]) { _, new in new }
        case "lfm2_embedding", "lfm2_colbert":
            text.merge([
                "norm_eps": 1e-5, "conv_L_cache": 3, "full_attn_idxs": [1],
                "block_ff_dim": 64, "block_multiple_of": 16,
            ]) { _, new in new }
        case "ministral3_embedding":
            text["rope_parameters"] = [
                "rope_type": "default", "rope_theta": 10000,
                "llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 2,
            ]
        case "xlm_roberta":
            text["pad_token_id"] = 0
        case "qwen3_vl_embedding", "colqwen2_5":
            text["rope_scaling"] = ["type": "default", "mrope_section": [2, 1, 1]]
            let vision: [String: Any] = [
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "out_hidden_size": 32, "num_heads": 2, "patch_size": 2, "spatial_patch_size": 2,
                "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0], "window_size": 8, "fullatt_block_indexes": [0],
                "tokens_per_second": 2,
            ]
            var result: [String: Any] = [
                "model_type": family, "text_config": text, "vision_config": vision,
                "image_token_id": 14, "video_token_id": 15, "vision_start_token_id": 12,
                "vision_end_token_id": 13, "vision_token_id": 11, "vocab_size": 64,
                "embedding_dim": 8,
            ]
            if family == "colqwen2_5" {
                result.merge(text) { current, _ in current }
                result["sliding_window"] = 32
                result["use_sliding_window"] = false
                result["max_window_layers"] = 2
            }
            return result
        case "llama_nemotron_vl_embedding":
            return [
                "model_type": family, "text_config": text,
                "vision_config": [
                    "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                    "num_attention_heads": 2, "image_size": 4, "patch_size": 2,
                ],
                "img_context_token_id": 14, "downsample_ratio": 0.5,
            ]
        default: break
        }
        return text
    }
    private func model(_ family: String, pooling: EmbeddingPooling? = nil) throws
        -> any EmbeddingModel
    {
        try ModelRegistry().createEmbeddingModel(
            configuration: JSONSerialization.data(withJSONObject: Self.configuration(family)),
            modelType: family, pooling: pooling)
    }
    private func array(_ result: EmbeddingOutput) -> MLXArray {
        switch result {
        case .pooled(let x), .tokens(let x): x
        }
    }
    private func assertClose(
        _ lhs: MLXArray, _ rhs: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)
        XCTAssertTrue(
            allClose(lhs, rhs, rtol: 3e-3, atol: 5e-4).item(Bool.self), file: file, line: line)
    }

    func testEncodersMaskPaddingAndProduceNormalizedResults() throws {
        for family in Self.families {
            let encoder = try model(family)
            let plain = array(try encoder.embed(LMInput(tokens: MLXArray([[2, 3, 4]]))))
            XCTAssertTrue(all(isFinite(plain)).item(Bool.self), family)
            assertClose(sum(square(plain), axis: -1), MLXArray.ones(Array(plain.shape.dropLast())))
            // ColBERT deliberately leaves convolution inputs unmasked; it does not promise padding invariance.
            if family == "lfm2_colbert" || family == "siglip" { continue }
            let padded = array(
                try encoder.embed(
                    LMInput(
                        text: .init(
                            tokens: MLXArray([[2, 3, 4, 0, 0]]), mask: MLXArray([[1, 1, 1, 0, 0]])))
                ))
            assertClose(plain, padded.ndim == 3 ? padded[0..., 0 ..< 3, 0...] : padded)
        }
    }

    func testBidirectionalEncodersAttendToFutureTokens() throws {
        for family in [
            "bert", "modernbert", "gemma3_embedding", "lfm2_embedding", "ministral3_embedding",
            "xlm_roberta", "lfm2_colbert",
        ] {
            let encoder = try model(family, pooling: .cls)
            let a = array(try encoder.embed(LMInput(tokens: MLXArray([[2, 3, 4]])))).flattened()
            let b = array(try encoder.embed(LMInput(tokens: MLXArray([[2, 3, 5]])))).flattened()
            let length = family == "lfm2_colbert" ? 8 : a.size
            XCTAssertGreaterThan(
                abs(a[0 ..< length] - b[0 ..< length]).max().item(Float.self), 1e-6, family)
        }
    }

    func testCenteredLFMConvolutionSeesBothNeighbors() throws {
        let config = try JSONDecoder().decode(
            LFM2Configuration.self,
            from: JSONSerialization.data(withJSONObject: Self.configuration("lfm2_embedding")))
        let encoder = try LFM2EmbeddingModel(config, pooling: .cls)
        let conv = try XCTUnwrap(encoder.model.layers[0].conv)
        let x = MLXArray.ones([1, 3, 32])
        let projected = conv.inProj(x).split(parts: 3, axis: -1)
        let gated = projected[0] * projected[2]
        let centered = conv.conv(padded(gated, widths: [0, [1, 1], 0]))
        let causal = conv.conv(padded(gated, widths: [0, [2, 0], 0]))
        XCTAssertFalse(allClose(centered, causal).item(Bool.self))
        // The ordinary generation convolution still pads only the left side.
        assertClose(conv(x, cache: nil), conv.outProj(projected[1] * causal))
    }

    func testVisualEncodersUseImageFeaturesAndHaveNoRequestState() throws {
        for family in [
            "qwen3_vl_embedding", "colqwen2_5", "colidefics3", "llama_nemotron_vl_embedding",
        ] {
            let encoder = try model(family)
            let tokens = MLXArray([[12, 14, 13, 3]])
            let shape =
                ["llama_nemotron_vl_embedding", "colidefics3"].contains(family)
                ? [1, 4, 4, 3] : [4, 24]
            func input(_ pixels: MLXArray) -> LMInput {
                LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: pixels, frames: [THW(1, 2, 2)]))
            }
            let first = array(try encoder.embed(input(MLXArray.zeros(shape))))
            let other = array(try encoder.embed(input(MLXArray.ones(shape))))
            XCTAssertTrue(all(isFinite(first)).item(Bool.self), family)
            XCTAssertGreaterThan(abs(first - other).max().item(Float.self), 1e-6, family)
            assertClose(first, array(try encoder.embed(input(MLXArray.zeros(shape)))))
            let textOnly = array(try encoder.embed(LMInput(tokens: MLXArray([[2, 3, 4]]))))
            XCTAssertTrue(all(isFinite(textOnly)).item(Bool.self), family)
        }
    }

    func testCheckpointLoadingIncludesProjectionSidecarsAndQuantization() async throws {
        for family in Self.families {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let encoder = try model(family)
            var config = Self.configuration(family)
            do {
                let mixedPath = "backbone.language_model.model.layers.0.self_attn.q_proj"
                try quantizeByParent(model: encoder) { path, module in
                    guard let linear = module as? Linear, linear.weight.dim(1) % 32 == 0 else {
                        return nil
                    }
                    return (32, path == mixedPath ? 8 : 4, .affine)
                }
                var quantization: [String: Any] = ["group_size": 32, "bits": 4]
                if family == "qwen3_vl_embedding" || family == "colqwen2_5" {
                    quantization[String(mixedPath.dropFirst("backbone.".count))] = [
                        "group_size": 32, "bits": 8,
                    ]
                }
                config["quantization"] = quantization
            }
            let expected = array(try encoder.embed(LMInput(tokens: MLXArray([[2, 3, 4]]))))
            var weights = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened())
            if family == "gemma3_embedding" {
                for index in 0 ..< 2 {
                    let folder = root.appendingPathComponent("\(index + 2)_Dense")
                    try FileManager.default.createDirectory(
                        at: folder, withIntermediateDirectories: true)
                    var dense: [String: MLXArray] = [:]
                    for key in Array(weights.keys) where key.hasPrefix("dense.\(index).") {
                        dense["linear." + key.dropFirst("dense.\(index).".count)] =
                            weights.removeValue(forKey: key)
                    }
                    try save(arrays: dense, url: folder.appendingPathComponent("model.safetensors"))
                }
                config["model_type"] = "gemma3_text"
            }
            if family == "lfm2_colbert" {
                let folder = root.appendingPathComponent("1_Dense")
                try FileManager.default.createDirectory(
                    at: folder, withIntermediateDirectories: true)
                var dense: [String: MLXArray] = [:]
                for key in Array(weights.keys) where key.hasPrefix("projection.") {
                    dense["linear." + key.dropFirst("projection.".count)] = weights.removeValue(
                        forKey: key)
                }
                try save(arrays: dense, url: folder.appendingPathComponent("model.safetensors"))
                try Data(#"{"out_features":8,"bias":false}"#.utf8).write(
                    to: folder.appendingPathComponent("config.json"))
                config["model_type"] = "lfm2"
            }
            weights = Dictionary(
                uniqueKeysWithValues: weights.map { key, tensor in
                    var key = key
                    var tensor = tensor
                    switch family {
                    case "qwen3_vl_embedding":
                        key = key.replacingOccurrences(
                            of: "backbone.vision_tower.", with: "model.visual."
                        )
                        .replacingOccurrences(
                            of: "backbone.language_model.model.", with: "model.language_model.")
                    case "colqwen2_5":
                        key = key.replacingOccurrences(
                            of: "backbone.vision_tower.", with: "vlm.model.visual."
                        )
                        .replacingOccurrences(
                            of: "backbone.language_model.model.", with: "vlm.model.language_model."
                        )
                        .replacingOccurrences(
                            of: "custom_text_proj.", with: "embedding_proj_layer.")
                    case "colidefics3":
                        key = key.replacingOccurrences(of: "backbone.", with: "model.")
                    case "llama_nemotron_vl_embedding":
                        key = key.replacingOccurrences(of: "mlp1.norm.", with: "mlp1.0.")
                            .replacingOccurrences(of: "mlp1.input.", with: "mlp1.1.")
                            .replacingOccurrences(of: "mlp1.output.", with: "mlp1.3.")
                    case "bert": key = "bert." + key
                    case "xlm_roberta": key = "roberta." + key
                    case "modernbert": key = "model." + key
                    case "siglip": key = String(key.dropFirst("text_model.".count))
                    default: break
                    }
                    if key.hasSuffix("patch_embedding.weight"), tensor.ndim == 4 {
                        tensor = tensor.transposed(0, 3, 1, 2)
                    } else if key.hasSuffix("patch_embed.proj.weight"), tensor.ndim == 5 {
                        tensor = tensor.transposed(0, 4, 1, 2, 3)
                    }
                    return (key, tensor)
                })
            try save(arrays: weights, url: root.appendingPathComponent("model.safetensors"))
            try JSONSerialization.data(withJSONObject: config).write(
                to: root.appendingPathComponent("config.json"))
            try Data(
                #"{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"min_pixels":16,"max_pixels":16,"merge_size":2,"patch_size":2,"temporal_patch_size":2,"image_processor_type":"Qwen2VLImageProcessor"}"#
                    .utf8
            )
            .write(to: root.appendingPathComponent("preprocessor_config.json"))
            let loadedModel = try await NativEmbeddingModel.loadModel(
                from: .directory(root),
                using: FamilyTokenizerLoader())
            let result = try await loadedModel.embed(input: .text("test")).result()
            let values: [Float]
            switch result.embeddings {
            case .vector(let v): values = v
            case .tokenVectors(let v): values = v.flatMap { $0 }
            }
            assertClose(expected.flattened(), MLXArray(values))
            if ["qwen3_vl_embedding", "colqwen2_5", "colidefics3", "llama_nemotron_vl_embedding"]
                .contains(family)
            {
                var imageResults: [EmbeddingRepresentation] = []
                for (name, color) in [
                    ("red", CIColor(red: 1, green: 0, blue: 0)),
                    ("green", CIColor(red: 0, green: 1, blue: 0)),
                ] {
                    let url = root.appendingPathComponent(name + ".png")
                    let image = CIImage(color: color).cropped(
                        to: CGRect(x: 0, y: 0, width: 4, height: 4))
                    try CIContext().writePNGRepresentation(
                        of: image, to: url, format: .RGBA8,
                        colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
                    let imageResult = try await loadedModel.embed(input: .image(url)).result()
                    imageResults.append(imageResult.embeddings)
                    XCTAssertGreaterThan(imageResult.tokenCount, 0)
                }
                XCTAssertNotEqual(imageResults[0], imageResults[1], family)
            }
        }
    }

    func testNemotronPixelShufflePreservesSpatialOrder() {
        let input = arange(16).reshaped(1, 16, 1)
        let shuffled = LlamaNemotronEmbeddingModel.pixelShuffle(input, factor: 2)
        XCTAssertEqual(
            shuffled.asArray(Int.self), [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    }
}

private struct FamilyTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { FamilyTokenizer() }
}
private struct FamilyTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let visualCount = text.components(separatedBy: "<|image_pad|>").count - 1
        if visualCount > 0 { return [12] + Array(repeating: 14, count: visualCount) + [13] }
        let count = max(
            text.components(separatedBy: "<image>").count - 1,
            text.components(separatedBy: "<IMG_CONTEXT>").count - 1)
        return count > 0 ? [12] + Array(repeating: 14, count: count) + [13, 3] : [2, 3, 4]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "<image>" }
    func convertTokenToId(_ token: String) -> Int? { token == "<|image_pad|>" ? 14 : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [12, 14, 13, 3] }
}
