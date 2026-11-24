import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct ImagePipelineTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { Int($0 % 16) + 1 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: messages.compactMap { $0["content"] as? String }.joined(), addSpecialTokens: false
        )
    }
}

private struct ImagePipelineTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { ImagePipelineTokenizer() }
}

final class ImagePipelineLoadingTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }
    private func json(_ values: [String: Any], _ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: values).write(to: url)
    }
    func testSavedZImageComponentsGenerateAndEditAfterEncoderReload() async throws {
        let root = try directory()
        let tc: [String: Any] = [
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "intermediate_size": 16, "head_dim": 4,
        ]
        let dc: [String: Any] = [
            "dim": 24, "n_heads": 2, "n_layers": 1, "n_refiner_layers": 1, "in_channels": 2,
            "cap_feat_dim": 8, "axes_dims": [4, 4, 4],
        ]
        let vc: [String: Any] = [
            "latent_channels": 2, "block_out_channels": [32, 32, 32, 32], "layers_per_block": 1,
        ]
        let text = Flux2TextEncoder(.init(diffusion: .init(tc)))
        let dit = ZImageTransformer(try .init(.init(dc)))
        let vae = ZImageVAE(.init(vc))
        for (component, config, model) in [
            ("text_encoder", tc, text as Module), ("transformer", dc, dit as Module),
            ("vae", vc, vae as Module),
        ] {
            try json(config, root.appending(path: "\(component)/config.json"))
            try save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                metadata: ["mlx_vlm_format": "z_image"],
                url: root.appending(path: "\(component)/model.safetensors"))
        }
        try json(["shift": 3], root.appending(path: "scheduler/scheduler_config.json"))
        try json(["_class_name": "ZImagePipeline"], root.appending(path: "model_index.json"))
        let loadedModel = try await NativImageModel.loadModel(
            from: .directory(root),
            using: ImagePipelineTokenizerLoader())
        let options = ImageGenerationParameters(seed: 42, steps: 2, width: 16, height: 16)
        let first = try await loadedModel.generate(prompt: "sun", parameters: options).result()
        _ = try await loadedModel.generate(prompt: "moon", parameters: options).result()
        let repeated = try await loadedModel.generate(prompt: "sun", parameters: options).result()
        XCTAssertEqual(first.image.pixels, repeated.image.pixels)
        // Repository identity is retained, but already-resolved assets never download it.
        let assets = ResolvedModelAssets(
            source: .repository("resolved/image-model"), modelDirectory: root,
            tokenizerDirectory: root)
        let localModel: NativImageModel = try await NativImageModel.loadModel(
            assets: assets, tokenizerLoader: ImagePipelineTokenizerLoader())
        XCTAssertEqual(localModel.source, assets.source)
        let localResult = try await localModel.generate(prompt: "sun", parameters: options).result()
        XCTAssertEqual(first.image.pixels, localResult.image.pixels)
        let reference = try ModelImageData(
            pixels: Array(repeating: 0, count: 3 * 16 * 16), shape: [1, 3, 16, 16])
        let edited = try await loadedModel.generate(
            prompt: "sun", referenceImages: [reference], parameters: options
        ).result()
        XCTAssertEqual(edited.image.shape, [16, 16, 3])
        XCTAssertNotEqual(first.image.pixels, edited.image.pixels)

        var expanding = options
        expanding.promptExpansion = .defaultModel
        do {
            _ = try await loadedModel.generate(prompt: "sun", parameters: expanding).result()
            XCTFail("Z-Image should reject a request for prompt expansion")
        } catch let error as ModelExecutionError {
            XCTAssertEqual(error, .unsupportedOperation("image prompt expansion"))
        }
    }

    func testQuantizedComponentMetadataAndStrictMissingWeights() throws {
        let root = try directory()
        let c = DiffusionConfiguration([
            "context_in_dim": 64, "num_attention_heads": 4, "attention_head_dim": 16,
            "axes_dims_rope": [4, 6, 6], "in_channels": 64, "out_channels": 64, "num_layers": 1,
        ])
        let source = try QwenImageTransformer(c)
        try quantizeByParent(model: source, groupSize: 32, bits: 4) { path, _ in
            path == "img_in" || path == "modulation.0"
        }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        try json(
            ["quantization": ["bits": 4, "group_size": 32, "mode": "affine"]],
            root.appending(path: "config.json"))
        try save(arrays: weights, url: root.appending(path: "model.safetensors"))
        let target = try QwenImageTransformer(c)
        try DiffusionCheckpoint(directory: root).apply(to: target)
        let x = MLXArray.ones([1, 4, 64])
        let text = MLXArray.ones([1, 3, 64])
        let time = MLXArray([Float(0.5)])
        let a = try source(x, text: text, timestep: time, height: 2, width: 2)
        let b = try target(x, text: text, timestep: time, height: 2, width: 2)
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5)
        var incomplete = try DiffusionCheckpoint(directory: root)
        incomplete.weights.removeValue(forKey: "img_in.scales")
        XCTAssertThrowsError(try incomplete.apply(to: QwenImageTransformer(c)))
    }
    func testCaptionExpansionRejectsInvalidStructuredContent() throws {
        let valid =
            #"{"high_level_description":"A sign","compositional_deconstruction":{"background":"A brick wall","elements":[{"type":"text","text":"OPEN","desc":"White letters","bbox":[0,0,1000,1000]}]}}"#
        XCTAssertNoThrow(try Ideogram4Prompting.validateExpandedCaption(valid))
        XCTAssertThrowsError(
            try Ideogram4Prompting.validateExpandedCaption(
                valid.replacingOccurrences(of: "1000", with: "2000")))
        XCTAssertThrowsError(
            try Ideogram4Prompting.validateExpandedCaption(
                #"{"compositional_deconstruction":{"elements":[]}}"#))
    }

    func testIdeogramPromptExpansionUsesSharedSelection() async throws {
        let root = try directory()
        let legacy = ModelSource.directory(root.appending(path: "legacy"))
        let override = ModelSource.directory(root.appending(path: "override"))
        let model = NativImageModel(
            model: Ideogram4ImagePipeline(), tokenizer: ImagePipelineTokenizer(),
            source: .directory(root), modelDirectory: root)
        var parameters = ImageGenerationParameters(ideogram4: .init(autoJSONCaption: false))
        parameters.ideogram4.legacyPromptExpansionModel = legacy

        for source in [legacy, override] {
            if source == override { parameters.promptExpansion = .model(source) }
            do {
                _ = try await model.generate(prompt: "A sign", parameters: parameters).result()
                XCTFail("The selected expansion model has no configuration")
            } catch ModelFactoryError.configurationFileError(let file, let name, _) {
                XCTAssertEqual(file, "config.json")
                XCTAssertEqual(name, source.name)
            }
        }

        parameters.promptExpansion = .defaultModel
        do {
            _ = try await model.generate(prompt: "A sign", parameters: parameters).result()
            XCTFail("Ideogram 4 requires an explicit expansion model")
        } catch let error as ModelExecutionError {
            XCTAssertEqual(
                error, .invalidOptions("Ideogram 4 requires an explicit prompt expansion model"))
        }

        parameters.promptExpansion = .disabled
        do {
            _ = try await model.generate(prompt: "A sign", parameters: parameters).result()
            XCTFail("The fixture has no image text encoder")
        } catch Ideogram4Error.missingComponent(let name) {
            XCTAssertEqual(name, "text_encoder")
        }
    }

    func testReferenceResizingPreservesStraightAlphaAndChannelRange() throws {
        let pixels = stacked([
            MLXArray.ones([4, 6]), MLXArray.zeros([4, 6]), -MLXArray.ones([4, 6]),
            MLXArray.zeros([4, 6]),
        ]).expandedDimensions(axis: 0)
        let resized = try diffusionResizeRGB(pixels, height: 8, width: 12, preserveAlpha: true)
        XCTAssertEqual(resized.shape, [1, 4, 8, 12])
        for (channel, expected) in [Float(1), 0, -1, 0].enumerated() {
            XCTAssertLessThan(abs(resized[0, channel] - expected).max().item(Float.self), 0.02)
        }
    }

    func testImageTokenCommitmentCrossesBlocksAndMasksUnconditionalPadding() throws {
        var calls = 0
        let output = try lladaImageSampleVQ(
            conditional: Array(repeating: 1, count: 30), unconditional: [2],
            count: 9, codebookSize: 4, imageTokenOffset: 10, maskToken: 9, endToken: 8
        ) { ids, mask, positions in
            calls += 1
            XCTAssertEqual(ids[0, ..<30].asArray(Int.self), Array(repeating: 1, count: 30))
            XCTAssertFalse(mask[1, 0, 0, 0].item(Bool.self))
            XCTAssertTrue(mask[0, 0, 0, 31].item(Bool.self))
            XCTAssertEqual(positions[1, 29].item(Int.self), 0)
            if ids.dim(1) == 64 { XCTAssertFalse(mask[0, 0, 0, 32].item(Bool.self)) }
            var logits = Array(repeating: Float(-10), count: 2 * 32 * 14)
            for i in 0 ..< 64 { logits[i * 14 + 11] = 10 }
            return MLXArray(logits, [2, 32, 14])
        }
        XCTAssertEqual(output.asArray(Int.self), Array(repeating: 1, count: 9))
        XCTAssertEqual(calls, 2)
        XCTAssertThrowsError(
            try lladaImageSampleVQ(
                conditional: [1], unconditional: [2], count: 4,
                codebookSize: 4, imageTokenOffset: 10, maskToken: 9, endToken: 8
            ) { _, _, _ in
                var logits = Array(repeating: Float(-10), count: 2 * 32 * 14)
                for i in 0 ..< 64 { logits[i * 14 + 8] = 10 }
                return MLXArray(logits, [2, 32, 14])
            })
    }
    func testFP8ExpertBlocksRestoreScalesAndRejectIncompleteMetadata() throws {
        let name = "layers.0.mlp.experts.gate_proj"
        let packed = MLXArray.full([2, 128, 130], values: MLXArray(UInt8(0x38)))
        let scales = MLXArray([Float(1), 2, 3, 4], [2, 1, 2])
        let config = DiffusionConfiguration([
            "llada_fp8_experts": [
                "enabled": true, "format": "e4m3fn", "weight_granularity": "per_expert_2d_block",
                "weight_block_size": [128, 128],
            ]
        ])
        let restored = try lladaImageRestoreFP8(
            [name: packed, name + "_scale": scales], config: config)[name]!
        XCTAssertEqual(restored[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(restored[0, 0, 129].item(Float.self), 2)
        XCTAssertEqual(restored[1, 127, 0].item(Float.self), 3)
        XCTAssertEqual(restored[1, 127, 129].item(Float.self), 4)
        XCTAssertThrowsError(try lladaImageRestoreFP8([name: packed], config: config))
        XCTAssertThrowsError(
            try lladaImageRestoreFP8([name: packed, name + "_scale": scales], config: .init()))
    }
}
