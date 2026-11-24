import Foundation
import MLX
import XCTest

@testable import NativSDK

private struct DownloadCall: Equatable {
    let id: String
    let revision: String?
    let patterns: [String]

}

private actor RecordingDownloader: ModelSnapshotProvider {
    let directory: URL
    private var recordedCalls = [DownloadCall]()

    init(directory: URL) {
        self.directory = directory
    }

    var calls: [DownloadCall] {
        recordedCalls
    }

    func snapshot(
        id: String,
        revision: String?,
        matching patterns: [String],

        progress _: @Sendable @escaping (Progress) -> Void,
    )
        async throws -> URL
    {
        recordedCalls.append(
            DownloadCall(id: id, revision: revision, patterns: patterns),
        )
        return directory
    }
}

final class ImageGenerationModelLoaderTests: XCTestCase {
    func testMOSSImageUsesDistinctSmallVAEDecoderChannels() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "MOSSImagePipeline"], "model_index.json", in: directory)
        try Self.write(
            ["_class_name": "MOSSImageTransformer"], "transformer/config.json", in: directory)
        try Self.write(["shift": 5], "scheduler/scheduler_config.json", in: directory)
        try Self.write(
            ["text_config": ["hidden_size": 4096]], "text_encoder/config.json", in: directory)
        try Self.write(
            [
                "block_out_channels": [128, 256, 512, 512],
                "decoder_block_out_channels": [96, 192, 384, 384],
                "latent_channels": 32, "layers_per_block": 2, "norm_num_groups": 32,
            ], "vae/config.json", in: directory)
        let vae = try Flux2VAEConfiguration.read(from: directory)
        XCTAssertEqual(vae.decoderBlockOutChannels, [96, 192, 384, 384])
        XCTAssertEqual(vae.encoderBlockOutChannels, [128, 256, 512, 512])
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .mossImage)
        let downloader = RecordingDownloader(directory: directory)
        let model = try await ImageModelLoader(provider: downloader).load(
            from: .repository("OpenMOSS-Team/MOSS-Image-Base"), using: TextFixtureTokenizerLoader())
        XCTAssertFalse(model.supportsImageEditing)
        let calls = await downloader.calls
        let call = try XCTUnwrap(calls.first)
        XCTAssertTrue(call.patterns.contains("transformer/*.safetensors"))
        XCTAssertTrue(call.patterns.contains("scheduler/*.json"))
        XCTAssertFalse(call.patterns.contains("**/*.safetensors"))
    }

    func testErnieImageDownloadsPromptEnhancerOnlyOnRequest() async throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "snapshots/resolved-revision")
        try Self.write(["_class_name": "ErnieImagePipeline"], "model_index.json", in: directory)
        try Self.write(["text_in_dim": 8], "transformer/config.json", in: directory)
        try Self.write(
            [
                "text_config": [
                    "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                    "num_attention_heads": 1, "vocab_size": 16, "rms_norm_eps": 1e-5,
                    "rope_parameters": [
                        "llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 16,
                    ],
                ]
            ], "text_encoder/config.json", in: directory)
        try Self.write(
            [
                "block_out_channels": [4, 4, 8, 8], "latent_channels": 32,
                "layers_per_block": 1, "norm_num_groups": 1,
            ], "vae/config.json", in: directory)
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .ernieImage)
        let vae = directory.appending(path: "vae/diffusion_pytorch_model.safetensors")
        try save(arrays: ["decoder.conv_in.weight": MLXArray(1)], url: vae)
        let decoderOnly = try ErnieImagePipeline(
            directory: directory, source: .directory(directory))
        XCTAssertFalse(decoderOnly.supportsImageEditing)
        try save(
            arrays: ["encoder.conv_in.weight": MLXArray(1), "quant_conv.weight": MLXArray(1)],
            url: vae)
        let originalVAE = directory.appending(path: "vae")
        let linkedVAE = root.appending(path: "vae")
        try FileManager.default.moveItem(at: originalVAE, to: linkedVAE)
        try FileManager.default.createSymbolicLink(at: originalVAE, withDestinationURL: linkedVAE)
        let downloader = RecordingDownloader(directory: directory)
        let model = try await ImageModelLoader(provider: downloader).load(
            from: .repository("ernie-image-turbo", revision: "pinned"),
            using: TextFixtureTokenizerLoader())
        XCTAssertTrue(model.supportsImageEditing)
        let initialCalls = await downloader.calls
        let initial = try XCTUnwrap(initialCalls.first)
        XCTAssertEqual(initialCalls.count, 1)
        XCTAssertEqual(initial.id, "baidu/ERNIE-Image-Turbo")
        XCTAssertEqual(initial.revision, "pinned")
        XCTAssertFalse(initial.patterns.contains("pe/*"))

        let p = ImageGenerationParameters(width: 16, height: 16, promptExpansion: .defaultModel)
        let reference = try ModelImageData(pixels: [0, 0, 0], shape: [1, 3, 1, 1])
        do {
            _ = try await model.generate(
                prompt: "caption", referenceImages: [reference, reference], parameters: p
            ).result()
            XCTFail("ERNIE-Image must reject a second reference")
        } catch ModelExecutionError.invalidOptions(let message) {
            XCTAssertEqual(message, "ERNIE-Image accepts one reference image")
        }
        let invalidCalls = await downloader.calls
        XCTAssertEqual(invalidCalls, initialCalls)

        do {
            _ = try await model.generate(prompt: "caption", parameters: p).result()
            XCTFail("Missing prompt enhancer weights must be reported")
        } catch ModelExecutionError.invalidOptions(let message) {
            XCTAssertTrue(message.contains("requires the PE checkpoint"))
        }
        let calls = await downloader.calls
        XCTAssertEqual(
            calls,
            initialCalls + [
                DownloadCall(
                    id: initial.id, revision: "resolved-revision",
                    patterns: ["pe/*", "pe_tokenizer/*"])
            ])
    }

    func testMageFlowEditAliasLoadsAnOpaqueSnapshot() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "MageFlowPipeline"], "model_index.json", in: directory)
        try Self.write(["_class_name": "MageFlow"], "transformer/config.json", in: directory)
        try Self.write(["_class_name": "MageVAE"], "vae/config.json", in: directory)
        let downloader = RecordingDownloader(directory: directory)
        let model = try await ImageModelLoader(provider: downloader).load(
            from: .repository("mage-flow-edit-turbo", revision: "pinned"),
            using: TextFixtureTokenizerLoader())
        XCTAssertTrue(model.supportsImageEditing)
        let calls = await downloader.calls
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(call.id, "mage-flow-community/Mage-Flow-Edit-Turbo")
        XCTAssertEqual(call.revision, "pinned")
        XCTAssertTrue(call.patterns.contains("transformer/*"))
        XCTAssertTrue(call.patterns.contains("text_encoder/*"))
    }

    func testFlux3UsesRootCheckpointLayoutAndScopedDownload() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(
            [
                "model": "flux-3-image-klein",
                "dit": ["in_channels": ["image": 128, "image_cond": 128, "text": 20480]],
            ], "config.json", in: directory)
        try Self.write(["model_type": "qwen3_vl"], "text_encoder/config.json", in: directory)
        for file in ["model.safetensors", "ae.safetensors"] {
            try Data().write(to: directory.appending(path: file))
        }
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .flux3)
        let downloader = RecordingDownloader(directory: directory)
        let model = try await ImageModelLoader(provider: downloader).load(
            from: .repository("black-forest-labs/flux-3-image-klein", revision: "pinned"),
            using: TextFixtureTokenizerLoader())
        XCTAssertTrue(model.supportsImageEditing)
        let calls = await downloader.calls
        XCTAssertEqual(
            calls,
            [
                DownloadCall(
                    id: "black-forest-labs/flux-3-image-klein", revision: "pinned",
                    patterns: [
                        "config.json", "model.safetensors", "ae.safetensors", "text_encoder/*",
                    ])
            ])
    }

    func testImageExecutorResolvesRepositoryRevisionAndComponentFiles() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let downloader = RecordingDownloader(directory: directory)
        do {
            _ = try await ImageModelLoader(provider: downloader).load(
                from: .repository("example/image-model", revision: "custom-revision"))
            XCTFail("A snapshot without image metadata must fail family detection")
        } catch ModelExecutionError.invalidOptions(let message) {
            XCTAssertTrue(message.contains("Cannot identify the image pipeline"))
        }
        let calls = await downloader.calls
        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.id, "example/image-model")
        XCTAssertEqual(call.revision, "custom-revision")
        XCTAssertTrue(call.patterns.contains("**/*.safetensors"))
        XCTAssertTrue(call.patterns.contains("**/*.json"))
        XCTAssertTrue(call.patterns.contains("**/*.jinja"))
    }

    func testMiniMaxH3ImageResolvesDiffusersComponentsAndLinkedConditioner() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(
            [
                "_class_name": "MiniMaxH3ImagePipeline",
                "text_encoder": [
                    "transformers", "Qwen3VLForConditionalGeneration",
                    [
                        "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                        "subfolder": "text_encoder", "revision": "conditioner-revision",
                    ],
                ],
            ], "model_index.json", in: directory)
        try Self.write(
            ["_class_name": "MiniMaxH3ImageTransformer2DModel"],
            "transformer_diffusers/config.json", in: directory)
        try Self.write(
            [
                "latents_mean": Array(repeating: 0, count: 24),
                "latents_std": Array(repeating: 1, count: 24),
            ], "vae_diffusers/config.json", in: directory)
        let downloader = RecordingDownloader(directory: directory)
        let model = try await ImageModelLoader(provider: downloader).load(
            from: .repository("MiniMaxAI/MiniMax-H3-Image"), using: TextFixtureTokenizerLoader())
        XCTAssertTrue(model.supportsImageEditing)
        let calls = await downloader.calls
        XCTAssertEqual(calls.count, 2)
        let checkpoint = try XCTUnwrap(calls.first)
        XCTAssertTrue(checkpoint.patterns.contains("transformer_diffusers/*"))
        XCTAssertFalse(checkpoint.patterns.contains("transformer/*"))
        XCTAssertEqual(
            calls.last,
            DownloadCall(
                id: "MiniMaxAI/MiniMax-H3", revision: "conditioner-revision",
                patterns: ["text_encoder/*"]))
    }

    func testMissingLocalImageDirectoryNeverDownloads() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let downloader = RecordingDownloader(directory: directory)
        do {
            _ = try await ImageModelLoader(provider: downloader).load(
                from: .directory(directory))
            XCTFail("A missing directory must fail locally")
        } catch ModelExecutionError.invalidOptions(let message) {
            XCTAssertTrue(message.contains("Image model directory does not exist"))
        }
        let calls = await downloader.calls
        XCTAssertTrue(calls.isEmpty)
    }

    func testMetadataDetectionDoesNotDependOnDirectoryName() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "flux2-base-9b")
        try Self.write(["_class_name": "QwenImage21Pipeline"], "model_index.json", in: directory)
        try Self.write(
            ["model_type": "qwen_image"], "mlx_qwen_image.json",
            in: directory)
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .qwenImage)
        let renamed = root.appending(path: "opaque-checkpoint")
        try FileManager.default.moveItem(at: directory, to: renamed)
        XCTAssertEqual(try ImagePipelineKind.detect(in: renamed), .qwenImage)
        try Self.write(["model_type": "z_image"], "transformer/config.json", in: renamed)
        XCTAssertThrowsError(try ImagePipelineKind.detect(in: renamed))
    }

    func testQwenImage21PublishedMetadataSelectsImagePipeline() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "QwenImage21Pipeline"], "model_index.json", in: directory)
        try Self.write(
            ["_class_name": "QwenImage21Transformer2DModel"], "transformer/config.json",
            in: directory)
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .qwenImage)
        try FileManager.default.removeItem(at: directory.appending(path: "model_index.json"))
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .qwenImage)
    }

    func testBonsaiPackedLayoutTakesPrecedenceOverSharedFluxClass() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "Flux2KleinPipeline"], "model_index.json", in: directory)
        try Self.write(
            ["_class_name": "Flux2Transformer2DModel"], "transformer-packed-mflux/config.json",
            in: directory)
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .bonsai)
    }

    func testFluxBehaviorRequiresMetadataAndPreservesBaseAndReferenceCacheModes() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "Flux2KleinPipeline"], "model_index.json", in: directory)
        XCTAssertThrowsError(try ImageCheckpointMetadata.fluxBehavior(in: directory))
        try Self.write(
            ["_class_name": "Flux2KleinPipeline", "is_distilled": false], "model_index.json",
            in: directory)
        let base = try ImageCheckpointMetadata.fluxBehavior(in: directory)
        XCTAssertEqual(base.steps, 50)
        XCTAssertEqual(base.guidance, 4)
        XCTAssertFalse(base.referenceCache)
        try Self.write(
            ["_class_name": "Flux2KleinKVPipeline", "is_distilled": true], "model_index.json",
            in: directory)
        let distilled = try ImageCheckpointMetadata.fluxBehavior(in: directory)
        XCTAssertEqual(distilled.steps, 4)
        XCTAssertEqual(distilled.guidance, 1)
        XCTAssertTrue(distilled.referenceCache)
    }

    func testInvalidConversionMetadataJSONIsNotIgnored() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(["_class_name": "QwenImage21Pipeline"], "model_index.json", in: directory)
        try Self.write(
            ["model_type": "qwen_image"], "mlx_qwen_image.json",
            in: directory)
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .qwenImage)
        try Data("invalid".utf8).write(to: directory.appending(path: "mlx_qwen_image.json"))
        XCTAssertThrowsError(try ImagePipelineKind.detect(in: directory))
    }

    func testComponentConfigurationsBuildCheckpointDimensions() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(
            [
                "in_channels": 8, "num_layers": 1, "num_single_layers": 2,
                "attention_head_dim": 8, "num_attention_heads": 2, "joint_attention_dim": 24,
                "axes_dims_rope": [2, 2, 2, 2], "timestep_guidance_channels": 8,
            ], "transformer/config.json", in: directory)
        let config = try Flux2TransformerConfiguration.read(from: directory)
        let model = Flux2Transformer(config)
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["x_embedder.weight"], [16, 8])
        XCTAssertNotNil(shapes.keys.first { $0.hasPrefix("single_transformer_blocks.1.") })
        XCTAssertNil(shapes.keys.first { $0.hasPrefix("single_transformer_blocks.2.") })
        try Self.write(["hidden_size": 8], "text_encoder/config.json", in: directory)
        XCTAssertThrowsError(try Flux2TextEncoderConfiguration.read(from: directory))
        try Self.write(
            [
                "block_out_channels": [4, 8], "latent_channels": 2,
                "layers_per_block": 1, "norm_num_groups": 1,
            ], "vae/config.json", in: directory)
        let vae = try Flux2VAEConfiguration.read(from: directory, includeEncoder: true)
        XCTAssertTrue(Flux2VAE(vae).supportsEncoding)
        XCTAssertEqual(vae.spatialScale, 2)
    }

    private static func write(_ object: [String: Any], _ path: String, in directory: URL) throws {
        let url = directory.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: object).write(to: url)
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }
}
