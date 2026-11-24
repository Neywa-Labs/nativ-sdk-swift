import MLX
import XCTest

@testable import NativSDK

private struct Flux2TestTokenizer: Tokenizer {
    var bosToken: String? {
        "<bos>"
    }

    var eosToken: String? {
        "<eos>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let core = text.unicodeScalars.map { Int($0.value % 32) + 3 }
        return addSpecialTokens ? [1] + core + [2] : core
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<bos>": 1
        case "<eos>": 2
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 1: "<bos>"
        case 2: "<eos>"
        default: nil
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        var content = ""
        for message in messages {
            content += message["content"] as? String ?? ""
        }
        return encode(text: "<user>\(content)<assistant>", addSpecialTokens: true)
    }
}

final class Flux2Tests: XCTestCase {
    func testSchedulerAndLatentPackingShapes() throws {
        try Flux2Dimensions.validate(width: 256, height: 512)
        let scheduler = Flux2FlowMatchEulerDiscreteScheduler(
            imageSequenceLength: 16 * 32,
            inferenceSteps: 4,
        )
        XCTAssertEqual(scheduler.timesteps.shape, [4])
        XCTAssertEqual(scheduler.sigmas.shape, [5])

        let prepared = Flux2Latents.preparePackedLatents(
            seed: 7,
            height: 256,
            width: 256,
            numLatentChannels: 2,
        )
        XCTAssertEqual(prepared.latents.shape, [1, 256, 8])
        XCTAssertEqual(prepared.ids.shape, [1, 256, 4])
        XCTAssertEqual(prepared.latentHeight, 16)
        XCTAssertEqual(prepared.latentWidth, 16)
    }

    func testPromptTokenizerBucketsAndTextEncoderEmbeddings() throws {
        let tokenizer = Flux2TestTokenizer()
        let tokens = try Flux2PromptTokenizer.tokenize(
            prompt: "tiny prompt",
            tokenizer: tokenizer,
            maxLength: 32,
            bucketed: true,
        )
        XCTAssertEqual(tokens.inputIDs.shape, [1, 32])
        XCTAssertEqual(tokens.attentionMask.shape, [1, 32])

        let encoder = Flux2TextEncoder(Self.tinyTextConfig())
        let embeddings = try encoder.promptEmbeddings(
            inputIDs: MLXArray([1, 2, 3, 4].map(Int32.init), [1, 4]),
            attentionMask: MLXArray([1, 1, 1, 1].map(Int32.init), [1, 4]),
            hiddenStateLayers: [1, 2, 3],
        )
        XCTAssertEqual(embeddings.shape, [1, 4, 24])
        XCTAssertEqual(Flux2PromptIDs.prepareTextIDs(for: embeddings).shape, [1, 4, 4])
    }

    func testPipelineRunsTinyKVEditPath() throws {
        let pipeline = Flux2ImagePipeline(
            textEncoder: Flux2TextEncoder(Self.tinyTextConfig()),
            transformer: Flux2Transformer(Self.tinyTransformerConfig()),
            vae: Flux2VAE(Self.tinyVAEEditConfig()), usesReferenceKVCache: true,
        )
        let prompt = MLXArray.zeros([1, 32, 24], dtype: .float32)
        let referenceImage = MLXArray.zeros([1, 3, 256, 256], dtype: .float32)
        let result = try pipeline.edit(
            promptEmbeddings: prompt,
            textIDs: Flux2PromptIDs.prepareTextIDs(for: prompt),
            referenceImages: [referenceImage],
            parameters: ImageGenerationParameters(seed: 11, steps: 2, width: 256, height: 256),
        )
        XCTAssertEqual(result.image.shape, [256, 256, 3])
        XCTAssertEqual(result.metadata["uses_reference_kv_cache"], "true")
    }

    func testOwnedExecutorRunsTinyPromptAndEditingPipeline() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let encoderWeights = directory.appending(path: "model.safetensors")
        let vae = Flux2VAE(Self.tinyVAEEditConfig(includeEncoder: false))
        vae.encoderDirectory = directory
        let textConfig = Flux2TextEncoderConfiguration(
            vocabSize: 64, hiddenSize: 8,
            numHiddenLayers: 28, numAttentionHeads: 2, numKeyValueHeads: 1,
            intermediateSize: 16, maxPositionEmbeddings: 512, headDim: 4)
        let pipeline = Flux2ImagePipeline(
            textEncoder: Flux2TextEncoder(textConfig),
            transformer: Flux2Transformer(Self.tinyTransformerConfig()),
            vae: vae, usesReferenceKVCache: true)
        let loadedModel = NativImageModel(
            model: pipeline, tokenizer: Flux2TestTokenizer(),
            source: .repository("example/custom-image", revision: "pinned"),
            modelDirectory: URL(fileURLWithPath: "/tmp"))
        XCTAssertTrue(loadedModel.supportsImageEditing)
        let shapes = try await loadedModel.promptShapes(
            prompt: "cat", parameters: .init(maxPromptLength: 32, bucketedPrompt: false))
        XCTAssertEqual(shapes, [[1, 32, 24], [1, 32, 4]])
        let parameters = ImageGenerationParameters(seed: 11, steps: 1, width: 256, height: 256)
        let first = loadedModel.generate(prompt: "cat", parameters: parameters)
        let generated = try await first.result()
        XCTAssertEqual(generated.image.shape, [256, 256, 3])
        XCTAssertEqual(generated.source, .repository("example/custom-image", revision: "pinned"))
        try save(arrays: Self.encoderWeights(), url: encoderWeights)
        let reference = try ModelImageData(
            pixels: Array(repeating: 0, count: 3 * 256 * 256), shape: [1, 3, 256, 256])
        let edit = loadedModel.generate(
            prompt: "cat", referenceImages: [reference], parameters: parameters)
        let edited = try await edit.result()
        XCTAssertEqual(edited.image.shape, [256, 256, 3])
        XCTAssertEqual(edited.metadata["uses_reference_kv_cache"], "true")
        try FileManager.default.removeItem(at: encoderWeights)
        let repeatedEdit = try await loadedModel.generate(
            prompt: "cat", referenceImages: [reference], parameters: parameters
        ).result()
        XCTAssertEqual(repeatedEdit.image.pixels, edited.image.pixels)
        let repeatRequest = loadedModel.generate(prompt: "cat", parameters: parameters)
        let repeated = try await repeatRequest.result()
        XCTAssertEqual(repeated.image.pixels, generated.image.pixels)
        XCTAssertTrue(edited.image.pixels.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 255 })
    }

    func testFailedEncoderLoadPreservesDecoderAndCanRetry() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = Self.encoderWeights()
        let file = directory.appending(path: "model.safetensors")
        try save(arrays: weights.filter { !$0.key.hasPrefix("quant_conv.") }, url: file)
        let vae = Flux2VAE(Self.tinyVAEEditConfig(includeEncoder: false))
        vae.encoderDirectory = directory
        let latents = MLXArray.zeros([1, 2, 2, 2])
        let before = vae.decode(latents).asArray(Float.self)
        let pixels = MLXArray.zeros([1, 3, 16, 16])
        XCTAssertThrowsError(try vae.encode(pixels))
        XCTAssertFalse(vae.supportsEncoding)
        XCTAssertEqual(vae.decode(latents).asArray(Float.self), before)
        try save(arrays: weights, url: file)
        let encoded = try vae.encode(pixels)
        XCTAssertEqual(encoded.shape, [1, 2, 2, 2])
        XCTAssertTrue(encoded.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(vae.decode(latents).asArray(Float.self), before)
    }

    private static func encoderWeights() -> [String: MLXArray] {
        let source = Flux2VAE(tinyVAEEditConfig())
        return Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().compactMap { key, value in
                guard key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") else { return nil }
                let key = key.replacingOccurrences(
                    of: "encoder.conv_in.conv.", with: "encoder.conv_in."
                )
                .replacingOccurrences(of: "encoder.conv_out.conv.", with: "encoder.conv_out.")
                .replacingOccurrences(
                    of: "encoder.conv_norm_out.norm.", with: "encoder.conv_norm_out.")
                return (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
            })
    }

    func testSanitizersMatchPythonFlux2KeyRewrites() {
        let text = Flux2WeightLoader.sanitizeTextEncoderWeights([
            "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
            "lm_head.weight": MLXArray.zeros([8, 4]),
        ])
        XCTAssertNotNil(text["embed_tokens.weight"])
        XCTAssertNotNil(text["layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(text["lm_head.weight"])

        let transformer = Flux2WeightLoader.sanitizeTransformerWeights([
            "time_guidance_embed.timestep_embedder.linear_1.weight": MLXArray.zeros([4, 4]),
            "transformer_blocks.0.attn.to_out.0.weight": MLXArray.zeros([4, 4]),
        ])
        XCTAssertNotNil(transformer["time_guidance_embed.linear_1.weight"])
        XCTAssertNotNil(transformer["transformer_blocks.0.attn.to_out.weight"])

        let vae = Flux2WeightLoader.sanitizeVAEWeights([
            "decoder.conv_in.weight": MLXArray.zeros([4, 2, 3, 3]),
            "decoder.conv_norm_out.weight": MLXArray.zeros([4]),
            "encoder.conv_in.weight": MLXArray.zeros([4, 3, 3, 3]),
            "bn.running_mean": MLXArray.zeros([8]),
            "decoder.block.num_batches_tracked": MLXArray.zeros([1]),
        ])
        XCTAssertEqual(vae["decoder.conv_in.conv.weight"]?.shape, [4, 3, 3, 2])
        XCTAssertNotNil(vae["decoder.conv_norm_out.norm.weight"])
        XCTAssertNotNil(vae["bn.running_mean"])
        XCTAssertNil(vae["encoder.conv_in.conv.weight"])
        XCTAssertNil(vae["decoder.block.num_batches_tracked"])
    }

    private static func tinyTextConfig() -> Flux2TextEncoderConfiguration {
        Flux2TextEncoderConfiguration(
            vocabSize: 64,
            hiddenSize: 8,
            numHiddenLayers: 3,
            numAttentionHeads: 2,
            numKeyValueHeads: 1,
            intermediateSize: 16,
            maxPositionEmbeddings: 64,
            headDim: 4,
        )
    }

    private static func tinyTransformerConfig() -> Flux2TransformerConfiguration {
        Flux2TransformerConfiguration(
            inChannels: 8,
            numLayers: 1,
            numSingleLayers: 1,
            attentionHeadDim: 8,
            numAttentionHeads: 2,
            jointAttentionDim: 24,
            timestepGuidanceChannels: 8,
            mlpRatio: 1.0,
            axesDimsRoPE: [2, 2, 2, 2],
        )
    }

    private static func tinyVAEEditConfig(includeEncoder: Bool = true) -> Flux2VAEConfiguration {
        Flux2VAEConfiguration(
            decoderBlockOutChannels: [4, 4, 4, 4],
            encoderBlockOutChannels: [4, 4, 4, 4],
            includeEncoder: includeEncoder,
            latentChannels: 2,
            layersPerBlock: 1,
            normGroups: 1,
            midBlockAddAttention: true,
        )
    }
}
