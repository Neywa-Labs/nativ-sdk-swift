import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class BonsaiTests: XCTestCase {
    func testSanitizersPreservePackedWeightDTypesAndMapFlux2Keys() {
        let text = BonsaiWeightLoader.sanitizeTextEncoderWeights([
            "model.embed_tokens.weight": MLXArray.zeros([4, 1], dtype: .uint32),
            "model.embed_tokens.scales": MLXArray.zeros([4, 1]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 1], dtype: .uint32),
            "model.layers.0.self_attn.q_proj.biases": MLXArray.zeros([4, 1]),
            "lm_head.weight": MLXArray.zeros([4, 4]),
        ])
        XCTAssertEqual(text["embed_tokens.weight"]?.dtype, .uint32)
        XCTAssertEqual(text["embed_tokens.scales"]?.dtype, Flux2Precision.dtype)
        XCTAssertEqual(text["layers.0.self_attn.q_proj.weight"]?.dtype, .uint32)
        XCTAssertNil(text["lm_head.weight"])

        let transformer = BonsaiWeightLoader.sanitizeTransformerWeights([
            "time_guidance_embed.timestep_embedder.linear_1.weight": MLXArray.zeros([4, 4]),
            "transformer_blocks.0.attn.to_out.0.weight": MLXArray.zeros([4, 1], dtype: .uint32),
            "transformer_blocks.0.attn.to_out.0.scales": MLXArray.zeros([4, 1]),
            "transformer_blocks.0.attn.to_out.0.biases": MLXArray.zeros([4, 1]),
        ])
        XCTAssertNotNil(transformer["time_guidance_embed.linear_1.weight"])
        XCTAssertEqual(transformer["transformer_blocks.0.attn.to_out.weight"]?.dtype, .uint32)
        XCTAssertEqual(
            transformer["transformer_blocks.0.attn.to_out.scales"]?.dtype, Flux2Precision.dtype)
        XCTAssertEqual(
            transformer["transformer_blocks.0.attn.to_out.biases"]?.dtype, Flux2Precision.dtype)
    }

    func testPipelineRunsOneTinyDenoisingStep() throws {
        let pipeline = BonsaiImagePipeline(
            textEncoder: Flux2TextEncoder(Self.tinyTextConfig()),
            transformer: Flux2Transformer(Self.tinyTransformerConfig()),
            vae: Flux2VAE(Self.tinyVAEConfigForGeneration()),
        )
        let prompt = MLXArray.zeros([1, 32, 24], dtype: .float32)
        let result = try pipeline.generate(
            promptEmbeddings: prompt,
            textIDs: Flux2PromptIDs.prepareTextIDs(for: prompt),
            parameters: ImageGenerationParameters(seed: 3, steps: 1, width: 256, height: 256),
        )
        XCTAssertEqual(result.metadata["transformer_precision"], "2bit")
        XCTAssertEqual(result.image.shape, [256, 256, 3])
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

    private static func tinyVAEConfigForGeneration() -> Flux2VAEConfiguration {
        Flux2VAEConfiguration(
            decoderBlockOutChannels: [4, 4, 4, 4],
            latentChannels: 2,
            layersPerBlock: 1,
            normGroups: 1,
            midBlockAddAttention: true,
        )
    }
}
