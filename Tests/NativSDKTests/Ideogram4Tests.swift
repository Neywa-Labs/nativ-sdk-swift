import MLX
import XCTest

@testable import NativSDK

final class Ideogram4Tests: XCTestCase {
    func testPromptWrapperProducesStructuredCaption() throws {
        let prepared = try Ideogram4Prompting.preparePrompt("A clean poster with the word SWIFT.")
        XCTAssertTrue(prepared.isJSONCaption)
        XCTAssertTrue(prepared.isStructuredCaption)
        XCTAssertTrue(prepared.wasWrapped)
        XCTAssertTrue(prepared.text.contains("high_level_description"))
        XCTAssertTrue(prepared.text.contains("compositional_deconstruction"))
    }

    func testScheduleUsesBoundedLogitNormalValues() {
        let schedule = Ideogram4LogitNormalSchedule.forResolution(
            width: 256,
            height: 256,
            knownMean: 0.0,
            std: 1.75,
        )
        XCTAssertGreaterThan(schedule.value(at: 0.0), schedule.value(at: 0.5))
        XCTAssertGreaterThan(schedule.value(at: 0.5), schedule.value(at: 1.0))
    }

    func testFP8ConversionHandlesE4M3WeightsAndRowScales() throws {
        let weight = MLXArray(
            [UInt8(0x38), UInt8(0xB8), UInt8(0x40), UInt8(0xC0)],
            [2, 2],
        )
        let scale = MLXArray([Float(1.0), Float(0.5)])
        let converted = Ideogram4WeightLoader.dequantizeFP8WeightOnly(
            [
                "projection.weight": weight,
                "projection.weight_scale": scale,
            ],
            precision: .float32,
        )
        let decoded = try XCTUnwrap(converted["projection.weight"])
        eval(decoded)
        XCTAssertNil(converted["projection.weight_scale"])
        XCTAssertEqual(decoded.shape, [2, 2])
        XCTAssertEqual(decoded[0, 0].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(decoded[0, 1].item(Float.self), -1.0, accuracy: 0.001)
        XCTAssertEqual(decoded[1, 0].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(decoded[1, 1].item(Float.self), -1.0, accuracy: 0.001)
    }

    func testPipelineRunsOneTinyDenoisingStep() throws {
        let config = Self.tinyTransformerConfig(inChannels: 128)
        let pipeline = Ideogram4ImagePipeline(
            textEncoder: Flux2TextEncoder(
                .init(
                    vocabSize: 32, hiddenSize: 8, numHiddenLayers: 1,
                    numAttentionHeads: 2, numKeyValueHeads: 1, intermediateSize: 16, headDim: 4)),
            conditionalTransformer: Ideogram4Transformer(config),
            unconditionalTransformer: Ideogram4Transformer(config),
            vae: Flux2VAE(Self.tinyVAEConfigForGeneration()),
        )
        pipeline.componentDirectory = URL(fileURLWithPath: "/unused/already-loaded-components")
        let parameters = ImageGenerationParameters(seed: 5, steps: 1, width: 256, height: 256)
        let positionIDs = Self.positionIDs(
            textTokens: 2, width: parameters.width, height: parameters.height)
        let promptEmbeddings = MLXArray.zeros(
            [1, positionIDs.dim(1), config.llmFeaturesDim], dtype: .float32)
        let result = try pipeline.generate(
            promptEmbeddings: promptEmbeddings,
            textIDs: positionIDs,
            parameters: parameters,
        )
        XCTAssertEqual(result.steps, 1)
        XCTAssertEqual(result.image.shape, [256, 256, 3])
        XCTAssertEqual(result.metadata["architecture"], "single_stream_dit")
        XCTAssertNil(
            pipeline.textEncoder, "Staged generation releases the text encoder before denoising")
    }

    private static func tinyTransformerConfig(inChannels: Int) -> Ideogram4TransformerConfiguration
    {
        Ideogram4TransformerConfiguration(
            embDim: 16,
            numLayers: 1,
            numHeads: 2,
            intermediateSize: 32,
            adaLNDim: 8,
            inChannels: inChannels,
            llmFeaturesDim: 12,
            ropeTheta: 1000,
            mropeSection: [1, 1, 1],
        )
    }

    private static func tinyVAEConfigForGeneration() -> Flux2VAEConfiguration {
        Flux2VAEConfiguration(
            decoderBlockOutChannels: [4, 4, 4, 4],
            latentChannels: 32,
            layersPerBlock: 1,
            normGroups: 1,
            midBlockAddAttention: true,
        )
    }

    private static func positionIDs(textTokens: Int, width: Int, height: Int) -> MLXArray {
        let gridHeight = height / 16
        let gridWidth = width / 16
        var values: [Int32] = []
        for index in 0 ..< textTokens {
            let value = Int32(index)
            values.append(contentsOf: [value, value, value])
        }
        for h in 0 ..< gridHeight {
            for w in 0 ..< gridWidth {
                values.append(contentsOf: [
                    Int32(65536),
                    Int32(65536 + h),
                    Int32(65536 + w),
                ])
            }
        }
        return MLXArray(values, [1, textTokens + gridHeight * gridWidth, 3])
    }
}
