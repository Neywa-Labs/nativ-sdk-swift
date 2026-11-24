import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct BOSFragmentTokenizer: Tokenizer {
    var bosToken: String? { "<bos>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (addSpecialTokens ? [99] : []) + text.unicodeScalars.map { Int($0.value) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private final class ProjectorFixture: Module, UnaryLayer {
    @ModuleInfo var mlp1: [UnaryLayer]
    override init() {
        _mlp1.wrappedValue = [
            LayerNorm(dimensions: 64), Linear(64, 64),
            SequentialActivation(GELU()), Linear(64, 64),
        ]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp1.reduce(x) { $1($0) } }
}

final class SmallModelCompatibilityTests: XCTestCase {
    func testImageFragmentsDoNotInsertBOS() throws {
        let tokenizer = BOSFragmentTokenizer()
        let fragment = "<|vision_start|><|image_pad|><|vision_end|>"
        let prompt =
            [99] + tokenizer.encode(text: "prefix" + fragment + "suffix", addSpecialTokens: false)
        let actual = try QwenVL.replacePaddingTokens(
            in: prompt, frames: [.init(1, 4, 4)], paddingToken: "<|image_pad|>",
            mergeSize: 2, tokenizer: tokenizer)
        let expanded =
            "<|vision_start|>" + String(repeating: "<|image_pad|>", count: 4)
            + "<|vision_end|>"
        XCTAssertEqual(
            actual,
            [99] + tokenizer.encode(text: "prefix" + expanded + "suffix", addSpecialTokens: false))
    }

    func testApertusDefaultRopeDoesNotRequireScalingFactor() throws {
        let json = """
            {"hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
             "num_attention_heads":4,"rms_norm_eps":0.00001,"vocab_size":48,
             "rope_scaling":{"rope_type":"default"}}
            """
        let config = try JSONDecoder().decode(ApertusConfiguration.self, from: Data(json.utf8))
        XCTAssertNil(config.ropeScaling)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ApertusConfiguration.self,
                from: Data(json.replacingOccurrences(of: "default", with: "linear").utf8)))
    }

    func testPaliGemmaVisionResolutionDefaultsAndExplicitOverride() throws {
        let json = """
            {"model_type":"siglip_vision_model","hidden_size":16,"num_hidden_layers":1,
             "intermediate_size":32,"num_attention_heads":2,"patch_size":14,"projection_dim":32}
            """
        let config = try JSONDecoder().decode(
            PaliGemmaConfiguration.VisionConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.imageSize, 224)
        let explicit = json.dropLast() + ",\"image_size\":448}"
        XCTAssertEqual(
            try JSONDecoder().decode(
                PaliGemmaConfiguration.VisionConfiguration.self, from: Data(explicit.utf8)
            ).imageSize, 448)
    }

    func testNanoChatDerivedFrequenciesAreNotCheckpointParameters() throws {
        let json = """
            {"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":4,
             "vocab_size":48,"max_position_embeddings":32,"intermediate_size":64}
            """
        let config = try JSONDecoder().decode(NanoChatConfiguration.self, from: Data(json.utf8))
        let model = NanoChatModel(config)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertFalse(weights.keys.contains { $0.contains("freqs") })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let logits = model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }

    func testDisabledLFMProjectorNormDropsOnlyItsUnusedWeights() throws {
        let json = """
            {"model_type":"lfm2-vl","projector_use_layernorm":false,"projector_hidden_size":32,
             "text_config":{"model_type":"lfm2","hidden_size":32,"num_hidden_layers":1,
                "num_attention_heads":4,"num_key_value_heads":2,"vocab_size":48},
             "vision_config":{"model_type":"siglip","hidden_size":16,"intermediate_size":32,
                "num_hidden_layers":1,"num_attention_heads":2,"image_size":32,"patch_size":8}}
            """
        for enabled in [false, true] {
            let config = try JSONDecoder().decode(
                LFM2VLConfiguration.self,
                from: Data(
                    json.replacingOccurrences(
                        of: "\"projector_use_layernorm\":false",
                        with: "\"projector_use_layernorm\":\(enabled)"
                    ).utf8))
            let model = LFM2VL(config)
            let weights = model.sanitize(weights: [
                "model.multi_modal_projector.layer_norm.weight": MLXArray.ones([64]),
                "model.multi_modal_projector.linear_1.weight": MLXArray.ones([32, 64]),
            ])
            XCTAssertEqual(weights["multi_modal_projector.layer_norm.weight"] != nil, enabled)
            XCTAssertNotNil(weights["multi_modal_projector.linear_1.weight"])
        }
    }

    func testSparseProjectorKeysSurviveQuantizationAndStrictLoading() throws {
        let source = ProjectorFixture()
        try quantizeByParent(model: source, groupSize: 32, bits: 4) { _, module in module is Linear
        }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        XCTAssertNotNil(weights["mlp1.3.scales"])
        let target = ProjectorFixture()
        try quantizeByParent(model: target, groupSize: 32, bits: 4) { _, module in module is Linear
        }
        try target.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let x = MLXArray.ones([1, 2, 64])
        let expected = source(x)
        let actual = target(x)
        eval(expected, actual)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-6)
    }

    func testInvalidSamplingOptionsAreRejectedAndBoundariesRemainValid() throws {
        for value: Float in [-1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(
                try GenerationDriver.validate(.init(sampling: .init(temperature: value))))
        }
        for value: Float in [-0.1, 1.1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try GenerationDriver.validate(.init(sampling: .init(topP: value))))
            XCTAssertThrowsError(try GenerationDriver.validate(.init(sampling: .init(minP: value))))
        }
        XCTAssertThrowsError(try GenerationDriver.validate(.init(sampling: .init(topK: -1))))
        for value: Float in [0, 1] {
            XCTAssertNoThrow(
                try GenerationDriver.validate(
                    .init(
                        sampling: .init(
                            temperature: value, topP: value, topK: 0, minP: value))))
        }
    }

    func testQuantizingOneArraySlotPreservesLeadingAndTrailingModules() throws {
        let projector = ProjectorFixture()
        let norm = projector.mlp1[0] as! LayerNorm
        let trailing = projector.mlp1[3] as! Linear
        try quantizeByParent(model: projector, groupSize: 32, bits: 4) { path, _ in path == "mlp1.1"
        }
        XCTAssertEqual(projector.mlp1.count, 4)
        XCTAssertTrue(projector.mlp1[1] is QuantizedLinear)
        XCTAssertTrue(projector.mlp1[0] as! LayerNorm === norm)
        XCTAssertTrue(projector.mlp1[3] as! Linear === trailing)
    }
}
