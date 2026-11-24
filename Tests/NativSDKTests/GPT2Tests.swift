import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class GPT2Tests: XCTestCase {
    func testQuantizedCheckpointPreservesLogits() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json.utf8), modelType: "gpt2", quantized: true)
    }

    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json.utf8), modelType: "gpt2")
    }

    private static let json = """
        {
          "model_type": "gpt2",
          "n_ctx": 1024,
          "n_embd": 32,
          "n_head": 4,
          "n_layer": 2,
          "n_positions": 64,
          "layer_norm_epsilon": 0.00001,
          "vocab_size": 48
        }
        """

    private static func config() throws -> GPT2Configuration {
        try JSONDecoder.json5().decode(
            GPT2Configuration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    /// Positions are looked up here, not rotated, so the cache offset has to reach `wpe`.
    ///
    /// Both runs below decode the same token against a cache holding the same five
    /// entries; the only difference is the weight sitting in position row 5. If the offset
    /// were dropped and every token embedded as position 0, both runs would read the same
    /// row of zeros and agree. Comparing one warm run against one cold run would not show
    /// this: a deeper cache also gives attention more to read.
    func testPositionEmbeddingFollowsTheCacheOffset() throws {
        let config = try Self.config()
        // One model, so the only thing that changes between the two runs is the weight in
        // position row 5. Building a fresh model per run would compare two random
        // initializations and pass whether or not the offset was ever applied.
        let model = GPT2Model(config)
        let inner = model.languageModel.model

        func output(positionRowFive: Float) throws -> MLXArray {
            let wpe = MLXArray.zeros([config.positions, config.embeddingDimensions])
            // A pattern, not a constant: the block's first LayerNorm subtracts the mean, so
            // a uniform row is removed before it reaches attention and would make this test
            // pass with the offset deleted.
            let ramp = MLXArray(0 ..< config.embeddingDimensions).asType(.float32)
            wpe[5] = (ramp - Float(config.embeddingDimensions) / 2) * positionRowFive
            inner.positionEmbedding.update(
                parameters: ModuleParameters.unflattened([("weight", wpe)]))

            let cache = try model.newCache(configuration: nil)
            _ = inner(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5), cache: cache)
            XCTAssertEqual(cache[0].offset, 5)

            let out = inner(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
            eval(out)
            return out
        }

        let quiet = try output(positionRowFive: 0)
        let loud = try output(positionRowFive: 8)
        let difference = abs(quiet - loud).max().item(Float.self)

        XCTAssertGreaterThan(
            difference, 1e-3,
            "the two runs differ only in position row 5, so an identical answer means the "
                + "cache offset never reached the position embedding")
    }

    /// HuggingFace stores these as `Conv1D`, which is the transpose of what a Linear wants.
    /// Every one of them is square or paired with its own transpose, so loading them
    /// untransposed binds cleanly and answers nonsense.
    func testSanitizerTransposesConv1DWeightsAndDropsTheMaskBuffer() throws {
        let model = GPT2Model(try Self.config())
        let sanitized = model.sanitize(weights: [
            "h.0.attn.c_attn.weight": MLXArray.zeros([32, 96]),
            "h.0.attn.c_attn.bias": MLXArray.zeros([96]),
            "h.0.attn.c_proj.weight": MLXArray.zeros([32, 32]),
            "h.0.mlp.c_fc.weight": MLXArray.zeros([32, 128]),
            "h.0.mlp.c_proj.weight": MLXArray.zeros([128, 32]),
            "h.0.attn.bias": MLXArray.ones([1, 1, 1024, 1024]),
            "h.0.attn.masked_bias": MLXArray.ones([1]),
            "wte.weight": MLXArray.zeros([48, 32]),
            "wpe.weight": MLXArray.zeros([64, 32]),
        ])

        XCTAssertEqual(sanitized["language_model.model.h.0.attn.c_attn.weight"]?.shape, [96, 32])
        XCTAssertEqual(sanitized["language_model.model.h.0.attn.c_proj.weight"]?.shape, [32, 32])
        XCTAssertEqual(sanitized["language_model.model.h.0.mlp.c_fc.weight"]?.shape, [128, 32])
        XCTAssertEqual(sanitized["language_model.model.h.0.mlp.c_proj.weight"]?.shape, [32, 128])

        // Biases are one-dimensional and must pass through untouched.
        XCTAssertEqual(sanitized["language_model.model.h.0.attn.c_attn.bias"]?.shape, [96])

        // The causal mask the original implementation kept as a buffer is not a parameter.
        XCTAssertNil(sanitized["language_model.model.h.0.attn.bias"])
        XCTAssertNil(sanitized["language_model.model.h.0.attn.masked_bias"])

        XCTAssertNotNil(sanitized["language_model.model.wte.weight"])
        XCTAssertNotNil(sanitized["language_model.model.wpe.weight"])
    }

    func testSanitizerAcceptsAlreadyPrefixedKeys() throws {
        let model = GPT2Model(try Self.config())
        let sanitized = model.sanitize(weights: [
            "language_model.model.wte.weight": MLXArray.zeros([48, 32]),
            "model.wpe.weight": MLXArray.zeros([64, 32]),
        ])
        XCTAssertNotNil(sanitized["language_model.model.wte.weight"])
        XCTAssertNotNil(sanitized["language_model.model.wpe.weight"])
        XCTAssertEqual(sanitized.count, 2)
    }

    /// A sanitized checkpoint has to actually bind. Loading the model's own freshly
    /// initialized weights back through the published spelling is the cheapest way to say
    /// so without a download: every key the module tree wants is present, and nothing the
    /// sanitizer emits is left over.
    func testSanitizedWeightsBindToTheModuleTree() throws {
        let model = GPT2Model(try Self.config())
        var published: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var name = key.replacingOccurrences(of: "language_model.model.", with: "")
            // Publish the projections in Conv1D order, which is what the sanitizer undoes.
            if [
                "attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight",
                "mlp.c_proj.weight",
            ].contains(where: { name.hasSuffix($0) }) {
                published[name] = value.transposed(1, 0)
                continue
            }
            name = name.replacingOccurrences(of: "language_model.", with: "")
            published[name] = value
        }
        published["h.0.attn.bias"] = MLXArray.ones([1, 1, 1024, 1024])

        let sanitized = model.sanitize(weights: published)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }

}

extension GPT2Tests {

    /// Converted projections must not be transposed again, including square matrices.
    func testTransposeFollowsTheCheckpointLayout() throws {
        let config = try Self.config()
        let model = GPT2LanguageModel(config)
        let dimensions = config.embeddingDimensions

        func weights(fused: [Int]) -> [String: MLXArray] {
            [
                "h.0.attn.c_attn.weight": MLXArray.zeros(fused),
                "h.0.attn.c_proj.weight": MLXArray.zeros([dimensions, dimensions]),
                "h.0.mlp.c_fc.weight": MLXArray.zeros([4 * dimensions, dimensions]),
            ]
        }

        // Already converted: shapes must survive untouched.
        let converted = model.sanitize(
            weights: weights(fused: [3 * dimensions, dimensions]))
        XCTAssertEqual(
            converted["language_model.model.h.0.attn.c_attn.weight"]?.shape,
            [3 * dimensions, dimensions])
        XCTAssertEqual(
            converted["language_model.model.h.0.mlp.c_fc.weight"]?.shape,
            [4 * dimensions, dimensions])

        // Raw Conv1D layout: every listed projection is transposed.
        var raw = weights(fused: [dimensions, 3 * dimensions])
        raw["h.0.mlp.c_fc.weight"] = MLXArray.zeros([dimensions, 4 * dimensions])
        let fixed = model.sanitize(weights: raw)
        XCTAssertEqual(
            fixed["language_model.model.h.0.attn.c_attn.weight"]?.shape,
            [3 * dimensions, dimensions])
        XCTAssertEqual(
            fixed["language_model.model.h.0.mlp.c_fc.weight"]?.shape,
            [4 * dimensions, dimensions])

        // The causal-mask buffer is dropped either way rather than bound.
        let withBuffer = model.sanitize(
            weights: weights(fused: [3 * dimensions, dimensions])
                .merging(["h.0.attn.bias": MLXArray.zeros([1, 1, 8, 8])]) { a, _ in a })
        XCTAssertFalse(withBuffer.keys.contains { $0.hasSuffix(".attn.bias") })
    }
}
