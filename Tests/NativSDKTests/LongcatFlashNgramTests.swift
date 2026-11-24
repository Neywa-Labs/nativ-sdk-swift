import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class LongcatFlashNgramTests: XCTestCase {
    private static func config() throws -> LongcatFlashNgramConfiguration {
        try JSONDecoder.json5().decode(
            LongcatFlashNgramConfiguration.self,
            from: ModelTestConfigurations.configuration("longcat_flash_ngram"))
    }

    private static func embedding(_ config: LongcatFlashNgramConfiguration)
        -> LongcatNgramEmbedding
    {
        LongcatNgramEmbedding(
            vocabularySize: config.base.vocabularySize,
            hiddenSize: config.base.hiddenSize,
            vocabularyRatio: config.ngramVocabularyRatio,
            splits: config.embeddingSplits,
            neighbours: config.embeddingNeighbours)
    }

    /// `k * (n - 1)` tables, each with its own modulus `m + index * 2 + 1`, and each
    /// narrower than the hidden size by that count.
    func testEachEmbedderHasItsOwnModulus() throws {
        let config = try Self.config()
        let embedding = Self.embedding(config)

        XCTAssertEqual(embedding.embedders.count, 4)
        XCTAssertEqual(embedding.embedderModuli, [97, 99, 101, 103])
        XCTAssertEqual(
            embedding.embedders.map { $0.weight.dim(0) }, [97, 99, 101, 103],
            "the tables must be sized to their own moduli")
        XCTAssertEqual(
            embedding.embedders[0].weight.dim(1), 24 / 4,
            "each table is the hidden size divided by the table count")
        XCTAssertEqual(embedding.postProjections[0].weight.dim(0), 24)
    }

    /// The powers are `vocab_size^j` taken modulo *this* embedder's modulus.
    ///
    /// The first power is the vocabulary size itself whenever that is below the modulus, so
    /// the two-neighbour tables share it; what separates the embedders is the modulus
    /// applied at the end, and -- once a power exceeds it -- the powers themselves.
    func testPowersAreTakenModuloEachEmbedder() throws {
        let embedding = Self.embedding(try Self.config())

        XCTAssertEqual(embedding.vocabularyModuli[0], [32])
        XCTAssertEqual(embedding.vocabularyModuli[1], [32])
        XCTAssertNotEqual(
            embedding.embedderModuli[0], embedding.embedderModuli[1],
            "the two-neighbour tables are separated by their moduli alone")

        XCTAssertEqual(embedding.vocabularyModuli[2], [32, 14])
        XCTAssertEqual(embedding.vocabularyModuli[3], [32, 97])
        XCTAssertNotEqual(
            embedding.vocabularyModuli[2], embedding.vocabularyModuli[3],
            "the three-neighbour tables must hash differently once a power wraps")
    }

    /// A sequence no longer than the shift becomes all zeros, not a partial shift.
    func testShiftRight() {
        let x = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let byOne = LongcatNgramEmbedding.shiftedRight(x, by: 1)
        eval(byOne)
        XCTAssertEqual(byOne.reshaped(4).asArray(Int32.self), [0, 1, 2, 3])

        let byFour = LongcatNgramEmbedding.shiftedRight(x, by: 4)
        eval(byFour)
        XCTAssertEqual(byFour.reshaped(4).asArray(Int32.self), [0, 0, 0, 0])

        XCTAssertEqual(
            LongcatNgramEmbedding.shiftedRight(x, by: 0).reshaped(4).asArray(Int32.self),
            [1, 2, 3, 4])
    }

    /// The sum of the word embedding and every projected n-gram embedding is divided by
    /// `1 + k * (n - 1)`.
    func testTheSumIsAveraged() throws {
        let config = try Self.config()
        let embedding = Self.embedding(config)
        var updates: [(String, MLXArray)] = []
        for index in 0 ..< 4 {
            updates.append(("post_projs.\(index).weight", MLXArray.zeros([24, 6])))
        }
        embedding.update(parameters: ModuleParameters.unflattened(updates))

        let tokens = MLXArray([Int32(3), 5]).reshaped(1, 2)
        let out = embedding(tokens, cache: nil)
        let words = embedding.wordEmbeddings(tokens)
        eval(out, words)

        XCTAssertLessThan(
            abs(out - words / 5).max().item(Float.self), 1e-6,
            "the averaging divisor is not 1 + k * (n - 1)")
        XCTAssertGreaterThan(abs(words).max().item(Float.self), 1e-3)
    }

    /// A decode step's n-grams must reach back into the cached context, so the same token
    /// answers differently depending on what preceded it.
    func testDecodeStepUsesTheCachedContext() throws {
        let config = try Self.config()
        let embedding = Self.embedding(config)

        func step(after prefix: [Int32]) -> MLXArray {
            let cache = ArraysCache(size: 1)
            _ = embedding(MLXArray(prefix).reshaped(1, prefix.count), cache: cache)
            return embedding(MLXArray([Int32(7)]).reshaped(1, 1), cache: cache)
        }

        let a = step(after: [1, 2, 3])
        let b = step(after: [9, 8, 7])
        eval(a, b)
        XCTAssertGreaterThan(
            abs(a - b).max().item(Float.self), 1e-5,
            "the same token embedded identically after different context")
    }

    /// The word table moves under the n-gram embedding, and the base model's expert
    /// stacking still applies.
    func testSanitizeRenamesTheWordTable() throws {
        let model = LongcatFlashNgramModel(try Self.config())
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([32, 24]),
            "model.mtp.layers.0.weight": MLXArray.zeros([2, 2]),
        ])
        XCTAssertNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.ngram_embeddings.word_embeddings.weight"])
        XCTAssertNil(sanitized["model.mtp.layers.0.weight"])
    }
}
