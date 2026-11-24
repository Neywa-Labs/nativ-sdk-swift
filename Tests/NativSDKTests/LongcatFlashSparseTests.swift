import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class LongcatFlashSparseTests: XCTestCase {

    private static func config(
        method: String = "LSA", topK: Int = 3, sinks: Int = 0, local: Int = 0,
        ngramRatio: Int = 0, keyNorm: String = "rms"
    ) throws -> LongcatFlashSparseConfiguration {
        try JSONDecoder.json5().decode(
            LongcatFlashSparseConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "longcat_flash_sparse",
                overrides: [
                    "attention_method": method, "index_topk": topK, "index_init_tokens": sinks,
                    "index_local_tokens": local, "oe_vocab_size_ratio": ngramRatio,
                    "index_k_norm_type": keyNorm,
                ]))
    }

    /// Only the first sub-layer owns an indexer; the second reuses its indices. A second
    /// indexer would declare weights no checkpoint has.
    func testOnlyTheFirstSubLayerOwnsAnIndexer() throws {
        let model = LongcatFlashSparseModel(try Self.config())
        XCTAssertNotNil(model.model.layers[0].attentions[0].indexer)
        XCTAssertNil(model.model.layers[0].attentions[1].indexer)

        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.0.indexer.wq_b.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.self_attn.1.indexer.wq_b.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.1.kv_a_proj_with_mqa.weight"))
    }

    /// `MLA` leaves attention dense and builds no indexer at all.
    func testDenseMethodBuildsNoIndexer() throws {
        let dense = LongcatFlashSparseModel(try Self.config(method: "MLA"))
        XCTAssertFalse(dense.configuration.usesSparseAttention)
        XCTAssertNil(dense.model.layers[0].attentions[0].indexer)

        let cache = try dense.newCache(configuration: nil)
        XCTAssertEqual((cache[0] as? CacheList)?.children.count, 2)
    }

    /// A ratio of zero means plain `embed_tokens`; above zero the n-gram tables replace it.
    func testNgramEmbeddingIsOptional() throws {
        let plain = LongcatFlashSparseModel(try Self.config(ngramRatio: 0))
        XCTAssertNotNil(plain.model.embedTokens)
        XCTAssertNil(plain.model.ngramEmbeddings)
        XCTAssertTrue(
            Set(plain.parameters().flattened().map(\.0))
                .contains("model.embed_tokens.weight"))

        let ngram = LongcatFlashSparseModel(try Self.config(ngramRatio: 3))
        XCTAssertNil(ngram.model.embedTokens)
        XCTAssertNotNil(ngram.model.ngramEmbeddings)
        XCTAssertTrue(
            Set(ngram.parameters().flattened().map(\.0))
                .contains("model.ngram_embeddings.word_embeddings.weight"))
    }

    /// Everything inside the budget needs no selection at all.
    func testIndexerReturnsNothingWhenEverythingFits() throws {
        let indexer = LongcatSparseIndexer(try Self.config(topK: 8))
        let x = MLXRandom.normal([1, 4, 24], key: MLXRandom.key(3))
        let rank = MLXRandom.normal([1, 4, 16], key: MLXRandom.key(4))
        XCTAssertNil(
            indexer(x, queryRank: rank, mask: nil, cache: nil),
            "four positions under a budget of eight need no selection")
    }

    /// The sinks and the local window are forced in *before* the top-k, so they are
    /// selected however low they score.
    func testStreamingSinksAreAlwaysSelected() throws {
        let indexer = LongcatSparseIndexer(try Self.config(topK: 2, sinks: 1))
        let x = MLXRandom.normal([1, 6, 24], key: MLXRandom.key(5))
        let rank = MLXRandom.normal([1, 6, 16], key: MLXRandom.key(6))
        let selected = try XCTUnwrap(indexer(x, queryRank: rank, mask: nil, cache: nil))
        eval(selected)

        XCTAssertEqual(selected.shape, [1, 1, 6, 2])
        let includesZero = (selected .== 0).any(axis: -1)
        eval(includesZero)
        XCTAssertTrue(
            includesZero.all().item(Bool.self),
            "a forced sink was outscored and dropped")

        let unforced = LongcatSparseIndexer(try Self.config(topK: 2, sinks: 0))
        unforced.update(
            parameters: ModuleParameters.unflattened(
                indexer.parameters().flattened()))
        let free = try XCTUnwrap(unforced(x, queryRank: rank, mask: nil, cache: nil))
        eval(free)
        XCTAssertFalse(
            (free .== 0).any(axis: -1).all().item(Bool.self),
            "column zero wins on score anyway, so the forcing test proves nothing")
    }

    /// The local window forces the current position and the ones just before it.
    func testLocalWindowIsAlwaysSelected() throws {
        let indexer = LongcatSparseIndexer(try Self.config(topK: 2, local: 2))
        let x = MLXRandom.normal([1, 5, 24], key: MLXRandom.key(7))
        let rank = MLXRandom.normal([1, 5, 16], key: MLXRandom.key(8))
        let selected = try XCTUnwrap(indexer(x, queryRank: rank, mask: nil, cache: nil))
        eval(selected)

        let row = selected[0, 0, 4].asArray(Int32.self).sorted()
        XCTAssertEqual(row, [3, 4], "the local window was not forced in")
    }

    /// `index_k_norm_type` chooses the indexer's key norm.
    func testKeyNormTypeIsConfigurable() throws {
        XCTAssertTrue(
            LongcatSparseIndexer(try Self.config(keyNorm: "rms")).kNorm is RMSNorm)
        XCTAssertTrue(
            LongcatSparseIndexer(try Self.config(keyNorm: "layer")).kNorm is LayerNorm)
    }

    /// A decode step gathers the rows it selected rather than masking them, so the
    /// attention it runs is only as wide as the budget.
    func testDecodeStepGathersTheSelectedRows() throws {
        let config = try Self.config(topK: 2)
        let model = LongcatFlashSparseModel(config)
        let cache = try model.newCache(configuration: nil)

        _ = model(MLXArray(0 ..< 6).reshaped(1, 6), cache: cache)
        let step = model(MLXArray([Int32(9)]).reshaped(1, 1), cache: cache)
        eval(step)

        XCTAssertEqual(step.shape, [1, 1, 32])
        XCTAssertFalse(isNaN(step.asType(.float32)).any().item(Bool.self))
        XCTAssertEqual((cache[0] as? CacheList)?[0].offset, 7)
        XCTAssertEqual((cache[0] as? CacheList)?[1].offset, 7, "the indexer cache too")
        XCTAssertEqual((cache[0] as? CacheList)?[2].offset, 7, "and the second sub-layer's")
    }

}
