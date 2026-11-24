import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func qwen4ExpConfig(_ json: String) throws -> Qwen4ExpTextConfiguration {
    try JSONDecoder.json5().decode(Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
}

@Suite("Qwen4-Exp")
struct Qwen4ExpTests {

    /// Layers alternate linear and sparse attention, and a checkpoint may spell the sparse
    /// kind either way.
    @Test("Layer types are derived or normalized")
    func layerTypes() throws {
        let derived = try qwen4ExpConfig(
            #"{"num_hidden_layers": 8, "full_attention_interval": 4}"#)
        #expect(
            derived.layerTypes == [
                "linear_attention", "linear_attention", "linear_attention",
                "qwen_sparse_attention", "linear_attention", "linear_attention",
                "linear_attention", "qwen_sparse_attention",
            ])
        #expect(derived.layerIsSparse(3))
        #expect(derived.layerIsSparse(0) == false)

        // `full_attention` names the same layer as `qwen_sparse_attention`.
        let declared = try qwen4ExpConfig(
            #"{"num_hidden_layers": 2, "layer_types": ["full_attention", "linear_attention"]}"#)
        #expect(declared.layerTypes == ["qwen_sparse_attention", "linear_attention"])

        // The PLE layer list is deduplicated and sorted.
        let ple = try qwen4ExpConfig(#"{"ple_layer_ids": [5, 1, 5, 3], "eos_token_id": 1}"#)
        #expect(ple.pleLayerIDs == [1, 3, 5])

        let sized = try qwen4ExpConfig(#"{"hidden_size": 64, "num_attention_heads": 8}"#)
        #expect(sized.resolvedHeadDim == 8)
        #expect(sized.hyperConnectionWidth == 4 * 64)
    }

    /// The norm's weights are centred at zero and applied as `1 + weight`, so a fresh one
    /// is the identity. Values captured from mlx-vlm.
    @Test("The norm is zero-centred")
    func rmsNorm() {
        let norm = Qwen4ExpRMSNorm(dimensions: 4)
        norm.update(parameters: .unflattened(["weight": MLXArray([Float(0), 0.1, -0.1, 0.2])]))

        let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
        let out = norm(x)
        eval(out)
        let expected: [Float] = [0.36515, 0.80333, 0.9859, 1.75271]
        for (got, want) in zip(out.reshaped(-1).asArray(Float.self), expected) {
            #expect(abs(got - want) < 1e-4, "\(got) != \(want)")
        }

        // A zero weight is the identity scaling, not a zeroing.
        let identity = Qwen4ExpRMSNorm(dimensions: 4)
        let plain = identity(x)
        eval(plain)
        #expect(plain.reshaped(-1).asArray(Float.self).allSatisfy { $0 != 0 })
    }

    /// With a group size the norm works within groups along the last axis, which is how one
    /// norm covers several hyper-connection streams without mixing them.
    @Test("A grouped norm does not mix across groups")
    func groupedNorm() {
        let norm = Qwen4ExpRMSNorm(dimensions: 4, groupSize: 2)
        norm.update(parameters: .unflattened(["weight": MLXArray([Float(0), 0.1, -0.1, 0.2])]))

        let out = norm(MLXArray([Float(1), 2, 3, 4], [1, 4]))
        eval(out)
        let expected: [Float] = [0.63246, 1.3914, 0.76368, 1.35765]
        for (got, want) in zip(out.reshaped(-1).asArray(Float.self), expected) {
            #expect(abs(got - want) < 1e-4, "\(got) != \(want)")
        }

        // Changing one group leaves the other untouched, which a whole-row norm would not.
        let changed = norm(MLXArray([Float(1), 2, 30, 40], [1, 4]))
        eval(changed)
        let values = changed.reshaped(-1).asArray(Float.self)
        for (index, want) in expected.prefix(2).enumerated() {
            #expect(abs(values[index] - want) < 1e-4)
        }
    }

    /// The mixer averages the gated streams. Without a combine it returns only the mixed
    /// input, which is all the MTP head takes.
    @Test("The gated residual mixes streams and optionally combines")
    func gatedResidual() throws {
        let config = try qwen4ExpConfig(
            #"{"hidden_size": 8, "hc_count": 4, "hc_lowrank": 6, "ple_embed_dim": 16}"#)

        let mixer = Qwen4ExpGatedResidual(config, useCombine: false)
        let names = Set(mixer.parameters().flattened().map(\.0))
        #expect(names.contains("hc_norm.weight"))
        #expect(names.contains("input_mix_weight_down.weight"))
        #expect(!names.contains("block_inject_weight.weight"))

        let hyper = MLXArray.zeros([1, 3, 4 * 8])
        let (mixed, streams, injection) = mixer(hyper)
        eval(mixed)
        // One hidden-width vector per token, from four streams.
        #expect(mixed.shape == [1, 3, 8])
        #expect(streams == nil)
        #expect(injection == nil)

        let combining = Qwen4ExpGatedResidual(config, useCombine: true)
        #expect(
            Set(combining.parameters().flattened().map(\.0))
                .contains("block_inject_weight.weight"))
        let (combinedMixed, combinedStreams, combinedInjection) = combining(hyper)
        eval(combinedMixed)
        #expect(combinedMixed.shape == [1, 3, 8])
        #expect(combinedStreams?.shape == [1, 3, 4 * 8])
        // One injection weight per stream, centred at one.
        let weights = try #require(combinedInjection)
        eval(weights)
        #expect(weights.shape == [1, 3, 4])
        for value in weights.reshaped(-1).asArray(Float.self) {
            #expect(abs(value - 1.0) < 1e-4)
        }
    }

}

/// The sparse-attention cache carries the indexer's unprojected keys and the positions they
/// were seen at, beside the ordinary KV. They have to move together.
@Suite("Qwen4-Exp sparse-attention cache")
struct QSAKVCacheTests {

    private func fill(_ cache: QSAKVCache, tokens: Int) throws {
        _ = cache.update(
            keys: MLXArray.zeros([1, 1, tokens, 4]),
            values: MLXArray.zeros([1, 1, tokens, 4]))
        _ = try cache.updateIndexer(
            keys: MLXArray(Int32(0) ..< Int32(tokens)).reshaped(1, tokens, 1).asType(
                .float32),
            positionIDs: MLXArray(Int32(0) ..< Int32(tokens)).reshaped(1, tokens))
    }

    /// Indexer keys and positions accumulate alongside the KV.
    @Test("The indexer state accumulates with the KV")
    func accumulates() throws {
        let cache = QSAKVCache()
        try fill(cache, tokens: 3)
        try fill(cache, tokens: 2)

        #expect(cache.offset == 5)
        #expect(cache.indexKeys?.dim(1) == 5)
        #expect(cache.indexPositionIDs?.dim(-1) == 5)
    }

    /// A rejected speculative block trims the KV; leaving the indexer's keys behind would
    /// have it select against positions that no longer exist.
    @Test("Trimming takes the indexer state with it")
    func trimKeepsIndexerInStep() throws {
        let cache = QSAKVCache()
        try fill(cache, tokens: 6)
        cache.indexBlockKeys = MLXArray.zeros([1, 2, 4])
        cache.indexBlockRatio = 4

        let trimmed = cache.trim(2)
        #expect(trimmed == 2)
        #expect(cache.offset == 4)
        #expect(cache.indexKeys?.dim(1) == 4)
        #expect(cache.indexPositionIDs?.dim(-1) == 4)

        // The surviving positions are the leading ones.
        let positions = try #require(cache.indexPositionIDs)
        eval(positions)
        #expect(positions.reshaped(-1).asArray(Int32.self) == [0, 1, 2, 3])

        // Block-compressed keys are a compression of what was just trimmed, so they are
        // dropped rather than kept consistent.
        #expect(cache.indexBlockKeys == nil)
        #expect(cache.indexBlockRatio == nil)
    }

    /// A trim of more than the cache holds takes it to empty rather than going negative.
    @Test("Over-trimming clamps")
    func overTrim() throws {
        let cache = QSAKVCache()
        try fill(cache, tokens: 3)
        #expect(cache.trim(10) == 3)
        #expect(cache.offset == 0)
        #expect(cache.indexKeys?.dim(1) == 0)
    }

    /// A sequence can start with text positions and continue with multimodal ones; the
    /// narrower side is broadcast rather than the mismatch being an error.
    @Test("Text and multimodal positions concatenate")
    func positionRanks() throws {
        let text = MLXArray(Int32(0) ..< Int32(3)).reshaped(1, 3)
        let mrope = MLXArray(Int32(0) ..< Int32(6)).reshaped(2, 1, 3)

        let promoted = try appendQSAIndexerPositions(cached: mrope, positionIDs: text)
        #expect(promoted.shape == [2, 1, 6])

        let widened = try appendQSAIndexerPositions(cached: text, positionIDs: mrope)
        #expect(widened.shape == [2, 1, 6])

        // Nothing cached yet is just the current positions.
        let first = try appendQSAIndexerPositions(cached: nil, positionIDs: text)
        #expect(first.shape == [1, 3])

        // A rank that is neither is reported rather than silently broadcast.
        #expect(throws: Qwen4ExpError.self) {
            _ = try appendQSAIndexerPositions(
                cached: MLXArray.zeros([1, 1, 1, 3]), positionIDs: text)
        }
    }
}

/// The indexer pools keys into blocks, scores them, and attends only the best few. Below
/// the budget there is nothing to choose between, so the layer stays dense.
@Suite("Qwen4-Exp sparse-attention indexer")
struct Qwen4ExpQSAIndexerTests {

    /// budget 8 over a ratio of 2 gives a block budget of 4, so five complete blocks --
    /// ten cached tokens -- is the first point where selection does anything.
    private func makeIndexer() throws -> (Qwen4ExpQSAIndexer, Qwen4ExpTextConfiguration) {
        let config = try JSONDecoder.json5().decode(
            Qwen4ExpTextConfiguration.self,
            from: Data(
                #"{"hidden_size": 32, "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8, "indexer_budget": 8, "indexer_compress_ratio": 2}"#
                    .utf8))
        let rotary = Qwen35Language.RotaryEmbedding(
            dim: config.indexerHeadDim, base: 10000, mropeSection: [3, 3, 2])
        return (Qwen4ExpQSAIndexer(config, rotary: rotary), config)
    }

    /// Below the block budget every block would be selected, so the layer stays dense and
    /// the indexer declines to build a mask at all.
    @Test("Below the budget the indexer declines")
    func belowBudgetIsDense() throws {
        let (indexer, _) = try makeIndexer()
        #expect(indexer.blockTopK == 4)

        let cache = QSAKVCache()
        // Eight tokens is four complete blocks, which is exactly the budget.
        let mask = try indexer(MLXArray.zeros([1, 8, 32]), cache: cache, positionIDs: nil)
        #expect(mask == nil)
        // The indexer still recorded its keys, so a later call sees the whole history.
        #expect(cache.indexKeys?.dim(1) == 8)
    }

    /// Past the budget it selects exactly the block budget and builds a per-token mask.
    @Test("Past the budget it selects and builds a mask")
    func selectsPastBudget() throws {
        let (indexer, _) = try makeIndexer()
        let cache = QSAKVCache()
        let hidden = (MLXArray(0 ..< (12 * 32)).asType(.float32) / 97).reshaped(1, 12, 32)

        let selection = try #require(
            try indexer.select(
                projected: indexer.indexQKProj(hidden), cache: cache, positionIDs: nil))
        // Twelve tokens is six complete blocks; four are chosen.
        #expect(selection.keyLength == 12)
        #expect(selection.selectedBlocks.shape == [1, 12, 4])

        let mask = indexer.buildMask(selection)
        eval(mask)
        #expect(mask.shape == [1, 1, 12, 12])
    }

    /// What the mask guarantees, and what it does not.
    ///
    /// Causality always holds, and a row with too few complete blocks to choose between
    /// falls back to dense. A sparse row is a different matter: its own position is only
    /// visible if the block holding it was selected, and once the tail is empty -- which it
    /// is whenever the query sits exactly on a block boundary -- nothing else covers it.
    /// That is the algorithm's behaviour, not this port's, so the test states it rather
    /// than asserting something stronger that random weights would flake on.
    @Test("The mask is causal, dense below the budget, and always keeps the tail")
    func maskShape() throws {
        let (indexer, _) = try makeIndexer()
        // Pinned weights: which blocks win is a discrete choice, and with the random
        // initializer the answer changes between runs.
        indexer.update(
            parameters: .unflattened([
                "index_qk_proj.weight":
                    (MLXArray(0 ..< (24 * 32)).asType(.float32) / 371).reshaped(24, 32)
            ]))

        let cache = QSAKVCache()
        let hidden = (MLXArray(0 ..< (12 * 32)).asType(.float32) / 97).reshaped(1, 12, 32)
        let mask = try #require(try indexer(hidden, cache: cache, positionIDs: nil))
        eval(mask)

        let values = mask.asType(.int32).reshaped(12, 12).asArray(Int32.self)
        for query in 0 ..< 12 {
            // Nothing after the query is ever visible.
            for key in (query + 1) ..< 12 {
                #expect(values[query * 12 + key] == 0, "query \(query) saw key \(key)")
            }
        }

        // Rows with at most the block budget of complete blocks attend densely, so they
        // keep every earlier position including their own.
        for query in 0 ... 8 where (query + 1) / 2 <= indexer.blockTopK {
            for key in 0 ... query {
                #expect(values[query * 12 + key] == 1, "dense row \(query) lost key \(key)")
            }
        }

        // A sparse row always keeps whatever sits past its last complete block. A query
        // sitting exactly on a block boundary has no such tail, which is the case that
        // leaves its own position to the selection.
        for query in 9 ..< 12 {
            let tailStart = ((query + 1) / 2) * 2
            guard tailStart <= query else { continue }
            for key in tailStart ... query {
                #expect(
                    values[query * 12 + key] == 1,
                    "sparse row \(query) lost tail key \(key)")
            }
        }
    }

    /// Pooled blocks are kept on the cache so a later prefill chunk only summarizes the
    /// blocks it completed, and the result must match doing it in one pass.
    @Test("Block keys are reused across chunks")
    func blockReuse() throws {
        let (indexer, _) = try makeIndexer()
        let hidden = (MLXArray(0 ..< (12 * 32)).asType(.float32) / 97).reshaped(1, 12, 32)

        // One pass over everything.
        let single = QSAKVCache()
        let whole = try #require(try indexer(hidden, cache: single, positionIDs: nil))
        eval(whole)

        // The same tokens in two chunks, which makes the second reuse the first's blocks.
        // The first has to clear the budget itself -- below it the indexer returns before
        // pooling anything, so there would be nothing cached to reuse.
        let chunked = QSAKVCache()
        _ = try indexer(hidden[0..., ..<10, 0...], cache: chunked, positionIDs: nil)
        // The attention advances the KV around the indexer, and the indexer reads that
        // offset as the past length, so a standalone test has to do the same.
        _ = chunked.update(
            keys: MLXArray.zeros([1, 1, 10, 4]), values: MLXArray.zeros([1, 1, 10, 4]))
        let reusedBlocks = chunked.indexBlockKeys
        let second = try #require(
            try indexer(hidden[0..., 10..., 0...], cache: chunked, positionIDs: nil))
        eval(second)

        #expect(reusedBlocks != nil)
        #expect(chunked.indexBlockRatio == 2)
        // Six complete blocks either way, and the tail rows agree token for token.
        #expect(chunked.indexBlockKeys?.dim(2) == 6)
        #expect(second.shape == [1, 1, 2, 12])
        let wholeTail = whole[0..., 0..., 10..., 0...].asType(.int32).reshaped(-1)
            .asArray(Int32.self)
        let chunkTail = second.asType(.int32).reshaped(-1).asArray(Int32.self)
        #expect(wholeTail == chunkTail)
    }
}

/// The sparse-attention block, which is the only kind the MTP head builds.
@Suite("Qwen4-Exp decoder layer")
struct Qwen4ExpDecoderLayerTests {

    private func config() throws -> Qwen4ExpTextConfiguration {
        try JSONDecoder.json5().decode(
            Qwen4ExpTextConfiguration.self,
            from: Data(
                #"{"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "hc_count": 2, "hc_lowrank": 6, "layer_types": ["qwen_sparse_attention"], "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16, "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8, "indexer_budget": 8, "indexer_compress_ratio": 2, "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]}}"#
                    .utf8))
    }

}

// MARK: - Per-layer embeddings

/// Fills every float parameter with the same index-derived pattern mlx-vlm's reference run
/// used, so the numbers below can be compared without shipping a checkpoint.
private func fillDeterministically(_ module: Module) {
    var updates: [String: MLXArray] = [:]
    for (key, value) in module.parameters().flattened() where value.dtype == .float32 {
        let k = MLXArray(0 ..< value.size)
        updates[key] =
            (((k * 37) % 71) - 35).asType(.float32).reshaped(value.shape) / 100.0
    }
    module.update(parameters: .unflattened(updates))
}

private func deterministic(_ shape: [Int]) -> MLXArray {
    let count = shape.reduce(1, *)
    let k = MLXArray(0 ..< count)
    return (((k * 37) % 71) - 35).asType(.float32).reshaped(shape) / 100.0
}

private let plePLEConfig = #"""
    {
      "hidden_size": 8, "num_hidden_layers": 4, "num_attention_heads": 4,
      "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64,
      "hc_count": 2, "hc_lowrank": 8, "ple_layer_ids": [1, 2], "ple_embed_dim": 8,
      "ple_conv_kernel_size": 2, "ngram_size": 3, "heads_per_ngram": 2,
      "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
      "split_ngram_parts": 3, "seed": 1234, "eos_token_id": 1,
      "linear_num_value_heads": 4, "linear_num_key_heads": 2,
      "linear_key_head_dim": 32, "linear_value_head_dim": 32,
      "linear_conv_kernel_dim": 2, "num_experts": 4, "num_experts_per_tok": 2,
      "shared_expert_intermediate_size": 8, "moe_intermediate_size": 8
    }
    """#

@Suite("Qwen4-Exp per-layer embeddings")
struct Qwen4ExpPLETests {

    /// The hash multipliers are derived from the seed and the layer's position in
    /// `ple_layer_ids`, never loaded, so they have to agree exactly with upstream's or the
    /// same checkpoint would address different rows. Values from mlx-vlm at 61bcc727.
    @Test("Hash multipliers match upstream")
    func multipliers() {
        #expect(
            Qwen4ExpHash.layerMultipliers(
                unigramVocabularySize: 64, ngramSize: 3, pleIndex: 0, seed: 1234)
                == [112_381_549_946_653_559, 107_512_210_695_146_523, 55_314_113_489_879_299])
        #expect(
            Qwen4ExpHash.layerMultipliers(
                unigramVocabularySize: 64, ngramSize: 3, pleIndex: 1, seed: 1234)
                == [125_324_136_723_312_099, 74_677_362_759_709_025, 33_105_235_034_841_755])

        // Every multiplier is odd, and bounded so `token * multiplier` cannot overflow.
        let bounded = Qwen4ExpHash.layerMultipliers(
            unigramVocabularySize: 151_936, ngramSize: 4, pleIndex: 7, seed: 1234)
        #expect(bounded.allSatisfy { $0 % 2 == 1 })
        #expect(
            bounded.allSatisfy { $0.multipliedReportingOverflow(by: 151_936).overflow == false })
    }

    /// Each n-gram head takes the next prime after the base, counting across every PLE
    /// layer, so a head's vocabulary depends on the layer it belongs to.
    @Test("Head vocabularies are successive primes")
    func headTables() {
        #expect(
            (0 ..< 8).map { Qwen4ExpHash.nthPrimeAfter(999, count: $0 + 1) }
                == [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049])

        let first = Qwen4ExpHash.headTables(
            ngramHeads: 4, vocabularySizeBase: 1000, pleIndex: 0)
        #expect(first.sizes == [1009, 1013, 1019, 1021])
        #expect(first.offsets == [0, 1009, 2022, 3041])
        #expect(first.total == 4062)

        // The second PLE layer continues the sequence rather than restarting it.
        let second = Qwen4ExpHash.headTables(
            ngramHeads: 4, vocabularySizeBase: 1000, pleIndex: 1)
        #expect(second.sizes == [1031, 1033, 1039, 1049])
    }

    /// Rows are split evenly with the remainder going to the leading shards, which is the
    /// layout the checkpoint's shard shapes assume.
    @Test("Shards split rows evenly, remainder first")
    func shardLayout() {
        let table = Qwen4ExpShardedEmbedding(rows: 10, dimensions: 3, shardCount: 3)
        #expect(table.shards.map { $0.weight.dim(0) } == [4, 3, 3])
        #expect(table.shardOffsets == [0, 4, 7, 10])

        // A lookup routes each row to its owning shard and keeps the caller's order.
        for (index, shard) in table.shards.enumerated() {
            shard.update(
                parameters: .unflattened([
                    "weight": MLXArray.full(
                        [shard.weight.dim(0), 3], values: MLXArray(Float(index)))
                ]))
        }
        let gathered = table(MLXArray([Int32(9), 0, 5, 3, 4]))
        eval(gathered)
        #expect(gathered.shape == [5, 3])
        #expect(gathered[0..., 0].asArray(Float.self) == [2, 0, 1, 0, 1])
    }

    /// The shift that builds an n-gram's history refuses to cross an end-of-sequence
    /// token: a position fewer than `shift` tokens into its own segment takes the
    /// end-of-sequence id instead. Values from mlx-vlm at 61bcc727.
    @Test("The history shift stops at an end-of-sequence token")
    func shiftStopsAtEOS() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        let embedding = Qwen4ExpNGramEmbedding(config, embeddingDim: 8, pleIndex: 0)

        let tokens = MLXArray([Int64(5), 1, 7, 9], [1, 4])
        let expected: [[Int64]] = [[5, 1, 7, 9], [1, 5, 1, 7], [1, 1, 1, 1]]
        for (shift, want) in expected.enumerated() {
            let got = embedding.shiftRightIgnoringEOS(tokens, shift: shift)
            eval(got)
            #expect(got.reshaped(-1).asArray(Int64.self) == want, "shift \(shift)")
        }
    }

    /// One PLE layer, forward, against mlx-vlm at 61bcc727 with every float parameter
    /// filled from the same index pattern.
    @Test("A per-layer embedding matches upstream")
    func forward() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        let layer = Qwen4ExpPLELayer(config, pleIndex: 0)
        fillDeterministically(layer)

        let hidden = deterministic([1, 5, config.hyperConnectionWidth])
        let cache = ArraysCache(size: 4)
        let out = layer(
            hidden[0..., ..<4], inputIDs: MLXArray([Int32(5), 1, 7, 9], [1, 4]),
            cache: cache, mask: nil)
        eval(out)

        #expect(out.shape == [1, 4, config.hyperConnectionWidth])
        let prefill: [Float] = [
            0.050553, 0.07744, 0.087864, -0.10381, -0.086112, -0.105614, 0.120611,
            0.205322, 0.135727, 0.241506, 0.212081, -0.201149, 0.109984, 0.13787,
            -0.066631, -0.097248,
        ]
        for (got, want) in zip(out[0, 0].asArray(Float.self), prefill) {
            #expect(abs(got - want) < 2e-5, "\(got) != \(want)")
        }

        // The token history keeps the last `ngram_size - 1` ids, and the convolution its
        // dilated window.
        #expect(cache[3]!.reshaped(-1).asArray(Int64.self) == [7, 9])
        #expect(cache[2]!.shape == [1, 3, config.hyperConnectionWidth])

        let step = layer(
            hidden[0..., 4...], inputIDs: MLXArray([Int32(11)], [1, 1]), cache: cache,
            mask: nil)
        eval(step)
        let expected: [Float] = [
            -0.071815, -0.018304, -0.090353, 0.115841, 0.162471, 0.213827, -0.237294,
            -0.235747, -0.13288, -0.164819, -0.13016, 0.196733, -0.051353, -0.078683,
            0.037531, 0.027169,
        ]
        for (got, want) in zip(step.reshaped(-1).asArray(Float.self), expected) {
            #expect(abs(got - want) < 2e-5, "\(got) != \(want)")
        }
        #expect(cache[3]!.reshaped(-1).asArray(Int64.self) == [9, 11])
    }

    /// A stepped decode and one unchunked forward agree to the same bound upstream itself
    /// holds: the key and query projections run at a different sequence length in the two
    /// regimes, and MLX's float32 matmul is not shape-invariant. Measured at 3.1e-4 on
    /// mlx-vlm at 61bcc727 for this fixture, so the bound is not a port artifact.
    @Test("Stepping matches an unchunked forward")
    func steppedMatchesUnchunked() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        let layer = Qwen4ExpPLELayer(config, pleIndex: 0)
        fillDeterministically(layer)

        let hidden = deterministic([1, 5, config.hyperConnectionWidth])
        let ids = MLXArray([Int32(5), 1, 7, 9, 11], [1, 5])

        let flat = layer(hidden, inputIDs: ids, cache: ArraysCache(size: 4), mask: nil)
        let cache = ArraysCache(size: 4)
        _ = layer(hidden[0..., ..<4], inputIDs: ids[0..., ..<4], cache: cache, mask: nil)
        let step = layer(hidden[0..., 4...], inputIDs: ids[0..., 4...], cache: cache, mask: nil)
        eval(flat, step)

        let difference = abs(flat[0..., 4...] - step).max().item(Float.self)
        #expect(difference < 1e-3, "chunked and flat diverged by \(difference)")
    }

    /// `ple_layer_ids` is one-based, and a layer's position in the sorted list is what
    /// seeds its hash multipliers.
    @Test("PLE layer ids are one-based")
    func pleIndices() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        #expect(config.pleLayerIDs == [1, 2])
        #expect(config.pleIndex(forLayer: 0) == 0)
        #expect(config.pleIndex(forLayer: 1) == 1)
        #expect(config.pleIndex(forLayer: 2) == nil)

        let layer = Qwen4ExpDecoderLayer(config, layerIndex: 0)
        #expect(layer.ple != nil)
        #expect(Qwen4ExpDecoderLayer(config, layerIndex: 2).ple == nil)
    }
}

// MARK: - Text model

@Suite("Qwen4-Exp text model")
struct Qwen4ExpTextModelTests {

    /// The checkpoint names each n-gram shard in the key and stacks the gate and up halves
    /// of the expert projection; `conv1d` is stored channels-last.
    @Test("Sanitize moves the checkpoint onto the tree")
    func sanitize() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        let model = Qwen4ExpLanguageModel(config)

        let weights: [String: MLXArray] = [
            "mtp.layers.0.mlp.gate.weight": MLXArray.zeros([4, 8]),
            "model.layers.0.ple.ple_embedding.ngram_embedding.shard_2.weight":
                MLXArray.zeros([1354, 2]),
            "model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([160, 1, 2]),
            "model.layers.3.mlp.experts.gate_up_proj": MLXArray.zeros([4, 16, 8]),
            "model.layers.3.mlp.experts.down_proj": MLXArray.zeros([4, 8, 8]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains { $0.hasPrefix("mtp.") } == false)
        #expect(
            sanitized["model.layers.0.ple.ple_embedding.ngram_embedding.shards.2.weight"]
                != nil)
        #expect(
            sanitized["model.layers.0.linear_attn.conv1d.weight"]?.shape == [160, 2, 1])
        #expect(sanitized["model.layers.3.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 8, 8])
        #expect(sanitized["model.layers.3.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 8, 8])
        #expect(sanitized["model.layers.3.mlp.switch_mlp.down_proj.weight"]?.shape == [4, 8, 8])
        #expect(sanitized["model.layers.3.mlp.experts.gate_up_proj"] == nil)
    }

    /// Every parameter the module publishes has to be somewhere in the checkpoint, so the
    /// derived hash tables must not be published as parameters.
    @Test("The derived hash tables are not parameters")
    func derivedTablesAreNotParameters() throws {
        let config = try qwen4ExpConfig(plePLEConfig)
        let layer = Qwen4ExpPLELayer(config, pleIndex: 0)
        let published = Set(layer.parameters().flattened().map(\.0))

        #expect(published.contains("ple_embedding.layer_multipliers") == false)
        #expect(published.contains("ple_embedding.ngram_heads_vocab_sizes") == false)
        #expect(published.contains("ple_embedding.ngram_heads_offsets") == false)
        #expect(published.contains("ple_embedding.ngram_embedding.shards.0.weight"))
        #expect(published.contains("norm_key.weight"))
        #expect(published.contains("conv1d.weight"))
    }

    /// Configurations the architecture cannot express are refused rather than built into a
    /// model whose weights would never load.
    @Test("Invalid configurations are refused")
    func refusals() {
        let bad = [
            #"{"num_hidden_layers": 2, "layer_types": ["linear_attention"]}"#,
            #"{"num_hidden_layers": 1, "layer_types": ["sliding_attention"]}"#,
            #"{"hc_count": 1}"#,
            #"{"output_gate_type": "gelu"}"#,
            #"{"indexer_kv_heads": 2}"#,
            #"{"indexer_budget": 2050, "indexer_compress_ratio": 4}"#,
            #"{"ple_embed_dim": 100, "ngram_size": 3, "heads_per_ngram": 8}"#,
            #"{"num_hidden_layers": 4, "ple_layer_ids": [5], "eos_token_id": 1}"#,
            #"{"num_hidden_layers": 4, "ple_layer_ids": [4], "eos_token_id": 1}"#,
            #"{"num_hidden_layers": 4, "ple_layer_ids": [1]}"#,
        ]
        for json in bad {
            #expect(throws: (any Error).self, "accepted \(json)") {
                try qwen4ExpConfig(json)
            }
        }
    }
}

extension Qwen4ExpPLETests {
    @Test("Image prefill preserves per-layer token IDs and multimodal decode positions")
    func imageContinuation() throws {
        var text = try #require(
            JSONSerialization.jsonObject(with: Data(plePLEConfig.utf8)) as? [String: Any])
        text["indexer_n_heads"] = 2
        text["indexer_head_dim"] = 8
        text["indexer_budget"] = 8
        text["indexer_compress_ratio"] = 2
        text["rope_parameters"] = [
            "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
        ]
        let object: [String: Any] = [
            "model_type": "qwen4_exp", "text_config": text,
            "image_token_id": 62, "video_token_id": 63,
            "vision_start_token_id": 60, "vision_end_token_id": 61,
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "out_hidden_size": 8,
                "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [],
            ],
        ]
        let configuration = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        let model = Qwen4Exp(configuration)
        fillDeterministically(model)
        let pixels = deterministic([16, 12])
        func input(_ ids: [Int]) -> LMInput {
            LMInput(
                text: .init(tokens: MLXArray(ids)[.newAxis]),
                image: .init(pixels: pixels, frames: [THW(1, 4, 4)]))
        }
        let prompt = [1, 60, 62, 62, 62, 62, 61, 8]
        let fullCache = try model.newCache(configuration: nil)
        guard
            case .logits(let full) = try model.prepare(
                input(prompt + [9]), cache: fullCache, state: nil, prefill: .init())
        else {
            Issue.record("Expected image logits")
            return
        }
        eval(full.logits)
        for step in [2, 32] {
            let cache = try model.newCache(configuration: nil)
            guard
                case .logits(let prefilled) = try model.prepare(
                    input(prompt), cache: cache, state: nil, prefill: .init(stepSize: step))
            else {
                Issue.record("Expected image logits")
                return
            }
            let decoded = model(.init(tokens: MLXArray([9])), cache: cache, state: prefilled.state)
            eval(decoded.logits)
            #expect(abs(decoded.logits[0, -1] - full.logits[0, -1]).max().item(Float.self) < 1e-3)
        }
    }
}
