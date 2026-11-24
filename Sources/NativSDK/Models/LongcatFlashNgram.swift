import Foundation
import MLX
import MLXNN

/// Word embeddings plus one narrow embedding per (neighbour count, split) pair, indexed by
/// a hash of the n-gram ending at each position.
class LongcatNgramEmbedding: Module {

    let vocabularySize: Int
    let modulusBase: Int
    let splits: Int
    let neighbours: Int

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "embedders") var embedders: [Embedding]
    @ModuleInfo(key: "post_projs") var postProjections: [Linear]

    /// Per embedder, the powers of the vocabulary size taken modulo *that embedder's*
    /// modulus. Held as plain integers so nothing lands in the parameter tree.
    let vocabularyModuli: [[Int]]
    let embedderModuli: [Int]

    /// Takes plain sizes rather than a configuration: `longcat_flash_sparse` builds the
    /// same embedding from keys named `oe_*` instead of `ngram_*`.
    init(
        vocabularySize: Int, hiddenSize: Int, vocabularyRatio: Int, splits: Int,
        neighbours: Int
    ) {
        self.vocabularySize = vocabularySize
        self.modulusBase = vocabularyRatio * vocabularySize
        self.splits = splits
        self.neighbours = neighbours

        let count = splits * (neighbours - 1)
        precondition(count > 0, "the split and neighbour counts must give at least one table")
        let width = hiddenSize / count

        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: hiddenSize)

        var moduli: [Int] = []
        var tables: [Embedding] = []
        var projections: [Linear] = []
        for index in 0 ..< count {
            let modulus = modulusBase + index * 2 + 1
            moduli.append(modulus)
            tables.append(Embedding(embeddingCount: modulus, dimensions: width))
            projections.append(Linear(width, hiddenSize, bias: false))
        }
        _embedders.wrappedValue = tables
        _postProjections.wrappedValue = projections
        self.embedderModuli = moduli

        var powers: [[Int]] = []
        for neighbour in 2 ... neighbours {
            for split in 0 ..< splits {
                let index = (neighbour - 2) * splits + split
                let modulus = modulusBase + index * 2 + 1
                var running = 1
                var terms: [Int] = []
                for _ in 0 ..< (neighbour - 1) {
                    running = (running * self.vocabularySize) % modulus
                    terms.append(running)
                }
                powers.append(terms)
            }
        }
        self.vocabularyModuli = powers

        super.init()
    }

    /// The sequence shifted right by `distance`, zero filled -- and all zeros when the
    /// sequence is no longer than the shift.
    static func shiftedRight(_ x: MLXArray, by distance: Int) -> MLXArray {
        guard distance > 0 else { return x }
        let length = x.dim(-1)
        if length <= distance {
            return MLXArray.zeros(x.shape, dtype: x.dtype)
        }
        return concatenated(
            [
                MLXArray.zeros([x.dim(0), distance], dtype: x.dtype),
                x[0..., ..<(length - distance)],
            ], axis: -1)
    }

    func callAsFunction(_ inputIds: MLXArray, cache: ArraysCache?) -> MLXArray {
        let length = inputIds.dim(-1)
        let ids = inputIds.asType(.int64)

        var context = ids
        if let cache {
            if let previous = cache[0] {
                context = concatenated([previous, ids], axis: -1)
            }
            let keep = max(0, context.dim(-1) - neighbours + 1)
            cache[0] = context[0..., keep...]
        }

        var x = wordEmbeddings(inputIds)

        var shifted: [Int: MLXArray] = [:]
        for neighbour in 2 ... neighbours {
            shifted[neighbour] = Self.shiftedRight(context, by: neighbour - 1)
        }

        for neighbour in 2 ... neighbours {
            for split in 0 ..< splits {
                let index = (neighbour - 2) * splits + split
                let modulus = embedderModuli[index]
                let powers = vocabularyModuli[index]

                var hashed = context
                for k in 2 ... neighbour {
                    hashed = hashed + shifted[k]! * powers[k - 2]
                }
                let rows = (hashed % modulus)[0..., (context.dim(-1) - length)...]
                x = x + postProjections[index](embedders[index](rows))
            }
        }

        return x / Float(1 + splits * (neighbours - 1))
    }
}

public class LongcatFlashNgramModelInner: Module {

    @ModuleInfo(key: "ngram_embeddings") var ngramEmbeddings: LongcatNgramEmbedding

    let layers: [LongcatFlashBlock]
    let norm: RMSNorm

    public init(_ args: LongcatFlashNgramConfiguration) {
        precondition(args.base.vocabularySize > 0)
        _ngramEmbeddings.wrappedValue = LongcatNgramEmbedding(
            vocabularySize: args.base.vocabularySize, hiddenSize: args.base.hiddenSize,
            vocabularyRatio: args.ngramVocabularyRatio, splits: args.embeddingSplits,
            neighbours: args.embeddingNeighbours)
        self.layers = (0 ..< args.base.layerCount).map { _ in
            LongcatFlashBlock(args.base)
        }
        self.norm = RMSNorm(dimensions: args.base.hiddenSize, eps: args.base.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = ngramEmbeddings(inputs, cache: cache?.first as? ArraysCache)

        let first = (cache?.count ?? 0) > 1 ? (cache?[1] as? CacheList)?[0] : nil
        let mask: MLXArray? = createAttentionMask(h: h, cache: first.map { [$0] })

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i + 1] as? CacheList)
        }

        return norm(h)
    }
}

public class LongcatFlashNgramModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: LongcatFlashNgramModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: LongcatFlashNgramConfiguration

    public init(_ args: LongcatFlashNgramConfiguration) {
        self.configuration = args
        self.vocabularySize = args.base.vocabularySize
        self.model = LongcatFlashNgramModelInner(args)
        _lmHead.wrappedValue = Linear(
            args.base.hiddenSize, args.base.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// One array slot for the n-gram context, then two caches per layer.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        [ArraysCache(size: 1)]
            + model.layers.map { _ in
                CacheList(
                    MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
            }
    }

    /// The base model's remapping, plus the word table's move under the n-gram embedding.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = LongcatFlashModel.sanitize(
            weights: weights, configuration: configuration.base)
        if let table = sanitized.removeValue(forKey: "model.embed_tokens.weight") {
            sanitized["model.ngram_embeddings.word_embeddings.weight"] = table
        }
        return sanitized
    }
}

public struct LongcatFlashNgramConfiguration: Decodable, Sendable {

    /// Everything the decoder needs, decoded from the same object.
    var base: LongcatFlashConfiguration

    var ngramVocabularyRatio: Int
    var embeddingNeighbours: Int
    var embeddingSplits: Int

    enum CodingKeys: String, CodingKey {
        case ngramVocabularyRatio = "ngram_vocab_size_ratio"
        case embeddingNeighbours = "emb_neighbor_num"
        case embeddingSplits = "emb_split_num"
    }

    public init(from decoder: Decoder) throws {
        base = try LongcatFlashConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ngramVocabularyRatio =
            try c.decodeIfPresent(Int.self, forKey: .ngramVocabularyRatio) ?? 78
        embeddingNeighbours =
            try c.decodeIfPresent(Int.self, forKey: .embeddingNeighbours) ?? 4
        embeddingSplits = try c.decodeIfPresent(Int.self, forKey: .embeddingSplits) ?? 4
    }
}

extension LongcatFlashNgramModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
