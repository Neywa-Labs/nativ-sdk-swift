import Foundation
import MLX
import MLXNN

/// Bucket layout of the n-gram hash tables.
///
/// A position is hashed as `maxNgramSize - 1` n-grams (2-gram up to
/// `maxNgramSize`-gram), each split over `heads` heads. Every (n-gram size, head) pair
/// owns its own prime-sized bucket range in the layer's table; the primes are drawn in
/// order and never reused, which keeps the ranges disjoint.
public struct DeepSeekV41EngramLayout: Sendable {
    public let maxNgramSize: Int
    public let layerIds: [Int]
    public let numEmbeddings: [Int]
    /// `[layer][n-gram size][head]`.
    public let primes: [[[Int]]]
    public let heads: Int
    public let headDim: Int

    public var hashColumns: Int { (maxNgramSize - 1) * heads }

    /// The number of buckets a layer's table must have: the ranges are disjoint and
    /// contiguous, so the highest id one layer can produce is the sum of its primes
    /// minus one. The released `engram_num_embeddings` are exactly these sums.
    public func requiredEmbeddings(layer: Int) -> Int {
        primes[layer].reduce(0) { $0 + $1.reduce(0, +) }
    }

    /// A table smaller than its hash space reads rows that are not there, and MLX
    /// answers an out-of-bounds gather with zeros rather than an error, so the model
    /// would run and quietly contribute nothing.
    public func validate() throws {
        guard numEmbeddings.count == layerIds.count else {
            throw DeepSeekV41EngramError.tableCount(numEmbeddings.count, layerIds.count)
        }
        for index in layerIds.indices where numEmbeddings[index] < requiredEmbeddings(layer: index)
        {
            throw DeepSeekV41EngramError.tableTooSmall(
                layerIds[index], numEmbeddings[index], requiredEmbeddings(layer: index))
        }
    }

    /// Trial-division primality, for table-size integers.
    static func isPrime(_ n: Int) -> Bool {
        if n < 2 { return false }
        if n % 2 == 0 { return n == 2 }
        var divisor = 3
        while divisor * divisor <= n {
            if n % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// The smallest prime above `start` that has not been handed out yet.
    static func nextPrime(after start: Int, seen: Set<Int>) -> Int {
        var candidate = start + 1
        while !isPrime(candidate) || seen.contains(candidate) {
            candidate += 1
        }
        return candidate
    }

    public init?(
        layerIds: [Int], maxNgramSize: Int, heads: Int, headDim: Int,
        vocabularySize: Int, numEmbeddings: [Int]
    ) {
        guard !layerIds.isEmpty else { return nil }
        self.maxNgramSize = maxNgramSize
        self.layerIds = layerIds
        self.numEmbeddings = numEmbeddings
        self.heads = heads
        self.headDim = headDim

        var primes: [[[Int]]] = []
        var seen: Set<Int> = []
        for _ in layerIds {
            var perNgram: [[Int]] = []
            for _ in 0 ..< (maxNgramSize - 1) {
                var sizes: [Int] = []
                var current = vocabularySize - 1
                for _ in 0 ..< heads {
                    current = DeepSeekV41EngramLayout.nextPrime(after: current, seen: seen)
                    seen.insert(current)
                    sizes.append(current)
                }
                perNgram.append(sizes)
            }
            primes.append(perNgram)
        }
        self.primes = primes
    }
}

/// Maps each position to the hash ids of the n-grams ending there.
///
/// Ids go through the compressed table, then each position is hashed with the
/// `maxNgramSize - 1` tokens before it. Look-back stops at the start of the sequence and
/// at any dead token (an image span), so an n-gram never spans one. The token history
/// lives on the caller's cache rather than on this object, so two generations do not
/// share it.
public final class DeepSeekV41NgramHashState {
    static let dead: Int32 = -1

    public let layout: DeepSeekV41EngramLayout
    public let padId: Int

    /// `[layer][n-gram size][head]` flattened to `[layer, n-gram size, head]`.
    let primes: MLXArray
    /// `[layer, hashColumns]`, the running start of each bucket range.
    let offsets: MLXArray
    /// `[layer, maxNgramSize]`.
    let multipliers: MLXArray
    let tokenMap: MLXArray

    /// One multiplier per (layer, lookback), from a per-layer generator so layers hash
    /// differently. Kept odd, and bounded so `tokenId * multiplier` cannot overflow.
    public static func hashMultipliers(
        layerIds: [Int], maxNgramSize: Int, vocabularySize: Int
    ) -> [[Int64]] {
        let bound = max(1, (Int64.max / Int64(vocabularySize)) / 2)
        return layerIds.map { layerId in
            var generator = NumPyPCG64(seed: UInt64(10007 * layerId))
            return generator.integers(low: 0, high: bound, count: maxNgramSize)
                .map { $0 * 2 + 1 }
        }
    }

    /// `tokenMap` is the released `engram_token_map.json`. Deriving it instead would mean
    /// decoding and normalizing the whole vocabulary through the same NFKC/strip-accents/
    /// lowercase pipeline the reference uses; upstream prefers the shipped map for that
    /// reason, and this port requires it.
    public init(
        layout: DeepSeekV41EngramLayout, tokenMap: [Int], padTokenId: Int,
        compressedVocabularySize: Int
    ) throws {
        let vocabularySize = (tokenMap.max() ?? -1) + 1
        guard vocabularySize == compressedVocabularySize else {
            throw DeepSeekV41EngramError.compressedVocabularyMismatch(
                vocabularySize, compressedVocabularySize)
        }
        guard padTokenId >= 0, padTokenId < tokenMap.count else {
            throw DeepSeekV41EngramError.padTokenOutOfRange(padTokenId, tokenMap.count)
        }
        self.layout = layout
        self.padId = tokenMap[padTokenId]

        let flat: [[Int]] = layout.primes.map { $0.flatMap { $0 } }
        var runningOffsets: [Int32] = []
        for sizes in flat {
            var running: Int32 = 0
            for size in sizes {
                runningOffsets.append(running)
                running += Int32(size)
            }
        }
        self.offsets = MLXArray(runningOffsets)
            .reshaped(flat.count, layout.hashColumns).asType(.int64)

        self.primes = MLXArray(layout.primes.flatMap { $0.flatMap { $0 } }.map(Int32.init))
            .reshaped(layout.layerIds.count, layout.maxNgramSize - 1, layout.heads)
            .asType(.int64)

        let rows = DeepSeekV41NgramHashState.hashMultipliers(
            layerIds: layout.layerIds, maxNgramSize: layout.maxNgramSize,
            vocabularySize: vocabularySize)
        self.multipliers = MLXArray(rows.flatMap { $0 })
            .reshaped(rows.count, layout.maxNgramSize)
        self.tokenMap = MLXArray(tokenMap.map(Int32.init)).asType(.int64)
    }

    /// `tokenMask` is `[B, L]`, false for tokens that take no part in an n-gram (an image
    /// span). Returns the hash ids, shaped `[B, L, layers, hashColumns]`. `history` is
    /// the caller's token history and is extended in place.
    public func callAsFunction(
        _ inputIds: MLXArray, startPosition: Int, history: inout MLXArray?,
        tokenMask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length) = (inputIds.dim(0), inputIds.dim(1))
        var compressed = take(tokenMap, inputIds.flattened(), axis: 0)
            .reshaped(inputIds.shape)
        if let tokenMask {
            compressed = MLX.where(tokenMask, compressed, MLXArray(Self.dead).asType(.int64))
        }

        let needed = startPosition + length
        var cache = history ?? MLXArray.zeros([batch, needed], dtype: .int64)
        if cache.dim(0) < batch {
            cache = concatenated(
                [cache, MLXArray.zeros([batch - cache.dim(0), cache.dim(1)], dtype: .int64)],
                axis: 0)
        }
        if cache.dim(1) < needed {
            cache = concatenated(
                [cache, MLXArray.zeros([cache.dim(0), needed - cache.dim(1)], dtype: .int64)],
                axis: 1)
        }

        var parts: [MLXArray] = []
        if startPosition > 0 { parts.append(cache[0 ..< batch, 0 ..< startPosition]) }
        parts.append(compressed.asType(.int64))
        if cache.dim(1) > needed { parts.append(cache[0 ..< batch, needed...]) }
        let head = parts.count > 1 ? concatenated(parts, axis: 1) : parts[0]
        cache = cache.dim(0) > batch ? concatenated([head, cache[batch...]], axis: 0) : head
        history = cache

        let current = cache[0 ..< batch]
        let positions = broadcast(
            MLXArray(startPosition ..< needed), to: [batch, length])
        var tokens: [MLXArray] = []
        var blocked = MLXArray.zeros([batch, length], dtype: .bool)
        for shift in 0 ..< layout.maxNgramSize {
            let index = clip(positions - shift, min: 0, max: current.dim(1) - 1)
            let source = takeAlong(current, index, axis: 1)
            blocked = blocked .|| (positions .< shift) .|| (source .== Self.dead)
            tokens.append(MLX.where(blocked, MLXArray(Int64(padId)), source))
        }

        let stackedTokens = stacked(tokens, axis: -1)
        let products =
            stackedTokens[0..., 0..., .newAxis, 0...]
            * multipliers[.newAxis, .newAxis, 0..., 0...]
        var rolling = products[.ellipsis, 0]
        var hashes: [MLXArray] = []
        for index in 1 ..< layout.maxNgramSize {
            rolling = rolling ^ products[.ellipsis, index]
            hashes.append(rolling[.ellipsis, .newAxis] % primes[0..., index - 1])
        }
        return concatenated(hashes, axis: -1) + offsets
    }
}

public enum DeepSeekV41EngramError: LocalizedError {
    case compressedVocabularyMismatch(Int, Int)
    case padTokenOutOfRange(Int, Int)
    case tableCount(Int, Int)
    case tableTooSmall(Int, Int, Int)

    public var errorDescription: String? {
        switch self {
        case .compressedVocabularyMismatch(let derived, let declared):
            "the token map compresses to \(derived) ids, but the config declares \(declared)"
        case .padTokenOutOfRange(let padTokenId, let count):
            "engram_pad_token_id \(padTokenId) is outside a token map of \(count)"
        case .tableCount(let tables, let layers):
            "`engram_num_embeddings` has \(tables) entries for \(layers) engram layers"
        case .tableTooSmall(let layer, let given, let required):
            "engram layer \(layer) declares \(given) buckets but its primes span \(required)"
        }
    }
}

/// Writes an n-gram lookup into the residual stream, gated by how well it matches that
/// stream.
///
/// The hash ids fetch `hashColumns` rows; `wkv` turns them into one key per hc copy plus
/// a shared value. The gate is a normalized dot product of stream against key.
public final class DeepSeekV41Engram: Module {
    public let layerId: Int
    let layerHashIndex: Int
    let dimensions: Int
    let hcMultiplier: Int
    let headDim: Int
    let eps: Float

    static let clampValue: Float = 1e-6

    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray

    public init(
        layout: DeepSeekV41EngramLayout, layerId: Int, hiddenSize: Int,
        hcMultiplier: Int, rmsNormEps: Float
    ) {
        self.layerId = layerId
        self.layerHashIndex = layout.layerIds.firstIndex(of: layerId) ?? 0
        self.dimensions = hiddenSize
        self.hcMultiplier = hcMultiplier
        self.headDim = layout.headDim
        self.eps = rmsNormEps

        _embed.wrappedValue = Embedding(
            embeddingCount: layout.numEmbeddings[layerHashIndex],
            dimensions: layout.headDim)
        _wkv.wrappedValue = Linear(
            layout.hashColumns * layout.headDim, hiddenSize * (hcMultiplier + 1),
            bias: false)
        _qWeight.wrappedValue = MLXArray.ones([hcMultiplier, hiddenSize])
        _kWeight.wrappedValue = MLXArray.ones([hcMultiplier, hiddenSize])
    }

    /// `x` is `[B, L, hcMult, dim]` and `hashIds` is `[B, L, hashColumns]`. `tokenMask`
    /// is `[B, L]`; false shuts the gate so those positions pass through untouched.
    public func callAsFunction(
        _ x: MLXArray, hashIds: MLXArray, tokenMask: MLXArray? = nil
    ) -> MLXArray {
        let dtype = x.dtype
        let columns = hashIds.dim(-1)
        let embeddings = embed(hashIds).asType(dtype)
            .reshaped(hashIds.shape.dropLast() + [columns * headDim])
        let kv = wkv(DeepSeekV41FakeQuant.fp8UE8M0(embeddings))
        let split = hcMultiplier * dimensions
        let key = kv[.ellipsis, ..<split]
            .reshaped(kv.shape.dropLast() + [hcMultiplier, dimensions])
        let value = kv[.ellipsis, split...]

        let score = gateScore(x, key: key)
        return applyGate(x, value: value, gate: sigmoid(score), tokenMask: tokenMask)
    }

    /// The eps and the head scale are passed in rather than captured: `compile` freezes
    /// a captured Swift scalar at its first call and would keep one model's values for
    /// every model loaded after it.
    private static let compiledGateScore:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
            compile { x, key, qWeight, kWeight, eps, scale in
                let h = x.asType(.float32)
                let key = key.asType(.float32)
                let weight = qWeight.asType(.float32) * kWeight.asType(.float32)
                let rstd =
                    rsqrt(h.square().mean(axis: -1) + eps)
                    * rsqrt(key.square().mean(axis: -1) + eps)
                let dot = (h * weight * key).sum(axis: -1) * rstd * scale
                let magnitude = sqrt(maximum(abs(dot), DeepSeekV41Engram.clampValue))
                return MLX.where(dot .>= 0, magnitude, -magnitude)
            }

    func gateScore(_ x: MLXArray, key: MLXArray) -> MLXArray {
        DeepSeekV41Engram.compiledGateScore(
            x, key, qWeight, kWeight, MLXArray(eps),
            MLXArray(pow(Float(dimensions), -0.5)))
    }

    func applyGate(
        _ x: MLXArray, value: MLXArray, gate: MLXArray, tokenMask: MLXArray?
    ) -> MLXArray {
        var gate = gate
        if let tokenMask {
            gate = MLX.where(tokenMask[.ellipsis, .newAxis], gate, MLXArray(Float(0)))
        }
        let expanded = value.asType(.float32)[0..., 0..., .newAxis, 0...]
        return (x.asType(.float32) + gate[.ellipsis, .newAxis] * expanded).asType(x.dtype)
    }
}
