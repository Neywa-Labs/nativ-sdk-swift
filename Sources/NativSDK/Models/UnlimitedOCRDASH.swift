// DASH inference adapted from mlx-vlm and Baidu's Unlimited-OCR-2 (MIT).
import Foundation
import MLX
import MLXNN

struct UnlimitedOCRSparseConfiguration: Decodable, Sendable {
    private let trainingStage: String?
    private let useSparseAttention: Bool?
    private let dashStage0: String?
    private let sparseAttentionFreq: [Int]?
    private let sparseNumIndexHeads: Int?
    private let sparseBlockSize: Int?
    private let sparseLocalBlock: Int?
    private let sparseIndexDim: Int?
    private let dashTopkBlocks: Int?
    private let dashSigma: Float?
    private let dashImageOnly: Bool?
    private let imageTokenId: Int?
    let pageTokenId: Int?
    private let perPageImageTokens: Int?
    private let sparseMainRopeMode: String?
    private let sparseMainRopeTokFreqMode: String?
    private let sparseRopeTokTheta: Float?
    private let sparseRopePageTheta: Float?
    private let sparseRopePagePairs: Int?
    private let sparseRopePageMod: Int?
    private let dashScalePerLayer: [Float]?
    private let dashFloorPerLayer: [Float]?
    private let dashScalingFactor: Float?
    private let dashPriorFloor: Float?
    private let dashQueryChunk: Int?

    enum CodingKeys: String, CodingKey {
        case trainingStage = "training_stage"
        case useSparseAttention = "use_sparse_attention"
        case dashStage0 = "dash_stage0"
        case sparseAttentionFreq = "sparse_attention_freq"
        case sparseNumIndexHeads = "sparse_num_index_heads"
        case sparseBlockSize = "sparse_block_size"
        case sparseLocalBlock = "sparse_local_block"
        case sparseIndexDim = "sparse_index_dim"
        case dashTopkBlocks = "dash_topk_blocks"
        case dashSigma = "dash_sigma"
        case dashImageOnly = "dash_image_only"
        case imageTokenId = "image_token_id"
        case pageTokenId = "page_token_id"
        case perPageImageTokens = "per_page_image_tokens"
        case sparseMainRopeMode = "sparse_main_rope_mode"
        case sparseMainRopeTokFreqMode = "sparse_main_rope_tok_freq_mode"
        case sparseRopeTokTheta = "sparse_rope_tok_theta"
        case sparseRopePageTheta = "sparse_rope_page_theta"
        case sparseRopePagePairs = "sparse_rope_page_pairs"
        case sparseRopePageMod = "sparse_rope_page_mod"
        case dashScalePerLayer = "dash_scale_per_layer"
        case dashFloorPerLayer = "dash_floor_per_layer"
        case dashScalingFactor = "dash_scaling_factor"
        case dashPriorFloor = "dash_prior_floor"
        case dashQueryChunk = "dash_query_chunk"
    }

    var isDASH: Bool { trainingStage == "dash" && useSparseAttention == true }

    func parameters(headDim: Int, layerCount: Int, imageToken: Int) throws
        -> UnlimitedOCRDASHParameters
    {
        let pageMode = sparseMainRopeMode ?? "1d"
        let frequencyMode = sparseMainRopeTokFreqMode ?? "truncate"
        let block = sparseBlockSize ?? 128
        let local = sparseLocalBlock ?? 1
        let dimension = sparseIndexDim ?? headDim
        let topK = dashTopkBlocks ?? 12
        let sigma = dashSigma ?? 1
        let pairs = sparseRopePagePairs ?? 8
        let modulus = sparseRopePageMod ?? 0
        let tokenTheta = sparseRopeTokTheta ?? 10000
        let pageTheta = sparseRopePageTheta ?? 256
        let prefillStep = dashQueryChunk ?? 2048
        let perPage = perPageImageTokens ?? 273
        let scales = dashScalePerLayer ?? []
        let floors = dashFloorPerLayer ?? []
        let scale = dashScalingFactor ?? 2
        let floor = dashPriorFloor ?? 0.01
        guard dashStage0 == "head_cls", (sparseNumIndexHeads ?? 1) == 1,
            sparseAttentionFreq?.allSatisfy({ $0 == 1 }) ?? true,
            block > 0, local > 0, dimension == headDim, topK > 0,
            prefillStep > 0, perPage > 0, modulus >= 0,
            ["page2d", "1d"].contains(pageMode),
            ["truncate", "resample"].contains(frequencyMode),
            pageMode != "page2d" || (pairs > 0 && pairs < headDim / 2),
            (scales + floors + [scale, floor, sigma, tokenTheta, pageTheta])
                .allSatisfy({ $0.isFinite && $0 > 0 }),
            imageTokenId == nil || imageTokenId == imageToken
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid or unsupported Unlimited-OCR-2 DASH configuration")
        }

        return UnlimitedOCRDASHParameters(
            blockSize: block, localBlocks: local, topK: topK, sigma: sigma,
            imageOnly: dashImageOnly ?? true, imageToken: imageToken,
            pageToken: pageTokenId ?? -1, perPageImageTokens: perPage,
            page2D: pageMode == "page2d", tokenTheta: tokenTheta, pageTheta: pageTheta,
            pagePairs: pairs, truncateTokenFrequencies: frequencyMode == "truncate",
            pageModulus: modulus, prefillStep: prefillStep,
            scales: (0 ..< layerCount).map {
                (scales.indices.contains($0) ? scales[$0] : scale) / sqrt(Float(headDim))
            },
            floors: (0 ..< layerCount).map { floors.indices.contains($0) ? floors[$0] : floor })
    }
}

struct UnlimitedOCRDASHParameters: Sendable {
    let blockSize: Int
    let localBlocks: Int
    let topK: Int
    let sigma: Float
    let imageOnly: Bool
    let imageToken: Int
    let pageToken: Int
    let perPageImageTokens: Int
    let page2D: Bool
    let tokenTheta: Float
    let pageTheta: Float
    let pagePairs: Int
    let truncateTokenFrequencies: Bool
    let pageModulus: Int
    let prefillStep: Int
    let scales: [Float]
    let floors: [Float]

    var localWindow: Int { blockSize * localBlocks }

    func blockCount(_ length: Int) -> Int { (length + blockSize - 1) / blockSize }
}

/// Prompt positions and image-block ownership are computed once and shared by the decoder.
struct UnlimitedOCRPromptPages {
    let length: Int
    let promptPageCount: Int
    let markerCount: Int32
    let queryPages: MLXArray
    let minimumPages: MLXArray
    let maximumPages: MLXArray
    let decodeMinimumPages: MLXArray
    let decodeMaximumPages: MLXArray
    let coordinates: Coordinates?

    struct Coordinates {
        let token: MLXArray
        let page: MLXArray
        let prefillCounts: MLXArray
        let pageCount: Int
        let lastPage: Int32
        let lastToken: Int32
    }

    init(ids: [Int32], parameters: UnlimitedOCRDASHParameters) {
        length = ids.count
        let lastImage = ids.lastIndex(of: Int32(parameters.imageToken)) ?? -1
        var cumulative: [Int32] = []
        cumulative.reserveCapacity(length)
        var markers: Int32 = 0
        for id in ids {
            if id == parameters.pageToken { markers += 1 }
            cumulative.append(markers)
        }
        markerCount = markers
        let promptPages = lastImage >= 0 ? cumulative[lastImage] : 0
        promptPageCount = Int(promptPages)

        let blocks = parameters.blockCount(length)
        var minima = [Int32](repeating: .max, count: blocks)
        var maxima = [Int32](repeating: -1, count: blocks)
        var decodeMinima = minima
        var decodeMaxima = maxima
        var imageIndex: Int32 = 0
        var queries: [Int32] = []
        queries.reserveCapacity(length)
        for (index, id) in ids.enumerated() {
            let rawPage = cumulative[index] - 1
            queries.append(max(-1, index <= lastImage ? rawPage : rawPage - promptPages))
            guard id == parameters.imageToken else { continue }
            let fallback = imageIndex / Int32(parameters.perPageImageTokens)
            let imagePage = promptPages > 0 ? rawPage : fallback
            let decodePage = cumulative[index] > 0 ? rawPage : fallback
            let block = index / parameters.blockSize
            if imagePage >= 0 {
                minima[block] = min(minima[block], imagePage)
                maxima[block] = max(maxima[block], imagePage)
            }
            decodeMinima[block] = min(decodeMinima[block], decodePage)
            decodeMaxima[block] = max(decodeMaxima[block], decodePage)
            imageIndex += 1
        }
        queryPages = MLXArray(queries)
        minimumPages = MLXArray(minima.map { $0 == .max ? -1 : $0 })
        maximumPages = MLXArray(maxima)
        decodeMinimumPages = MLXArray(decodeMinima.map { $0 == .max ? -1 : $0 })
        decodeMaximumPages = MLXArray(decodeMaxima)

        if parameters.page2D, promptPages > 0 {
            let pages = cumulative.enumerated().map { index, page in
                max(0, index <= lastImage ? page : page - promptPages)
            }
            let count = Int(pages.max() ?? 0) + 1
            var counts = [Int32](repeating: 0, count: count)
            var prefillCounts = counts
            let tokens = pages.enumerated().map { index, page in
                let page = Int(page)
                let token = counts[page]
                counts[page] += 1
                if index <= lastImage { prefillCounts[page] += 1 }
                return token
            }
            coordinates = Coordinates(
                token: MLXArray(tokens), page: MLXArray(pages),
                prefillCounts: MLXArray(prefillCounts),
                pageCount: count, lastPage: pages.last!, lastToken: tokens.last!)
        } else {
            coordinates = nil
        }
    }

    func prefillCandidates(range: Range<Int>, blocks: Int) -> MLXArray {
        let page = queryPages[range, .newAxis]
        let lower = minimumPages[.newAxis, ..<blocks]
        let upper = maximumPages[.newAxis, ..<blocks]
        let same = (page .>= 0) & (lower .>= 0) & (lower .<= page) & (upper .>= page)
        return same | logicalNot(same.any(axis: -1, keepDims: true))
    }

    func decodeCandidates(markers: MLXArray, capacity: Int, imageOnly: Bool) -> MLXArray {
        guard imageOnly else { return .ones([1, capacity], dtype: .bool) }
        let page = markers - 1 - promptPageCount
        let same =
            (page .>= 0) & (decodeMinimumPages .>= 0)
            & (decodeMinimumPages .<= page) & (decodeMaximumPages .>= page)
        // All prompt blocks fit because the decode capacity includes the prompt.
        let allow = concatenated([same, .zeros([capacity - same.dim(0)], dtype: .bool)], axis: 0)
        return (allow | logicalNot(same.any())).expandedDimensions(axis: 0)
    }
}

struct UnlimitedOCRResponsePosition {
    var markers: MLXArray
    var page: MLXArray
    var token: MLXArray

    init(_ prompt: UnlimitedOCRPromptPages) {
        markers = MLXArray(prompt.markerCount)
        page = MLXArray(prompt.coordinates?.lastPage ?? 0)
        token = MLXArray(prompt.coordinates?.lastToken ?? 0)
    }

    var arrays: [MLXArray] { [markers, page, token] }
}

struct UnlimitedOCRDASHContext {
    let offset: Int
    let length: Int
    let cosine: MLXArray
    let sine: MLXArray
    let candidates: MLXArray
    let decode: Bool
}

final class UnlimitedOCRDASH {
    let parameters: UnlimitedOCRDASHParameters
    private let tokenFrequencies: MLXArray
    private let pageFrequencies: MLXArray
    private let linearFrequencies: MLXArray
    private let linearScale: Float
    private let selectDecode: @Sendable ([MLXArray]) -> [MLXArray]

    init(
        _ parameters: UnlimitedOCRDASHParameters, headDim: Int, base: Float,
        maxPositions: Int, scaling: [String: StringOrNumber]?
    ) {
        self.parameters = parameters
        let tokenPairs = parameters.page2D ? headDim / 2 - parameters.pagePairs : headDim / 2
        let denominator = parameters.truncateTokenFrequencies ? headDim / 2 : tokenPairs
        tokenFrequencies = MLXArray(
            (0 ..< tokenPairs).map {
                1 / pow(parameters.tokenTheta, Float($0) / Float(denominator))
            })
        let pagePairs = parameters.page2D ? parameters.pagePairs : 0
        pageFrequencies = MLXArray(
            (0 ..< pagePairs).map {
                1 / pow(parameters.pageTheta, Float($0) / Float(pagePairs))
            })
        (linearFrequencies, linearScale) = Self.linearRoPE(
            headDim: headDim, base: base, maxPositions: maxPositions, scaling: scaling)
        selectDecode = compile { arrays in
            let (query, summaries, candidates, position, scale, floor) =
                (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5])
            let remote =
                Self.range(0 ..< summaries.dim(0))[.newAxis, 0...]
                .<= (floorDivide(position, parameters.blockSize) - parameters.localBlocks)
            let (blocks, bias, valid) = Self.select(
                scores: Self.scores(query, summaries), remote: remote, candidates: candidates,
                scale: scale, floor: floor, parameters: parameters)
            let blockIndices =
                blocks[0, 0..., .newAxis] * parameters.blockSize
                + Self.range(0 ..< parameters.blockSize)[.newAxis, 0...]
            let keep =
                valid[0, 0..., .newAxis]
                & (blockIndices .<= position - parameters.localWindow)
            let blockMask = MLX.where(keep, bias[0, 0..., .newAxis], -Float.infinity)
            let local =
                position - parameters.localWindow + 1 + Self.range(0 ..< parameters.localWindow)
            let localMask = MLX.where(local .>= 0, Float(0), -Float.infinity)
            return [
                concatenated(
                    [minimum(blockIndices.flattened(), position), maximum(local, 0)], axis: 0),
                concatenated([blockMask.flattened(), localMask], axis: 0),
            ]
        }
    }

    static func blockCapacity(_ count: Int) -> Int { (count + 63) / 64 * 64 }

    private static func range(_ range: Range<Int>) -> MLXArray {
        MLXArray(Int32(range.lowerBound) ..< Int32(range.upperBound))
    }

    func makeContext(cache: UnlimitedOCRDASHCache, tokens: MLXArray?, length: Int, dtype: DType)
        -> UnlimitedOCRDASHContext
    {
        let prompt = cache.prompt!
        let offset = cache.offset
        let end = offset + length
        let range = offset ..< end
        let blocks = parameters.blockCount(end)
        let cosine: MLXArray
        let sine: MLXArray
        let candidates: MLXArray
        let decode = end > prompt.length
        if decode {
            precondition(length == 1 && offset >= prompt.length)
            var position = cache.responsePosition!
            let isPage = tokens!.reshaped(-1)[0] .== parameters.pageToken
            position.markers = position.markers + isPage.asType(.int32)
            if let coords = prompt.coordinates {
                position.page = MLX.where(
                    isPage, minimum(position.page + 1, coords.pageCount - 1), position.page)
                position.token = MLX.where(
                    isPage, coords.prefillCounts.take(position.page, axis: 0), position.token + 1)
                (cosine, sine) = pageRoPE(
                    token: position.token.reshaped(1), page: position.page.reshaped(1), dtype: dtype
                )
            } else {
                (cosine, sine) = linearRoPE(positions: Self.range(range), dtype: dtype)
            }
            cache.responsePosition = position
            candidates = prompt.decodeCandidates(
                markers: position.markers, capacity: Self.blockCapacity(blocks),
                imageOnly: parameters.imageOnly)
        } else {
            if let coords = prompt.coordinates {
                (cosine, sine) = pageRoPE(
                    token: coords.token[range], page: coords.page[range], dtype: dtype)
            } else {
                (cosine, sine) = linearRoPE(positions: Self.range(range), dtype: dtype)
            }
            candidates = prompt.prefillCandidates(range: range, blocks: blocks)
        }
        return UnlimitedOCRDASHContext(
            offset: offset, length: length, cosine: cosine, sine: sine,
            candidates: candidates, decode: decode)
    }

    private func pageRoPE(token: MLXArray, page: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let page = parameters.pageModulus > 0 ? page % parameters.pageModulus : page
        let angles = concatenated(
            [
                token.asType(.float32)[0..., .newAxis] * tokenFrequencies,
                page.asType(.float32)[0..., .newAxis] * pageFrequencies,
            ], axis: -1)
        return Self.cosineSine(angles, scale: 1, dtype: dtype)
    }

    private func linearRoPE(positions: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let angles = positions.asType(.float32)[0..., .newAxis] * linearFrequencies
        return Self.cosineSine(angles, scale: linearScale, dtype: dtype)
    }

    private static func cosineSine(_ angles: MLXArray, scale: Float, dtype: DType)
        -> (MLXArray, MLXArray)
    {
        let angles = concatenated([angles, angles], axis: -1)
        return (
            (cos(angles) * scale).asType(dtype)[.newAxis, .newAxis],
            (sin(angles) * scale).asType(dtype)[.newAxis, .newAxis]
        )
    }

    private static func linearRoPE(
        headDim: Int, base: Float, maxPositions: Int, scaling: [String: StringOrNumber]?
    ) -> (MLXArray, Float) {
        let frequencies = (0 ..< headDim / 2).map { pow(base, Float(2 * $0) / Float(headDim)) }
        guard let scaling,
            (scaling["rope_type"] ?? scaling["type"]) == .string("yarn"),
            let factor = scaling["factor"]?.asFloat()
        else { return (MLXArray(frequencies.map { 1 / $0 }), 1) }
        let original = scaling["original_max_position_embeddings"]?.asFloat() ?? Float(maxPositions)
        func correction(_ rotations: Float) -> Float {
            Float(headDim) * log(original / (rotations * 2 * .pi)) / (2 * log(base))
        }
        let low = max(floor(correction(scaling["beta_fast"]?.asFloat() ?? 32)), 0)
        let high = min(ceil(correction(scaling["beta_slow"]?.asFloat() ?? 1)), Float(headDim - 1))
        let width = low == high ? 0.001 : high - low
        let adjusted = frequencies.enumerated().map { i, f in
            let ramp = min(max((Float(i) - low) / width, 0), 1)
            return ramp / (factor * f) + (1 - ramp) / f
        }
        return (
            MLXArray(adjusted),
            scaling["attention_factor"]?.asFloat() ?? (factor > 1 ? 1 + 0.1 * log(factor) : 1)
        )
    }

    fileprivate static func rotate(_ x: MLXArray, _ cosine: MLXArray, _ sine: MLXArray) -> MLXArray
    {
        let halves = split(x, parts: 2, axis: -1)
        return x * cosine + concatenated([-halves[1], halves[0]], axis: -1) * sine
    }

    private static let rotateDecode = compile { x, cosine, sine in rotate(x, cosine, sine) }

    private static func scores(_ query: MLXArray, _ summaries: MLXArray) -> MLXArray {
        clip(matmul(query, summaries.T) * pow(Float(query.dim(-1)), -0.5), min: -1e4, max: 1e4)
    }

    static func entmax15(_ scores: MLXArray) -> MLXArray {
        let x = (scores - scores[0..., ..<1]) / 2
        let count = range(1 ..< (x.dim(-1) + 1)).asType(.float32)
        let mean = cumsum(x, axis: -1) / count
        let meanSquared = cumsum(x * x, axis: -1) / count
        let delta = (1 - count * (meanSquared - mean * mean)) / count
        let tau = mean - sqrt(maximum(delta, 0))
        let support = maximum((tau .<= x).sum(axis: -1, keepDims: true), 1)
        let finiteTau = takeAlong(tau, support - 1, axis: -1)
        let result = maximum(x - finiteTau, 0)
        return result * result
    }

    private static func select(
        scores: MLXArray, remote: MLXArray, candidates: MLXArray, scale: MLXArray,
        floor: MLXArray, parameters: UnlimitedOCRDASHParameters
    ) -> (MLXArray, MLXArray, MLXArray) {
        let allowed = remote & candidates
        let selected = MLX.where(allowed.any(axis: -1, keepDims: true), allowed, remote)
        let logits = MLX.where(selected, scores * scale, Float(-1e4))
        let pool = min(16, scores.dim(-1))
        let count = min(parameters.topK, pool)
        let order = argSort(-logits, axis: -1)[0..., ..<pool]
        let prior = entmax15(takeAlong(logits, order, axis: -1))[0..., ..<count]
        let top = order[0..., ..<count]
        let valid = takeAlong(selected, top, axis: -1)
        let logp = log(maximum(prior, floor))
        let n = maximum(valid.sum(axis: -1, keepDims: true), 1)
        let mean = MLX.where(valid, logp, 0).sum(axis: -1, keepDims: true) / n
        return (top, MLX.where(valid, (logp - mean) / parameters.sigma, 0), valid)
    }

    func attend(
        x: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, router: UnlimitedOCRDASHRouter, layer: Int,
        cache: UnlimitedOCRDASHCache, context: UnlimitedOCRDASHContext
    ) -> MLXArray {
        let applyRoPE = context.decode ? Self.rotateDecode : Self.rotate
        let q = applyRoPE(queries, context.cosine, context.sine)
        let k = applyRoPE(keys, context.cosine, context.sine)
        let (cachedKeys, cachedValues) = cache.update(keys: k, values: values)
        let (query, indexKeys) = router(x, context: context)
        cache.appendRouterKeys(indexKeys, head: router.head, blockSize: parameters.blockSize)
        let indexScale = MLXArray(parameters.scales[layer])
        let floor = MLXArray(parameters.floors[layer])
        let blocks = context.candidates.dim(-1)
        if context.decode {
            let result = selectDecode([
                query, cache.summaryBuffer(capacity: blocks, dimension: query.dim(-1)),
                context.candidates, MLXArray(Int32(context.offset)), indexScale, floor,
            ])
            let indices = result[0]
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: cachedKeys.take(indices, axis: 2),
                values: cachedValues.take(indices, axis: 2), scale: scale,
                mask: .array(result[1].asType(q.dtype)[.newAxis, .newAxis, .newAxis]))
        }

        let end = context.offset + context.length
        let kPositions = Self.range(0 ..< end)[.newAxis, 0...]
        let kBlocks = floorDivide(kPositions, parameters.blockSize)[0]
        let summaries = cache.summaryBuffer(capacity: blocks, dimension: query.dim(-1))
        let blockIndices = Self.range(0 ..< blocks)[.newAxis, 0...]
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: context.length, by: parameters.prefillStep) {
            let stop = min(start + parameters.prefillStep, context.length)
            let range = start ..< stop
            let positions = Self.range((context.offset + start) ..< (context.offset + stop))
                .expandedDimensions(axis: -1)
            let qBlocks = floorDivide(positions, parameters.blockSize)
            let remote = blockIndices .<= (qBlocks - parameters.localBlocks)
            let (selected, bias, valid) = Self.select(
                scores: Self.scores(query[range, 0...], summaries), remote: remote,
                candidates: context.candidates[range, 0...], scale: indexScale, floor: floor,
                parameters: parameters)
            let blockMask = putAlong(
                MLXArray.full([range.count, blocks], values: MLXArray(-Float.infinity)), selected,
                values: MLX.where(valid, bias, -Float.infinity), axis: -1)
            let causal = kPositions .<= positions
            let local = causal & (kPositions .> positions - parameters.localWindow)
            let mask = MLX.where(
                local, 0, MLX.where(causal, blockMask.take(kBlocks, axis: -1), -Float.infinity))
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., range, 0...], keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: .array(mask.asType(q.dtype)[.newAxis, .newAxis])))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }
}

final class UnlimitedOCRDASHRouter: Module {
    final class Norm: Module {
        let weight: MLXArray

        init(_ dimension: Int) {
            weight = .zeros([dimension])
        }
    }

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Norm
    @ModuleInfo(key: "k_norm") var keyNorm: Norm
    @ParameterInfo(key: "head_cls") var head: MLXArray
    let epsilon: Float

    init(hiddenSize: Int, dimension: Int, epsilon: Float) {
        self.epsilon = epsilon
        _query.wrappedValue = Linear(hiddenSize, dimension, bias: false)
        _key.wrappedValue = Linear(hiddenSize, dimension, bias: false)
        _queryNorm.wrappedValue = Norm(dimension)
        _keyNorm.wrappedValue = Norm(dimension)
        _head.wrappedValue = .zeros([dimension])
    }

    private static func finite(_ x: MLXArray) -> MLXArray { MLX.where(isFinite(x), x, 0) }

    private static func project(_ arrays: [MLXArray]) -> [MLXArray] {
        let x = finite(arrays[0])
        let (cosine, sine, epsilon) = (arrays[5], arrays[6], arrays[7])
        func apply(_ projection: MLXArray, _ weight: MLXArray) -> MLXArray {
            let projected = matmul(x, projection.T)
            let value = finite(projected.asType(.float32))
            let normalized = clip(
                value * rsqrt((value * value).mean(axis: -1, keepDims: true) + epsilon),
                min: -1e4, max: 1e4)
            let scaled = clip(normalized * (1 + weight.asType(.float32)), min: -1e4, max: 1e4)
            let rotated = UnlimitedOCRDASH.rotate(
                scaled.asType(projected.dtype).expandedDimensions(axis: 1), cosine, sine)
            return finite(rotated[0, 0].asType(.float32))
        }
        return [apply(arrays[1], arrays[3]), apply(arrays[2], arrays[4])]
    }

    private static let projectDecode = compile { arrays in project(arrays) }

    func callAsFunction(_ x: MLXArray, context: UnlimitedOCRDASHContext) -> (MLXArray, MLXArray) {
        let args = [
            x, query.weight, key.weight, queryNorm.weight, keyNorm.weight,
            context.cosine, context.sine, MLXArray(epsilon),
        ]
        let projected = context.decode ? Self.projectDecode(args) : Self.project(args)
        return (projected[0], projected[1])
    }
}
