import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Moondream3Configuration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let vocabularySize: Int
        public let maxPositionEmbeddings: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let ropeTheta: Float
        public let ropeDim: Int
        public let layerNormEps: Float
        public let numExperts: Int
        public let numExpertsPerToken: Int
        public let moeIntermediateSize: Int
        public let moeStartLayer: Int
        public let attentionBias: Bool
        public let prefixAttentionLength: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case ropeTheta = "rope_theta"
            case ropeDim = "rope_dim"
            case layerNormEps = "rms_norm_eps"
            case numExperts = "num_experts"
            case numExpertsPerToken = "num_experts_per_tok"
            case moeIntermediateSize = "moe_intermediate_size"
            case moeStartLayer = "moe_start_layer"
            case attentionBias = "attention_bias"
            case prefixAttentionLength = "prefix_attn"
        }

        public init(
            modelType: String = "moondream3",
            hiddenSize: Int = 2048,
            intermediateSize: Int = 8192,
            hiddenLayers: Int = 24,
            vocabularySize: Int = 51200,
            maxPositionEmbeddings: Int = 4096,
            attentionHeads: Int = 32,
            kvHeads: Int = 32,
            headDim: Int = 64,
            ropeTheta: Float = 1_500_000,
            ropeDim: Int = 32,
            layerNormEps: Float = 1e-5,
            numExperts: Int = 64,
            numExpertsPerToken: Int = 8,
            moeIntermediateSize: Int = 1024,
            moeStartLayer: Int = 4,
            attentionBias: Bool = true,
            prefixAttentionLength: Int = 730,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.hiddenLayers = hiddenLayers
            self.vocabularySize = vocabularySize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.ropeTheta = ropeTheta
            self.ropeDim = ropeDim
            self.layerNormEps = layerNormEps
            self.numExperts = numExperts
            self.numExpertsPerToken = numExpertsPerToken
            self.moeIntermediateSize = moeIntermediateSize
            self.moeStartLayer = moeStartLayer
            self.attentionBias = attentionBias
            self.prefixAttentionLength = prefixAttentionLength
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream3"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 51200
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / max(attentionHeads, 1)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_500_000
            ropeDim = try c.decodeIfPresent(Int.self, forKey: .ropeDim) ?? headDim / 2
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
            numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 64
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1024
            moeStartLayer = try c.decodeIfPresent(Int.self, forKey: .moeStartLayer) ?? 4
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
            prefixAttentionLength =
                try c.decodeIfPresent(Int.self, forKey: .prefixAttentionLength) ?? 730
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let cropSize: Int
        public let maxCrops: Int
        public let overlapMargin: Int
        public let channels: Int
        public let projectionInnerDimension: Int
        public let projectionOutputDimension: Int
        public let attentionBias: Bool
        public let layerNormEps: Float

        public var visionTokenCount: Int {
            let grid = cropSize / patchSize
            return grid * grid
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case cropSize = "crop_size"
            case maxCrops = "max_crops"
            case overlapMargin = "overlap_margin"
            case channels = "in_channels"
            case projectionInnerDimension = "proj_inner_dim"
            case projectionOutputDimension = "proj_out_dim"
            case attentionBias = "attention_bias"
            case layerNormEps = "layer_norm_eps"
        }

        public init(
            modelType: String = "moondream3_vision",
            hiddenSize: Int = 1152,
            intermediateSize: Int = 4304,
            hiddenLayers: Int = 27,
            attentionHeads: Int = 16,
            patchSize: Int = 14,
            cropSize: Int = 378,
            maxCrops: Int = 12,
            overlapMargin: Int = 4,
            channels: Int = 3,
            projectionInnerDimension: Int = 8192,
            projectionOutputDimension: Int = 2048,
            attentionBias: Bool = true,
            layerNormEps: Float = 1e-6,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.patchSize = patchSize
            self.cropSize = cropSize
            self.maxCrops = maxCrops
            self.overlapMargin = overlapMargin
            self.channels = channels
            self.projectionInnerDimension = projectionInnerDimension
            self.projectionOutputDimension = projectionOutputDimension
            self.attentionBias = attentionBias
            self.layerNormEps = layerNormEps
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream3_vision"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            cropSize = try c.decodeIfPresent(Int.self, forKey: .cropSize) ?? 378
            maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? 12
            overlapMargin = try c.decodeIfPresent(Int.self, forKey: .overlapMargin) ?? 4
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            projectionInnerDimension =
                try c.decodeIfPresent(Int.self, forKey: .projectionInnerDimension) ?? 8192
            projectionOutputDimension =
                try c.decodeIfPresent(Int.self, forKey: .projectionOutputDimension) ?? 2048
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    private struct RuntimeConfiguration: Decodable {
        let text: TextConfiguration?
        let vision: VisionConfiguration?
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let eosTokenID: Int
    public let bosTokenID: Int

    public var visionTokenCount: Int {
        visionConfiguration.visionTokenCount
    }

    public var visionPrefixLength: Int {
        1 + visionTokenCount
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case runtimeConfiguration = "config"
        case eosTokenID = "eos_token_id"
        case bosTokenID = "bos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream3"
        let runtime = try c.decodeIfPresent(
            RuntimeConfiguration.self, forKey: .runtimeConfiguration)
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? runtime?.text
            ?? TextConfiguration()
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? runtime?.vision
            ?? VisionConfiguration()
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 0
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 0
    }
}

public struct Moondream3ProcessorConfiguration: Decodable, Sendable {
    private struct RuntimeConfiguration: Decodable {
        let vision: Moondream3Configuration.VisionConfiguration?
    }

    public let processorClass: String
    public let cropSize: Int
    public let maxCrops: Int
    public let overlapMargin: Int
    public let patchSize: Int
    public let bosTokenID: Int
    public let answerTokenID: Int

    public var visionTokenCount: Int {
        let grid = cropSize / patchSize
        return grid * grid
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case cropSize = "crop_size"
        case maxCrops = "max_crops"
        case overlapMargin = "overlap_margin"
        case patchSize = "patch_size"
        case bosTokenID = "bos_token_id"
        case answerTokenID = "answer_token_id"
        case runtimeConfiguration = "config"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "Moondream3Processor"
        let runtime = try c.decodeIfPresent(
            RuntimeConfiguration.self, forKey: .runtimeConfiguration)
        let rootVision = try c.decodeIfPresent(
            Moondream3Configuration.VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        let vision = rootVision ?? runtime?.vision
        cropSize =
            try c.decodeIfPresent(Int.self, forKey: .cropSize)
            ?? vision?.cropSize
            ?? 378
        maxCrops =
            try c.decodeIfPresent(Int.self, forKey: .maxCrops)
            ?? vision?.maxCrops
            ?? 12
        overlapMargin =
            try c.decodeIfPresent(Int.self, forKey: .overlapMargin)
            ?? vision?.overlapMargin
            ?? 4
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? vision?.patchSize
            ?? 14
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 0
        answerTokenID = try c.decodeIfPresent(Int.self, forKey: .answerTokenID) ?? 3
    }
}

// MARK: - Processor

public struct Moondream3Processor: UserInputProcessor {
    private let config: Moondream3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Moondream3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)

        guard !images.isEmpty else {
            let tokens =
                [config.bosTokenID]
                + tokenizer.encode(
                    text: prompt,
                    addSpecialTokens: false,
                )
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let processed = try preprocess(
            image: images[0].asCIImage(),
            processing: input.processing,
        )
        let formatted = "\n\nQuestion: \(prompt)\n\nAnswer: "
        let textTokens = tokenizer.encode(text: formatted, addSpecialTokens: false)
        let tokens =
            [config.bosTokenID]
            + Array(repeating: 0, count: config.visionTokenCount)
            + textTokens
            + [config.answerTokenID]
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: processed.pixels),
            video: .init(pixels: processed.metadata),
        )
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let metadata: MLXArray
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing).toSRGB()
        let cropSize = config.cropSize
        let patchSize = config.patchSize
        guard cropSize > 0, patchSize > 0 else {
            throw VLMError.processing("Moondream3 crop and patch sizes must be positive")
        }

        let canonical = MediaProcessing.zeroOrigin(image)
        let width = max(Int(canonical.extent.width.rounded()), 1)
        let height = max(Int(canonical.extent.height.rounded()), 1)
        var crops = [[Float]]()
        crops.append(
            Self.resizeImage(image: canonical, targetHeight: cropSize, targetWidth: cropSize))

        let layout = Self.selectCropGrid(
            width: width,
            height: height,
            cropSize: cropSize,
            maxCrops: config.maxCrops,
            overlapMargin: config.overlapMargin,
            patchSize: patchSize,
        )
        if let layout, layout.rows * layout.cols + 1 <= config.maxCrops {
            let overlapPixels = config.overlapMargin * patchSize
            let effectiveSize = cropSize - 2 * overlapPixels
            let denominatorWidth = max(layout.cols * effectiveSize + 2 * overlapPixels, 1)
            let denominatorHeight = max(layout.rows * effectiveSize + 2 * overlapPixels, 1)
            let scaleX = Double(width) / Double(denominatorWidth)
            let scaleY = Double(height) / Double(denominatorHeight)
            let scale = max(scaleX, scaleY)
            let cropWidth = max(Int(Double(cropSize) * scale), 1)
            let cropHeight = max(Int(Double(cropSize) * scale), 1)

            for row in 0 ..< layout.rows {
                for col in 0 ..< layout.cols {
                    var x0 = Int(Double(col * effectiveSize) * scale)
                    var y0 = Int(Double(row * effectiveSize) * scale)
                    x0 = min(x0, max(width - cropWidth, 0))
                    y0 = min(y0, max(height - cropHeight, 0))
                    let x1 = min(x0 + cropWidth, width)
                    let y1 = min(y0 + cropHeight, height)
                    let rect = CGRect(x: x0, y: y0, width: max(x1 - x0, 1), height: max(y1 - y0, 1))
                    let crop = MediaProcessing.zeroOrigin(canonical.cropped(to: rect))
                    crops.append(
                        Self.resizeImage(image: crop, targetHeight: cropSize, targetWidth: cropSize)
                    )
                }
            }
        }

        let pixelArray = MLXArray(crops.flatMap(\.self), [crops.count, cropSize, cropSize, 3])
        let metadata = MLXArray(
            [
                Int32(crops.count), Int32(layout?.rows ?? 1),
                Int32(layout?.cols ?? max(1, crops.count - 1)),
            ],
            [1, 3],
        )
        return ProcessedImage(pixels: pixelArray, metadata: metadata)
    }

    private static func selectCropGrid(
        width: Int,
        height: Int,
        cropSize: Int,
        maxCrops: Int,
        overlapMargin: Int,
        patchSize: Int,
    )
        -> (rows: Int, cols: Int)?
    {
        let effectiveCrop = cropSize - 2 * overlapMargin * patchSize
        guard effectiveCrop > 0, maxCrops > 1 else {
            return nil
        }

        var best: (rows: Int, cols: Int)?
        var bestScore = Double.infinity
        let aspectRatio = Double(max(width, 1)) / Double(max(height, 1))
        for rows in 1 ... maxCrops {
            for cols in 1 ... maxCrops {
                let totalCrops = rows * cols + 1
                if totalCrops > maxCrops {
                    continue
                }
                let effectiveHeight = rows * effectiveCrop + 2 * overlapMargin * patchSize
                let effectiveWidth = cols * effectiveCrop + 2 * overlapMargin * patchSize
                let scaleHeight = Double(effectiveHeight) / Double(max(height, 1))
                let scaleWidth = Double(effectiveWidth) / Double(max(width, 1))
                let gridRatio = Double(cols) / Double(rows)
                let ratioDiff = abs(log(aspectRatio / gridRatio))
                let scaleDiff = abs(log(max(min(scaleHeight, scaleWidth), 1e-6)))
                let score = ratioDiff + 0.5 * scaleDiff + 0.1 * Double(totalCrops)
                if score < bestScore {
                    bestScore = score
                    best = (rows, cols)
                }
            }
        }
        return best
    }

    private static func resizeImage(image: CIImage, targetHeight: Int, targetWidth: Int) -> [Float]
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        var pixels = MediaProcessing.renderRGBFloat(
            resized, width: targetWidth, height: targetHeight)
        for index in stride(from: 0, to: pixels.count, by: 3) {
            pixels[index] = (pixels[index] - 0.5) / 0.5
            pixels[index + 1] = (pixels[index + 1] - 0.5) / 0.5
            pixels[index + 2] = (pixels[index + 2] - 0.5) / 0.5
        }
        return pixels
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.last(where: { $0.role == .user })?.content
                ?? messages.map(\.content).joined(separator: "\n")
        }
    }

}

// MARK: - Language

private enum Moondream3Language {
    final class Tau: Module {
        @ModuleInfo var wq: MLXArray
        @ModuleInfo var wv: MLXArray
        @ModuleInfo var alpha: MLXArray
        let activation = GELU(approximation: .none)

        init(heads: Int, qkvDimension: Int) {
            _wq.wrappedValue = MLXArray.zeros([heads, qkvDimension])
            _wv.wrappedValue = MLXArray.zeros([heads, qkvDimension])
            _alpha.wrappedValue = MLXArray.zeros([heads])
            super.init()
        }

        func callAsFunction(_ qkv: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
            let hidden = activation(qkv)
            let dtype = qkv.dtype
            let tokenQ = tanh(hidden.matmul(wq.asType(dtype).T))
            let tokenV = tanh(hidden.matmul(wv.asType(dtype).T))
            let logPositions = log(positions.asType(dtype) + 1)
            let alphaLogPositions =
                alpha.asType(dtype)[.newAxis, 0..., .newAxis]
                * logPositions[.newAxis, .newAxis, 0...]
            let tauPosition = (1 + (sigmoid(alphaLogPositions) - 0.5)).asType(dtype)
            let tauQ = tokenQ.transposed(0, 2, 1) + tauPosition
            let tauV = tokenV.transposed(0, 2, 1) + tauPosition
            return (
                expandedDimensions(tauQ, axis: -1),
                expandedDimensions(tauV, axis: -1),
            )
        }
    }

    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo var tau: Tau

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE
        let fused: (Int, Int)

        init(_ config: Moondream3Configuration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            let qDim = heads * headDim
            let kvDim = kvHeads * headDim
            let qkvDim = qDim + 2 * kvDim
            fused = (qDim, qDim + kvDim)
            _qkv.wrappedValue = Linear(config.hiddenSize, qkvDim, bias: config.attentionBias)
            _proj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: config.attentionBias)
            _tau.wrappedValue = Tau(heads: heads, qkvDimension: qkvDim)
            rope = RoPE(
                dimensions: config.ropeDim,
                traditional: false,
                base: config.ropeTheta,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkvOutput = qkv(x)
            let offset = cache?.offset ?? 0
            let positions = MLXArray(offset ..< offset + length)
            let (tauQ, tauV) = tau(qkvOutput, positions: positions)
            let parts = split(qkvOutput, indices: [fused.0, fused.1], axis: -1)
            var queries = parts[0].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            var keys = parts[1].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            var values = parts[2].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            queries = queries * tauQ
            values = values * tauV
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            return proj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class DenseMLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .tanh)

        init(_ config: Moondream3Configuration.TextConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class MoEMLP: Module, UnaryLayer {
        @ModuleInfo var router: Linear
        @ModuleInfo var fc1: SwitchLinear
        @ModuleInfo var fc2: SwitchLinear
        let topK: Int
        let activation = GELU(approximation: .tanh)

        init(_ config: Moondream3Configuration.TextConfiguration) {
            topK = config.numExpertsPerToken
            _router.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: true)
            _fc1.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: 2 * config.moeIntermediateSize,
                numExperts: config.numExperts,
                bias: false,
            )
            _fc2.wrappedValue = SwitchLinear(
                inputDims: config.moeIntermediateSize,
                outputDims: config.hiddenSize,
                numExperts: config.numExperts,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gates = router(x)
            let indices = argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
            let scores = softmax(takeAlong(gates, indices, axis: -1), axis: -1, precise: true)
            var expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            let shouldSort = indices.size >= 64
            var expertIndices = indices
            var inverseOrder = MLXArray()
            if shouldSort {
                (expanded, expertIndices, inverseOrder) = gatherSort(x: expanded, indices: indices)
            }
            let hidden = fc1(expanded, expertIndices, sortedIndices: shouldSort)
            let parts = split(hidden, parts: 2, axis: -1)
            var output = fc2(
                activation(parts[0]) * (parts[1] + 1),
                expertIndices,
                sortedIndices: shouldSort,
            )
            if shouldSort {
                output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
            }
            output = squeezed(output, axis: -2)
            return (output * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        }
    }

    final class DecoderBlock: Module {
        @ModuleInfo var ln: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: Module

        init(_ config: Moondream3Configuration.TextConfiguration, layerIndex: Int) {
            _ln.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            if layerIndex < config.moeStartLayer {
                _mlp.wrappedValue = DenseMLP(config)
            } else {
                _mlp.wrappedValue = MoEMLP(config)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let hidden = ln(x)
            return x + attn(hidden, mask: mask, cache: cache) + (mlp as! UnaryLayer)(hidden)
        }
    }

    final class TextModel: Module {
        @ModuleInfo var wte: Embedding
        @ModuleInfo var blocks: [DecoderBlock]
        @ModuleInfo(key: "post_ln") var postLayerNorm: LayerNorm

        init(_ config: Moondream3Configuration.TextConfiguration) {
            _wte.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderBlock(config, layerIndex: $0)
            }
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            prefixAttentionLength: Int? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbedding ?? wte(inputs!)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                if cache == nil, let prefixAttentionLength,
                    hidden.dim(1) > 1
                {
                    .array(
                        Self.prefixAttentionMask(
                            sequenceLength: hidden.dim(1),
                            prefixLength: prefixAttentionLength,
                            dtype: hidden.dtype,
                        ),
                    )
                } else {
                    createAttentionMask(h: hidden, cache: cache?.first)
                }
            for (index, block) in blocks.enumerated() {
                hidden = block(hidden, mask: mask, cache: cache?[index])
            }
            return postLayerNorm(hidden)
        }

        private static func prefixAttentionMask(
            sequenceLength: Int,
            prefixLength: Int,
            dtype: DType,
        )
            -> MLXArray
        {
            var values = Array(repeating: Float32(0), count: sequenceLength * sequenceLength)
            for row in 0 ..< sequenceLength {
                for col in 0 ..< sequenceLength {
                    let visible = col <= row || (row < prefixLength && col < prefixLength)
                    if !visible {
                        values[row * sequenceLength + col] = -Float32.infinity
                    }
                }
            }
            return MLXArray(values, [sequenceLength, sequenceLength]).asType(dtype)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let kvHeads: [Int]

        init(_ config: Moondream3Configuration.TextConfiguration) {
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: true)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            prefixAttentionLength: Int? = nil,
        )
            -> LMOutput
        {
            LMOutput(
                logits: lmHead(
                    model(
                        inputs,
                        cache: cache,
                        inputEmbedding: inputEmbedding,
                        prefixAttentionLength: prefixAttentionLength,
                    ),
                ),
            )
        }
    }
}

// MARK: - Vision

private enum Moondream3Vision {
    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        let heads: Int
        let headDim: Int
        let scale: Float

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize,
                3 * config.hiddenSize,
                bias: config.attentionBias,
            )
            _proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkv = split(qkv(x), parts: 3, axis: -1)
            let queries = qkv[0].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let keys = qkv[1].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let values = qkv[2].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: nil,
            )
            return proj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .tanh)

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class Block: Module {
        @ModuleInfo var ln1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var ln2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            _ln1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _ln2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = x + attn(ln1(x))
            hidden = hidden + mlp(ln2(hidden))
            return hidden
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "patch_emb") var patchEmbedding: Linear
        @ModuleInfo(key: "pos_emb") var positionEmbedding: MLXArray
        @ModuleInfo var blocks: [Block]
        @ModuleInfo(key: "post_ln") var postLayerNorm: LayerNorm

        let config: Moondream3Configuration.VisionConfiguration

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            self.config = config
            _patchEmbedding.wrappedValue = Linear(
                config.patchSize * config.patchSize * config.channels,
                config.hiddenSize,
                bias: true,
            )
            _positionEmbedding.wrappedValue = MLXArray.zeros(
                [1, config.visionTokenCount, config.hiddenSize],
            )
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let height = x.dim(1)
            let width = x.dim(2)
            let channels = x.dim(3)
            let patch = config.patchSize
            let patchH = height / patch
            let patchW = width / patch
            var hidden =
                x
                .reshaped(batch, patchH, patch, patchW, patch, channels)
                .transposed(0, 1, 3, 5, 2, 4)
                .reshaped(batch, patchH * patchW, channels * patch * patch)
            hidden = patchEmbedding(hidden)
            hidden = hidden + positionEmbedding.asType(hidden.dtype)
            for block in blocks {
                hidden = block(hidden)
            }
            return postLayerNorm(hidden)
        }
    }

    final class Projection: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .tanh)

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                2 * config.hiddenSize,
                config.projectionInnerDimension,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.projectionInnerDimension,
                config.projectionOutputDimension,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "proj_mlp") var projection: Projection

        let config: Moondream3Configuration.VisionConfiguration

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            self.config = config
            _encoder.wrappedValue = Encoder(config)
            _projection.wrappedValue = Projection(config)
            super.init()
        }

        func callAsFunction(pixelValues: MLXArray, metadata: MLXArray?) -> MLXArray {
            let allFeatures = encoder(pixelValues)
            guard let metadata else {
                let combined = concatenated([allFeatures, allFeatures], axis: -1)
                return projection(combined)
            }

            let values = metadata.asArray(Int32.self).map(Int.init)
            let numCrops = values.first ?? allFeatures.dim(0)
            let rows = values.count > 1 ? values[1] : 1
            let cols = values.count > 2 ? values[2] : max(1, numCrops - 1)
            let globalFeatures = allFeatures[0, 0..., 0...]
            let reconstructed: MLXArray
            if numCrops > 1 {
                let localFeatures = (1 ..< numCrops).map { allFeatures[$0, 0..., 0...] }
                reconstructed = reconstructLocalFeatures(localFeatures, rows: rows, cols: cols)
            } else {
                reconstructed = globalFeatures
            }
            let combined = concatenated([globalFeatures, reconstructed], axis: -1)
            return projection(combined)[.newAxis, 0..., 0...]
        }

        private func reconstructLocalFeatures(
            _ localFeatures: [MLXArray],
            rows: Int,
            cols: Int,
        )
            -> MLXArray
        {
            let gridSize = config.cropSize / config.patchSize
            let margin = config.overlapMargin
            var cropRows = [MLXArray]()
            var index = 0
            for row in 0 ..< rows {
                var rowFeatures = [MLXArray]()
                for col in 0 ..< cols {
                    var feature = localFeatures[min(index, localFeatures.count - 1)]
                        .reshaped(gridSize, gridSize, -1)
                    let top = row > 0 ? margin : 0
                    let bottom = row < rows - 1 ? gridSize - margin : gridSize
                    let left = col > 0 ? margin : 0
                    let right = col < cols - 1 ? gridSize - margin : gridSize
                    feature = feature[top ..< bottom, left ..< right, 0...]
                    rowFeatures.append(feature)
                    index += 1
                }
                cropRows.append(concatenated(rowFeatures, axis: 1))
            }
            let fullGrid = concatenated(cropRows, axis: 0)
            let height = fullGrid.dim(0)
            let width = fullGrid.dim(1)
            var pooled = [MLXArray]()
            pooled.reserveCapacity(gridSize * gridSize)
            for row in 0 ..< gridSize {
                let hStart = min(Int(round(Float(row * height) / Float(gridSize))), height - 1)
                let hEnd = max(
                    hStart + 1,
                    min(Int(round(Float((row + 1) * height) / Float(gridSize))), height),
                )
                for col in 0 ..< gridSize {
                    let wStart = min(Int(round(Float(col * width) / Float(gridSize))), width - 1)
                    let wEnd = max(
                        wStart + 1,
                        min(Int(round(Float((col + 1) * width) / Float(gridSize))), width),
                    )
                    let cell = mean(
                        mean(fullGrid[hStart ..< hEnd, wStart ..< wEnd, 0...], axis: 0),
                        axis: 0,
                    )
                    pooled.append(cell)
                }
            }
            return concatenated(pooled.map { $0[.newAxis, 0...] }, axis: 0)
        }
    }
}

// MARK: - Model

public final class Moondream3: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo private var vision: Moondream3Vision.VisionModel
    @ModuleInfo private var text: Moondream3Language.LanguageModel

    public let config: Moondream3Configuration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        text.kvHeads
    }

    public var loraLayers: [Module] {
        text.model.blocks
    }

    public init(_ config: Moondream3Configuration) {
        self.config = config
        _vision.wrappedValue = Moondream3Vision.VisionModel(config.visionConfiguration)
        _text.wrappedValue = Moondream3Language.LanguageModel(config.textConfiguration)
        super.init()
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?, metadata: MLXArray?)
        -> MLXArray
    {
        let embeddings = text.model.wte(inputIDs)
        guard let pixelValues else {
            return embeddings
        }
        let dtype = vision.encoder.patchEmbedding.weight.dtype
        let imageFeatures = vision(
            pixelValues: pixelValues.asType(dtype),
            metadata: metadata,
        )
        .asType(embeddings.dtype)
        let textStart = 1 + imageFeatures.dim(1)
        let bosEmbedding = embeddings[0..., 0 ..< 1, 0...]
        if embeddings.dim(1) > textStart {
            return concatenated(
                [
                    bosEmbedding,
                    imageFeatures,
                    embeddings[0..., textStart..., 0...],
                ],
                axis: 1,
            )
        }
        return concatenated([bosEmbedding, imageFeatures], axis: 1)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            metadata: input.video?.pixels,
        )
        let result = text(
            nil,
            cache: cache,
            inputEmbedding: embeddings,
            prefixAttentionLength: input.image == nil ? nil : config.visionPrefixLength,
        )
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        text(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("freqs_cis") {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("region.") || newKey.hasPrefix("region_model.") {
                continue
            }

            if newKey == "text.wte" {
                newKey = "text.model.wte.weight"
            } else if newKey.hasPrefix("text.model.") || newKey.hasPrefix("text.lm_head.") {
                // Already in the MLX runtime layout.
            } else if newKey.hasPrefix("text.") {
                newKey = "text.model." + newKey.dropFirst("text.".count)
            } else if newKey.hasPrefix("vision.encoder.") || newKey.hasPrefix("vision.proj_mlp.") {
                // Already in the MLX runtime layout.
            } else if newKey.hasPrefix("vision.") {
                newKey = "vision.encoder." + newKey.dropFirst("vision.".count)
            }

            sanitized[String(newKey)] = value
        }
        return sanitized
    }
}
