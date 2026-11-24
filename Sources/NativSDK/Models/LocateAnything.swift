import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct LocateAnythingConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 1152
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 27
        }

        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 16
        }

        private let _intermediateSize: Int?
        public var intermediateSize: Int {
            _intermediateSize ?? 4304
        }

        private let _patchSize: Int?
        public var patchSize: Int {
            _patchSize ?? 14
        }

        private let _initPosEmbeddingHeight: Int?
        public var initPosEmbeddingHeight: Int {
            _initPosEmbeddingHeight ?? 64
        }

        private let _initPosEmbeddingWidth: Int?
        public var initPosEmbeddingWidth: Int {
            _initPosEmbeddingWidth ?? 64
        }

        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _mergeKernelSize: [Int]?
        public var mergeKernelSize: [Int] {
            _mergeKernelSize ?? [2, 2]
        }

        public var spatialMergeSize: Int {
            mergeKernelSize.first ?? 2
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case _attentionHeads = "num_attention_heads"
            case _intermediateSize = "intermediate_size"
            case _patchSize = "patch_size"
            case _initPosEmbeddingHeight = "init_pos_emb_height"
            case _initPosEmbeddingWidth = "init_pos_emb_width"
            case _channels = "num_channels"
            case _mergeKernelSize = "merge_kernel_size"
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 2048
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 36
        }

        private let _intermediateSize: Int?
        public var intermediateSize: Int {
            _intermediateSize ?? 11008
        }

        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 16
        }

        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? 2
        }

        private let _vocabularySize: Int?
        public var vocabularySize: Int {
            _vocabularySize ?? 152_681
        }

        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-6
        }

        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 1_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 32768
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? true
        }

        private let _blockSize: Int?
        public var blockSize: Int {
            _blockSize ?? 6
        }

        private let _causalAttention: Bool?
        public var causalAttention: Bool {
            _causalAttention ?? false
        }

        private let _textMaskTokenID: Int?
        public var textMaskTokenID: Int {
            _textMaskTokenID ?? 151_676
        }

        private let _nullTokenID: Int?
        public var nullTokenID: Int {
            _nullTokenID ?? 152_678
        }

        private let _switchTokenID: Int?
        public var switchTokenID: Int {
            _switchTokenID ?? 152_679
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case _intermediateSize = "intermediate_size"
            case _attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case _vocabularySize = "vocab_size"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _blockSize = "block_size"
            case _causalAttention = "causal_attn"
            case _textMaskTokenID = "text_mask_token_id"
            case _nullTokenID = "null_token_id"
            case _switchTokenID = "switch_token_id"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 151_665
    }

    private let _boxStartTokenID: Int?
    public var boxStartTokenID: Int {
        _boxStartTokenID ?? 151_668
    }

    private let _boxEndTokenID: Int?
    public var boxEndTokenID: Int {
        _boxEndTokenID ?? 151_669
    }

    private let _coordStartTokenID: Int?
    public var coordStartTokenID: Int {
        _coordStartTokenID ?? 151_677
    }

    private let _coordEndTokenID: Int?
    public var coordEndTokenID: Int {
        _coordEndTokenID ?? 152_677
    }

    private let _refStartTokenID: Int?
    public var refStartTokenID: Int {
        _refStartTokenID ?? 151_672
    }

    private let _refEndTokenID: Int?
    public var refEndTokenID: Int {
        _refEndTokenID ?? 151_673
    }

    private let _noneTokenID: Int?
    public var noneTokenID: Int {
        _noneTokenID ?? 4064
    }

    private let _connectorLayers: Int?
    public var connectorLayers: Int {
        _connectorLayers ?? 2
    }

    private let _vocabularySize: Int?
    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    private let _eosTokenID: IntOrIntArray?
    public var eosTokenIDs: [Int]? {
        _eosTokenID?.values
    }

    private let _futureTokens: Int?
    public var futureTokens: Int {
        _futureTokens ?? 6
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _boxStartTokenID = "box_start_token_id"
        case _boxEndTokenID = "box_end_token_id"
        case _coordStartTokenID = "coord_start_token_id"
        case _coordEndTokenID = "coord_end_token_id"
        case _refStartTokenID = "ref_start_token_id"
        case _refEndTokenID = "ref_end_token_id"
        case _noneTokenID = "none_token_id"
        case _connectorLayers = "mlp_connector_layers"
        case _vocabularySize = "vocab_size"
        case _eosTokenID = "eos_token_id"
        case _futureTokens = "n_future_tokens"
    }
}

public struct LocateAnythingProcessorConfiguration: Decodable, Sendable {
    private struct NestedImageProcessor: Decodable {
        let patchSize: Int?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let inTokenLimit: Int?
        let mergeKernelSize: [Int]?

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case inTokenLimit = "in_token_limit"
            case mergeKernelSize = "merge_kernel_size"
        }
    }

    public let processorClass: String
    public let patchSize: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let inTokenLimit: Int
    public let mergeKernelSize: [Int]
    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<IMG_CONTEXT>"
    }

    private let _imageStartToken: String?
    public var imageStartToken: String {
        _imageStartToken ?? "<img>"
    }

    private let _imageEndToken: String?
    public var imageEndToken: String {
        _imageEndToken ?? "</img>"
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case patchSize = "patch_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case inTokenLimit = "in_token_limit"
        case mergeKernelSize = "merge_kernel_size"
        case _imageToken = "image_token"
        case _imageStartToken = "image_start_token"
        case _imageEndToken = "image_end_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "LocateAnythingProcessor"
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? nested?.patchSize
            ?? 14
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd
            ?? [0.5, 0.5, 0.5]
        inTokenLimit =
            try c.decodeIfPresent(Int.self, forKey: .inTokenLimit)
            ?? nested?.inTokenLimit
            ?? 25600
        mergeKernelSize =
            try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize)
            ?? nested?.mergeKernelSize
            ?? [2, 2]
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _imageStartToken = try c.decodeIfPresent(String.self, forKey: ._imageStartToken)
        _imageEndToken = try c.decodeIfPresent(String.self, forKey: ._imageEndToken)
    }
}

// MARK: - Processor

public struct LocateAnythingProcessor: UserInputProcessor {
    private let config: LocateAnythingProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int

    public init(_ config: LocateAnythingProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID = tokenizer.convertTokenToId(config.imageToken) ?? 151_665
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        guard !images.isEmpty else {
            let prompt = renderPrompt(input: input, imageCount: 0)
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        var patchArrays: [MLXArray] = []
        var frames: [THW] = []
        var placeholderCounts: [Int] = []
        for image in images {
            let processed = try preprocess(
                image: image.asCIImage(),
                processing: input.processing,
            )
            patchArrays.append(processed.pixels)
            frames.append(THW(1, processed.gridHeight, processed.gridWidth))
            placeholderCounts.append(processed.placeholderCount)
        }

        let prompt = expandImagePlaceholders(
            in: renderPrompt(input: input, imageCount: images.count),
            counts: placeholderCounts,
        )
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: concatenated(patchArrays, axis: 0), frames: frames),
        )
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let gridHeight: Int
        let gridWidth: Int
        let placeholderCount: Int
    }

    private func renderPrompt(input: UserInput, imageCount: Int) -> String {
        let messages: [Chat.Message]
        switch input.prompt {
        case .text(let text):
            messages = [.user(text)]
        case .chat(let chat):
            messages = chat
        case .messages:
            let placeholders = (1 ... max(imageCount, 1)).map { "<image-\($0)>" }.joined()
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                + "<|im_start|>user\n\(placeholders)\(input.prompt.description)<|im_end|>\n"
                + "<|im_start|>assistant\n"
        }

        var result = ""
        var nextImageIndex = 1
        if messages.first?.role != .system {
            result += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        }
        for message in messages {
            result += "<|im_start|>\(message.role.rawValue)\n"
            for _ in message.images {
                result += "<image-\(nextImageIndex)>"
                nextImageIndex += 1
            }
            result += message.content
            result += "<|im_end|>\n"
        }
        result += "<|im_start|>assistant\n"
        return result
    }

    private func expandImagePlaceholders(in prompt: String, counts: [Int]) -> String {
        var rendered = prompt
        for (index, count) in counts.enumerated() {
            let placeholder = "<image-\(index + 1)>"
            let replacement =
                config.imageStartToken
                + String(repeating: config.imageToken, count: count)
                + config.imageEndToken
            rendered = rendered.replacingOccurrences(of: placeholder, with: replacement)
        }
        return rendered
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.zeroOrigin(MediaProcessing.apply(image, processing: processing))
        let originalWidth = max(Int(image.extent.width.rounded()), 1)
        let originalHeight = max(Int(image.extent.height.rounded()), 1)
        let resizedSize = targetSize(width: originalWidth, height: originalHeight)
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: resizedSize.width, height: resizedSize.height),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(
            resized, width: resizedSize.width, height: resizedSize.height)
        let normalized = MediaProcessing.normalizeCHW(
            raw,
            width: resizedSize.width,
            height: resizedSize.height,
            mean: config.imageMean,
            std: config.imageStd,
        )
        let patches = patchify(
            normalized,
            width: resizedSize.width,
            height: resizedSize.height,
        )
        let gridHeight = resizedSize.height / config.patchSize
        let gridWidth = resizedSize.width / config.patchSize
        let mergeLength = max(
            (config.mergeKernelSize.first ?? 2) * (config.mergeKernelSize.dropFirst().first ?? 2), 1
        )
        return ProcessedImage(
            pixels: MLXArray(
                patches, [gridHeight * gridWidth, 3, config.patchSize, config.patchSize]),
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            placeholderCount: (gridHeight * gridWidth) / mergeLength,
        )
    }

    private func targetSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let patchSize = config.patchSize
        var width = width
        var height = height
        if (width / patchSize) * (height / patchSize) > config.inTokenLimit {
            let scale = sqrt(
                Float(config.inTokenLimit)
                    / Float(max((width / patchSize) * (height / patchSize), 1)),
            )
            width = max(Int(Float(width) * scale), patchSize)
            height = max(Int(Float(height) * scale), patchSize)
        }

        let padWidth = (config.mergeKernelSize.dropFirst().first ?? 2) * patchSize
        let padHeight = (config.mergeKernelSize.first ?? 2) * patchSize
        let targetWidth = Int(ceil(Double(width) / Double(padWidth))) * padWidth
        let targetHeight = Int(ceil(Double(height) / Double(padHeight))) * padHeight
        return (max(targetWidth, padWidth), max(targetHeight, padHeight))
    }

    private func patchify(_ pixels: [Float], width: Int, height: Int) -> [Float] {
        let patch = config.patchSize
        let gridHeight = height / patch
        let gridWidth = width / patch
        var output = [Float]()
        output.reserveCapacity(gridHeight * gridWidth * 3 * patch * patch)
        for patchRow in 0 ..< gridHeight {
            for patchCol in 0 ..< gridWidth {
                for channel in 0 ..< 3 {
                    let channelOffset = channel * width * height
                    for row in 0 ..< patch {
                        let sourceRow = patchRow * patch + row
                        for col in 0 ..< patch {
                            let sourceCol = patchCol * patch + col
                            output.append(pixels[channelOffset + sourceRow * width + sourceCol])
                        }
                    }
                }
            }
        }
        return output
    }

}

// MARK: - Language

private enum LocateAnythingLanguage {
    final class Attention: Module {
        let config: LocateAnythingConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        let rope: RoPE

        init(_ config: LocateAnythingConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * headDim,
                bias: true,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: true,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: true,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: false,
            )
            rope = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = queries.reshaped(batch, length, config.attentionHeads, headDim)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batch, length, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batch, length, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return oProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(_ config: LocateAnythingConfiguration.TextConfiguration) {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LocateAnythingConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let hidden = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return hidden + mlp(postAttentionLayerNorm(hidden))
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: LocateAnythingConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]?,
            inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            let h0: MLXArray
            if let inputEmbedding {
                h0 = inputEmbedding
            } else if let inputs {
                h0 = embedTokens(inputs)
            } else {
                preconditionFailure("inputs or inputEmbedding must be provided")
            }

            var h = h0
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        let config: LocateAnythingConfiguration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: LocateAnythingConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum LocateAnythingVision {
    final class Learnable2DInterpPosEmb: Module {
        @ModuleInfo var weight: MLXArray

        init(height: Int, width: Int, dim: Int) {
            _weight.wrappedValue = MLXArray.ones([height, width, dim])
        }

        func callAsFunction(frames: [THW]) -> MLXArray {
            let sourceHeight = weight.dim(0)
            let sourceWidth = weight.dim(1)
            let dim = weight.dim(2)
            let source = weight.transposed(2, 0, 1).reshaped(1, dim, sourceHeight, sourceWidth)
            let parts = frames.map { frame -> MLXArray in
                if frame.h == sourceHeight, frame.w == sourceWidth {
                    return weight.reshaped(sourceHeight * sourceWidth, dim)
                }
                let interpolated = bicubicInterpolate(
                    source,
                    size: (frame.h, frame.w),
                )
                return interpolated.reshaped(dim, frame.h * frame.w).transposed(1, 0)
            }
            return concatenated(parts, axis: 0)
        }
    }

    final class PatchEmbed: Module {
        @ModuleInfo var proj: Conv2d
        @ModuleInfo(key: "pos_emb") var posEmbedding: Learnable2DInterpPosEmb

        init(_ config: LocateAnythingConfiguration.VisionConfiguration) {
            _proj.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _posEmbedding.wrappedValue = Learnable2DInterpPosEmb(
                height: config.initPosEmbeddingHeight,
                width: config.initPosEmbeddingWidth,
                dim: config.hiddenSize,
            )
        }

        func callAsFunction(_ x: MLXArray, frames: [THW]) -> MLXArray {
            let hidden = proj(x).reshaped(x.dim(0), -1)
            return hidden + posEmbedding(frames: frames).asType(hidden.dtype)
        }
    }

    final class Rope2DPosEmb {
        let dim: Int
        let thetaBase: Float

        init(dim: Int, thetaBase: Float = 10000) {
            self.dim = dim
            self.thetaBase = thetaBase
        }

        func cosSin(frames: [THW]) -> (MLXArray, MLXArray) {
            let half = dim / 2
            let quarter = dim / 4
            var cosValues: [Float] = []
            var sinValues: [Float] = []
            let total = frames.reduce(0) { $0 + $1.h * $1.w }
            cosValues.reserveCapacity(total * half)
            sinValues.reserveCapacity(total * half)

            for frame in frames {
                for row in 0 ..< frame.h {
                    for col in 0 ..< frame.w {
                        for i in 0 ..< quarter {
                            let freq = pow(thetaBase, -Float(i * 4) / Float(dim))
                            let xAngle = Float(col) * freq
                            let yAngle = Float(row) * freq
                            cosValues.append(cos(xAngle))
                            sinValues.append(sin(xAngle))
                            cosValues.append(cos(yAngle))
                            sinValues.append(sin(yAngle))
                        }
                    }
                }
            }

            return (
                MLXArray(cosValues, [total, half]),
                MLXArray(sinValues, [total, half]),
            )
        }
    }

    static func apply2DRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let sequence = x.dim(0)
        let heads = x.dim(1)
        let headDim = x.dim(2)
        let half = headDim / 2
        let paired = x.asType(.float32).reshaped(sequence, heads, half, 2)
        let real = paired[0..., 0..., 0..., 0]
        let imag = paired[0..., 0..., 0..., 1]
        let cos = cos.expandedDimensions(axis: 1)
        let sin = sin.expandedDimensions(axis: 1)
        let rotated = stacked(
            [
                real * cos - imag * sin,
                real * sin + imag * cos,
            ], axis: -1,
        )
        return rotated.reshaped(sequence, heads, headDim).asType(x.dtype)
    }

    static func blockAttentionMask(frames: [THW], dtype: DType) -> MLXArray? {
        guard frames.count > 1 else {
            return nil
        }
        let lengths = frames.map { $0.h * $0.w }
        let total = lengths.reduce(0, +)
        var values = Array(repeating: Float(-1e9), count: total * total)
        var offset = 0
        for length in lengths {
            for row in offset ..< offset + length {
                for col in offset ..< offset + length {
                    values[row * total + col] = 0
                }
            }
            offset += length
        }
        return MLXArray(values, [1, 1, total, total]).asType(dtype)
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var wqkv: Linear
        @ModuleInfo var wo: Linear

        init(_ config: LocateAnythingConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _wqkv.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize * 3,
                bias: true,
            )
            _wo.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?)
            -> MLXArray
        {
            let sequence = x.dim(0)
            let qkv = wqkv(x)
                .reshaped(sequence, 3, heads, headDim)
            let parts = split(qkv, parts: 3, axis: 1)
            var query = parts[0].squeezed(axis: 1)
            var key = parts[1].squeezed(axis: 1)
            var value = parts[2].squeezed(axis: 1)

            query = LocateAnythingVision.apply2DRotary(query, cos: cos, sin: sin)
            key = LocateAnythingVision.apply2DRotary(key, cos: cos, sin: sin)

            query = query.transposed(1, 0, 2).expandedDimensions(axis: 0)
            key = key.transposed(1, 0, 2).expandedDimensions(axis: 0)
            value = value.transposed(1, 0, 2).expandedDimensions(axis: 0)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
                mask.map { .array($0) } ?? .none
            let output = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: key,
                values: value,
                scale: scale,
                mask: attentionMask,
            )
            .squeezed(axis: 0)
            .transposed(1, 0, 2)
            .reshaped(sequence, -1)
            return wo(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc0: Linear
        @ModuleInfo var fc1: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: LocateAnythingConfiguration.VisionConfiguration) {
            _fc0.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: true,
            )
            _fc1.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc1(activation(fc0(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo var norm0: LayerNorm
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        let mlp: MLP

        init(_ config: LocateAnythingConfiguration.VisionConfiguration) {
            _norm0.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
            _attn.wrappedValue = Attention(config)
            mlp = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?)
            -> MLXArray
        {
            let hidden = x + attn(norm0(x), cos: cos, sin: sin, mask: mask)
            return hidden + mlp(norm1(hidden))
        }
    }

    final class VisionModel: Module {
        let config: LocateAnythingConfiguration.VisionConfiguration
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let blocks: [VisionBlock]
        @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm
        let ropePosEmbedding: Rope2DPosEmb

        init(_ config: LocateAnythingConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbed(config)
            blocks = (0 ..< config.hiddenLayers).map { _ in VisionBlock(config) }
            _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
            ropePosEmbedding = Rope2DPosEmb(
                dim: config.hiddenSize / config.attentionHeads,
            )
        }

        var patchEmbeddingDtype: DType {
            patchEmbed.proj.weight.dtype
        }

        func callAsFunction(_ x: MLXArray, frames: [THW]) -> [MLXArray] {
            var hidden = patchEmbed(x, frames: frames)
            let (cos, sin) = ropePosEmbedding.cosSin(frames: frames)
            let mask = LocateAnythingVision.blockAttentionMask(frames: frames, dtype: hidden.dtype)
            for block in blocks {
                hidden = block(
                    hidden, cos: cos.asType(hidden.dtype), sin: sin.asType(hidden.dtype), mask: mask
                )
            }
            hidden = finalLayerNorm(hidden)
            return patchMerger(hidden, frames: frames, mergeKernelSize: config.mergeKernelSize)
        }

        private func patchMerger(
            _ x: MLXArray,
            frames: [THW],
            mergeKernelSize: [Int],
        )
            -> [MLXArray]
        {
            let mergeHeight = mergeKernelSize.first ?? 2
            let mergeWidth = mergeKernelSize.dropFirst().first ?? 2
            var outputs: [MLXArray] = []
            var offset = 0
            for frame in frames {
                let length = frame.h * frame.w
                let sequence = x[offset ..< offset + length, 0...]
                let newHeight = frame.h / mergeHeight
                let newWidth = frame.w / mergeWidth
                let hidden = sequence.dim(-1)
                let merged =
                    sequence
                    .reshaped(newHeight, mergeHeight, newWidth, mergeWidth, hidden)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(newHeight * newWidth, mergeHeight * mergeWidth, hidden)
                outputs.append(merged)
                offset += length
            }
            return outputs
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embed.proj.weight"), value.ndim == 4 {
                    let shape = value.shape
                    if shape[1] == 3 {
                        sanitized[key] = value.transposed(0, 2, 3, 1)
                    } else {
                        sanitized[key] = value
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Projector

private final class LocateAnythingMultiModalProjector: Module {
    let inputDim: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU()

    init(_ config: LocateAnythingConfiguration) {
        let visionHidden = config.visionConfiguration.hiddenSize
        let merge = config.visionConfiguration.mergeKernelSize
        inputDim = visionHidden * (merge.first ?? 2) * (merge.dropFirst().first ?? 2)
        let textHidden = config.textConfiguration.hiddenSize
        _layerNorm.wrappedValue = LayerNorm(dimensions: inputDim)
        _linear1.wrappedValue = Linear(inputDim, textHidden, bias: true)
        _linear2.wrappedValue = Linear(textHidden, textHidden, bias: true)
    }

    func callAsFunction(_ imageFeatures: [MLXArray]) -> MLXArray {
        let hidden = concatenated(imageFeatures, axis: 0).reshaped(-1, inputDim)
        return linear2(activation(linear1(layerNorm(hidden))))
    }
}

// MARK: - Model

public final class LocateAnything: Module, LanguageModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let config: LocateAnythingConfiguration
    @ModuleInfo(key: "vision_tower") private var visionTower: LocateAnythingVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel:
        LocateAnythingLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        LocateAnythingMultiModalProjector

    public init(_ config: LocateAnythingConfiguration) {
        self.config = config
        vocabularySize = config.textConfiguration.vocabularySize
        kvHeads = (0 ..< config.textConfiguration.hiddenLayers).map {
            _ in config.textConfiguration.kvHeads
        }
        _visionTower.wrappedValue = LocateAnythingVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LocateAnythingLanguage.LanguageModel(config.textConfiguration)
        _multiModalProjector.wrappedValue = LocateAnythingMultiModalProjector(config)
    }

    private func inputEmbeddings(inputIDs: MLXArray, image: LMInput.ProcessedImage?) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let image, let frames = image.frames else {
            return embeddings
        }

        let pixels = image.pixels.asType(visionTower.patchEmbeddingDtype)
            .transposed(0, 2, 3, 1)
        let imageFeatures = multiModalProjector(visionTower(pixels, frames: frames))
            .asType(embeddings.dtype)
        let ids = inputIDs.flattened().asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        precondition(
            positions.count == imageFeatures.dim(0),
            "LocateAnything image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0))",
        )

        embeddings[0..., MLXArray(positions), 0...] = imageFeatures.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(inputIDs: input.text.tokens, image: input.image)
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(nil, cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        eval(cache)

        let result = languageModel(
            nil, cache: cache, inputEmbedding: embeddings[0..., processed..., 0...])
        prefill.progress?(totalPositions, totalPositions)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key == "language_model.lm_head.weight" {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("vision_model.") {
                newKey =
                    newKey
                    .replacingOccurrences(of: "vision_model.encoder.", with: "vision_tower.")
                    .replacingOccurrences(of: "vision_model.", with: "vision_tower.")
            } else if newKey.hasPrefix("mlp1.") {
                newKey =
                    newKey
                    .replacingOccurrences(of: "mlp1.0.", with: "multi_modal_projector.layer_norm.")
                    .replacingOccurrences(of: "mlp1.1.", with: "multi_modal_projector.linear_1.")
                    .replacingOccurrences(of: "mlp1.3.", with: "multi_modal_projector.linear_2.")
            }
            sanitized[newKey] = value
        }
        return visionTower.sanitize(weights: sanitized)
    }
}
