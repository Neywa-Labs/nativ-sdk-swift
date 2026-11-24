import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct JinaVLMConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vocabularySize: Int
        public let additionalVocabularySize: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let maxPositionEmbeddings: Int
        public let useQKNorm: Bool
        public let tieWordEmbeddings: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "n_layers"
            case numHiddenLayers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case additionalVocabularySize = "additional_vocab_size"
            case ropeTheta = "rope_theta"
            case maxSequenceLength = "max_sequence_length"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case blockConfiguration = "block_config"
        }

        public init(
            modelType: String = "jina_vlm",
            hiddenSize: Int = 2048,
            hiddenLayers: Int = 28,
            attentionHeads: Int = 16,
            kvHeads: Int = 8,
            headDim: Int = 128,
            vocabularySize: Int = 151_936,
            additionalVocabularySize: Int = 128,
            intermediateSize: Int = 6144,
            rmsNormEps: Float = 1e-6,
            ropeTheta: Float = 1_000_000,
            maxPositionEmbeddings: Int = 40960,
            useQKNorm: Bool = true,
            tieWordEmbeddings: Bool = false,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.vocabularySize = vocabularySize
            self.additionalVocabularySize = additionalVocabularySize
            self.intermediateSize = intermediateSize
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.useQKNorm = useQKNorm
            self.tieWordEmbeddings = tieWordEmbeddings
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let block = try? c.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: .blockConfiguration,
            )
            let attention = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("attn_config"),
            )
            let ffn = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("ffn_config"),
            )
            let norm = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("lnorm_config"),
            )

            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "jina_vlm"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? c.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
                ?? 28
            attentionHeads =
                try attention?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("n_heads")) ?? 16
            kvHeads =
                try attention?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("n_kv_heads")) ?? 8
            headDim =
                try attention?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("head_dim")) ?? 128
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            additionalVocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .additionalVocabularySize) ?? 128
            intermediateSize =
                try ffn?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("size")) ?? 6144
            rmsNormEps =
                try norm?.decodeIfPresent(Float.self, forKey: JinaVLMJSONKey("eps")) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxSequenceLength)
                ?? c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? 40960
            useQKNorm =
                try attention?.decodeIfPresent(Bool.self, forKey: JinaVLMJSONKey("q_lnorm"))
                ?? true
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let headDim: Int
        public let patchSize: Int
        public let imageSize: Int
        public let channels: Int
        public let intermediateSize: Int
        public let layerNormEps: Float
        public let useBias: Bool
        public let useCLSToken: Bool
        public let postLayerNorm: Bool
        public let activation: String
        public let vitLayers: [Int]
        public let outputSize: Int
        public let poolingH: Int
        public let poolingW: Int
        public let connectorHiddenSize: Int

        public var cropPatches: Int {
            imageSize / patchSize
        }

        public var tokenLengthH: Int {
            (cropPatches + poolingH - 1) / poolingH
        }

        public var tokenLengthW: Int {
            (cropPatches + poolingW - 1) / poolingW
        }

        public var tokensPerImage: Int {
            tokenLengthH * tokenLengthW
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "n_layers"
            case patchSize = "patch_size"
            case inputSize = "input_size"
            case channels = "n_channels"
            case useCLSToken = "use_cls_token"
            case postLayerNorm = "post_lnorm"
            case vitLayers = "vit_layers"
            case outputSize = "output_size"
            case blockConfiguration = "block_config"
            case connectorConfiguration = "vl_connector_config"
        }

        public init(
            modelType: String = "jina_vlm",
            hiddenSize: Int = 1152,
            hiddenLayers: Int = 27,
            attentionHeads: Int = 16,
            headDim: Int = 72,
            patchSize: Int = 14,
            imageSize: Int = 378,
            channels: Int = 3,
            intermediateSize: Int = 4304,
            layerNormEps: Float = 1e-6,
            useBias: Bool = true,
            useCLSToken: Bool = false,
            postLayerNorm: Bool = true,
            activation: String = "gelu_pytorch_tanh",
            vitLayers: [Int] = [-4, -10],
            outputSize: Int = 2048,
            poolingH: Int = 2,
            poolingW: Int = 2,
            connectorHiddenSize: Int = 6144,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.headDim = headDim
            self.patchSize = patchSize
            self.imageSize = imageSize
            self.channels = channels
            self.intermediateSize = intermediateSize
            self.layerNormEps = layerNormEps
            self.useBias = useBias
            self.useCLSToken = useCLSToken
            self.postLayerNorm = postLayerNorm
            self.activation = activation
            self.vitLayers = vitLayers
            self.outputSize = outputSize
            self.poolingH = poolingH
            self.poolingW = poolingW
            self.connectorHiddenSize = connectorHiddenSize
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let block = try? c.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: .blockConfiguration,
            )
            let attention = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("attn_config"),
            )
            let ffn = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("ffn_config"),
            )
            let norm = try? block?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("lnorm_config"),
            )
            let connector = try? c.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: .connectorConfiguration,
            )
            let connectorMLP = try? connector?.nestedContainer(
                keyedBy: JinaVLMJSONKey.self,
                forKey: JinaVLMJSONKey("mlp_projector_config"),
            )

            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "jina_vlm"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads =
                try attention?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("n_heads")) ?? 16
            headDim =
                try attention?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("head_dim")) ?? 72
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            let inputSize = try c.decodeIfPresent([Int].self, forKey: .inputSize) ?? [378, 378]
            imageSize = inputSize.first ?? 378
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            intermediateSize =
                try ffn?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("size")) ?? 4304
            layerNormEps =
                try norm?.decodeIfPresent(Float.self, forKey: JinaVLMJSONKey("eps")) ?? 1e-6
            useBias =
                try attention?.decodeIfPresent(Bool.self, forKey: JinaVLMJSONKey("q_bias")) ?? true
            useCLSToken = try c.decodeIfPresent(Bool.self, forKey: .useCLSToken) ?? false
            postLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .postLayerNorm) ?? true
            activation =
                try ffn?.decodeIfPresent(String.self, forKey: JinaVLMJSONKey("activation_type"))
                ?? "gelu_pytorch_tanh"
            vitLayers = try c.decodeIfPresent([Int].self, forKey: .vitLayers) ?? [-4, -10]
            outputSize = try c.decodeIfPresent(Int.self, forKey: .outputSize) ?? 2048
            poolingH =
                try connector?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("pooling_h")) ?? 2
            poolingW =
                try connector?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("pooling_w")) ?? 2
            connectorHiddenSize =
                try connectorMLP?.decodeIfPresent(Int.self, forKey: JinaVLMJSONKey("size"))
                ?? 6144
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let vocabularySize: Int
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let padTokenID: Int
    public let imageTokenIndex: Int
    public let imageTokenID: Int
    public let imageStartTokenID: Int
    public let imageEndTokenID: Int
    public let imagePatchTokenID: Int
    public let imageColumnTokenID: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
        case imageTokenIndex = "image_token_index"
        case imageTokenID = "image_token_id"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case imagePatchTokenID = "image_patch_token_id"
        case imageColumnTokenID = "image_column_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "jvlm"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration()
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 151_643
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 151_643
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 151_643
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 151_940
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? imageTokenIndex
        imageStartTokenID = try c.decodeIfPresent(Int.self, forKey: .imageStartTokenID) ?? 151_936
        imageEndTokenID = try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 151_937
        imagePatchTokenID = try c.decodeIfPresent(Int.self, forKey: .imagePatchTokenID) ?? 151_938
        imageColumnTokenID = try c.decodeIfPresent(Int.self, forKey: .imageColumnTokenID) ?? 151_939
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
    }
}

private struct JinaVLMJSONKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init(_ stringValue: String) {
        self.stringValue = stringValue
        intValue = nil
    }

    init?(stringValue: String) {
        self.init(stringValue)
    }

    init?(intValue: Int) {
        stringValue = "\(intValue)"
        self.intValue = intValue
    }
}

// MARK: - Processor

public struct JinaVLMProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let baseInputSize: [Int]
    public let patchSize: Int
    public let maxCrops: Int
    public let minPixels: Int
    public let maxPixels: Int
    public let overlapMargins: [Int]
    public let poolingH: Int
    public let poolingW: Int
    public let useColumnTokens: Bool
    public let startTokenID: Int
    public let endTokenID: Int
    public let patchTokenID: Int
    public let columnTokenID: Int
    public let imageToken: String

    public var cropPatches: Int {
        baseInputSize[0] / patchSize
    }

    public var tokenLengthH: Int {
        (cropPatches + poolingH - 1) / poolingH
    }

    public var tokenLengthW: Int {
        (cropPatches + poolingW - 1) / poolingW
    }

    public var tokensPerImage: Int {
        tokenLengthH * tokenLengthW
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case baseInputSize = "base_input_size"
        case patchSize = "patch_size"
        case maxCrops = "max_crops"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case overlapMargins = "overlap_margins"
        case poolingH = "pooling_h"
        case poolingW = "pooling_w"
        case useColumnTokens = "use_column_tokens"
        case startTokenID = "start_token_id"
        case endTokenID = "end_token_id"
        case patchTokenID = "patch_token_id"
        case columnTokenID = "column_token_id"
        case imageToken = "image_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "JinaVLMProcessor"
        baseInputSize = try c.decodeIfPresent([Int].self, forKey: .baseInputSize) ?? [378, 378]
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? 12
        minPixels = try c.decodeIfPresent(Int.self, forKey: .minPixels) ?? 3136
        maxPixels = try c.decodeIfPresent(Int.self, forKey: .maxPixels) ?? 1_003_520
        overlapMargins = try c.decodeIfPresent([Int].self, forKey: .overlapMargins) ?? [4, 4]
        poolingH = try c.decodeIfPresent(Int.self, forKey: .poolingH) ?? 2
        poolingW = try c.decodeIfPresent(Int.self, forKey: .poolingW) ?? 2
        useColumnTokens = try c.decodeIfPresent(Bool.self, forKey: .useColumnTokens) ?? true
        startTokenID = try c.decodeIfPresent(Int.self, forKey: .startTokenID) ?? 151_936
        endTokenID = try c.decodeIfPresent(Int.self, forKey: .endTokenID) ?? 151_937
        patchTokenID = try c.decodeIfPresent(Int.self, forKey: .patchTokenID) ?? 151_938
        columnTokenID = try c.decodeIfPresent(Int.self, forKey: .columnTokenID) ?? 151_939
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|image|>"
    }
}

public struct JinaVLMProcessor: UserInputProcessor {
    private let config: JinaVLMProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: JinaVLMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(
            from: input, imageToken: config.imageToken, imageCount: images.count)

        guard !images.isEmpty else {
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        let processedImages = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let splits = prompt.components(separatedBy: config.imageToken)
        var inputIDs = [Int]()
        var pixelValues = [MLXArray]()
        var metadataValues = [MLXArray]()
        var imageIndex = 0

        for index in splits.indices {
            if !splits[index].isEmpty {
                inputIDs.append(
                    contentsOf: tokenizer.encode(text: splits[index], addSpecialTokens: false))
            }
            if index < splits.count - 1, imageIndex < processedImages.count {
                let processed = processedImages[imageIndex]
                let offsetMetadata = offsetMetadata(processed.metadata, by: inputIDs.count)
                inputIDs.append(contentsOf: processed.imageTokens)
                pixelValues.append(processed.patches)
                metadataValues.append(offsetMetadata)
                imageIndex += 1
            }
        }

        if imageIndex < processedImages.count {
            for image in processedImages[imageIndex...] {
                let offsetMetadata = offsetMetadata(image.metadata, by: inputIDs.count)
                inputIDs.append(contentsOf: image.imageTokens)
                pixelValues.append(image.patches)
                metadataValues.append(offsetMetadata)
            }
        }

        let tokenArray = MLXArray(inputIDs).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: concatenated(pixelValues, axis: 0)),
            video: .init(pixels: concatenated(metadataValues, axis: 0)),
        )
    }

    private struct ProcessedImage {
        let patches: MLXArray
        let imageTokens: [Int]
        let metadata: MLXArray
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing).toSRGB()
        let canonical = MediaProcessing.zeroOrigin(image)
        var width = max(Int(canonical.extent.width.rounded()), 1)
        var height = max(Int(canonical.extent.height.rounded()), 1)
        var source = canonical

        let smart = Self.smartResize(
            height: height,
            width: width,
            factor: config.patchSize,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )
        if smart.height != height || smart.width != width {
            source = source.resampled(
                to: CGSize(width: smart.width, height: smart.height),
                method: .bicubic,
            )
            width = smart.width
            height = smart.height
        }

        let (patches, imageTokens, patchOrdering, masks) = try imageToPatchesAndTokens(
            image: source,
            width: width,
            height: height,
        )
        let imageInputIdx = buildImageInputIdx(imageTokens: imageTokens, patchOrder: patchOrdering)
        let metadata = packMetadata(imageInputIdx: imageInputIdx, imageMasks: masks)
        return ProcessedImage(patches: patches, imageTokens: imageTokens, metadata: metadata)
    }

    private func imageToPatchesAndTokens(
        image: CIImage,
        width originalWidth: Int,
        height originalHeight: Int,
    )
        throws -> (MLXArray, [Int], [Int], [[Float]])
    {
        let baseH = config.baseInputSize[0]
        let baseW = config.baseInputSize.count > 1 ? config.baseInputSize[1] : baseH
        let patchSize = config.patchSize
        let tokensPerImage = config.tokensPerImage
        let cropPatches = config.cropPatches
        let leftMargin = config.overlapMargins.first ?? 0
        let rightMargin = config.overlapMargins.count > 1 ? config.overlapMargins[1] : leftMargin
        let totalMarginPixels = patchSize * (rightMargin + leftMargin)
        let cropWindowPatches = cropPatches - (rightMargin + leftMargin)
        guard cropWindowPatches > 0 else {
            throw VLMError.processing("Jina-VLM overlap margins leave no crop interior")
        }
        let cropWindowSize = cropWindowPatches * patchSize

        let tiling = Self.selectTiling(
            height: originalHeight - totalMarginPixels,
            width: originalWidth - totalMarginPixels,
            patchSize: cropWindowSize,
            maxCrops: config.maxCrops,
        )

        let targetHeight = tiling.rows * cropWindowSize + totalMarginPixels
        let targetWidth = tiling.cols * cropWindowSize + totalMarginPixels
        let resized = Self.resizeExact(image: image, width: targetWidth, height: targetHeight)
        let normalized = Self.minMaxNormalize(resized)
        let fullMask = Array(repeating: Float(1), count: targetWidth * targetHeight)

        var patchValues = [Float]()
        var maskValues = [[Float]]()
        var localPatchOrder = [Int]()
        var orderingOffset = 0

        for row in 0 ..< tiling.rows {
            let y0 = row * cropWindowSize
            let cropY0 = row == 0 ? 0 : leftMargin / config.poolingH
            var cropH = cropPatches - (rightMargin + leftMargin)
            if row == 0 {
                cropH += leftMargin
            }
            if row == tiling.rows - 1 {
                cropH += rightMargin
            }
            let pooledH = (cropH + config.poolingH - 1) / config.poolingH

            for col in 0 ..< tiling.cols {
                let x0 = col * cropWindowSize
                let cropX0 = col == 0 ? 0 : leftMargin / config.poolingW
                var cropW = cropPatches - (rightMargin + leftMargin)
                if col == 0 {
                    cropW += leftMargin
                }
                if col == tiling.cols - 1 {
                    cropW += rightMargin
                }
                let pooledW = (cropW + config.poolingW - 1) / config.poolingW

                var ordering = Array(repeating: -1, count: tokensPerImage)
                for r in 0 ..< pooledH {
                    for c in 0 ..< pooledW {
                        let targetR = cropY0 + r
                        let targetC = cropX0 + c
                        if targetR < config.tokenLengthH, targetC < config.tokenLengthW {
                            ordering[targetR * config.tokenLengthW + targetC] =
                                orderingOffset + r * pooledW + c
                        }
                    }
                }
                localPatchOrder.append(contentsOf: ordering)

                let cropPixels = MediaProcessing.cropPixels(
                    normalized,
                    sourceWidth: targetWidth,
                    x: x0,
                    y: y0,
                    width: baseW,
                    height: baseH,
                    channels: 3,
                )
                patchValues.append(
                    contentsOf: MediaProcessing.patchifyPixels(
                        cropPixels,
                        width: baseW,
                        height: baseH,
                        channels: 3,
                        patchSize: patchSize,
                    ),
                )

                let cropMask = MediaProcessing.cropPixels(
                    fullMask,
                    sourceWidth: targetWidth,
                    x: x0,
                    y: y0,
                    width: baseW,
                    height: baseH,
                    channels: 1,
                )
                maskValues.append(
                    MediaProcessing.patchifyMask(
                        cropMask,
                        width: baseW,
                        height: baseH,
                        patchSize: patchSize,
                    ),
                )

                orderingOffset += pooledH * pooledW
            }
        }

        let highResH = Self.patchesFromTiling(
            tiles: tiling.rows,
            pooling: config.poolingH,
            cropPatches: cropPatches,
            cropWindowPatches: cropWindowPatches,
            leftMargin: leftMargin,
            rightMargin: rightMargin,
        )
        let highResW = Self.patchesFromTiling(
            tiles: tiling.cols,
            pooling: config.poolingW,
            cropPatches: cropPatches,
            cropWindowPatches: cropWindowPatches,
            leftMargin: leftMargin,
            rightMargin: rightMargin,
        )
        var highResRow = Array(repeating: config.patchTokenID, count: highResW / config.poolingW)
        if config.useColumnTokens {
            highResRow.append(config.columnTokenID)
        }
        var highResTokens = [config.startTokenID]
        for _ in 0 ..< (highResH / config.poolingH) {
            highResTokens.append(contentsOf: highResRow)
        }
        highResTokens.append(config.endTokenID)

        let thumbnail = Self.minMaxNormalize(
            Self.resizeExact(image: image, width: baseW, height: baseH))
        let thumbnailPatches = MediaProcessing.patchifyPixels(
            thumbnail,
            width: baseW,
            height: baseH,
            channels: 3,
            patchSize: patchSize,
        )
        patchValues.insert(contentsOf: thumbnailPatches, at: 0)

        let patchOrderRows = localPatchOrder.count / tokensPerImage
        var transposedOrder = Array(repeating: -1, count: localPatchOrder.count)
        if patchOrderRows == tiling.rows * tiling.cols {
            for tileRow in 0 ..< tiling.rows {
                for localR in 0 ..< config.tokenLengthH {
                    for tileCol in 0 ..< tiling.cols {
                        for localC in 0 ..< config.tokenLengthW {
                            let originalIndex =
                                ((tileRow * tiling.cols + tileCol) * tokensPerImage)
                                + localR * config.tokenLengthW + localC
                            let transposedIndex =
                                (((tileRow * config.tokenLengthH + localR) * tiling.cols
                                    + tileCol) * config.tokenLengthW) + localC
                            if transposedIndex < transposedOrder.count {
                                transposedOrder[transposedIndex] = localPatchOrder[originalIndex]
                            }
                        }
                    }
                }
            }
        }

        var patchOrdering = localPatchOrder.enumerated().map { index, value in
            value >= 0 && index < transposedOrder.count && transposedOrder[index] >= 0
                ? transposedOrder[index] + tokensPerImage
                : -1
        }
        patchOrdering.insert(contentsOf: 0 ..< tokensPerImage, at: 0)

        var globalRow = Array(repeating: config.patchTokenID, count: config.tokenLengthW)
        if config.useColumnTokens {
            globalRow.append(config.columnTokenID)
        }
        var imageTokens = [config.startTokenID]
        for _ in 0 ..< config.tokenLengthH {
            imageTokens.append(contentsOf: globalRow)
        }
        imageTokens.append(config.endTokenID)
        imageTokens.append(contentsOf: highResTokens)

        maskValues.insert(
            Array(repeating: 1, count: (baseH / patchSize) * (baseW / patchSize)),
            at: 0,
        )
        let cropCount = maskValues.count
        let patchDim = patchSize * patchSize * 3
        let patchArray = MLXArray(
            patchValues,
            [cropCount, (baseH / patchSize) * (baseW / patchSize), patchDim],
        )
        return (patchArray, imageTokens, patchOrdering, maskValues)
    }

    private func buildImageInputIdx(imageTokens: [Int], patchOrder: [Int]) -> [[Int]] {
        let tokensPerImage = config.tokensPerImage
        let imageInputPositions = imageTokens.indices.filter {
            imageTokens[$0] == config.patchTokenID
        }
        let validPatchOrder = patchOrder.filter { $0 >= 0 }
        var sortedPatchIndices = Array(repeating: 0, count: imageInputPositions.count)
        for (value, sourceIndex) in validPatchOrder.enumerated()
        where sourceIndex < sortedPatchIndices.count {
            sortedPatchIndices[sourceIndex] = value
        }

        var flat = Array(repeating: -10000, count: patchOrder.count)
        for (index, order) in patchOrder.enumerated() where order >= 0 {
            let sortedIndex = sortedPatchIndices[order]
            if sortedIndex < imageInputPositions.count {
                flat[index] = imageInputPositions[sortedIndex]
            }
        }

        return stride(from: 0, to: flat.count, by: tokensPerImage).map { start in
            Array(flat[start ..< min(start + tokensPerImage, flat.count)])
        }
    }

    private func packMetadata(imageInputIdx: [[Int]], imageMasks: [[Float]]) -> MLXArray {
        precondition(imageInputIdx.count == imageMasks.count)
        let patchCount = imageMasks.first?.count ?? 0
        var values = [Float]()
        values.reserveCapacity(imageInputIdx.count * (config.tokensPerImage + patchCount))
        for row in 0 ..< imageInputIdx.count {
            values.append(contentsOf: imageInputIdx[row].map(Float.init))
            values.append(contentsOf: imageMasks[row])
        }
        return MLXArray(values, [imageInputIdx.count, config.tokensPerImage + patchCount])
    }

    private func offsetMetadata(_ metadata: MLXArray, by offset: Int) -> MLXArray {
        if offset == 0 {
            return metadata
        }
        let result = metadata
        let tokens = config.tokensPerImage
        let offsets = MLXArray.full(
            [metadata.dim(0), tokens],
            values: MLXArray(Float32(offset)),
            dtype: metadata.dtype,
        )
        let positions = result[0..., 0 ..< tokens]
        let valid = (positions .>= 0).asType(positions.dtype)
        result[0..., 0 ..< tokens] = positions + offsets * valid
        return result
    }

    private static func prompt(from input: UserInput, imageToken: String, imageCount: Int) -> String
    {
        switch input.prompt {
        case .text(let text):
            if imageCount > 0 {
                return "<|user|>\n" + String(repeating: imageToken, count: imageCount) + "\n" + text
                    + "\n<|assistant|>\n"
            }
            return text
        case .messages:
            if imageCount > 0 {
                return "<|user|>\n" + String(repeating: imageToken, count: imageCount) + "\n"
                    + input.prompt.description + "\n<|assistant|>\n"
            }
            return input.prompt.description
        case .chat(let messages):
            var rendered = ""
            for message in messages {
                switch message.role {
                case .system:
                    rendered += "<|system|>\n\(message.content)\n"
                case .user:
                    rendered += "<|user|>\n"
                    for _ in message.images {
                        rendered += imageToken
                    }
                    rendered += "\(message.content)\n"
                case .assistant:
                    rendered += "<|assistant|>\n\(message.content)\n"
                default:
                    rendered += "\(message.content)\n"
                }
            }
            rendered += "<|assistant|>\n"
            return rendered
        }
    }

    private static func smartResize(
        height: Int,
        width: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
    )
        -> (height: Int, width: Int)
    {
        var hBar = max(factor, Int((Float(height) / Float(factor)).rounded()) * factor)
        var wBar = max(factor, Int((Float(width) / Float(factor)).rounded()) * factor)
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = max(factor, Int(floor(Float(height) / beta / Float(factor))) * factor)
            wBar = max(factor, Int(floor(Float(width) / beta / Float(factor))) * factor)
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = max(factor, Int(ceil(Float(height) * beta / Float(factor))) * factor)
            wBar = max(factor, Int(ceil(Float(width) * beta / Float(factor))) * factor)
        }
        return (hBar, wBar)
    }

    private static func resizeExact(image: CIImage, width: Int, height: Int) -> [Float] {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: width, height: height), method: .bicubic)
        return MediaProcessing.renderRGBFloat(resized, width: width, height: height)
    }

    private static func minMaxNormalize(_ pixels: [Float]) -> [Float] {
        pixels.map { ($0 - 0.5) * 2 }
    }

    private static func selectTiling(
        height: Int,
        width: Int,
        patchSize: Int,
        maxCrops: Int,
    )
        -> (rows: Int, cols: Int)
    {
        var tilings = [(Int, Int)]()
        for rows in 1 ... maxCrops {
            for cols in 1 ... maxCrops where rows * cols <= maxCrops {
                tilings.append((rows, cols))
            }
        }
        tilings.sort {
            if $0.0 * $0.1 == $1.0 * $1.1 {
                return $0.0 < $1.0
            }
            return $0.0 * $0.1 < $1.0 * $1.1
        }

        let originalH = max(Float(height), 1)
        let originalW = max(Float(width), 1)
        let scales = tilings.map { tiling in
            min(Float(tiling.0 * patchSize) / originalH, Float(tiling.1 * patchSize) / originalW)
        }
        if scales.allSatisfy({ $0 < 1 }) {
            let index = scales.indices.max { scales[$0] < scales[$1] } ?? 0
            return tilings[index]
        }
        let adjusted = scales.map { $0 < 1 ? Float.greatestFiniteMagnitude : $0 }
        let index = adjusted.indices.min { adjusted[$0] < adjusted[$1] } ?? 0
        return tilings[index]
    }

    private static func patchesFromTiling(
        tiles: Int,
        pooling: Int,
        cropPatches: Int,
        cropWindowPatches: Int,
        leftMargin: Int,
        rightMargin: Int,
    )
        -> Int
    {
        if tiles > 1 {
            let left = ((cropWindowPatches + leftMargin + pooling - 1) / pooling) * pooling
            let middle = ((cropWindowPatches + pooling - 1) / pooling) * pooling
            let right = ((cropWindowPatches + rightMargin + pooling - 1) / pooling) * pooling
            return left + (tiles - 2) * middle + right
        }
        return ((cropPatches + pooling - 1) / pooling) * pooling
    }
}

// MARK: - Language

private enum JinaVLMLanguage {
    final class Embeddings: Module {
        @ModuleInfo var embedding: MLXArray
        @ModuleInfo(key: "new_embedding") var newEmbedding: MLXArray

        init(vocabularySize: Int, additionalVocabularySize: Int, dimensions: Int) {
            _embedding.wrappedValue = MLXArray.zeros([vocabularySize, dimensions])
            _newEmbedding.wrappedValue = MLXArray.zeros([additionalVocabularySize, dimensions])
        }

        func callAsFunction(_ tokens: MLXArray) -> MLXArray {
            concatenated([embedding, newEmbedding], axis: 0)[tokens]
        }
    }

    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var out: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

        let rope: RoPE
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let fused: (Int, Int, Int)

        init(_ config: JinaVLMConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            fused = (heads * headDim, kvHeads * headDim, kvHeads * headDim)
            rope = RoPE(dimensions: headDim, base: config.ropeTheta)
            _qkv.wrappedValue = Linear(config.hiddenSize, fused.0 + fused.1 + fused.2, bias: false)
            _out.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            if config.useQKNorm {
                _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
                _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            }
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
            let parts = split(qkv(x), indices: [fused.0, fused.0 + fused.1], axis: -1)
            var queries = parts[0].reshaped(batch, length, heads, headDim)
            var keys = parts[1].reshaped(batch, length, kvHeads, headDim)
            let values = parts[2].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            if let qNorm, let kNorm {
                queries = qNorm(queries)
                keys = kNorm(keys)
            }
            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            return attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            .applying(out)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up") var gateUp: Linear
        @ModuleInfo var down: Linear

        init(_ config: JinaVLMConfiguration.TextConfiguration) {
            _gateUp.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize * 2, bias: false)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = split(gateUp(x), parts: 2, axis: -1)
            return down(silu(parts[1]) * parts[0])
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
        @ModuleInfo var ffn: MLP

        init(_ config: JinaVLMConfiguration.TextConfiguration) {
            _attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _attn.wrappedValue = Attention(config)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _ffn.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let h = x + attn(attentionNorm(x), mask: mask, cache: cache)
            return h + ffn(ffnNorm(h))
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var embedding: Embeddings
        @ModuleInfo(key: "layers") var layers: [Layer]
        @ModuleInfo(key: "ln_f") var norm: RMSNorm
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let kvHeads: [Int]

        init(_ config: JinaVLMConfiguration.TextConfiguration) {
            _embedding.wrappedValue = Embeddings(
                vocabularySize: config.vocabularySize,
                additionalVocabularySize: config.additionalVocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Layer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        @discardableResult
        override func update(
            modules: ModuleChildren,
            verify: VerifyUpdate,
            path: [String] = [],
            modulePath: [String] = [],
        )
            throws -> Self
        {
            var modules = modules
            if case .array(let values)? = modules["layers"] {
                let hasDictionary = values.contains { value in
                    if case .dictionary = value {
                        return true
                    }
                    return false
                }
                if hasDictionary {
                    modules["layers"] = .array(
                        values.map { value in
                            if case .none = value {
                                return .dictionary([:])
                            }
                            return value
                        },
                    )
                }
            }
            return try super.update(
                modules: modules,
                verify: verify,
                path: path,
                modulePath: modulePath,
            )
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var h = inputEmbedding ?? embedding(inputs!)
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return LMOutput(logits: lmHead(norm(h)))
        }
    }
}

// MARK: - Vision

private enum JinaVLMVision {
    final class PatchEmbedding: Module, UnaryLayer {
        @ModuleInfo var proj: Linear

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            _proj.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize,
                config.hiddenSize,
                bias: config.useBias,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var up: Linear
        @ModuleInfo var down: Linear
        let gelu = GELU(approximation: .tanh)

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            _up.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.useBias)
            _down.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.useBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(up(x)))
        }
    }

    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var out: Linear
        let heads: Int
        let headDim: Int
        let scale: Float

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(config.hiddenSize, heads * headDim * 3, bias: config.useBias)
            _out.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.useBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkvStates = qkv(x)
                .reshaped(batch, length, 3, heads, headDim)
                .transposed(2, 0, 3, 1, 4)
            let output = MLXFast.scaledDotProductAttention(
                queries: qkvStates[0],
                keys: qkvStates[1],
                values: qkvStates[2],
                scale: scale,
                mask: nil,
            )
            return out(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm
        @ModuleInfo var ffn: MLP

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            _attentionNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _ffnNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _ffn.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(attentionNorm(x))
            return h + ffn(ffnNorm(h))
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
        @ModuleInfo(key: "layers") var layers: [Layer]
        @ModuleInfo(key: "post_norm") var postNorm: LayerNorm?

        let config: JinaVLMConfiguration.VisionConfiguration
        let vitLayerIndices: [Int]

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbedding(config)
            var patchCount = config.cropPatches * config.cropPatches
            if config.useCLSToken {
                patchCount += 1
            }
            _posEmbed.wrappedValue = MLXArray.zeros([patchCount, config.hiddenSize])
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Layer(config) }
            if config.postLayerNorm {
                _postNorm.wrappedValue = LayerNorm(
                    dimensions: config.hiddenSize, eps: config.layerNormEps)
            }
            let hiddenCount = config.hiddenLayers + (config.postLayerNorm ? 1 : 0)
            vitLayerIndices = config.vitLayers.map { $0 >= 0 ? $0 : hiddenCount + $0 }
        }

        func callAsFunction(_ images: MLXArray) -> [MLXArray] {
            var h = patchEmbed(images)
            h = h + posEmbed.expandedDimensions(axis: 0).asType(h.dtype)
            var states = [MLXArray]()
            for layer in layers {
                h = layer(h)
                states.append(h)
            }
            if let postNorm {
                h = postNorm(h)
                states.append(h)
            }
            return states
        }

        func features(_ images: MLXArray) -> MLXArray {
            let states = self(images)
            return concatenated(vitLayerIndices.map { states[$0] }, axis: -1)
        }
    }

    final class CrossAttention: Module {
        @ModuleInfo var q: Linear
        @ModuleInfo var kv: Linear
        @ModuleInfo var out: Linear
        let heads: Int
        let headDim: Int
        let scale: Float

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            let inputSize = config.hiddenSize * config.vitLayers.count
            heads = config.attentionHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _q.wrappedValue = Linear(inputSize, heads * headDim, bias: true)
            _kv.wrappedValue = Linear(inputSize, 2 * heads * headDim, bias: true)
            _out.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ query: MLXArray, keyValue: MLXArray) -> MLXArray {
            let batch = query.dim(0)
            let queryLength = query.dim(1)
            let keyValueLength = keyValue.dim(1)
            let queries = q(query).reshaped(batch, queryLength, heads, headDim).transposed(
                0, 2, 1, 3)
            let kvStates = kv(keyValue)
                .reshaped(batch, keyValueLength, 2, heads, headDim)
                .transposed(2, 0, 3, 1, 4)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: kvStates[0],
                values: kvStates[1],
                scale: scale,
                mask: nil,
            )
            return out(output.transposed(0, 2, 1, 3).reshaped(batch, queryLength, -1))
        }
    }

    final class Projector: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up") var gateUp: Linear
        @ModuleInfo var down: Linear

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            _gateUp.wrappedValue = Linear(
                config.hiddenSize, config.connectorHiddenSize * 2, bias: false)
            _down.wrappedValue = Linear(config.connectorHiddenSize, config.outputSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = split(gateUp(x), parts: 2, axis: -1)
            return down(silu(parts[1]) * parts[0])
        }
    }

    final class Connector: Module {
        @ModuleInfo(key: "pad_embed") var padEmbed: MLXArray
        @ModuleInfo var pooling: CrossAttention
        @ModuleInfo var projector: Projector

        let config: JinaVLMConfiguration.VisionConfiguration

        init(_ config: JinaVLMConfiguration.VisionConfiguration) {
            self.config = config
            _padEmbed.wrappedValue = MLXArray.zeros([
                2, config.hiddenSize * config.vitLayers.count,
            ])
            _pooling.wrappedValue = CrossAttention(config)
            _projector.wrappedValue = Projector(config)
        }

        func callAsFunction(_ imageFeatures: MLXArray, imageMasks: MLXArray?) -> MLXArray {
            var features = imageFeatures
            if let imageMasks {
                let mask = imageMasks.asType(features.dtype)
                let allPad = (mask .== 0).asType(features.dtype)
                let partialPad =
                    ((mask .< 1).asType(features.dtype) * (mask .!= 0).asType(features.dtype))
                features =
                    features
                    + padEmbed[0].reshaped(1, 1, 1, -1) * allPad.expandedDimensions(axis: -1)
                    + padEmbed[1].reshaped(1, 1, 1, -1) * partialPad.expandedDimensions(axis: -1)
            }

            let batch = features.dim(0)
            let crops = features.dim(1)
            let patchH = config.cropPatches
            let patchW = config.cropPatches
            let channels = features.dim(-1)
            features = features.reshaped(batch, crops, patchH, patchW, channels)

            let padH =
                patchH % config.poolingH == 0 ? 0 : config.poolingH - (patchH % config.poolingH)
            let padW =
                patchW % config.poolingW == 0 ? 0 : config.poolingW - (patchW % config.poolingW)
            if padH > 0 || padW > 0 {
                features = padded(features, widths: [0, 0, .init((0, padH)), .init((0, padW)), 0])
            }

            let height = features.dim(2)
            let width = features.dim(3)
            let newH = height / config.poolingH
            let newW = width / config.poolingW
            features =
                features
                .reshaped(batch, crops, newH, config.poolingH, newW, config.poolingW, channels)
                .transposed(0, 1, 2, 4, 3, 5, 6)
                .reshaped(batch * crops * newH * newW, config.poolingH * config.poolingW, channels)
            let query = features.mean(axis: 1, keepDims: true)
            let pooled = pooling(query, keyValue: features)
            return projector(pooled).reshaped(batch, crops, newH * newW, -1)
        }
    }
}

// MARK: - Model

public final class JinaVLM: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: JinaVLMVision.VisionModel
    @ModuleInfo(key: "vl_connector") private var connector: JinaVLMVision.Connector
    @ModuleInfo(key: "language_model") private var languageModel: JinaVLMLanguage.LanguageModel

    public let config: JinaVLMConfiguration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.layers
    }

    public init(_ config: JinaVLMConfiguration) {
        self.config = config
        _visionModel.wrappedValue = JinaVLMVision.VisionModel(config.visionConfiguration)
        _connector.wrappedValue = JinaVLMVision.Connector(config.visionConfiguration)
        _languageModel.wrappedValue = JinaVLMLanguage.LanguageModel(config.textConfiguration)
    }

    private func imageFeatures(pixelValues: MLXArray, imageMasks: MLXArray?) -> MLXArray {
        let batch = 1
        let crops = pixelValues.dim(0)
        let patches = pixelValues.dim(1)
        let patchDim = pixelValues.dim(2)
        let dtype = visionModel.patchEmbed.proj.weight.dtype
        let features = visionModel.features(
            pixelValues.asType(dtype).reshaped(batch * crops, patches, patchDim),
        ).reshaped(batch, crops, patches, -1)
        return connector(features, imageMasks: imageMasks)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?, metadata: MLXArray?)
        -> MLXArray
    {
        var embeddings = languageModel.embedding(inputIDs)
        guard let pixelValues, let metadata else {
            return embeddings
        }
        let tokensPerImage = config.visionConfiguration.tokensPerImage
        let imageInputIdx = metadata[0..., 0 ..< tokensPerImage].asType(.int32)
        let imageMasks = metadata[0..., tokensPerImage...].expandedDimensions(axis: 0)
        let imageFeatures = imageFeatures(pixelValues: pixelValues, imageMasks: imageMasks)
            .reshaped(-1, config.textConfiguration.hiddenSize)
            .asType(embeddings.dtype)

        let positions = imageInputIdx.flattened().asArray(Int32.self).map(Int.init)
        let validPairs = positions.enumerated().filter {
            $0.element >= 0 && $0.element < inputIDs.dim(1)
        }
        guard !validPairs.isEmpty else {
            return embeddings
        }

        let featureIndices = validPairs.map(\.offset)
        let tokenPositions = validPairs.map(\.element)
        let featureIndexArray = MLXArray(featureIndices, [featureIndices.count])
        let tokenPositionArray = MLXArray(tokenPositions, [tokenPositions.count])
        let flat = embeddings.reshaped(-1, embeddings.dim(-1))
        flat[tokenPositionArray] = flat[tokenPositionArray] + imageFeatures[featureIndexArray]
        embeddings = flat.reshaped(embeddings.shape)
        return embeddings
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
        defer { prefill.progress?(embeddings.dim(1), embeddings.dim(1)) }
        return .logits(languageModel(nil, cache: cache, inputEmbedding: embeddings))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            if key.hasPrefix("lm_head.") {
                sanitized["language_model." + key] = value
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }
}

extension MLXArray {
    fileprivate func applying(_ layer: Linear) -> MLXArray {
        layer(self)
    }
}
