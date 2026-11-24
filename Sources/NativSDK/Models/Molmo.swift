import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MolmoConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let maxPositionEmbeddings: Int
        public let modelDimension: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let hiddenLayers: Int
        public let mlpHiddenSize: Int
        public let vocabularySize: Int
        public let embeddingSize: Int
        public let additionalVocabularySize: Int
        public let attentionDropout: Float
        public let residualDropout: Float
        public let embeddingDropout: Float
        public let layerNormEps: Float
        public let initializerRange: Float
        public let ropeTheta: Float
        public let weightTying: Bool
        public let actOutputMultiplier: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case maxPositionEmbeddings = "max_position_embeddings"
            case modelDimension = "d_model"
            case attentionHeads = "n_heads"
            case kvHeads = "n_kv_heads"
            case hiddenLayers = "n_layers"
            case mlpHiddenSize = "mlp_hidden_size"
            case vocabularySize = "vocab_size"
            case embeddingSize = "embedding_size"
            case additionalVocabularySize = "additional_vocab_size"
            case attentionDropout = "attention_dropout"
            case residualDropout = "residual_dropout"
            case embeddingDropout = "embedding_dropout"
            case layerNormEps = "layer_norm_eps"
            case initializerRange = "initializer_range"
            case ropeTheta = "rope_theta"
            case weightTying = "weight_tying"
            case actOutputMultiplier = "act_output_multiplier"
        }

        public init(
            modelType: String = "molmo",
            maxPositionEmbeddings: Int = 4096,
            modelDimension: Int = 3584,
            attentionHeads: Int = 28,
            kvHeads: Int = 4,
            hiddenLayers: Int = 28,
            mlpHiddenSize: Int = 37888,
            vocabularySize: Int = 152_064,
            embeddingSize: Int = 152_064,
            additionalVocabularySize: Int = 128,
            attentionDropout: Float = 0.1,
            residualDropout: Float = 0.1,
            embeddingDropout: Float = 0.1,
            layerNormEps: Float = 1e-5,
            initializerRange: Float = 0.02,
            ropeTheta: Float = 1_000_000,
            weightTying: Bool = false,
            actOutputMultiplier: Float = 0.5,
        ) {
            self.modelType = modelType
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.modelDimension = modelDimension
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.hiddenLayers = hiddenLayers
            self.mlpHiddenSize = mlpHiddenSize
            self.vocabularySize = vocabularySize
            self.embeddingSize = embeddingSize
            self.additionalVocabularySize = additionalVocabularySize
            self.attentionDropout = attentionDropout
            self.residualDropout = residualDropout
            self.embeddingDropout = embeddingDropout
            self.layerNormEps = layerNormEps
            self.initializerRange = initializerRange
            self.ropeTheta = ropeTheta
            self.weightTying = weightTying
            self.actOutputMultiplier = actOutputMultiplier
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo"
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
            modelDimension =
                try c.decodeIfPresent(Int.self, forKey: .modelDimension) ?? 3584
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 28
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 28
            mlpHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .mlpHiddenSize) ?? 37888
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 152_064
            embeddingSize =
                try c.decodeIfPresent(Int.self, forKey: .embeddingSize) ?? vocabularySize
            additionalVocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .additionalVocabularySize) ?? 128
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.1
            residualDropout =
                try c.decodeIfPresent(Float.self, forKey: .residualDropout) ?? 0.1
            embeddingDropout =
                try c.decodeIfPresent(Float.self, forKey: .embeddingDropout) ?? 0.1
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
            initializerRange =
                try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            weightTying = try c.decodeIfPresent(Bool.self, forKey: .weightTying) ?? false
            actOutputMultiplier =
                try c.decodeIfPresent(Float.self, forKey: .actOutputMultiplier) ?? 0.5
        }

        func withVocabularySize(_ vocabularySize: Int) -> Self {
            Self(
                modelType: modelType,
                maxPositionEmbeddings: maxPositionEmbeddings,
                modelDimension: modelDimension,
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                hiddenLayers: hiddenLayers,
                mlpHiddenSize: mlpHiddenSize,
                vocabularySize: vocabularySize,
                embeddingSize: vocabularySize,
                additionalVocabularySize: additionalVocabularySize,
                attentionDropout: attentionDropout,
                residualDropout: residualDropout,
                embeddingDropout: embeddingDropout,
                layerNormEps: layerNormEps,
                initializerRange: initializerRange,
                ropeTheta: ropeTheta,
                weightTying: weightTying,
                actOutputMultiplier: actOutputMultiplier,
            )
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let channels: Int
        public let imageDefaultInputSize: [Int]
        public let imagePatchSize: Int
        public let imagePositionPatchSize: Int
        public let hiddenSize: Int
        public let imageEmbeddingDimension: Int
        public let imageAttentionHeads: Int
        public let imageKVHeads: Int
        public let imageLayers: Int
        public let imageHeadDimension: Int
        public let imageMLPDimension: Int
        public let imageNumPositions: Int
        public let imageNormEps: Float
        public let attentionDropout: Float
        public let residualDropout: Float
        public let modelDimension: Int
        public let imagePoolingH: Int
        public let imagePoolingW: Int
        public let vitLayers: [Int]?
        public let imagePooling2D: String
        public let imagePaddingEmbedding: String?
        public let intermediateSize: Int

        public var imageNumPatch: (Int, Int) {
            (imageDefaultInputSize[0] / imagePatchSize, imageDefaultInputSize[1] / imagePatchSize)
        }

        public var llmPatchesPerCrop: (Int, Int) {
            let patch = imageNumPatch
            let h = (patch.0 + imagePoolingH - 1) / imagePoolingH
            let w = (patch.1 + imagePoolingW - 1) / imagePoolingW
            return (h, w)
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case channels = "num_channels"
            case imageDefaultInputSize = "image_default_input_size"
            case imagePatchSize = "image_patch_size"
            case imagePositionPatchSize = "image_pos_patch_size"
            case hiddenSize = "hidden_size"
            case imageEmbeddingDimension = "image_emb_dim"
            case imageAttentionHeads = "image_num_heads"
            case imageKVHeads = "image_num_key_value_heads"
            case imageLayers = "image_num_layers"
            case imageHeadDimension = "image_head_dim"
            case imageMLPDimension = "image_mlp_dim"
            case imageNumPositions = "image_num_pos"
            case imageNormEps = "image_norm_eps"
            case attentionDropout = "attention_dropout"
            case residualDropout = "residual_dropout"
            case modelDimension = "d_model"
            case imagePoolingH = "image_pooling_h"
            case imagePoolingW = "image_pooling_w"
            case vitLayers = "vit_layers"
            case imagePooling2D = "image_pooling_2d"
            case imagePaddingEmbedding = "image_padding_embed"
            case intermediateSize = "intermediate_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo"
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageDefaultInputSize =
                try c.decodeIfPresent([Int].self, forKey: .imageDefaultInputSize)
                ?? [336, 336]
            imagePatchSize = try c.decodeIfPresent(Int.self, forKey: .imagePatchSize) ?? 14
            imagePositionPatchSize =
                try c.decodeIfPresent(Int.self, forKey: .imagePositionPatchSize) ?? 14
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 18944
            imageEmbeddingDimension =
                try c.decodeIfPresent(Int.self, forKey: .imageEmbeddingDimension) ?? 1024
            imageAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .imageAttentionHeads) ?? 16
            imageKVHeads =
                try c.decodeIfPresent(Int.self, forKey: .imageKVHeads) ?? 16
            imageLayers = try c.decodeIfPresent(Int.self, forKey: .imageLayers) ?? 23
            imageHeadDimension =
                try c.decodeIfPresent(Int.self, forKey: .imageHeadDimension) ?? 64
            imageMLPDimension =
                try c.decodeIfPresent(Int.self, forKey: .imageMLPDimension) ?? 4096
            imageNumPositions =
                try c.decodeIfPresent(Int.self, forKey: .imageNumPositions) ?? 577
            imageNormEps = try c.decodeIfPresent(Float.self, forKey: .imageNormEps) ?? 1e-5
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            residualDropout =
                try c.decodeIfPresent(Float.self, forKey: .residualDropout) ?? 0
            modelDimension =
                try c.decodeIfPresent(Int.self, forKey: .modelDimension) ?? 3584
            imagePoolingH = try c.decodeIfPresent(Int.self, forKey: .imagePoolingH) ?? 2
            imagePoolingW = try c.decodeIfPresent(Int.self, forKey: .imagePoolingW) ?? 2
            vitLayers = try c.decodeIfPresent([Int].self, forKey: .vitLayers) ?? [-2, -9]
            imagePooling2D =
                try c.decodeIfPresent(String.self, forKey: .imagePooling2D) ?? "attention-meanq"
            imagePaddingEmbedding =
                try c.decodeIfPresent(String.self, forKey: .imagePaddingEmbedding)
                ?? "pad_and_partial_pad"
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? imagePatchSize * imagePatchSize * channels
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let imagePatchTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
        case imagePatchTokenID = "image_patch_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo"
        var text =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()
        if let topLevelVocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize),
            !c.contains(.textConfiguration)
        {
            text = text.withVocabularySize(topLevelVocabularySize)
        }
        textConfiguration = text
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? JSONDecoder.defaultValue(VisionConfiguration.self)
        imagePatchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imagePatchTokenID) ?? 152_066
    }
}

public struct MolmoProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let maxCrops: Int
    public let overlapMargins: [Int]
    public let baseImageInputSize: [Int]
    public let imageTokenLengthW: Int
    public let imageTokenLengthH: Int
    public let imagePatchSize: Int
    public let imagePaddingMask: Bool
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case maxCrops = "max_crops"
        case overlapMargins = "overlap_margins"
        case baseImageInputSize = "base_image_input_size"
        case imageTokenLengthW = "image_token_length_w"
        case imageTokenLengthH = "image_token_length_h"
        case imagePatchSize = "image_patch_size"
        case imagePaddingMask = "image_padding_mask"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "MolmoProcessor"
        maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? 12
        overlapMargins = try c.decodeIfPresent([Int].self, forKey: .overlapMargins) ?? [4, 4]
        baseImageInputSize =
            try c.decodeIfPresent([Int].self, forKey: .baseImageInputSize)
            ?? [336, 336]
        imageTokenLengthW = try c.decodeIfPresent(Int.self, forKey: .imageTokenLengthW) ?? 12
        imageTokenLengthH = try c.decodeIfPresent(Int.self, forKey: .imageTokenLengthH) ?? 12
        imagePatchSize = try c.decodeIfPresent(Int.self, forKey: .imagePatchSize) ?? 14
        imagePaddingMask = try c.decodeIfPresent(Bool.self, forKey: .imagePaddingMask) ?? true
        doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? true
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.26862954, 0.26130258, 0.27577711]
    }
}

extension JSONDecoder {
    fileprivate static func defaultValue<T: Decodable>(_: T.Type) -> T {
        try! JSONDecoder.json5().decode(T.self, from: "{}".data(using: .utf8)!)
    }
}

// MARK: - Processor

public struct MolmoProcessor: UserInputProcessor {
    private let config: MolmoProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MolmoProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)
        let textTokens = tokenizer.encode(text: prompt, addSpecialTokens: true)

        guard !images.isEmpty else {
            let tokens = MLXArray(textTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let imagePatchTokenID = try tokenID("<im_patch>")
        let imageColTokenID = try tokenID("<im_col>")
        let imageStartTokenID = try tokenID("<im_start>")
        let imageEndTokenID = try tokenID("<im_end>")

        let processedImage = try preprocess(
            image: images[0].asCIImage(),
            processing: input.processing,
            imagePatchTokenID: imagePatchTokenID,
            imageColTokenID: imageColTokenID,
            imageStartTokenID: imageStartTokenID,
            imageEndTokenID: imageEndTokenID,
        )

        let tokens = processedImage.imageTokens + textTokens
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: processedImage.patches),
            video: .init(pixels: processedImage.metadata),
        )
    }

    private struct ProcessedImage {
        let patches: MLXArray
        let metadata: MLXArray
        let imageTokens: [Int]
    }

    private func tokenID(_ token: String) throws -> Int {
        if let id = tokenizer.convertTokenToId(token) {
            return id
        }
        throw VLMError.processing("Molmo tokenizer is missing required token \(token)")
    }

    private func preprocess(
        image: CIImage,
        processing: UserInput.Processing?,
        imagePatchTokenID: Int,
        imageColTokenID: Int,
        imageStartTokenID: Int,
        imageEndTokenID: Int,
    )
        throws -> ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing).toSRGB()
        let (patches, imageTokens, patchOrdering, imageMasks) = try imageToPatchesAndTokens(
            image: image,
            imagePatchTokenID: imagePatchTokenID,
            imageColTokenID: imageColTokenID,
            imageStartTokenID: imageStartTokenID,
            imageEndTokenID: imageEndTokenID,
        )
        let imageInputIdx = buildImageInputIdx(
            imageTokens: imageTokens,
            patchOrder: patchOrdering,
            imagePatchTokenID: imagePatchTokenID,
        )
        let metadata = packMetadata(imageInputIdx: imageInputIdx, imageMasks: imageMasks)
        return ProcessedImage(patches: patches, metadata: metadata, imageTokens: imageTokens)
    }

    private func imageToPatchesAndTokens(
        image: CIImage,
        imagePatchTokenID: Int,
        imageColTokenID: Int,
        imageStartTokenID: Int,
        imageEndTokenID: Int,
    )
        throws -> (MLXArray, [Int], [Int], [[Float]])
    {
        let baseH = config.baseImageInputSize[0]
        let baseW = config.baseImageInputSize[1]
        let patchSize = config.imagePatchSize
        let tokensPerImage = config.imageTokenLengthW * config.imageTokenLengthH
        let imageBasePatchH = baseH / patchSize
        let imageBasePatchW = baseW / patchSize

        let canonical = MediaProcessing.zeroOrigin(image)
        let originalH = Int(canonical.extent.height.rounded())
        let originalW = Int(canonical.extent.width.rounded())
        let cropSize = baseH

        let leftMargin = config.overlapMargins[0]
        let rightMargin = config.overlapMargins.count > 1 ? config.overlapMargins[1] : leftMargin
        let totalMarginPixels = patchSize * (rightMargin + leftMargin)
        let cropPatches = baseH / patchSize
        let cropWindowPatches = cropPatches - (rightMargin + leftMargin)
        let cropWindowSize = cropWindowPatches * patchSize

        let tiling = Self.selectTiling(
            height: originalH - totalMarginPixels,
            width: originalW - totalMarginPixels,
            patchSize: cropWindowSize,
            maxCrops: config.maxCrops,
        )

        let source = Self.resizeAndPad(
            image: canonical,
            targetHeight: tiling.rows * cropWindowSize + totalMarginPixels,
            targetWidth: tiling.cols * cropWindowSize + totalMarginPixels,
            mean: config.imageMean,
            std: config.imageStd,
            normalize: config.doNormalize,
        )

        var patchValues: [Float] = []
        var maskValues: [[Float]] = []
        var patchOrdering = [Int]()
        var localPatchOrder = [Int]()
        var orderingOffset = 0

        for row in 0 ..< tiling.rows {
            let y0 = row * cropWindowSize
            let cropY0 = row == 0 ? 0 : leftMargin / 2
            var cropH = imageBasePatchH - (rightMargin + leftMargin)
            if row == 0 {
                cropH += leftMargin
            }
            if row == tiling.rows - 1 {
                cropH += rightMargin
            }
            let pooledH = (cropH + 1) / 2

            for col in 0 ..< tiling.cols {
                let x0 = col * cropWindowSize
                let cropX0 = col == 0 ? 0 : leftMargin / 2
                var cropW = imageBasePatchW - (rightMargin + leftMargin)
                if col == 0 {
                    cropW += leftMargin
                }
                if col == tiling.cols - 1 {
                    cropW += rightMargin
                }
                let pooledW = (cropW + 1) / 2

                var ordering = Array(
                    repeating: -1,
                    count: config.imageTokenLengthH * config.imageTokenLengthW,
                )
                for r in 0 ..< pooledH {
                    for c in 0 ..< pooledW {
                        let targetR = cropY0 + r
                        let targetC = cropX0 + c
                        if targetR < config.imageTokenLengthH, targetC < config.imageTokenLengthW {
                            ordering[targetR * config.imageTokenLengthW + targetC] =
                                orderingOffset + r * pooledW + c
                        }
                    }
                }
                localPatchOrder.append(contentsOf: ordering)

                let cropPixels = MediaProcessing.cropPixels(
                    source.pixels,
                    sourceWidth: source.width,
                    x: x0,
                    y: y0,
                    width: cropSize,
                    height: cropSize,
                    channels: 3,
                )
                patchValues.append(
                    contentsOf: MediaProcessing.patchifyPixels(
                        cropPixels,
                        width: cropSize,
                        height: cropSize,
                        channels: 3,
                        patchSize: patchSize,
                    ),
                )

                let cropMask = MediaProcessing.cropPixels(
                    source.mask,
                    sourceWidth: source.width,
                    x: x0,
                    y: y0,
                    width: cropSize,
                    height: cropSize,
                    channels: 1,
                )
                maskValues.append(
                    MediaProcessing.patchifyMask(
                        cropMask,
                        width: cropSize,
                        height: cropSize,
                        patchSize: patchSize,
                    ),
                )
                orderingOffset += pooledH * pooledW
            }
        }

        let h = tiling.rows * cropWindowPatches + (rightMargin + leftMargin)
        let w = tiling.cols * cropWindowPatches + (rightMargin + leftMargin)
        var highResRow = Array(repeating: imagePatchTokenID, count: (w + 1) / 2)
        highResRow.append(imageColTokenID)
        var highResTokens = [imageStartTokenID]
        for _ in 0 ..< ((h + 1) / 2) {
            highResTokens.append(contentsOf: highResRow)
        }
        highResTokens.append(imageEndTokenID)

        let global = Self.resizeAndPad(
            image: canonical,
            targetHeight: baseH,
            targetWidth: baseW,
            mean: config.imageMean,
            std: config.imageStd,
            normalize: config.doNormalize,
        )
        let globalPatches = MediaProcessing.patchifyPixels(
            global.pixels,
            width: baseW,
            height: baseH,
            channels: 3,
            patchSize: patchSize,
        )
        patchValues.insert(contentsOf: globalPatches, at: 0)

        let patchOrderRows = localPatchOrder.count / tokensPerImage
        var transposedOrder = Array(repeating: -1, count: localPatchOrder.count)
        if patchOrderRows == tiling.rows * tiling.cols {
            for tileRow in 0 ..< tiling.rows {
                for localR in 0 ..< config.imageTokenLengthH {
                    for tileCol in 0 ..< tiling.cols {
                        for localC in 0 ..< config.imageTokenLengthW {
                            let originalIndex =
                                ((tileRow * tiling.cols + tileCol) * tokensPerImage)
                                + localR * config.imageTokenLengthW + localC
                            let transposedIndex =
                                (((tileRow * config.imageTokenLengthH + localR) * tiling.cols
                                    + tileCol) * config.imageTokenLengthW) + localC
                            if transposedIndex < transposedOrder.count {
                                transposedOrder[transposedIndex] = localPatchOrder[originalIndex]
                            }
                        }
                    }
                }
            }
        }
        patchOrdering = localPatchOrder.enumerated().map { index, value in
            value >= 0 && index < transposedOrder.count && transposedOrder[index] >= 0
                ? transposedOrder[index] + tokensPerImage
                : -1
        }
        patchOrdering.insert(contentsOf: 0 ..< tokensPerImage, at: 0)

        var globalRow = Array(repeating: imagePatchTokenID, count: config.imageTokenLengthW)
        globalRow.append(imageColTokenID)
        var imageTokens = [imageStartTokenID]
        for _ in 0 ..< config.imageTokenLengthH {
            imageTokens.append(contentsOf: globalRow)
        }
        imageTokens.append(imageEndTokenID)
        imageTokens.append(contentsOf: highResTokens)

        maskValues.append(Array(repeating: -1, count: imageBasePatchH * imageBasePatchW))

        let cropCount =
            patchValues.count / (imageBasePatchH * imageBasePatchW * patchSize * patchSize * 3)
        let patchArray = MLXArray(
            patchValues,
            [cropCount, imageBasePatchH * imageBasePatchW, patchSize * patchSize * 3],
        )
        return (patchArray, imageTokens, patchOrdering, maskValues)
    }

    private func buildImageInputIdx(
        imageTokens: [Int],
        patchOrder: [Int],
        imagePatchTokenID: Int,
    )
        -> [[Int]]
    {
        let tokensPerImage = config.imageTokenLengthW * config.imageTokenLengthH
        let imageInputPositions = imageTokens.indices.filter {
            imageTokens[$0] == imagePatchTokenID
        }
        let validPatchOrder = patchOrder.filter { $0 >= 0 }
        let nValidPatches = validPatchOrder.count
        var sortedPatchIndices = Array(repeating: 0, count: imageInputPositions.count)
        for (value, sourceIndex) in validPatchOrder.enumerated()
        where sourceIndex < sortedPatchIndices.count {
            sortedPatchIndices[sourceIndex] = value
        }

        var flat = Array(repeating: -100, count: patchOrder.count)
        for (index, order) in patchOrder.enumerated() where order >= 0 {
            let sortedIndex = sortedPatchIndices[order]
            if sortedIndex < imageInputPositions.count, sortedIndex < nValidPatches {
                flat[index] = imageInputPositions[sortedIndex]
            }
        }

        return stride(from: 0, to: flat.count, by: tokensPerImage).map { start in
            Array(flat[start ..< min(start + tokensPerImage, flat.count)])
        }
    }

    private func packMetadata(imageInputIdx: [[Int]], imageMasks: [[Float]]) -> MLXArray {
        precondition(imageInputIdx.count == imageMasks.count)
        let tokensPerImage = config.imageTokenLengthW * config.imageTokenLengthH
        let patchCount = imageMasks.first?.count ?? 0
        var values = [Float]()
        values.reserveCapacity(imageInputIdx.count * (tokensPerImage + patchCount))
        for row in 0 ..< imageInputIdx.count {
            values.append(contentsOf: imageInputIdx[row].map(Float.init))
            values.append(contentsOf: imageMasks[row])
        }
        return MLXArray(values, [imageInputIdx.count, tokensPerImage + patchCount])
    }

    private struct ResizedImage {
        let pixels: [Float]
        let mask: [Float]
        let width: Int
        let height: Int
    }

    private static func resizeAndPad(
        image: CIImage,
        targetHeight: Int,
        targetWidth: Int,
        mean: [CGFloat],
        std: [CGFloat],
        normalize: Bool,
    )
        -> ResizedImage
    {
        let width = max(Int(image.extent.width.rounded()), 1)
        let height = max(Int(image.extent.height.rounded()), 1)
        let scale = min(
            CGFloat(targetWidth) / CGFloat(width), CGFloat(targetHeight) / CGFloat(height))
        let scaledWidth = max(Int((CGFloat(width) * scale).rounded(.down)), 1)
        let scaledHeight = max(Int((CGFloat(height) * scale).rounded(.down)), 1)
        let topPad = (targetHeight - scaledHeight) / 2
        let leftPad = (targetWidth - scaledWidth) / 2

        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: scaledWidth, height: scaledHeight),
            method: .bicubic,
        )
        let scaledPixels = MediaProcessing.renderRGBFloat(
            resized, width: scaledWidth, height: scaledHeight)

        var pixels = Array(repeating: Float(0), count: targetHeight * targetWidth * 3)
        var mask = Array(repeating: Float(0), count: targetHeight * targetWidth)
        for y in 0 ..< scaledHeight {
            for x in 0 ..< scaledWidth {
                let source = (y * scaledWidth + x) * 3
                let target = ((topPad + y) * targetWidth + (leftPad + x)) * 3
                pixels[target] = scaledPixels[source]
                pixels[target + 1] = scaledPixels[source + 1]
                pixels[target + 2] = scaledPixels[source + 2]
                mask[(topPad + y) * targetWidth + (leftPad + x)] = 1
            }
        }

        if normalize {
            for index in stride(from: 0, to: pixels.count, by: 3) {
                pixels[index] = (pixels[index] - Float(mean[0])) / Float(std[0])
                pixels[index + 1] = (pixels[index + 1] - Float(mean[1])) / Float(std[1])
                pixels[index + 2] = (pixels[index + 2] - Float(mean[2])) / Float(std[2])
            }
        }

        return ResizedImage(pixels: pixels, mask: mask, width: targetWidth, height: targetHeight)
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
                $0.0 < $1.0
            } else {
                $0.0 * $0.1 < $1.0 * $1.1
            }
        }

        let originalH = max(Float(height), 1)
        let originalW = max(Float(width), 1)
        let scales = tilings.map { tiling -> Float in
            min(Float(tiling.0 * patchSize) / originalH, Float(tiling.1 * patchSize) / originalW)
        }
        if scales.allSatisfy({ $0 < 1 }) {
            let index = scales.indices.max { scales[$0] < scales[$1] } ?? 0
            return (tilings[index].0, tilings[index].1)
        }
        var bestIndex = 0
        var bestScale = Float.greatestFiniteMagnitude
        for (index, scale) in scales.enumerated() {
            let candidate = scale < 1 ? Float(1e10) : scale
            if candidate < bestScale {
                bestScale = candidate
                bestIndex = index
            }
        }
        return (tilings[bestIndex].0, tilings[bestIndex].1)
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

private enum MolmoLanguage {
    final class SwiGLU: Module, UnaryLayer {
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = split(x, parts: 2, axis: -1)
            return silu(parts[1]) * parts[0]
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "attn_out") var attentionOutput: Linear
        @ModuleInfo(key: "ff_out") var feedForwardOutput: Linear
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "ff_norm") var feedForwardNorm: RMSNorm
        @ModuleInfo(key: "ff_proj") var feedForwardProjection: Linear
        @ModuleInfo(key: "att_proj") var attentionProjection: Linear

        let activation = SwiGLU()
        let rotaryEmbedding: RoPE
        let scale: Float
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let fusedDimensions: (Int, Int, Int)

        init(_ config: MolmoConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.modelDimension / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            fusedDimensions = (
                config.modelDimension,
                config.kvHeads * headDim,
                config.kvHeads * headDim,
            )
            rotaryEmbedding = RoPE(dimensions: headDim, base: config.ropeTheta)
            _attentionOutput.wrappedValue = Linear(
                config.modelDimension, config.modelDimension, bias: false,
            )
            _feedForwardOutput.wrappedValue = Linear(
                Int(config.actOutputMultiplier * Float(config.mlpHiddenSize)),
                config.modelDimension,
                bias: false,
            )
            _attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.modelDimension, eps: config.layerNormEps,
            )
            _feedForwardNorm.wrappedValue = RMSNorm(
                dimensions: config.modelDimension, eps: config.layerNormEps,
            )
            _feedForwardProjection.wrappedValue = Linear(
                config.modelDimension, config.mlpHiddenSize, bias: false,
            )
            _attentionProjection.wrappedValue = Linear(
                config.modelDimension,
                fusedDimensions.0 + fusedDimensions.1 + fusedDimensions.2,
                bias: true,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let dimensions = x.dim(2)

            let attentionInput = attentionNorm(x)
            let qkv = attentionProjection(attentionInput)
            let parts = split(
                qkv,
                indices: [fusedDimensions.0, fusedDimensions.0 + fusedDimensions.1],
                axis: -1,
            )

            var queries = parts[0]
                .reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            var keys = parts[1]
                .reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            let values = parts[2]
                .reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            let attention = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, dimensions)
            var h = x + attentionOutput(attention)

            let residual = h
            h = feedForwardNorm(h)
            h = feedForwardProjection(h)
            h = activation(h)
            h = feedForwardOutput(h)
            return residual + h
        }
    }

    final class MolmoEmbedding: Module {
        @ModuleInfo var embedding: MLXArray
        @ModuleInfo(key: "new_embedding") var newEmbedding: MLXArray

        init(
            embeddingCount: Int,
            newEmbeddingCount: Int,
            dimensions: Int,
            initializerRange: Float,
        ) {
            _embedding.wrappedValue = MLXRandom.normal(
                [embeddingCount, dimensions],
                scale: initializerRange,
            )
            _newEmbedding.wrappedValue = MLXRandom.normal(
                [newEmbeddingCount, dimensions],
                scale: initializerRange,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            concatenated([embedding, newEmbedding], axis: 0)[x]
        }

        func asLinear(_ x: MLXArray) -> MLXArray {
            matmul(x, concatenated([embedding, newEmbedding], axis: 0).T)
        }
    }

    final class Model: Module {
        @ModuleInfo var wte: MolmoEmbedding
        let blocks: [Block]
        @ModuleInfo(key: "ln_f") var norm: RMSNorm
        @ModuleInfo(key: "ff_out") var output: Linear?

        let config: MolmoConfiguration.TextConfiguration

        init(_ config: MolmoConfiguration.TextConfiguration) {
            self.config = config
            _wte.wrappedValue = MolmoEmbedding(
                embeddingCount: config.embeddingSize,
                newEmbeddingCount: config.additionalVocabularySize,
                dimensions: config.modelDimension,
                initializerRange: config.initializerRange,
            )
            blocks = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.modelDimension,
                eps: config.layerNormEps,
            )
            if !config.weightTying {
                _output.wrappedValue = Linear(
                    config.modelDimension,
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
            -> MLXArray
        {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = wte(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, block) in blocks.enumerated() {
                h = block(h, mask: mask, cache: cache?[index])
            }
            h = norm(h)
            if config.weightTying {
                return wte.asLinear(h)
            }
            return output!(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        let kvHeads: [Int]

        init(_ config: MolmoConfiguration.TextConfiguration) {
            _model.wrappedValue = Model(config)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            LMOutput(logits: model(inputs, cache: cache, inputEmbedding: inputEmbedding))
        }
    }
}

// MARK: - Vision

private enum MolmoVision {
    final class MLP: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w2: Linear
        @ModuleInfo var w3: Linear

        init(_ config: MolmoConfiguration.VisionConfiguration, inputDimension: Int) {
            _w1.wrappedValue = Linear(inputDimension, config.hiddenSize, bias: false)
            _w2.wrappedValue = Linear(config.hiddenSize, config.modelDimension, bias: false)
            _w3.wrappedValue = Linear(inputDimension, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    final class VisionMLP: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w2: Linear
        let activation = GELU(approximation: .fast)

        init(_ config: MolmoConfiguration.VisionConfiguration) {
            _w1.wrappedValue = Linear(
                config.imageEmbeddingDimension,
                config.imageMLPDimension,
                bias: true,
            )
            _w2.wrappedValue = Linear(
                config.imageMLPDimension,
                config.imageEmbeddingDimension,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(activation(w1(x)))
        }
    }

    final class Attention: Module {
        @ModuleInfo var wq: Linear
        @ModuleInfo var wk: Linear
        @ModuleInfo var wv: Linear
        @ModuleInfo var wo: Linear

        let embedDimension: Int
        let heads: Int
        let headDimension: Int
        let kvHeads: Int
        let scale: Float

        init(_ config: MolmoConfiguration.VisionConfiguration, isVisionLayer: Bool = true) {
            let layerCount =
                (isVisionLayer || config.vitLayers == nil) ? 1 : config.vitLayers!.count
            embedDimension = config.imageEmbeddingDimension
            heads = config.imageAttentionHeads
            headDimension = config.imageHeadDimension
            kvHeads = config.imageKVHeads
            scale = pow(Float(headDimension), -0.5)
            _wq.wrappedValue = Linear(
                layerCount * embedDimension,
                heads * headDimension,
                bias: true,
            )
            _wk.wrappedValue = Linear(
                layerCount * embedDimension,
                kvHeads * headDimension,
                bias: true,
            )
            _wv.wrappedValue = Linear(
                layerCount * embedDimension,
                kvHeads * headDimension,
                bias: true,
            )
            _wo.wrappedValue = Linear(
                heads * headDimension,
                embedDimension,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray, kv: MLXArray? = nil) -> MLXArray {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let kInput = kv ?? x
            let vInput = kv ?? x
            let kvLength = kInput.dim(1)

            let queries = wq(x)
                .reshaped(batchSize, sequenceLength, heads, headDimension)
                .transposed(0, 2, 1, 3)
            let keys = wk(kInput)
                .reshaped(batchSize, kvLength, kvHeads, headDimension)
                .transposed(0, 2, 1, 3)
            let values = wv(vInput)
                .reshaped(batchSize, kvLength, kvHeads, headDimension)
                .transposed(0, 2, 1, 3)

            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
            .applying(wo)
        }
    }

    final class ResidualAttentionBlock: Module {
        @ModuleInfo var attention: Attention
        @ModuleInfo(key: "feed_forward") var feedForward: VisionMLP
        @ModuleInfo(key: "attention_norm") var attentionNorm: LayerNorm
        @ModuleInfo(key: "ffn_norm") var feedForwardNorm: LayerNorm

        init(_ config: MolmoConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _feedForward.wrappedValue = VisionMLP(config)
            _attentionNorm.wrappedValue = LayerNorm(
                dimensions: config.imageEmbeddingDimension,
                eps: config.imageNormEps,
            )
            _feedForwardNorm.wrappedValue = LayerNorm(
                dimensions: config.imageEmbeddingDimension,
                eps: config.imageNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attention(attentionNorm(x))
            h = h + feedForward(feedForwardNorm(h))
            return h
        }
    }

    final class ResidualAttentionBlocks: Module {
        let resblocks: [ResidualAttentionBlock]

        init(_ config: MolmoConfiguration.VisionConfiguration) {
            resblocks = (0 ..< config.imageLayers).map { _ in ResidualAttentionBlock(config) }
        }

        func callAsFunction(_ x: MLXArray) -> [MLXArray] {
            var x = x
            var hiddenStates = [MLXArray]()
            hiddenStates.reserveCapacity(resblocks.count)
            for block in resblocks {
                x = block(x)
                hiddenStates.append(x)
            }
            return hiddenStates
        }
    }

    final class VisionTransformer: Module {
        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "pre_ln") var preLayerNorm: LayerNorm
        @ModuleInfo var transformer: ResidualAttentionBlocks

        let config: MolmoConfiguration.VisionConfiguration

        init(_ config: MolmoConfiguration.VisionConfiguration) {
            self.config = config
            _classEmbedding.wrappedValue = MLXArray.zeros([config.imageEmbeddingDimension])
            _positionalEmbedding.wrappedValue = MLXArray.zeros([
                config.imageNumPositions, config.imageEmbeddingDimension,
            ])
            _patchEmbedding.wrappedValue = Linear(
                config.intermediateSize,
                config.imageEmbeddingDimension,
                bias: false,
            )
            _preLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.imageEmbeddingDimension,
                eps: config.imageNormEps,
            )
            _transformer.wrappedValue = ResidualAttentionBlocks(config)
        }

        func callAsFunction(_ x: MLXArray, patchNum: (Int, Int)? = nil) -> [MLXArray] {
            let patchNum = patchNum ?? config.imageNumPatch
            var x = padToMultiple(x, targetSize: config.intermediateSize)
            x = patchEmbedding(x)

            let classEmbedding = broadcast(
                classEmbedding.reshaped(1, 1, -1),
                to: [x.dim(0), 1, classEmbedding.dim(0)],
            )
            x = concatenated([classEmbedding, x], axis: 1)
            x = x + positionEmbedding(patchNum: patchNum)
            x = preLayerNorm(x)
            return transformer(x)
        }

        private func positionEmbedding(patchNum: (Int, Int)) -> MLXArray {
            let cls = positionalEmbedding[0 ..< 1]
            var pos = positionalEmbedding[1...]
            let side = Int(Double(pos.dim(0)).squareRoot())
            if side * side == pos.dim(0), side != patchNum.0 || side != patchNum.1 {
                pos = pos.reshaped(side, side, pos.dim(-1))[
                    0 ..< patchNum.0, 0 ..< patchNum.1, 0...
                ]
                .reshaped(-1, pos.dim(-1))
            }
            return concatenated([cls, pos], axis: 0).expandedDimensions(axis: 0)
        }

        private func padToMultiple(_ x: MLXArray, targetSize: Int) -> MLXArray {
            let current = x.dim(-1)
            guard current < targetSize else {
                return x
            }
            return padded(x, widths: [0, 0, .init((0, targetSize - current))])
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "image_vit") var imageVIT: VisionTransformer
        @ModuleInfo(key: "image_pooling_2d") var imagePooling2D: Attention
        @ModuleInfo(key: "image_projector") var imageProjector: MLP
        @ModuleInfo(key: "pad_embed") var padEmbedding: MLXArray

        let config: MolmoConfiguration.VisionConfiguration

        init(_ config: MolmoConfiguration.VisionConfiguration) {
            self.config = config
            _imageVIT.wrappedValue = VisionTransformer(config)
            _imagePooling2D.wrappedValue = Attention(config, isVisionLayer: false)
            _imageProjector.wrappedValue = MLP(
                config,
                inputDimension: config.imageEmbeddingDimension,
            )
            _padEmbedding.wrappedValue = MLXArray.zeros([
                2, config.imageEmbeddingDimension * (config.vitLayers?.count ?? 1),
            ])
        }

        func callAsFunction(_ images: MLXArray, imageMasks: MLXArray) -> (MLXArray, MLXArray?) {
            let batchSize = images.dim(0)
            let numImages = images.dim(1)
            let (imageFeatures, classEmbedding) = encodeImage(images)
            var features = imageFeatures

            if let mode = config.imagePaddingEmbedding {
                let masks = imageMasks.reshaped(batchSize, numImages, imageMasks.dim(-1))
                if mode == "pad_and_partial_pad" {
                    let allPad = masks .== 0
                    let partialPad = (masks .< 1) & .!allPad
                    let pad = padEmbedding.reshaped(2, 1, 1, 1, -1)
                    features =
                        features + pad[0]
                        * allPad.expandedDimensions(axis: -1).asType(features.dtype)
                    features =
                        features
                        + pad[1] * partialPad.expandedDimensions(axis: -1).asType(features.dtype)
                } else if mode == "pad_embed" {
                    let allPad = masks .== 0
                    features =
                        features
                        + padEmbedding[0].reshaped(1, 1, 1, -1)
                        * allPad.expandedDimensions(axis: -1).asType(features.dtype)
                }
            }

            let patch = config.imageNumPatch
            features = features.reshaped(batchSize, numImages, patch.0, patch.1, -1)
            if patch.0 % config.imagePoolingH == 1 {
                features = padded(features, widths: [0, 0, .init((0, 1)), .init((0, 1)), 0])
            }

            let hBlocks = features.dim(2) / config.imagePoolingH
            let wBlocks = features.dim(3) / config.imagePoolingW
            features =
                features
                .reshaped(
                    batchSize,
                    numImages,
                    hBlocks,
                    config.imagePoolingH,
                    wBlocks,
                    config.imagePoolingW,
                    -1,
                )
                .transposed(0, 1, 2, 4, 3, 5, 6)
                .reshaped(
                    batchSize * numImages * hBlocks * wBlocks,
                    config.imagePoolingH * config.imagePoolingW,
                    -1,
                )

            if config.imagePooling2D == "attention-meanq" {
                let query = mean(features, axis: -2, keepDims: true)
                features = imagePooling2D(query, kv: features)
            } else if config.imagePooling2D != "none", config.imagePooling2D != "stack" {
                features = imagePooling2D(features[0..., 0 ..< 1, 0...], kv: features)
            }

            let llmPatch = config.llmPatchesPerCrop
            features = features.reshaped(batchSize, numImages, llmPatch.0 * llmPatch.1, -1)
            return (imageProjector(features), classEmbedding)
        }

        private func encodeImage(_ images: MLXArray) -> (MLXArray, MLXArray?) {
            let batchSize = images.dim(0)
            let numCrops = images.dim(1)
            let numPatches = images.dim(2)
            let patchDim = images.dim(3)
            let flatImages = images.reshaped(batchSize * numCrops, numPatches, patchDim)
            let hiddenStates = imageVIT(flatImages)

            let selected: [MLXArray] =
                if let vitLayers = config.vitLayers {
                    vitLayers.map { layer in
                        let index = layer >= 0 ? layer : hiddenStates.count + layer
                        return hiddenStates[index]
                    }
                } else {
                    [hiddenStates.last!]
                }
            var features = concatenated(selected, axis: -1)
            let classEmbedding = features[0..., 0, 0...].reshaped(batchSize, numCrops, -1)
            features = features[0..., 1..., 0...]
            features = features.reshaped(batchSize, numCrops, numPatches, -1)
            return (features, classEmbedding)
        }
    }
}

extension MLXArray {
    fileprivate func applying(_ layer: Linear) -> MLXArray {
        layer(self)
    }
}

// MARK: - Model

public final class Molmo: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: MolmoLanguage.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MolmoVision.VisionModel

    public let config: MolmoConfiguration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.blocks
    }

    public init(_ config: MolmoConfiguration) {
        self.config = config
        _languageModel.wrappedValue = MolmoLanguage.LanguageModel(config.textConfiguration)
        _visionTower.wrappedValue = MolmoVision.VisionModel(config.visionConfiguration)
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        metadata: MLXArray?,
    )
        -> MLXArray
    {
        var inputEmbeddings = languageModel.model.wte(inputIDs)
        guard let pixelValues, let metadata else {
            return inputEmbeddings
        }

        let tokensPerImage =
            config.visionConfiguration.llmPatchesPerCrop.0
            * config.visionConfiguration.llmPatchesPerCrop.1
        let imageInputIdx = metadata[0..., 0 ..< tokensPerImage].asType(.int32)
        let imageMasks = metadata[0..., tokensPerImage...]
        let dtype = visionTower.imageVIT.patchEmbedding.weight.dtype
        let batchedPixelValues = pixelValues.asType(dtype).expandedDimensions(axis: 0)
        let imageMasksBatch = imageMasks.expandedDimensions(axis: 0)
        var imageFeatures = visionTower(batchedPixelValues, imageMasks: imageMasksBatch).0
        imageFeatures = imageFeatures.reshaped(-1, imageFeatures.dim(-1)).asType(
            inputEmbeddings.dtype)

        let positions = imageInputIdx.flattened().asArray(Int32.self).map(Int.init)
        let validPairs = positions.enumerated().filter { $0.element >= 0 }
        guard !validPairs.isEmpty else {
            return inputEmbeddings
        }

        let featureIndices = validPairs.map(\.offset)
        let tokenPositions = validPairs.map(\.element)
        let featureIndexArray = MLXArray(featureIndices, [featureIndices.count])
        let tokenPositionArray = MLXArray(tokenPositions, [tokenPositions.count])
        let flatEmbeddings = inputEmbeddings.reshaped(-1, inputEmbeddings.dim(-1))
        flatEmbeddings[tokenPositionArray] =
            flatEmbeddings[tokenPositionArray] + imageFeatures[featureIndexArray]
        inputEmbeddings = flatEmbeddings.reshaped(inputEmbeddings.shape)
        return inputEmbeddings
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
        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            var newKey = key
            if newKey.contains("model.transformer") {
                newKey = newKey.replacingOccurrences(
                    of: "model.transformer",
                    with: "language_model.model",
                )
            }
            if newKey.contains("model.vision_backbone") {
                newKey = newKey.replacingOccurrences(
                    of: "model.vision_backbone",
                    with: "vision_tower",
                )
            }
            sanitized[newKey] = value
        }
        return sanitized
    }
}
