import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Molmo2Configuration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vocabularySize: Int
        public let additionalVocabularySize: Int
        public let layerNormEps: Float
        public let ropeTheta: Float
        public let qkvBias: Bool
        public let embeddingDropout: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case additionalVocabularySize = "additional_vocab_size"
            case layerNormEps = "layer_norm_eps"
            case ropeTheta = "rope_theta"
            case qkvBias = "qkv_bias"
            case embeddingDropout = "embedding_dropout"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2_text"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            additionalVocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .additionalVocabularySize) ?? 128
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
            embeddingDropout =
                try c.decodeIfPresent(Float.self, forKey: .embeddingDropout) ?? 0
        }
    }

    public struct VitConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let patchSize: Int
        public let numPositions: Int
        public let inputSize: [Int]
        public let layerNormEps: Float
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case patchSize = "image_patch_size"
            case numPositions = "image_num_pos"
            case inputSize = "image_default_input_size"
            case layerNormEps = "layer_norm_eps"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            let decodedLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 25
            hiddenLayers = min(decodedLayers, 25)
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 16
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 72
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            numPositions = try c.decodeIfPresent(Int.self, forKey: .numPositions) ?? 729
            inputSize = try c.decodeIfPresent([Int].self, forKey: .inputSize) ?? [378, 378]
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            float32Attention =
                try c.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }

        var imageNumPatch: (Int, Int) {
            (inputSize[0] / patchSize, inputSize[1] / patchSize)
        }

        init(
            modelType: String = "molmo2",
            hiddenSize: Int = 1152,
            intermediateSize: Int = 4304,
            hiddenLayers: Int = 25,
            attentionHeads: Int = 16,
            kvHeads: Int = 16,
            headDim: Int = 72,
            patchSize: Int = 14,
            numPositions: Int = 729,
            inputSize: [Int] = [378, 378],
            layerNormEps: Float = 1e-6,
            float32Attention: Bool = true,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.patchSize = patchSize
            self.numPositions = numPositions
            self.inputSize = inputSize
            self.layerNormEps = layerNormEps
            self.float32Attention = float32Attention
        }
    }

    public struct AdapterConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let textHiddenSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vitLayers: [Int]
        public let poolingAttentionMask: Bool
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case textHiddenSize = "text_hidden_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vitLayers = "vit_layers"
            case poolingAttentionMask = "pooling_attention_mask"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
            textHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? 4096
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 16
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 72
            vitLayers = try c.decodeIfPresent([Int].self, forKey: .vitLayers) ?? [-3, -9]
            poolingAttentionMask =
                try c.decodeIfPresent(Bool.self, forKey: .poolingAttentionMask) ?? true
            float32Attention =
                try c.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let vitConfiguration: VitConfiguration
    public let adapterConfiguration: AdapterConfiguration
    public let imageStartTokenID: Int
    public let lowResImageStartTokenID: Int
    public let imageEndTokenID: Int
    public let imagePatchTokenID: Int
    public let imageColumnTokenID: Int
    public let imagePlaceholderTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case vitConfiguration = "vit_config"
        case adapterConfiguration = "adapter_config"
        case imageStartTokenID = "image_start_token_id"
        case lowResImageStartTokenID = "low_res_image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case imagePatchTokenID = "image_patch_id"
        case imageColumnTokenID = "image_col_id"
        case imagePlaceholderTokenID = "image_placeholder_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? JSONDecoder.molmo2Default(TextConfiguration.self)
        vitConfiguration =
            try c.decodeIfPresent(VitConfiguration.self, forKey: .vitConfiguration)
            ?? JSONDecoder.molmo2Default(VitConfiguration.self)
        adapterConfiguration =
            try c.decodeIfPresent(AdapterConfiguration.self, forKey: .adapterConfiguration)
            ?? JSONDecoder.molmo2Default(AdapterConfiguration.self)
        imageStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageStartTokenID) ?? 151_936
        lowResImageStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .lowResImageStartTokenID) ?? 151_940
        imageEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 151_937
        imagePatchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imagePatchTokenID) ?? 151_938
        imageColumnTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageColumnTokenID) ?? 151_939
        imagePlaceholderTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imagePlaceholderTokenID) ?? 151_941
    }
}

extension JSONDecoder {
    fileprivate static func molmo2Default<T: Decodable>(_: T.Type) -> T {
        try! JSONDecoder.json5().decode(T.self, from: "{}".data(using: .utf8)!)
    }
}

public struct Molmo2ProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int
        public let width: Int
    }

    public struct ImageProcessor: Decodable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let maxCrops: Int
        public let overlapMargins: [Int]
        public let patchSize: Int
        public let poolingSize: [Int]
        public let size: Size

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case maxCrops = "max_crops"
            case overlapMargins = "overlap_margins"
            case patchSize = "patch_size"
            case poolingSize = "pooling_size"
            case size
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            imageMean =
                try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
            imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
            maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? 8
            overlapMargins = try c.decodeIfPresent([Int].self, forKey: .overlapMargins) ?? [4, 4]
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            poolingSize = try c.decodeIfPresent([Int].self, forKey: .poolingSize) ?? [2, 2]
            size = try c.decodeIfPresent(Size.self, forKey: .size) ?? Size(height: 378, width: 378)
        }
    }

    public let processorClass: String
    public let imageProcessor: ImageProcessor
    public let imageUseColumnTokens: Bool
    public let useSingleCropColumnTokens: Bool?
    public let useSingleCropStartToken: Bool
    public let useLowResTokenForGlobalCrops: Bool

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case imageUseColumnTokens = "image_use_col_tokens"
        case useSingleCropColumnTokens = "use_single_crop_col_tokens"
        case useSingleCropStartToken = "use_single_crop_start_token"
        case useLowResTokenForGlobalCrops = "use_low_res_token_for_global_crops"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Molmo2Processor"
        imageProcessor =
            try c.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)
            ?? ImageProcessor(from: decoder)
        imageUseColumnTokens =
            try c.decodeIfPresent(Bool.self, forKey: .imageUseColumnTokens) ?? true
        useSingleCropColumnTokens =
            try c.decodeIfPresent(Bool.self, forKey: .useSingleCropColumnTokens)
        useSingleCropStartToken =
            try c.decodeIfPresent(Bool.self, forKey: .useSingleCropStartToken) ?? true
        useLowResTokenForGlobalCrops =
            try c.decodeIfPresent(Bool.self, forKey: .useLowResTokenForGlobalCrops) ?? false
    }
}

// MARK: - Processor

public struct Molmo2Processor: UserInputProcessor {
    private let config: Molmo2ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Molmo2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)

        guard !images.isEmpty else {
            let tokens = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]],
                tools: input.tools,
                additionalContext: input.additionalContext,
            )
            return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let processed = try preprocess(image: images[0].asCIImage(), processing: input.processing)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: [
                [
                    "role": "user",
                    "content": [
                        ["type": "image"],
                        ["type": "text", "text": prompt],
                    ] as [[String: String]],
                ]
            ],
            tools: input.tools,
            additionalContext: input.additionalContext,
        )
        let imageTokens = imageTokens(for: processed.imageGrid)
        let placeholderTokenID = tokenizer.convertTokenToId("<|image|>") ?? 151_941
        if let placeholderIndex = promptTokens.firstIndex(of: placeholderTokenID) {
            promptTokens.replaceSubrange(placeholderIndex ... placeholderIndex, with: imageTokens)
        } else {
            promptTokens.insert(contentsOf: imageTokens, at: 0)
        }
        if config.useLowResTokenForGlobalCrops {
            let bosTokenID =
                tokenizer.bosToken.flatMap { tokenizer.convertTokenToId($0) }
                ?? tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) }
            if let bosTokenID, promptTokens.first != bosTokenID {
                promptTokens.insert(bosTokenID, at: 0)
            }
        }

        let tokenArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: processed.patches),
            video: .init(pixels: processed.pooling),
        )
    }

    private struct ProcessedImage {
        let imageGrid: [Int]
        let patches: MLXArray
        let pooling: MLXArray
    }

    private func imageTokens(for grid: [Int]) -> [Int] {
        let resizedH = grid[0]
        let resizedW = grid[1]
        let highH = grid[2]
        let highW = grid[3]
        let imageStartTokenID = tokenizer.convertTokenToId("<im_start>") ?? 151_936
        let imageEndTokenID = tokenizer.convertTokenToId("<im_end>") ?? 151_937
        let imagePatchTokenID = tokenizer.convertTokenToId("<im_patch>") ?? 151_938
        let imageColumnTokenID = tokenizer.convertTokenToId("<im_col>") ?? 151_939
        let imageLowResTokenID = tokenizer.convertTokenToId("<im_low>") ?? 151_942
        let lowResImageStartTokenID = tokenizer.convertTokenToId("<low_res_im_start>") ?? 151_940

        var highRow = Array(repeating: imagePatchTokenID, count: highW)
        if config.imageUseColumnTokens {
            highRow.append(imageColumnTokenID)
        }
        var high = [imageStartTokenID]
        for _ in 0 ..< highH {
            high.append(contentsOf: highRow)
        }
        high.append(imageEndTokenID)

        let useSingleCropCols = config.useSingleCropColumnTokens ?? config.imageUseColumnTokens
        let lowToken = config.useLowResTokenForGlobalCrops ? imageLowResTokenID : imagePatchTokenID
        var lowRow = Array(repeating: lowToken, count: resizedW)
        if useSingleCropCols {
            lowRow.append(imageColumnTokenID)
        }
        var low = [config.useSingleCropStartToken ? lowResImageStartTokenID : imageStartTokenID]
        for _ in 0 ..< resizedH {
            low.append(contentsOf: lowRow)
        }
        low.append(imageEndTokenID)
        return low + high
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing)
        let imageProcessor = config.imageProcessor
        let maxCrops =
            config.useLowResTokenForGlobalCrops && imageProcessor.maxCrops == 8
            ? 24
            : imageProcessor.maxCrops
        let baseSize = [imageProcessor.size.height, imageProcessor.size.width]
        let patchSize = imageProcessor.patchSize
        let poolingH = imageProcessor.poolingSize[0]
        let poolingW = imageProcessor.poolingSize[1]
        let cropPatchH = baseSize[0] / patchSize
        let cropPatchW = baseSize[1] / patchSize

        let (highCrops, highPatchIndex, _) = Self.buildOverlappingCrops(
            image: image,
            maxCrops: maxCrops,
            overlapMargins: imageProcessor.overlapMargins,
            baseImageInputSize: baseSize,
            mean: imageProcessor.imageMean,
            std: imageProcessor.imageStd,
            patchSize: patchSize,
        )
        let highPoolingGrid = Self.arangeForPooling(
            highPatchIndex,
            poolH: poolingH,
            poolW: poolingW,
        )
        let highH = highPoolingGrid.count
        let highW = highPoolingGrid.first?.count ?? 0
        let highPooling = Array(highPoolingGrid.joined())

        let resized = Self.resizeImage(
            image: image,
            targetHeight: baseSize[0],
            targetWidth: baseSize[1],
            mean: imageProcessor.imageMean,
            std: imageProcessor.imageStd,
        )
        let lowIndex = Array(0 ..< (cropPatchH * cropPatchW)).chunked(width: cropPatchW)
        let lowPoolingGrid = Self.arangeForPooling(lowIndex, poolH: poolingH, poolW: poolingW)
        let lowH = lowPoolingGrid.count
        let lowW = lowPoolingGrid.first?.count ?? 0
        let lowPooling = Array(lowPoolingGrid.joined())

        let offset = cropPatchH * cropPatchW
        let highPoolingOffset = highPooling.map { row in row.map { $0 >= 0 ? $0 + offset : -1 } }
        let pooling = lowPooling + highPoolingOffset
        let poolingValues = pooling.flatMap { $0.map(Int32.init) }
        let poolingArray = MLXArray(poolingValues, [pooling.count, pooling.first?.count ?? 0])

        let allCrops = [resized.pixels] + highCrops
        let patchValues = allCrops.flatMap {
            MediaProcessing.patchifyPixels(
                $0, width: baseSize[1], height: baseSize[0], channels: 3, patchSize: patchSize)
        }
        let patchArray = MLXArray(
            patchValues,
            [allCrops.count, cropPatchH * cropPatchW, patchSize * patchSize * 3],
        )
        return ProcessedImage(
            imageGrid: [lowH, lowW, highH, highW],
            patches: patchArray,
            pooling: poolingArray,
        )
    }

    private struct ResizedImage {
        let pixels: [Float]
        let width: Int
        let height: Int
    }

    private static func buildOverlappingCrops(
        image: CIImage,
        maxCrops: Int,
        overlapMargins: [Int],
        baseImageInputSize: [Int],
        mean: [CGFloat],
        std: [CGFloat],
        patchSize: Int,
    )
        -> ([[Float]], [[Int]], (rows: Int, cols: Int))
    {
        let cropSize = baseImageInputSize[0]
        let leftMargin = overlapMargins[0]
        let rightMargin = overlapMargins.count > 1 ? overlapMargins[1] : leftMargin
        let totalMarginPixels = patchSize * (rightMargin + leftMargin)
        let cropPatches = cropSize / patchSize
        let cropWindowPatches = cropPatches - (rightMargin + leftMargin)
        let cropWindowSize = cropWindowPatches * patchSize
        let cropPatchW = baseImageInputSize[1] / patchSize
        let cropPatchH = baseImageInputSize[0] / patchSize
        let canonical = MediaProcessing.zeroOrigin(image)
        let originalH = Int(canonical.extent.height.rounded())
        let originalW = Int(canonical.extent.width.rounded())
        let tiling = selectTiling(
            height: originalH - totalMarginPixels,
            width: originalW - totalMarginPixels,
            patchSize: cropWindowSize,
            maxCrops: maxCrops,
        )
        let source = resizeImage(
            image: canonical,
            targetHeight: tiling.rows * cropWindowSize + totalMarginPixels,
            targetWidth: tiling.cols * cropWindowSize + totalMarginPixels,
            mean: mean,
            std: std,
        )

        var crops = [[Float]]()
        var patchIndexFull = Array(
            repeating: Array(
                repeating: 0,
                count: tiling.cols * cropWindowPatches + leftMargin + rightMargin,
            ),
            count: tiling.rows * cropWindowPatches + leftMargin + rightMargin,
        )

        var cropIndex = 0
        for row in 0 ..< tiling.rows {
            let y0 = row * cropWindowSize
            for col in 0 ..< tiling.cols {
                let x0 = col * cropWindowSize
                crops.append(
                    MediaProcessing.cropPixels(
                        source.pixels,
                        sourceWidth: source.width,
                        x: x0,
                        y: y0,
                        width: cropSize,
                        height: cropSize,
                        channels: 3,
                    ))
                var patchIdx = Array(
                    repeating: Array(repeating: 0, count: cropPatchW), count: cropPatchH)
                for r in 0 ..< cropPatchH {
                    for c in 0 ..< cropPatchW {
                        patchIdx[r][c] = cropIndex * cropPatchH * cropPatchW + r * cropPatchW + c
                    }
                }
                if row != 0 {
                    for r in 0 ..< leftMargin {
                        for c in 0 ..< cropPatchW {
                            patchIdx[r][c] = -1
                        }
                    }
                }
                if col != 0 {
                    for r in 0 ..< cropPatchH {
                        for c in 0 ..< leftMargin {
                            patchIdx[r][c] = -1
                        }
                    }
                }
                if row != tiling.rows - 1 {
                    for r in (cropPatchH - rightMargin) ..< cropPatchH {
                        for c in 0 ..< cropPatchW {
                            patchIdx[r][c] = -1
                        }
                    }
                }
                if col != tiling.cols - 1 {
                    for r in 0 ..< cropPatchH {
                        for c in (cropPatchW - rightMargin) ..< cropPatchW {
                            patchIdx[r][c] = -1
                        }
                    }
                }

                let yStart = row * cropWindowPatches
                let xStart = col * cropWindowPatches
                for r in 0 ..< cropPatchH {
                    for c in 0 ..< cropPatchW where patchIdx[r][c] >= 0 {
                        patchIndexFull[yStart + r][xStart + c] = patchIdx[r][c]
                    }
                }
                cropIndex += 1
            }
        }
        return (crops, patchIndexFull, tiling)
    }

    private static func arangeForPooling(_ index: [[Int]], poolH: Int, poolW: Int) -> [[[Int]]] {
        let height = index.count
        let width = index.first?.count ?? 0
        let hPad = poolH * ((height + poolH - 1) / poolH) - height
        let wPad = poolW * ((width + poolW - 1) / poolW) - width
        let paddedH = height + hPad
        let paddedW = width + wPad
        var padded = Array(repeating: Array(repeating: -1, count: paddedW), count: paddedH)
        for r in 0 ..< height {
            for c in 0 ..< width {
                padded[hPad / 2 + r][wPad / 2 + c] = index[r][c]
            }
        }
        var result = Array(
            repeating: Array(
                repeating: Array(repeating: -1, count: poolH * poolW), count: paddedW / poolW),
            count: paddedH / poolH,
        )
        for r in 0 ..< (paddedH / poolH) {
            for c in 0 ..< (paddedW / poolW) {
                var values = [Int]()
                for dy in 0 ..< poolH {
                    for dx in 0 ..< poolW {
                        values.append(padded[r * poolH + dy][c * poolW + dx])
                    }
                }
                result[r][c] = values
            }
        }
        return result
    }

    private static func resizeImage(
        image: CIImage,
        targetHeight: Int,
        targetWidth: Int,
        mean: [CGFloat],
        std: [CGFloat],
    )
        -> ResizedImage
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        var pixels = MediaProcessing.renderRGB8(resized, width: targetWidth, height: targetHeight)
        for index in stride(from: 0, to: pixels.count, by: 3) {
            pixels[index] = (pixels[index] - Float(mean[0])) / Float(std[0])
            pixels[index + 1] = (pixels[index + 1] - Float(mean[1])) / Float(std[1])
            pixels[index + 2] = (pixels[index + 2] - Float(mean[2])) / Float(std[2])
        }
        return ResizedImage(pixels: pixels, width: targetWidth, height: targetHeight)
    }

    private static func selectTiling(height: Int, width: Int, patchSize: Int, maxCrops: Int) -> (
        rows: Int, cols: Int
    ) {
        var tilings = [(Int, Int)]()
        for rows in 1 ... maxCrops {
            for cols in 1 ... maxCrops where rows * cols <= maxCrops {
                tilings.append((rows, cols))
            }
        }
        tilings.sort { $0.0 * $0.1 == $1.0 * $1.1 ? $0.0 < $1.0 : $0.0 * $0.1 < $1.0 * $1.1 }
        let originalH = max(Float(height), 1)
        let originalW = max(Float(width), 1)
        let scales = tilings.map {
            min(Float($0.0 * patchSize) / originalH, Float($0.1 * patchSize) / originalW)
        }
        if scales.allSatisfy({ $0 < 1 }) {
            let index = scales.indices.max { scales[$0] < scales[$1] } ?? 0
            return tilings[index]
        }
        let index =
            scales.indices
            .min {
                (scales[$0] < 1 ? Float(1e9) : scales[$0])
                    < (scales[$1] < 1 ? Float(1e9) : scales[$1])
            } ?? 0
        return tilings[index]
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

extension Array {
    fileprivate func chunked(width: Int) -> [[Element]] {
        stride(from: 0, to: count, by: width).map { start in
            Array(self[start ..< Swift.min(start + width, count)])
        }
    }
}

// MARK: - Language

enum Molmo2Language {
    final class Embeddings: Module {
        @ModuleInfo var embedding: MLXArray
        @ModuleInfo(key: "new_embedding") var newEmbedding: MLXArray

        init(count: Int, newCount: Int, dimensions: Int) {
            _embedding.wrappedValue = MLXArray.zeros([count, dimensions])
            _newEmbedding.wrappedValue = MLXArray.zeros([newCount, dimensions])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            concatenated([embedding, newEmbedding], axis: 0)[x]
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "ff_proj") var projection: Linear
        @ModuleInfo(key: "ff_out") var output: Linear

        init(_ config: Molmo2Configuration.TextConfiguration) {
            _projection.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize * 2,
                bias: false,
            )
            _output.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = split(projection(x), parts: 2, axis: -1)
            return output(silu(parts[1]) * parts[0])
        }
    }

    final class Attention: Module {
        @ModuleInfo(key: "att_proj") var projection: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
        @ModuleInfo(key: "attn_out") var output: Linear

        let rotary: RoPE
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let fused: (Int, Int, Int)

        init(_ config: Molmo2Configuration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            fused = (
                config.attentionHeads * headDim, config.kvHeads * headDim, config.kvHeads * headDim
            )
            rotary = RoPE(dimensions: headDim, base: config.ropeTheta)
            _projection.wrappedValue = Linear(
                config.hiddenSize, fused.0 + fused.1 + fused.2, bias: config.qkvBias)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.layerNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.layerNormEps)
            _output.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
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
            let qkv = projection(x)
            let parts = split(qkv, indices: [fused.0, fused.0 + fused.1], axis: -1)
            var q = qNorm(parts[0].reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3)
            var k = kNorm(parts[1].reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3)
            let v = parts[2].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            let offset = cache?.offset ?? 0
            q = rotary(q, offset: offset)
            k = rotary(k, offset: offset)
            return attentionWithCacheUpdate(
                queries: q,
                keys: k,
                values: v,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            .applying(output)
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "ff_norm") var feedForwardNorm: RMSNorm
        @ModuleInfo var mlp: MLP

        init(_ config: Molmo2Configuration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _feedForwardNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            var h = x + attention(attentionNorm(x), mask: mask, cache: cache)
            h = h + mlp(feedForwardNorm(h))
            return h
        }
    }

    final class Transformer: Module {
        @ModuleInfo var wte: Embeddings
        let blocks: [Layer]
        @ModuleInfo(key: "ln_f") var norm: RMSNorm

        init(_ config: Molmo2Configuration.TextConfiguration) {
            _wte.wrappedValue = Embeddings(
                count: config.vocabularySize,
                newCount: config.additionalVocabularySize,
                dimensions: config.hiddenSize,
            )
            blocks = (0 ..< config.hiddenLayers).map { _ in Layer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? wte(inputs!)
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, block) in blocks.enumerated() {
                h = block(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Transformer
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let kvHeads: [Int]

        init(_ config: Molmo2Configuration.TextConfiguration) {
            _model.wrappedValue = Transformer(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            LMOutput(logits: lmHead(model(inputs, cache: cache, inputEmbedding: inputEmbedding)))
        }
    }
}

extension MLXArray {
    fileprivate func applying(_ layer: Linear) -> MLXArray {
        layer(self)
    }
}

// MARK: - Vision

enum Molmo2Vision {
    final class VisionMLP: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w2: Linear
        let activation = GELU(approximation: .fast)

        init(hidden: Int, intermediate: Int) {
            _w1.wrappedValue = Linear(hidden, intermediate, bias: true)
            _w2.wrappedValue = Linear(intermediate, hidden, bias: true)
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

        let hidden: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let float32Attention: Bool

        init(
            hidden: Int,
            heads: Int,
            kvHeads: Int,
            headDim: Int,
            inputDim: Int? = nil,
            outputLayer _: Bool = true,
            float32Attention: Bool = true,
        ) {
            self.hidden = hidden
            self.heads = heads
            self.kvHeads = kvHeads
            self.headDim = headDim
            scale = pow(Float(headDim), -0.5)
            self.float32Attention = float32Attention
            let input = inputDim ?? hidden
            _wq.wrappedValue = Linear(input, heads * headDim, bias: true)
            _wk.wrappedValue = Linear(input, kvHeads * headDim, bias: true)
            _wv.wrappedValue = Linear(input, kvHeads * headDim, bias: true)
            _wo.wrappedValue = Linear(heads * headDim, hidden, bias: true)
        }

        func callAsFunction(_ qInput: MLXArray, kvInput: MLXArray? = nil, mask: MLXArray? = nil)
            -> MLXArray
        {
            let kInput = kvInput ?? qInput
            let batch = qInput.dim(0)
            let qLength = qInput.dim(1)
            let kvLength = kInput.dim(1)
            var q = wq(qInput).reshaped(batch, qLength, heads, headDim)
            var k = wk(kInput).reshaped(batch, kvLength, kvHeads, headDim)
            var v = wv(kInput).reshaped(batch, kvLength, kvHeads, headDim)
            if heads != kvHeads {
                let groups = heads / kvHeads
                k = repeated(k, count: groups, axis: 2)
                v = repeated(v, count: groups, axis: 2)
            }
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
            let dtype = q.dtype
            if float32Attention {
                q = q.asType(.float32)
                k = k.asType(.float32)
                v = v.asType(.float32)
            }
            var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
            if let mask {
                scores = MLX.where(
                    mask,
                    scores,
                    MLXArray.full(
                        scores.shape, values: MLXArray(Float32(-1e9)), dtype: scores.dtype),
                )
            }
            let weights = softmax(scores, axis: -1)
            let output = matmul(weights, v).asType(dtype)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, qLength, -1)
            return wo(output)
        }
    }

    final class Block: Module {
        @ModuleInfo var attention: Attention
        @ModuleInfo(key: "feed_forward") var feedForward: VisionMLP
        @ModuleInfo(key: "attention_norm") var attentionNorm: LayerNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm

        init(_ config: Molmo2Configuration.VitConfiguration) {
            _attention.wrappedValue = Attention(
                hidden: config.hiddenSize,
                heads: config.attentionHeads,
                kvHeads: config.kvHeads,
                headDim: config.headDim,
                float32Attention: config.float32Attention,
            )
            _feedForward.wrappedValue = VisionMLP(
                hidden: config.hiddenSize,
                intermediate: config.intermediateSize,
            )
            _attentionNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _ffnNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attention(attentionNorm(x))
            h = h + feedForward(ffnNorm(h))
            return h
        }
    }

    final class VisionTransformer: Module {
        @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        let transformer: [Block]
        let config: Molmo2Configuration.VitConfiguration

        init(_ config: Molmo2Configuration.VitConfiguration) {
            self.config = config
            _positionalEmbedding.wrappedValue = MLXArray.zeros([
                config.numPositions, config.hiddenSize,
            ])
            _patchEmbedding.wrappedValue = Linear(
                config.patchSize * config.patchSize * 3,
                config.hiddenSize,
                bias: true,
            )
            transformer = (0 ..< config.hiddenLayers).map { _ in Block(config) }
        }

        func callAsFunction(_ x: MLXArray) -> [MLXArray] {
            var x = patchEmbedding(x)
            x = x + positionalEmbedding.expandedDimensions(axis: 0).asType(x.dtype)
            var states = [MLXArray]()
            for block in transformer {
                x = block(x)
                states.append(x)
            }
            return states
        }
    }

    final class Projector: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w2: Linear
        @ModuleInfo var w3: Linear

        init(input: Int, hidden: Int, output: Int) {
            _w1.wrappedValue = Linear(input, hidden, bias: false)
            _w2.wrappedValue = Linear(hidden, output, bias: false)
            _w3.wrappedValue = Linear(input, hidden, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "image_vit") var imageVIT: VisionTransformer
        @ModuleInfo(key: "image_pooling_2d") var imagePooling: Attention
        @ModuleInfo(key: "image_projector") var imageProjector: Projector

        let vitConfig: Molmo2Configuration.VitConfiguration
        let adapterConfig: Molmo2Configuration.AdapterConfiguration
        let vitLayerIndices: [Int]

        init(
            vit: Molmo2Configuration.VitConfiguration,
            adapter: Molmo2Configuration.AdapterConfiguration
        ) {
            vitConfig = vit
            adapterConfig = adapter
            _imageVIT.wrappedValue = VisionTransformer(vit)
            let poolDim = vit.hiddenSize * adapter.vitLayers.count
            _imagePooling.wrappedValue = Attention(
                hidden: adapter.hiddenSize,
                heads: adapter.attentionHeads,
                kvHeads: adapter.kvHeads,
                headDim: adapter.headDim,
                inputDim: poolDim,
                float32Attention: adapter.float32Attention,
            )
            _imageProjector.wrappedValue = Projector(
                input: adapter.hiddenSize,
                hidden: adapter.intermediateSize,
                output: adapter.textHiddenSize,
            )
            vitLayerIndices = adapter.vitLayers.map { $0 >= 0 ? $0 : vit.hiddenLayers + $0 }
        }

        func callAsFunction(_ images: MLXArray, tokenPooling: MLXArray) -> MLXArray {
            let numCrops = images.dim(0)
            let numPatches = images.dim(1)
            let patchDim = images.dim(2)
            let hiddenStates = imageVIT(images.reshaped(numCrops, numPatches, patchDim))
            let selected = vitLayerIndices.map { hiddenStates[$0] }
            let features = concatenated(selected, axis: -1).reshaped(numCrops * numPatches, -1)
            let dim = features.dim(-1)

            let pooling = tokenPooling.asArray(Int32.self).map(Int.init)
            let poolingWidth = tokenPooling.dim(1)
            var clipped = [Int]()
            var validValues = [Float]()
            var validTokenIndices = [Int]()
            for row in 0 ..< tokenPooling.dim(0) {
                var rowHasValid = false
                for col in 0 ..< poolingWidth {
                    let value = pooling[row * poolingWidth + col]
                    if value >= 0 {
                        clipped.append(value)
                        validValues.append(1)
                        rowHasValid = true
                    } else {
                        clipped.append(0)
                        validValues.append(0)
                    }
                }
                if rowHasValid {
                    validTokenIndices.append(row)
                }
            }

            let gathered = features[MLXArray(clipped, [clipped.count])]
                .reshaped(tokenPooling.dim(0), poolingWidth, dim)
            let valid = MLXArray(validValues, [tokenPooling.dim(0), poolingWidth])
            let toPool = gathered * valid.expandedDimensions(axis: -1).asType(gathered.dtype)
            let denom = maximum(sum(valid, axis: -1, keepDims: true), MLXArray(Float32(1)))
            let query =
                sum(toPool, axis: -2, keepDims: true)
                / denom.expandedDimensions(axis: -1).asType(toPool.dtype)
            let mask: MLXArray? =
                adapterConfig.poolingAttentionMask
                ? valid.reshaped(-1, 1, 1, poolingWidth).asType(.bool)
                : nil
            var pooled = imagePooling(query, kvInput: toPool, mask: mask).reshaped(
                tokenPooling.dim(0), -1)
            pooled = imageProjector(pooled)
            return pooled[MLXArray(validTokenIndices, [validTokenIndices.count])]
        }
    }
}

// MARK: - Model

public final class Molmo2: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: Molmo2Language.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Molmo2Vision.VisionModel

    public let config: Molmo2Configuration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.blocks
    }

    public init(_ config: Molmo2Configuration) {
        self.config = config
        _languageModel.wrappedValue = Molmo2Language.LanguageModel(config.textConfiguration)
        _visionTower.wrappedValue = Molmo2Vision.VisionModel(
            vit: config.vitConfiguration,
            adapter: config.adapterConfiguration,
        )
    }

    private func inputEmbeddings(
        inputIDs: MLXArray, pixelValues: MLXArray?, tokenPooling: MLXArray?
    ) -> MLXArray {
        let safeIDs = inputIDs * (inputIDs .!= -1).asType(inputIDs.dtype)
        var embeddings = languageModel.model.wte(safeIDs)
        guard let pixelValues, let tokenPooling else {
            return embeddings
        }
        let dtype = visionTower.imageVIT.patchEmbedding.weight.dtype
        let imageFeatures = Self.clipImageFeatures(
            visionTower(pixelValues.asType(dtype), tokenPooling: tokenPooling),
        ).asType(embeddings.dtype)
        let ids = inputIDs.flattened().asArray(Int32.self).map(Int.init)
        let positions = ids.indices.filter { ids[$0] == config.imagePatchTokenID }
        precondition(
            positions.count == imageFeatures.dim(0), "Molmo2 image feature/token count mismatch")
        let flat = embeddings.reshaped(-1, embeddings.dim(-1))
        let positionArray = MLXArray(positions, [positions.count])
        flat[positionArray] = flat[positionArray] + imageFeatures
        embeddings = flat.reshaped(embeddings.shape)
        return embeddings
    }

    private static func clipImageFeatures(_ imageFeatures: MLXArray) -> MLXArray {
        let dtypeMax = Float(
            imageFeatures.dtype.finfo?.max ?? Double(Float.greatestFiniteMagnitude))
        let clipLimit = imageFeatures.dtype == .float16 ? min(Float(65000), dtypeMax) : dtypeMax
        let cleaned = MLX.where(
            isNaN(imageFeatures),
            zeros(like: imageFeatures),
            imageFeatures,
        )
        return clip(cleaned, min: -clipLimit, max: clipLimit)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            tokenPooling: input.video?.pixels,
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
            var newKey = key
            if newKey.hasPrefix("model.transformer.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.transformer.",
                    with: "language_model.model.",
                    options: [.anchored],
                )
            }
            if newKey.hasPrefix("model.vision_backbone.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.vision_backbone.",
                    with: "vision_tower.",
                    options: [.anchored],
                )
            }
            if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.lm_head.",
                    options: [.anchored],
                )
            }
            newKey = newKey.replacingOccurrences(
                of: ".transformer.resblocks.", with: ".transformer.")
            sanitized[newKey] = value
        }
        return sanitized
    }
}
