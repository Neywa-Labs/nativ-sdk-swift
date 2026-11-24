import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

private struct ErnieIntList: Decodable {
    let values: [Int]

    init(_ values: [Int]) {
        self.values = values
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            values = [value]
        } else {
            values = try container.decode([Int].self)
        }
    }
}

public struct Ernie4_5VLConfiguration: Decodable, Sendable {
    public struct RopeScaling: Decodable, Sendable {
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let depth: Int
        public let embedDim: Int
        public let hiddenSize: Int
        public let hiddenAct: String
        public let mlpRatio: Float
        public let attentionHeads: Int
        public let channels: Int
        public let patchSize: Int
        public let spatialMergeSize: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case embedDim = "embed_dim"
            case hiddenSize = "hidden_size"
            case hiddenAct = "hidden_act"
            case mlpRatio = "mlp_ratio"
            case attentionHeads = "num_heads"
            case channels = "in_channels"
            case inChans = "in_chans"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "DFNRope_vision_transformer"
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 32
            embedDim = try c.decodeIfPresent(Int.self, forKey: .embedDim) ?? 1280
            let decodedHiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            hiddenSize =
                decodedHiddenSize == nil || decodedHiddenSize == 3584
                ? embedDim : decodedHiddenSize!
            hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "quick_gelu"
            mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels =
                try c.decodeIfPresent(Int.self, forKey: .channels)
                ?? c.decodeIfPresent(Int.self, forKey: .inChans)
                ?? 3
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let vocabularySize: Int
        public let rmsNormEps: Float
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let useBias: Bool
        public let tieWordEmbeddings: Bool
        private let _headDim: Int?
        public var headDim: Int {
            _headDim ?? hiddenSize / attentionHeads
        }

        public let moeNumExperts: [Int]
        public let moeLayerStartIndex: [Int]
        public let moeLayerEndIndex: [Int]?
        public let moeIntermediateSize: [Int]
        public let moeK: Int
        public let moeLayerInterval: Int
        public let moeNumSharedExperts: Int
        public let moeNormMin: Float
        public let mropeSection: [Int]

        public var textExperts: Int {
            moeNumExperts.first ?? 128
        }

        public var multimodalExperts: Int {
            moeNumExperts.count > 1 ? moeNumExperts[1] : 0
        }

        public var textMoeIntermediateSize: Int {
            moeIntermediateSize.first ?? 1408
        }

        public var multimodalMoeIntermediateSize: Int {
            moeIntermediateSize.count > 1 ? moeIntermediateSize[1] : textMoeIntermediateSize
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case rmsNormEps = "rms_norm_eps"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case useBias = "use_bias"
            case tieWordEmbeddings = "tie_word_embeddings"
            case _headDim = "head_dim"
            case moeNumExperts = "moe_num_experts"
            case moeLayerStartIndex = "moe_layer_start_index"
            case moeLayerEndIndex = "moe_layer_end_index"
            case moeIntermediateSize = "moe_intermediate_size"
            case moeK = "moe_k"
            case moeLayerInterval = "moe_layer_interval"
            case moeNumSharedExperts = "moe_num_shared_experts"
            case moeNormMin = "moe_norm_min"
            case ropeScaling = "rope_scaling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ernie"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3584
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18944
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 56
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 28
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            _headDim = try c.decodeIfPresent(Int.self, forKey: ._headDim)
            try moeNumExperts =
                (c.decodeIfPresent(ErnieIntList.self, forKey: .moeNumExperts))?.values
                ?? [128]
            try moeLayerStartIndex =
                (c.decodeIfPresent(ErnieIntList.self, forKey: .moeLayerStartIndex))?.values
                ?? [3]
            moeLayerEndIndex =
                try c.decodeIfPresent(ErnieIntList.self, forKey: .moeLayerEndIndex)?.values
            try moeIntermediateSize =
                (c.decodeIfPresent(ErnieIntList.self, forKey: .moeIntermediateSize))?.values
                ?? [1408]
            moeK = try c.decodeIfPresent(Int.self, forKey: .moeK) ?? 2
            moeLayerInterval = try c.decodeIfPresent(Int.self, forKey: .moeLayerInterval) ?? 1
            moeNumSharedExperts =
                try c.decodeIfPresent(Int.self, forKey: .moeNumSharedExperts) ?? 0
            moeNormMin = try c.decodeIfPresent(Float.self, forKey: .moeNormMin) ?? 1e-12
            let ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
            mropeSection = ropeScaling?.mropeSection ?? [22, 22, 20]
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let vocabularySize: Int
    public let imageTokenID: Int
    public let imageStartTokenID: Int
    public let imageEndTokenID: Int
    public let videoTokenID: Int
    public let videoStartTokenID: Int
    public let videoEndTokenID: Int
    public let pixelHiddenSize: Int
    public let hiddenSize: Int
    public let spatialConvSize: Int
    public let temporalConvSize: Int
    public let useTemporalConv: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
        case imPatchID = "im_patch_id"
        case imageTokenID = "image_token_id"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case videoTokenID = "video_token_id"
        case videoStartTokenID = "video_start_token_id"
        case videoEndTokenID = "video_end_token_id"
        case pixelHiddenSize = "pixel_hidden_size"
        case hiddenSize = "hidden_size"
        case spatialConvSize = "spatial_conv_size"
        case temporalConvSize = "temporal_conv_size"
        case useTemporalConv = "use_temporal_conv"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ernie4_5_moe_vl"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        visionConfiguration = try c.decode(
            VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        let imPatchID = try c.decodeIfPresent(Int.self, forKey: .imPatchID) ?? 100_295
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? imPatchID
        imageStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageStartTokenID) ?? 101_304
        imageEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 101_305
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? imageTokenID
        videoStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .videoStartTokenID) ?? 101_306
        videoEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .videoEndTokenID) ?? 101_307
        pixelHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .pixelHiddenSize)
            ?? visionConfiguration.embedDim
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? textConfiguration.hiddenSize
        spatialConvSize = try c.decodeIfPresent(Int.self, forKey: .spatialConvSize) ?? 2
        temporalConvSize = try c.decodeIfPresent(Int.self, forKey: .temporalConvSize) ?? 2
        useTemporalConv = try c.decodeIfPresent(Bool.self, forKey: .useTemporalConv) ?? true
    }
}

public struct Ernie4_5VLProcessorConfiguration: Decodable, Sendable {
    private struct NestedImageProcessor: Decodable {
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let minPixels: Int?
        let maxPixels: Int?
        let mergeSize: Int?
        let patchSize: Int?
        let temporalPatchSize: Int?
        let rescaleFactor: CGFloat?

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case minPixels = "min_pixels"
            case maxPixels = "max_pixels"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case rescaleFactor = "rescale_factor"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialConvSize: Int
    public let temporalConvSize: Int

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialConvSize = "spatial_conv_size"
        case temporalConvSize = "temporal_conv_size"
        case imageProcessor = "image_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd
            ?? [0.26862954, 0.26130258, 0.27577711]
        minPixels =
            try c.decodeIfPresent(Int.self, forKey: .minPixels) ?? nested?.minPixels ?? 3136
        maxPixels =
            try c.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? nested?.maxPixels
            ?? 1_003_520
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? nested?.mergeSize ?? 2
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? nested?.patchSize ?? 14
        temporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
            ?? nested?.temporalPatchSize
            ?? 2
        spatialConvSize =
            try c.decodeIfPresent(Int.self, forKey: .spatialConvSize) ?? 2
        temporalConvSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalConvSize) ?? 2
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
}

// MARK: - Shared Helpers

private enum Ernie4_5VLSupport {
    static func roundByFactor(_ value: Int, factor: Int) -> Int {
        Int(round(Double(value) / Double(factor))) * factor
    }

    static func ceilByFactor(_ value: Int, factor: Int) -> Int {
        Int(ceil(Double(value) / Double(factor))) * factor
    }

    static func floorByFactor(_ value: Int, factor: Int) -> Int {
        Int(floor(Double(value) / Double(factor))) * factor
    }

    static func targetSize(
        height originalHeight: Int,
        width originalWidth: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
    )
        -> (height: Int, width: Int)
    {
        var height = max(1, originalHeight)
        var width = max(1, originalWidth)
        if height / width > 200 {
            width = max(1, height / 200)
        } else if width / height > 200 {
            height = max(1, width / 200)
        }

        var targetHeight = max(factor, roundByFactor(height, factor: factor))
        var targetWidth = max(factor, roundByFactor(width, factor: factor))

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            targetHeight = floorByFactor(Int(Double(height) / beta), factor: factor)
            targetWidth = floorByFactor(Int(Double(width) / beta), factor: factor)
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            targetHeight = ceilByFactor(Int(Double(height) * beta), factor: factor)
            targetWidth = ceilByFactor(Int(Double(width) * beta), factor: factor)
        }

        return (max(factor, targetHeight), max(factor, targetWidth))
    }

    static func mergeInputIDsWithImageFeatures(
        inputIDs: MLXArray,
        inputEmbeds: MLXArray,
        imageFeatures: MLXArray,
        imageTokenID: Int,
        videoTokenID: Int,
    )
        -> MLXArray
    {
        let tokens = inputIDs.flattened().asArray(Int.self)
        let positions = tokens.indices.filter {
            tokens[$0] == imageTokenID || tokens[$0] == videoTokenID
        }
        precondition(
            positions.count == imageFeatures.dim(0),
            "ERNIE-4.5-VL image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0)).",
        )
        let embeddings = inputEmbeds
        embeddings[0..., MLXArray(positions), 0...] = imageFeatures.expandedDimensions(axis: 0)
        return embeddings
    }
}

// MARK: - Processor

public struct Ernie4_5VLProcessor: UserInputProcessor {
    private let config: Ernie4_5VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Ernie4_5VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)

        if images.isEmpty {
            let text = Self.renderPrompt(prompt: prompt, imageTokenCounts: [])
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        let processed = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let frames = processed.map(\.1)
        let tokenCounts = frames.map {
            $0.product / (config.spatialConvSize * config.spatialConvSize)
        }
        let text = Self.renderPrompt(prompt: prompt, imageTokenCounts: tokenCounts)
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: concatenated(processed.map(\.0)), frames: frames),
        )
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let image = MediaProcessing.apply(image, processing: processing)
        let size = image.extent.size
        let factor = config.patchSize * config.mergeSize
        let target = Ernie4_5VLSupport.targetSize(
            height: Int(size.height),
            width: Int(size.width),
            factor: factor,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )
        let targetSize = CGSize(width: target.width, height: target.height)
        let processed = image.toSRGB().resampled(to: targetSize, method: .bicubic)
        let normalized = normalizedCHWArray(processed)
        let patches = patchify(normalized)
        return (
            patches,
            THW(1, target.height / config.patchSize, target.width / config.patchSize),
        )
    }

    private func normalizedCHWArray(_ image: CIImage) -> MLXArray {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                image,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        let unpremultiplied = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        let mean = MLXArray(config.imageMean.map { Float32($0) })
        let std = MLXArray(config.imageStd.map { Float32($0) })
        return ((unpremultiplied - mean) / std).transposed(2, 0, 1)
    }

    private func patchify(_ image: MLXArray) -> MLXArray {
        let channels = image.dim(0)
        let height = image.dim(1)
        let width = image.dim(2)
        let gridH = height / config.patchSize
        let gridW = width / config.patchSize
        var patches = image.reshaped(
            channels,
            gridH / config.mergeSize,
            config.mergeSize,
            config.patchSize,
            gridW / config.mergeSize,
            config.mergeSize,
            config.patchSize,
        )
        patches = patches.transposed(1, 4, 2, 5, 0, 3, 6)
        return patches.reshaped(gridH * gridW, channels * config.patchSize * config.patchSize)
    }

    private static func renderPrompt(prompt: String, imageTokenCounts: [Int]) -> String {
        let system =
            "You are a multimodal AI assistant called ERNIE developed by Baidu based on the PaddlePaddle framework.\n"
        let imageContent = imageTokenCounts.enumerated().map { index, count in
            " Picture \(index + 1):<|IMAGE_START|>\(String(repeating: "<|IMAGE_PLACEHOLDER|>", count: count))<|IMAGE_END|>"
        }.joined()
        return "<|begin_of_sentence|>\(system)User:\(imageContent)\(prompt)\nAssistant: \n<think>\n"
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.map(\.content).joined(separator: "\n")
        }
    }

}

// MARK: - Language

private enum Ernie4_5VLLanguage {
    fileprivate final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let positionSelector: [Int]
        private let frequencySelector: [Int]

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration) {
            let dim = config.headDim
            let base = MLXArray(config.ropeTheta)
            let p = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
            let inv = 1.0 / pow(base, p)
            let tDim = config.mropeSection[2]
            if tDim > 0 {
                let hwFreqs = inv[..<(inv.dim(0) - tDim)]
                invFreq = concatenated([
                    hwFreqs[.ellipsis, stride(from: 0, to: hwFreqs.dim(0), by: 2)],
                    hwFreqs[.ellipsis, stride(from: 1, to: hwFreqs.dim(0), by: 2)],
                    inv[(inv.dim(0) - tDim)...],
                ])
            } else {
                let hwFreqs = inv
                invFreq = concatenated([
                    hwFreqs[.ellipsis, stride(from: 0, to: hwFreqs.dim(0), by: 2)],
                    hwFreqs[.ellipsis, stride(from: 1, to: hwFreqs.dim(0), by: 2)],
                ])
            }

            let sections = config.mropeSection
            let offsets = [0, sections[0], sections[0] + sections[1]]
            var pos = [Int]()
            var freq = [Int]()
            let interleaveLength = sections[0]
            for local in 0 ..< interleaveLength {
                pos.append(1)
                freq.append(offsets[0] + local)
                pos.append(2)
                freq.append(offsets[1] + local)
            }
            for local in 0 ..< sections[2] {
                pos.append(0)
                freq.append(offsets[2] + local)
            }
            positionSelector = pos
            frequencySelector = freq
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            var positionIDs = positionIDs
            if positionIDs.ndim == 2 {
                positionIDs = tiled(
                    positionIDs[.newAxis, 0..., 0...],
                    repetitions: [3, 1, 1],
                )
            }

            var chunks = [MLXArray]()
            chunks.reserveCapacity(positionSelector.count)
            for (axis, freqIndex) in zip(positionSelector, frequencySelector) {
                let positions = positionIDs[axis, 0..., 0...].asType(.float32)
                chunks.append(positions * invFreq[freqIndex])
            }
            let freqs = stacked(chunks, axis: -1)
            let emb = repeated(freqs, count: 2, axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyRotary(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = cos[0..., .newAxis, 0..., 0...]
        let sin = sin[0..., .newAxis, 0..., 0...]

        func apply(_ x: MLXArray) -> MLXArray {
            let originalShape = x.shape
            let rotaryDim = cos.dim(-1)
            let xRot = x[.ellipsis, ..<rotaryDim].asType(.float32)
            let xPass = x[.ellipsis, rotaryDim...]
            let paired = xRot.reshaped(originalShape.dropLast() + [rotaryDim / 2, 2])
            let x0 = paired[.ellipsis, 0]
            let x1 = paired[.ellipsis, 1]
            let c = cos.reshaped(cos.shape.dropLast() + [rotaryDim / 2, 2])[.ellipsis, 0]
            let s = sin.reshaped(sin.shape.dropLast() + [rotaryDim / 2, 2])[.ellipsis, 0]
            let rotated = stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1)
                .reshaped(originalShape.dropLast() + [rotaryDim])
                .asType(x.dtype)
            if xPass.dim(-1) == 0 {
                return rotated
            }
            return concatenated([rotated, xPass], axis: -1)
        }

        return (apply(q), apply(k))
    }

    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rotaryEmbedding: RotaryEmbedding

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            rotaryEmbedding = RotaryEmbedding(config)
            _wq.wrappedValue = Linear(dim, heads * headDim, bias: config.useBias)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: config.useBias)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)
            queries = queries.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyRotary(q: queries, k: keys, cos: cos, sin: sin)

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

            return wo(output)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int, bias: Bool) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: bias)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: bias)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: bias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate final class MoeMLP: Module {
        let k: Int
        let normMin: Float
        let hasDualExperts: Bool
        let numMultimodalExperts: Int

        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "gate_1") var gate1: Linear?
        @ModuleInfo(key: "e_score_correction_bias_1") var eScoreCorrectionBias1: MLXArray?
        @ModuleInfo(key: "switch_mlp_1") var switchMLP1: SwitchGLU?
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration) {
            k = config.moeK
            normMin = config.moeNormMin
            numMultimodalExperts = config.multimodalExperts
            hasDualExperts = config.multimodalExperts > 0
            _gate.wrappedValue = Linear(
                config.hiddenSize,
                config.textExperts,
                bias: false,
            )
            _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.textExperts])
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.textMoeIntermediateSize,
                numExperts: config.textExperts,
                bias: config.useBias,
            )

            if hasDualExperts {
                _gate1.wrappedValue = Linear(
                    config.hiddenSize,
                    config.multimodalExperts,
                    bias: false,
                )
                _eScoreCorrectionBias1.wrappedValue = MLXArray.zeros([
                    config.multimodalExperts
                ])
                _switchMLP1.wrappedValue = SwitchGLU(
                    inputDims: config.hiddenSize,
                    hiddenDims: config.multimodalMoeIntermediateSize,
                    numExperts: config.multimodalExperts,
                    bias: config.useBias,
                )
            }

            if config.moeNumSharedExperts > 0 {
                _sharedExperts.wrappedValue = MLP(
                    dimensions: config.hiddenSize,
                    hiddenDimensions: config.textMoeIntermediateSize
                        * config.moeNumSharedExperts,
                    bias: config.useBias,
                )
            }
        }

        private func route(
            _ x: MLXArray,
            gate: Linear,
            correctionBias: MLXArray,
        )
            -> (MLXArray, MLXArray)
        {
            let logits = gate(x).asType(.float32)
            let routingWeights = softmax(logits, axis: -1)
            let biasedWeights = routingWeights + correctionBias
            let selected = argPartition(-biasedWeights, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var scores = takeAlong(routingWeights, selected, axis: -1)
            scores = scores / maximum(scores.sum(axis: -1, keepDims: true), MLXArray(normMin))
            return (selected, scores)
        }

        func callAsFunction(_ x: MLXArray, tokenTypeIDs: MLXArray?) -> MLXArray {
            let (indices, scores) = route(x, gate: gate, correctionBias: eScoreCorrectionBias)
            var yText = switchMLP(x, indices)
            yText = (yText * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(yText.dtype)

            var y = yText
            if hasDualExperts,
                let gate1,
                let eScoreCorrectionBias1,
                let switchMLP1,
                let tokenTypeIDs
            {
                let (mmIndices, mmScores) = route(
                    x,
                    gate: gate1,
                    correctionBias: eScoreCorrectionBias1,
                )
                var yMM = switchMLP1(x, mmIndices)
                yMM = (yMM * mmScores[.ellipsis, .newAxis]).sum(axis: -2).asType(yMM.dtype)
                y = `where`((tokenTypeIDs .== 0)[.ellipsis, .newAxis], yText, yMM)
            }

            if let sharedExperts {
                y = y + sharedExperts(x)
            }
            return y
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        let mlp: Module
        let usesMoe: Bool

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )

            let start = config.moeLayerStartIndex.min() ?? 3
            let end = config.moeLayerEndIndex?.max() ?? (config.hiddenLayers - 1)
            let useMoe =
                ((layerIndex + 1) % config.moeLayerInterval == 0)
                && layerIndex >= start
                && layerIndex <= end
            usesMoe = useMoe
            if useMoe {
                mlp = MoeMLP(config)
            } else {
                mlp = MLP(
                    dimensions: config.hiddenSize,
                    hiddenDimensions: config.intermediateSize,
                    bias: config.useBias,
                )
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
            tokenTypeIDs: MLXArray?,
        )
            -> MLXArray
        {
            var residual = attention(
                inputLayerNorm(x),
                mask: mask,
                cache: cache,
                positionEmbeddings: positionEmbeddings,
            )
            let hidden = x + residual
            if usesMoe, let moe = mlp as? MoeMLP {
                residual = moe(postAttentionLayerNorm(hidden), tokenTypeIDs: tokenTypeIDs)
            } else if let dense = mlp as? MLP {
                residual = dense(postAttentionLayerNorm(hidden))
            } else {
                fatalError("unexpected ERNIE MLP type")
            }
            return hidden + residual
        }
    }

    fileprivate final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIDs: MLXArray,
            tokenTypeIDs: MLXArray?,
            rotaryEmbedding: RotaryEmbedding,
        )
            -> MLXArray
        {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let positionEmbeddings = rotaryEmbedding(h, positionIDs: positionIDs)
            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[i],
                    positionEmbeddings: positionEmbeddings,
                    tokenTypeIDs: tokenTypeIDs,
                )
            }
            return norm(h)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: Ernie4_5VLConfiguration.TextConfiguration
        let kvHeads: [Int]
        let rotaryEmbedding: RotaryEmbedding
        var positionIDs: MLXArray?
        var ropeDeltas: MLXArray?

        init(_ config: Ernie4_5VLConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
            rotaryEmbedding = RotaryEmbedding(config)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
            inputEmbedding: MLXArray? = nil,
            tokenTypeIDs: MLXArray? = nil,
        )
            -> LMOutput
        {
            let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
            let seqLen =
                inputEmbedding?.dim((inputEmbedding?.ndim ?? 0) - 2)
                ?? inputs?.dim((inputs?.ndim ?? 0) - 1)
                ?? 0
            let cacheOffset = cache?.first?.offset ?? 0
            let currentPositionIDs: MLXArray

            if let positionIDs {
                let storedLength = positionIDs.dim(2)
                if cacheOffset + seqLen <= storedLength {
                    currentPositionIDs =
                        positionIDs[0..., 0..., cacheOffset ..< (cacheOffset + seqLen)]
                } else {
                    let base = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                    let tiledBase = tiled(base[.newAxis, 0...], repetitions: [batchSize, 1])
                    let delta = ropeDeltas ?? MLXArray(Int32(0))
                    let positions = tiledBase + delta
                    currentPositionIDs = tiled(
                        positions[.newAxis, 0..., 0...],
                        repetitions: [3, 1, 1],
                    )
                }
            } else {
                let base = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                let positions = tiled(base[.newAxis, 0...], repetitions: [batchSize, 1])
                currentPositionIDs = tiled(
                    positions[.newAxis, 0..., 0...],
                    repetitions: [3, 1, 1],
                )
            }

            var output = model(
                inputs,
                cache: cache,
                inputEmbedding: inputEmbedding,
                positionIDs: currentPositionIDs,
                tokenTypeIDs: tokenTypeIDs,
                rotaryEmbedding: rotaryEmbedding,
            )
            if config.tieWordEmbeddings {
                output = model.embedTokens.asLinear(output)
            } else if let lmHead {
                output = lmHead(output)
            }
            return LMOutput(logits: output, state: state)
        }
    }
}

// MARK: - Vision

private enum Ernie4_5VLVision {
    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosines = cos(freqs)
        var sines = sin(freqs)
        cosines = tiled(cosines[0..., .newAxis, 0...], repetitions: [1, 1, 2])
        sines = tiled(sines[0..., .newAxis, 0...], repetitions: [1, 1, 2])
        cosines = cosines[.newAxis, 0..., 0..., 0...]
        sines = sines[.newAxis, 0..., 0..., 0...]
        return ((tensor * cosines) + (RotaryEmbeddingOps.rotateHalf(tensor) * sines)).asType(
            tensor.dtype)
    }

    fileprivate final class VisionRotaryEmbedding {
        let inverseFrequency: MLXArray

        init(dimensions: Int, theta: Float = 10000) {
            let p =
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / Float(dimensions)
            inverseFrequency = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFrequency.dtype)
            return outer(seq, inverseFrequency)
        }
    }

    fileprivate final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Linear

        init(_ config: Ernie4_5VLConfiguration.VisionConfiguration) {
            _proj.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize,
                config.embedDim,
                bias: false,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            proj(hiddenStates.asType(proj.weight.dtype))
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        let hiddenAct: String
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(_ config: Ernie4_5VLConfiguration.VisionConfiguration) {
            hiddenAct = config.hiddenAct
            let hiddenDim = Int(Float(config.embedDim) * config.mlpRatio)
            _fc1.wrappedValue = Linear(config.embedDim, hiddenDim, bias: true)
            _fc2.wrappedValue = Linear(hiddenDim, config.embedDim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let hidden = fc1(x)
            let activated =
                hiddenAct == "quick_gelu"
                ? hidden * sigmoid(MLXArray(Float32(1.702)) * hidden)
                : gelu(hidden)
            return fc2(activated)
        }
    }

    fileprivate final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: Ernie4_5VLConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.embedDim / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(config.embedDim, config.embedDim * 3, bias: true)
            _proj.wrappedValue = Linear(config.embedDim, config.embedDim, bias: true)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: [Int],
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = hiddenStates.dim(0)
            let qkv = qkv(hiddenStates)
                .reshaped(sequenceLength, 3, heads, -1)
                .transposed(1, 0, 2, 3)
            let parts = split(qkv, parts: 3, axis: 0)
            var q = parts[0].squeezed(axis: 0)
            var k = parts[1].squeezed(axis: 0)
            let v = parts[2].squeezed(axis: 0)

            q = applyRotary(q[.newAxis, 0..., 0..., 0...], freqs: rotaryPositionEmbedding)[0]
            k = applyRotary(k[.newAxis, 0..., 0..., 0...], freqs: rotaryPositionEmbedding)[0]

            let qT = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kT = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vT = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

            var outputs = [MLXArray]()
            for index in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[index]
                let end = cuSeqlens[index + 1]
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: qT[0..., 0..., start ..< end, 0...],
                        keys: kT[0..., 0..., start ..< end, 0...],
                        values: vT[0..., 0..., start ..< end, 0...],
                        scale: scale,
                        mask: .none,
                    ),
                )
            }

            return proj(
                concatenated(outputs, axis: 2)
                    .transposed(0, 2, 1, 3)
                    .reshaped(sequenceLength, -1),
            )
        }
    }

    fileprivate final class Block: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: Ernie4_5VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
            _norm2.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: [Int],
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            var hiddenStates =
                hiddenStates
                + attn(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate final class VisionModel: Module {
        let spatialMergeSize: Int
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo var blocks: [Block]
        @ModuleInfo var ln: LayerNorm
        let rotaryEmbedding: VisionRotaryEmbedding

        init(_ config: Ernie4_5VLConfiguration.VisionConfiguration) {
            spatialMergeSize = config.spatialMergeSize
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _blocks.wrappedValue = (0 ..< config.depth).map { _ in Block(config) }
            _ln.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
            let headDim = config.embedDim / config.attentionHeads
            rotaryEmbedding = VisionRotaryEmbedding(dimensions: headDim / 2)
        }

        var patchDType: DType {
            patchEmbed.proj.weight.dtype
        }

        func rotaryPositionEmbedding(frames: [THW]) -> MLXArray {
            var positionIDs = [MLXArray]()
            for frame in frames {
                let (t, h, w) = frame.values
                var hposIDs = MLXArray(0 ..< h).expandedDimensions(axis: 1)
                hposIDs = repeated(hposIDs, count: w, axis: 1)
                hposIDs =
                    hposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIDs = MLXArray(0 ..< w).expandedDimensions(axis: 0)
                wposIDs = repeated(wposIDs, count: h, axis: 0)
                wposIDs =
                    wposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIDs = stacked([hposIDs, wposIDs], axis: -1)
                positionIDs.append(tiled(stackedPosIDs, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIDs, axis: 0)
            let maxGridSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 1
            return rotaryEmbedding(sequenceLength: maxGridSize)[indices]
                .reshaped(indices.dim(0), -1)
        }

        func cuSeqlens(frames: [THW]) -> [Int] {
            var result = [0]
            var total = 0
            for frame in frames {
                total += frame.product
                result.append(total)
            }
            return result
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotary = rotaryPositionEmbedding(frames: frames)
            let cuSeqlens = cuSeqlens(frames: frames)
            for block in blocks {
                hiddenStates = block(
                    hiddenStates,
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotary,
                )
            }
            return ln(hiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights.filter { !$0.key.contains("position_ids") }
        }
    }
}

// MARK: - Resampler

private final class Ernie4_5VLResampler: Module {
    let spatialConvSize: Int
    let temporalConvSize: Int
    let useTemporalConv: Bool
    let spatialDim: Int
    let temporalDim: Int

    @ModuleInfo(key: "spatial_linear") var spatialLinear: Sequential
    @ModuleInfo(key: "temporal_linear") var temporalLinear: Sequential?
    @ModuleInfo var mlp: Linear
    @ModuleInfo(key: "after_norm") var afterNorm: RMSNorm

    init(_ config: Ernie4_5VLConfiguration) {
        spatialConvSize = config.spatialConvSize
        temporalConvSize = config.temporalConvSize
        useTemporalConv = config.useTemporalConv
        spatialDim = config.pixelHiddenSize * config.spatialConvSize * config.spatialConvSize
        temporalDim = spatialDim * config.temporalConvSize
        _spatialLinear.wrappedValue = Sequential(layers: [
            Linear(spatialDim, spatialDim, bias: true),
            GELU(),
            Linear(spatialDim, spatialDim, bias: true),
            LayerNorm(dimensions: spatialDim, eps: 1e-6),
        ])
        if useTemporalConv {
            _temporalLinear.wrappedValue = Sequential(layers: [
                Linear(temporalDim, spatialDim, bias: true),
                GELU(),
                Linear(spatialDim, spatialDim, bias: true),
                LayerNorm(dimensions: spatialDim, eps: 1e-6),
            ])
        }
        _mlp.wrappedValue = Linear(spatialDim, config.hiddenSize, bias: true)
        _afterNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
    }

    private func temporalIndices(frames: [THW]) -> ([Int], [Int]) {
        var first = [Int]()
        var second = [Int]()
        var batchOffset = 0
        for frame in frames {
            let spatial = frame.h * frame.w / (spatialConvSize * spatialConvSize)
            let tokens = frame.t * spatial
            for offset in stride(from: 0, to: frame.t, by: temporalConvSize) {
                first.append(
                    contentsOf: (batchOffset + offset * spatial)
                        ..< (batchOffset + (offset + 1) * spatial))
            }
            let secondStart = frame.t > 1 ? 1 : 0
            for offset in stride(from: secondStart, to: frame.t, by: temporalConvSize) {
                second.append(
                    contentsOf: (batchOffset + offset * spatial)
                        ..< (batchOffset + (offset + 1) * spatial))
            }
            batchOffset += tokens
        }
        return (first, second)
    }

    func callAsFunction(_ x: MLXArray, frames: [THW]) -> MLXArray {
        var x = x.reshaped(-1, spatialDim)
        x = spatialLinear(x)
        if useTemporalConv, let temporalLinear {
            let (first, second) = temporalIndices(frames: frames)
            x = concatenated([x[MLXArray(first), 0...], x[MLXArray(second), 0...]], axis: -1)
            x = temporalLinear(x)
        }
        return afterNorm(mlp(x))
    }
}

// MARK: - Model

public final class Ernie4_5VL: Module, VLMModel, KVCacheDimensionProvider {
    public let config: Ernie4_5VLConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    @ModuleInfo(key: "vision_tower") private var visionTower: Ernie4_5VLVision.VisionModel
    @ModuleInfo(key: "resampler_model") private var resamplerModel: Ernie4_5VLResampler
    @ModuleInfo(key: "language_model") private var languageModel: Ernie4_5VLLanguage.LanguageModel

    public init(_ config: Ernie4_5VLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = Ernie4_5VLVision.VisionModel(config.visionConfiguration)
        _resamplerModel.wrappedValue = Ernie4_5VLResampler(config)
        _languageModel.wrappedValue = Ernie4_5VLLanguage.LanguageModel(
            config.textConfiguration,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        guard let image = input.image, let frames = image.frames, !frames.isEmpty else {
            languageModel.positionIDs = nil
            languageModel.ropeDeltas = nil
            defer { prefill.progress?(input.text.tokens.size, input.text.tokens.size) }
            return .logits(languageModel(input.text.tokens, cache: cache))
        }

        let inputEmbeds = languageModel.model.embedTokens(input.text.tokens)
        let visionHidden = visionTower(image.pixels.asType(visionTower.patchDType), frames: frames)
        let imageFeatures = resamplerModel(visionHidden, frames: frames).asType(inputEmbeds.dtype)
        let embeddings = Ernie4_5VLSupport.mergeInputIDsWithImageFeatures(
            inputIDs: input.text.tokens,
            inputEmbeds: inputEmbeds,
            imageFeatures: imageFeatures,
            imageTokenID: config.imageTokenID,
            videoTokenID: config.videoTokenID,
        )
        let (positionIDs, ropeDeltas) = getRopeIndex(inputIDs: input.text.tokens, frames: frames)
        languageModel.positionIDs = positionIDs
        languageModel.ropeDeltas = ropeDeltas
        let tokenTypeIDs = buildTokenTypeIDs(inputIDs: input.text.tokens)

        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(
                nil,
                cache: cache,
                inputEmbedding: embeddings[0..., range, 0...],
                tokenTypeIDs: tokenTypeIDs[0..., range],
            )
            asyncEval(cache)
        }
        eval(cache)
        defer { prefill.progress?(totalPositions, totalPositions) }
        return .logits(
            languageModel(
                nil,
                cache: cache,
                inputEmbedding: embeddings[0..., processed..., 0...],
                tokenTypeIDs: tokenTypeIDs[0..., processed...],
            ),
        )
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    private func buildTokenTypeIDs(inputIDs: MLXArray) -> MLXArray {
        let imageMask = inputIDs .== MLXArray(config.imageTokenID)
        let videoMask = inputIDs .== MLXArray(config.videoTokenID)
        return `where`(imageMask .|| videoMask, ones(like: inputIDs), zeros(like: inputIDs))
    }

    private func getRopeIndex(inputIDs: MLXArray, frames: [THW]) -> (MLXArray, MLXArray) {
        let batchSize = inputIDs.dim(0)
        let seqLength = inputIDs.dim(1)
        precondition(batchSize == 1, "ERNIE-4.5-VL getRopeIndex only supports batchSize == 1")

        let positionIDs = zeros([3, batchSize, seqLength], type: Int32.self)
        var frameIndex = 0
        var ropeDelta: Int32 = 0

        for batchIndex in 0 ..< batchSize {
            let tokens = inputIDs[batchIndex].asArray(Int.self)
            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(tokens.count)
            dimH.reserveCapacity(tokens.count)
            dimW.reserveCapacity(tokens.count)
            var start = 0
            var lastMax: Int32 = -1

            func appendText(_ count: Int) {
                guard count > 0 else {
                    return
                }
                let base = lastMax + 1
                for offset in 0 ..< count {
                    let value = base + Int32(offset)
                    dimT.append(value)
                    dimH.append(value)
                    dimW.append(value)
                }
                lastMax = base + Int32(count) - 1
            }

            while frameIndex < frames.count,
                let imageIndex = tokens[start...].firstIndex(where: {
                    $0 == config.imageTokenID || $0 == config.videoTokenID
                })
            {
                let frame = frames[frameIndex]
                frameIndex += 1
                let llmGridT = frame.t
                let llmGridH = frame.h / config.visionConfiguration.spatialMergeSize
                let llmGridW = frame.w / config.visionConfiguration.spatialMergeSize
                appendText(imageIndex - start)

                let base = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(base + Int32(t))
                            dimH.append(base + Int32(h))
                            dimW.append(base + Int32(w))
                        }
                    }
                }
                lastMax = max(
                    base + Int32(llmGridT - 1),
                    max(base + Int32(llmGridH - 1), base + Int32(llmGridW - 1)),
                )
                start = imageIndex + llmGridT * llmGridH * llmGridW
            }

            appendText(tokens.count - start)
            positionIDs[0, batchIndex] = MLXArray(dimT)
            positionIDs[1, batchIndex] = MLXArray(dimH)
            positionIDs[2, batchIndex] = MLXArray(dimW)
            ropeDelta = lastMax + 1 - Int32(tokens.count)
        }

        return (positionIDs, MLXArray([ropeDelta]).reshaped(1, 1))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Ernie4_5VLSanitizer.sanitize(weights: weights, config: config)
    }
}

private enum Ernie4_5VLSanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: Ernie4_5VLConfiguration,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            if !newKey.contains("vision_tower"), newKey.contains("vision_model") {
                newKey = newKey.replacingOccurrences(of: "vision_model", with: "vision_tower")
            }
            if !newKey.contains("language_model") {
                if newKey.contains("model.layers")
                    || newKey.contains("model.embed_tokens")
                    || newKey.contains("model.norm")
                {
                    newKey = newKey.replacingOccurrences(
                        of: "model.",
                        with: "language_model.model.",
                    )
                } else if newKey.contains("lm_head") {
                    newKey = newKey.replacingOccurrences(
                        of: "lm_head",
                        with: "language_model.lm_head",
                    )
                }
            }
            if newKey.contains("model.resampler_model") {
                newKey = newKey.replacingOccurrences(
                    of: "model.resampler_model",
                    with: "resampler_model",
                )
            }
            newKey = mapSequentialKey(newKey, name: "spatial_linear")
            newKey = mapSequentialKey(newKey, name: "temporal_linear")
            if config.textConfiguration.tieWordEmbeddings,
                newKey == "language_model.lm_head.weight"
            {
                continue
            }
            sanitized[newKey] = value
        }
        stackExperts(weights: &sanitized, config: config.textConfiguration)
        return sanitized
    }

    private static func mapSequentialKey(_ key: String, name: String) -> String {
        var key = key
        for index in 0 ... 3 {
            key = key.replacingOccurrences(
                of: "\(name).\(index).",
                with: "\(name).layers.\(index).",
            )
        }
        return key
    }

    private static func stackExperts(
        weights: inout [String: MLXArray],
        config: Ernie4_5VLConfiguration.TextConfiguration,
    ) {
        let textExperts = config.textExperts
        let mmExperts = config.multimodalExperts
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(suffix)"
                    if weights[firstKey] != nil {
                        var toStack = [MLXArray]()
                        for expert in 0 ..< textExperts {
                            if let value = weights.removeValue(
                                forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)",
                            ) {
                                toStack.append(value)
                            }
                        }
                        if toStack.count == textExperts {
                            weights["\(prefix).switch_mlp.\(projection).\(suffix)"] =
                                stacked(toStack)
                        }
                    }
                    if mmExperts > 0,
                        weights["\(prefix).experts.\(textExperts).\(projection).\(suffix)"] != nil
                    {
                        var toStack = [MLXArray]()
                        for expert in textExperts ..< (textExperts + mmExperts) {
                            if let value = weights.removeValue(
                                forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)",
                            ) {
                                toStack.append(value)
                            }
                        }
                        if toStack.count == mmExperts {
                            weights["\(prefix).switch_mlp_1.\(projection).\(suffix)"] =
                                stacked(toStack)
                        }
                    }
                }
            }

            let gate1Key = "\(prefix).gate.weight_1"
            if let value = weights.removeValue(forKey: gate1Key) {
                weights["\(prefix).gate_1.weight"] =
                    value.dim(0) > value.dim(1) ? value.transposed() : value
            }

            let correctionKey = "\(prefix).moe_statics.e_score_correction_bias"
            if let bias = weights.removeValue(forKey: correctionKey) {
                if bias.ndim == 2, bias.dim(0) == 2 {
                    weights["\(prefix).e_score_correction_bias"] = bias[0]
                    if mmExperts > 0 {
                        weights["\(prefix).e_score_correction_bias_1"] = bias[1]
                    }
                } else {
                    weights["\(prefix).e_score_correction_bias"] =
                        bias.ndim > 1 ? bias.squeezed(axis: 0) : bias
                }
            }
        }
    }
}
