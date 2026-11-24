import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MageVLConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let ropeTheta: Float
        public let tieWordEmbeddings: Bool

        private struct RopeParameters: Decodable {
            let ropeTheta: Float?
            let ropeType: String?

            enum CodingKeys: String, CodingKey {
                case ropeTheta = "rope_theta"
                case ropeType = "rope_type"
            }
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case ropeParameters = "rope_parameters"
            case tieWordEmbeddings = "tie_word_embeddings"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9728
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

            // transformers 5.x nests rope under `rope_parameters`; a missed rope_theta
            // parity-passes at short lengths and diverges at long ones.
            let nested = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            ropeTheta =
                try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? nested?.ropeTheta ?? 5_000_000
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let intermediateSize: Int
        public let patchSize: Int
        public let imageSize: Int
        public let channels: Int
        public let layerNormEps: Float
        public let layerNormType: String
        public let ropeTheta: Float
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let outHiddenSize: Int
        public let frameWindowsSize: Int
        public let useHead: Bool
        public let maxPositionEmbeddings: Int
        public let usePatchPositionEncoding: Bool
        public let patchPositionEncodingType: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case layerNormType = "layer_norm_type"
            case ropeTheta = "rope_theta"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case outHiddenSize = "out_hidden_size"
            case frameWindowsSize = "frame_windows_size"
            case useHead = "use_head"
            case maxPositionEmbeddings = "max_position_embeddings"
            case usePatchPositionEncoding = "use_patch_position_encoding"
            case patchPositionEncodingType = "patch_position_encoding_type"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mage_vl_vision"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            layerNormType =
                try c.decodeIfPresent(String.self, forKey: .layerNormType) ?? "layer_norm"
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
            outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 2560
            frameWindowsSize = try c.decodeIfPresent(Int.self, forKey: .frameWindowsSize) ?? 4
            useHead = try c.decodeIfPresent(Bool.self, forKey: .useHead) ?? false
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
            usePatchPositionEncoding =
                try c.decodeIfPresent(Bool.self, forKey: .usePatchPositionEncoding) ?? false
            patchPositionEncodingType =
                try c.decodeIfPresent(String.self, forKey: .patchPositionEncodingType)
                ?? "absolute"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mage_vl"
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 151_655
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 151_656
        visionStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID)
            ?? 151_652
        visionEndTokenID = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenID) ?? 151_653
    }
}

// MARK: - Processor

public struct MageVLProcessorConfiguration: Codable, Sendable {
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let mergeSize: Int

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case mergeSize = "merge_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.26862954, 0.26130258, 0.27577711]
        minPixels = try c.decodeIfPresent(Int.self, forKey: .minPixels) ?? 3136
        maxPixels = try c.decodeIfPresent(Int.self, forKey: .maxPixels) ?? 4_000_000
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
    }
}

public struct MageVLProcessor: UserInputProcessor {
    private let config: MageVLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imagePadTokenID: Int

    public init(_ config: MageVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imagePadTokenID = tokenizer.convertTokenToId("<|image_pad|>") ?? 151_655
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.processing(
                "Mage-VL video input needs the upstream codec pipeline; images only for now.")
        }

        let messages = Qwen3VLMessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        let images = Self.images(from: input)
        guard !images.isEmpty else {
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        var pixelBatches: [MLXArray] = []
        var grids: [THW] = []
        for image in images {
            let (pixels, grid) = try preprocess(
                image: image.asCIImage(), processing: input.processing)
            pixelBatches.append(pixels)
            grids.append(grid)
        }

        let merged = config.mergeSize * config.mergeSize
        let counts = grids.map { $0.product / merged }
        let expandedTokens = expandImagePads(in: promptTokens, counts: counts)
        let tokensArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray).asType(.int8)),
            image: .init(pixels: concatenated(pixelBatches, axis: 0), frames: grids),
        )
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let applied = MediaProcessing.apply(image, processing: processing)
        let extent = applied.extent.size
        let factor = config.patchSize * config.mergeSize
        let (resizedHeight, resizedWidth) = try Self.targetSize(
            height: Int(extent.height),
            width: Int(extent.width),
            factor: factor,
            minPixels: processing?.minPixels ?? config.minPixels,
            maxPixels: processing?.maxPixels ?? config.maxPixels,
        )

        // The sRGB tone-curve step has to precede resample/normalize: CoreImage filters run in
        // linear light and asMLXArray renders raw working-space values, so skipping it feeds
        // linearized pixels to a model trained on gamma-encoded bytes.
        let normalized = MediaProcessing.asMLXArray(
            MediaProcessing.normalize(
                MediaProcessing.resampleBicubic(
                    MediaProcessing.inSRGBToneCurveSpace(applied),
                    to: CGSize(width: resizedWidth, height: resizedHeight),
                ),
                mean: config.imageMeanTuple,
                std: config.imageStdTuple,
            ),
        )

        return try Self.patchify(
            images: [normalized],
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
        )
    }

    /// Expands the i-th `<|image_pad|>` to its grid's merged-token count.
    private func expandImagePads(in promptTokens: [Int], counts: [Int]) -> [Int] {
        guard !counts.isEmpty else { return promptTokens }

        var tokens: [Int] = []
        var consumed = 0
        for token in promptTokens {
            guard token == imagePadTokenID, consumed < counts.count else {
                tokens.append(token)
                continue
            }
            tokens.append(
                contentsOf: Array(repeating: imagePadTokenID, count: max(counts[consumed], 1)))
            consumed += 1
        }
        return tokens
    }

    static func targetSize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int,
    )
        throws -> (Int, Int)
    {
        guard height >= factor, width >= factor else {
            throw VLMError.imageProcessingFailure(
                "Image \(width)x\(height) must be at least \(factor) on each side")
        }
        guard max(height, width) / min(height, width) <= 200 else {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) x \(height)")
        }

        var hBar = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var wBar = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor
            wBar = Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }

        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor
        guard hBar > 0, wBar > 0 else {
            throw VLMError.imageProcessingFailure("Invalid target dimensions: \(wBar) x \(hBar)")
        }
        return (hBar, wBar)
    }

    static func patchify(
        images: [MLXArray], mergeSize: Int, patchSize: Int, temporalPatchSize: Int,
    )
        throws -> (MLXArray, THW)
    {
        guard let first = images.first else {
            throw VLMError.imageProcessingFailure("No images to patchify")
        }
        let resizedHeight = first.dim(-2)
        let resizedWidth = first.dim(-1)
        var patches = concatenated(images)

        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let last = patches[-1, .ellipsis]
            patches = concatenated([
                patches, tiled(last, repetitions: [temporalPatchSize - mod, 1, 1, 1]),
            ])
        }

        let channels = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT,
            temporalPatchSize,
            channels,
            gridH / mergeSize,
            mergeSize,
            patchSize,
            gridW / mergeSize,
            mergeSize,
            patchSize,
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        return (
            patches.reshaped(
                gridT * gridH * gridW,
                channels * temporalPatchSize * patchSize * patchSize,
            ),
            THW(gridT, gridH, gridW)
        )
    }

    private static func images(from input: UserInput) -> [UserInput.Image] {
        if !input.images.isEmpty {
            return input.images
        }
        guard case .chat(let messages) = input.prompt else {
            return []
        }
        return messages.flatMap(\.images)
    }
}

// MARK: - Vision

// Three things below look like bugs and are not. The weights were trained with them:
//   1. the rotation is INTERLEAVED, (x1,x2,x3,x4) -> (-x2,x1,-x4,x3), not split-half;
//   2. the frequency table is concat([freqs, freqs]), so under an interleaved rotation
//      lanes 2i and 2i+1 take cos/sin from *different* frequencies;
//   3. the tap is the last hidden state, despite upstream's comment saying second-to-last.
// See mlx_vlm/models/mage_vl/vision.py; do not "fix" without changing it there too.
enum MageVLVision {
    static func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [shape[shape.count - 1] / 2, 2])
        let even = pairs[.ellipsis, 0]
        let odd = pairs[.ellipsis, 1]
        return stacked([-odd, even], axis: -1).reshaped(shape)
    }

    static func applyRotaryPositionEmbedding(
        queries: MLXArray, keys: MLXArray, freqs: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let originalDType = queries.dtype
        let q = queries.asType(.float32)
        let k = keys.asType(.float32)
        let cosines = cos(freqs).expandedDimensions(axis: 1).asType(.float32)
        let sines = sin(freqs).expandedDimensions(axis: 1).asType(.float32)
        let qEmbedded = (q * cosines) + (rotateHalfInterleaved(q) * sines)
        let kEmbedded = (k * cosines) + (rotateHalfInterleaved(k) * sines)
        return (qEmbedded.asType(originalDType), kEmbedded.asType(originalDType))
    }

    fileprivate final class RotaryEmbedding {
        private let inverseFrequencyT: MLXArray
        private let inverseFrequencyH: MLXArray
        private let inverseFrequencyW: MLXArray

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            let headDim = config.hiddenSize / config.attentionHeads
            let half = headDim / 2
            precondition(
                headDim % 16 == 0 && half % 16 == 0,
                "Mage-ViT needs head_dim and head_dim/2 divisible by 16 for the 4:6:6 split")
            let unit = half / 16

            // The exponent is arange(size)/size, not the conventional arange(0, d, 2)/d.
            func inverseFrequency(_ size: Int) -> MLXArray {
                let exponents = MLXArray(0 ..< size).asType(.float32) / Float(size)
                return 1 / pow(MLXArray(config.ropeTheta), exponents)
            }

            inverseFrequencyT = inverseFrequency(4 * unit)
            inverseFrequencyH = inverseFrequency(6 * unit)
            inverseFrequencyW = inverseFrequency(6 * unit)
        }

        func callAsFunction(patchPositions: MLXArray) -> MLXArray {
            let positions = patchPositions.asType(.float32)
            return concatenated(
                [
                    outer(positions[0..., 0], inverseFrequencyT),
                    outer(positions[0..., 1], inverseFrequencyH),
                    outer(positions[0..., 2], inverseFrequencyW),
                ],
                axis: -1,
            )
        }
    }

    /// Patch positions in the processor's 2x2 block order; a row-major t/h/w walk would put the
    /// right number of patches in the wrong places, visible only as degraded output.
    static func patchPositions(grids: [THW], mergeSize: Int) -> MLXArray {
        var rows: [Int32] = []
        for grid in grids {
            for t in 0 ..< grid.t {
                for hBlock in 0 ..< (grid.h / mergeSize) {
                    for wBlock in 0 ..< (grid.w / mergeSize) {
                        for dh in 0 ..< mergeSize {
                            for dw in 0 ..< mergeSize {
                                rows.append(Int32(t))
                                rows.append(Int32(hBlock * mergeSize + dh))
                                rows.append(Int32(wBlock * mergeSize + dw))
                            }
                        }
                    }
                }
            }
        }
        return MLXArray(rows, [rows.count / 3, 3])
    }

    /// Cumulative window boundaries; windows split a sample's frames into `frameWindow` chunks.
    static func windowBoundaries(grids: [THW], totalPatches: Int, frameWindow: Int) -> [Int] {
        guard !grids.isEmpty else { return [0, totalPatches] }

        var boundaries = [0]
        var current = 0
        for grid in grids {
            if frameWindow > 0, grid.t > frameWindow {
                for _ in 0 ..< (grid.t / frameWindow) {
                    current += frameWindow * grid.h * grid.w
                    boundaries.append(current)
                }
                if grid.t % frameWindow != 0 {
                    current += (grid.t % frameWindow) * grid.h * grid.w
                    boundaries.append(current)
                }
            } else {
                current += grid.t * grid.h * grid.w
                boundaries.append(current)
            }
        }
        return boundaries
    }

    static func blockDiagonalMask(boundaries: [Int], dtype: DType) -> MLXArray? {
        guard boundaries.count > 2 else { return nil }
        let total = boundaries[boundaries.count - 1]
        var ids = [Int32](repeating: 0, count: total)
        for index in 0 ..< (boundaries.count - 1) {
            for position in boundaries[index] ..< boundaries[index + 1] {
                ids[position] = Int32(index)
            }
        }
        let identifiers = MLXArray(ids, [total])
        let same =
            identifiers.expandedDimensions(axis: 1)
            .== identifiers.expandedDimensions(
                axis: 0)
        return which(
            same, MLXArray(0).asType(dtype), MLXArray(-Float.infinity).asType(dtype)
        )
        .expandedDimensions(axes: [0, 1])
    }

    fileprivate final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d

        private let patchSize: Int
        private let channels: Int
        private let embedDimensions: Int

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            patchSize = config.patchSize
            channels = config.channels
            embedDimensions = config.hiddenSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let reshaped = x.reshaped(-1, channels, patchSize, patchSize).transposed(0, 2, 3, 1)
            return patchEmbedding(reshaped).reshaped(-1, embedDimensions)
        }
    }

    fileprivate final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        private let numHeads: Int
        private let headDim: Int
        private let scale: Float

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            numHeads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, rotaryPositionEmbedding: MLXArray?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            // The QKV split is contiguous per projection, not head-interleaved.
            let fused = qkv(x).reshaped(batch, length, 3, numHeads, headDim)
                .transposed(2, 0, 3, 1, 4)
            var queries = fused[0]
            var keys = fused[1]
            let values = fused[2]

            if let rotaryPositionEmbedding {
                (queries, keys) = applyRotaryPositionEmbedding(
                    queries: queries, keys: keys, freqs: rotaryPositionEmbedding)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask.map { .array($0) } ?? .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return proj(output)
        }
    }

    /// `hidden_act` is "gelu": the exact erf form, not the tanh approximation that the
    /// shared GELUMLP applies.
    fileprivate final class SiglipMLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        private let activation = GELU()

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    fileprivate final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: UnaryLayer
        @ModuleInfo(key: "layer_norm2") var layerNorm2: UnaryLayer
        @ModuleInfo var mlp: SiglipMLP

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = Self.norm(config)
            _layerNorm2.wrappedValue = Self.norm(config)
            _mlp.wrappedValue = SiglipMLP(config)
        }

        private static func norm(_ config: MageVLConfiguration.VisionConfiguration) -> UnaryLayer {
            config.layerNormType == "rms_norm"
                ? RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
                : LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, rotaryPositionEmbedding: MLXArray?,
        )
            -> MLXArray
        {
            let attended =
                x
                + attention(
                    layerNorm1(x), mask: mask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding)
            return attended + mlp(layerNorm2(attended))
        }
    }

    fileprivate final class PatchMerger: Module {
        @ModuleInfo(key: "ln_q") var layerNormQ: LayerNorm
        // nn.Sequential upstream, so the activation occupies index 1 and the weights are
        // named mlp.0 / mlp.2. Keep it in the list or the second Linear loads as mlp.1.
        @ModuleInfo var mlp: [UnaryLayer]

        private let mergedSize: Int

        init(
            dimensions: Int,
            contextDimensions: Int,
            spatialMergeSize: Int,
            layerNormEps: Float,
        ) {
            mergedSize = contextDimensions * spatialMergeSize * spatialMergeSize
            _layerNormQ.wrappedValue = LayerNorm(
                dimensions: contextDimensions, eps: layerNormEps)
            _mlp.wrappedValue = [
                Linear(mergedSize, mergedSize),
                GELU(),
                Linear(mergedSize, dimensions),
            ]
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = layerNormQ(x).reshaped(-1, mergedSize)
            for layer in mlp {
                hidden = layer(hidden)
            }
            return hidden
        }
    }

    fileprivate final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "layernorm_pre") var layerNormPre: UnaryLayer
        @ModuleInfo var encoder: Encoder
        @ModuleInfo var merger: PatchMerger

        let config: MageVLConfiguration.VisionConfiguration

        fileprivate final class Encoder: Module {
            let layers: [EncoderLayer]

            init(_ config: MageVLConfiguration.VisionConfiguration) {
                layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            }
        }

        private let rotary: RotaryEmbedding

        init(_ config: MageVLConfiguration.VisionConfiguration) {
            self.config = config
            _embeddings.wrappedValue = Embeddings(config)
            _layerNormPre.wrappedValue =
                config.layerNormType == "rms_norm"
                ? RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
                : LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _encoder.wrappedValue = Encoder(config)
            rotary = RotaryEmbedding(config)
            _merger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize,
                contextDimensions: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize,
                layerNormEps: config.layerNormEps,
            )
        }

        func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> MLXArray {
            var hidden = embeddings(pixels)
            if hidden.ndim == 2 {
                hidden = hidden.expandedDimensions(axis: 0)
            }
            let totalPatches = hidden.dim(1)

            let positions = patchPositions(grids: grids, mergeSize: config.spatialMergeSize)
            var freqs = rotary(patchPositions: positions)
            freqs = concatenated([freqs, freqs], axis: -1)
            if freqs.ndim == 2 {
                freqs = freqs.expandedDimensions(axis: 0)
            }

            hidden = layerNormPre(hidden)

            let boundaries = windowBoundaries(
                grids: grids, totalPatches: totalPatches,
                frameWindow: config.frameWindowsSize)
            let mask = blockDiagonalMask(boundaries: boundaries, dtype: hidden.dtype)

            for layer in encoder.layers {
                hidden = layer(hidden, mask: mask, rotaryPositionEmbedding: freqs)
            }

            return merger(hidden)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("patch_embedding.weight"), value.ndim == 4,
                    value.dim(1) == config.channels
                {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Language

private enum MageVLLanguage {
    fileprivate final class Attention: Module {
        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        private let heads: Int
        private let kvHeads: Int
        private let headDim: Int
        private let scale: Float

        init(_ config: MageVLConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: false, base: config.ropeTheta)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = qNorm(wq(x).reshaped(batch, length, heads, headDim))
                .transposed(0, 2, 1, 3)
            var keys = kNorm(wk(x).reshaped(batch, length, kvHeads, headDim))
                .transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

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
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: MageVLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let attended = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return attended + mlp(postAttentionLayerNorm(attended))
        }
    }

    fileprivate final class Qwen3Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm

        init(_ config: MageVLConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbedding {
                hidden = inputEmbedding
            } else if let inputs {
                hidden = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: hidden, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen3Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let kvHeads: [Int]

        init(_ config: MageVLConfiguration.TextConfiguration) {
            model = Qwen3Model(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            let logits = lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
            return LMOutput(logits: logits, state: state)
        }
    }
}

// MARK: - Model

public final class MageVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: MageVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: MageVLLanguage.LanguageModel

    public let config: MageVLConfiguration

    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: MageVLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = MageVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = MageVLLanguage.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let pixels = input.image?.pixels, let grids = input.image?.frames else {
            return embeddings
        }

        let features = visionTower(pixels.asType(embeddings.dtype), grids: grids)
            .asType(embeddings.dtype)

        let isMedia = logicalOr(
            inputIDs .== config.imageTokenID, inputIDs .== config.videoTokenID)
        let placeholders = isMedia.sum().item(Int.self)
        guard placeholders == features.dim(0), placeholders > 0 else {
            return embeddings
        }

        let positions = (cumsum(isMedia.reshaped(-1).asType(.int32)) - 1)
            .reshaped(isMedia.shape)
        let gathered = features[clip(positions, min: 0, max: features.dim(0) - 1)]
        return which(isMedia[.ellipsis, .newAxis], gathered, embeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache],
        state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let embeddings = inputEmbeddings(input)
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

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if key.hasPrefix("model.language_model.") {
                name = "language_model.model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("model.visual.") {
                name = "vision_tower." + key.dropFirst("model.visual.".count)
            } else if key.hasPrefix("visual.") {
                name = "vision_tower." + key.dropFirst("visual.".count)
            } else if key.hasPrefix("lm_head.") {
                name = "language_model.lm_head." + key.dropFirst("lm_head.".count)
            } else if key.hasPrefix("model.") {
                name = "language_model.model." + key.dropFirst("model.".count)
            }
            renamed[name] = value
        }

        let visionPrefix = "vision_tower."
        var vision: [String: MLXArray] = [:]
        for (key, value) in renamed where key.hasPrefix(visionPrefix) {
            vision[String(key.dropFirst(visionPrefix.count))] = value
        }
        for (key, value) in visionTower.sanitize(weights: vision) {
            renamed[visionPrefix + key] = value
        }
        return renamed
    }
}
