import Foundation
import MLX
import MLXNN

/// One image's placement in the prompt, as the processor lays it out.
///
/// `types` runs one code per reserved token: the start marker, the image rows in
/// reading order, a newline at the end of each row, and the end marker.
public struct DeepSeekV41ImageSpan {
    public enum Code: Int32 {
        case start = 0
        case image = 1
        case newline = 2
        case end = 3
    }

    public let start: Int
    public let patches: MLXArray
    public let height: Int
    public let width: Int
    public let types: [Code]

    public init(
        start: Int, patches: MLXArray, height: Int, width: Int, types: [Code]
    ) {
        self.start = start
        self.patches = patches
        self.height = height
        self.width = width
        self.types = types
    }
}

public final class DeepSeekV41: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") var languageModel: DeepSeekV41Language.LanguageModel
    @ModuleInfo(key: "vision") var vision: DeepSeekV41Vision.ViT
    @ModuleInfo(key: "aligner") var aligner: DeepSeekV41Vision.Aligner
    @ParameterInfo(key: "image_start") var imageStart: MLXArray
    @ParameterInfo(key: "image_end") var imageEnd: MLXArray
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray

    public let config: DeepSeekV41Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: 1, count: config.hiddenLayers) }

    public init(_ config: DeepSeekV41Configuration) {
        self.config = config
        _languageModel.wrappedValue = DeepSeekV41Language.LanguageModel(config)
        _vision.wrappedValue = DeepSeekV41Vision.ViT(config)
        _aligner.wrappedValue = DeepSeekV41Vision.Aligner(config)
        _imageStart.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _imageEnd.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _imageNewline.wrappedValue = MLXArray.zeros([config.hiddenSize])
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache?.first as? DeepSeekV41Cache)
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        languageModel.newCache()
    }

    public func encodeImage(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        aligner(vision(patches, height: height, width: width), height: height, width: width)
    }

    /// Overwrite each image's token span with its aligner rows and span markers.
    ///
    /// Image slots take the rows in reading order; the three delimiters take the learned
    /// span embeddings.
    public func mergeImageEmbeddings(
        _ images: [[DeepSeekV41ImageSpan]], into h: MLXArray
    ) -> MLXArray {
        let dtype = h.dtype
        let marks: [DeepSeekV41ImageSpan.Code: MLXArray] = [
            .start: imageStart, .end: imageEnd, .newline: imageNewline,
        ]
        var rows: [MLXArray] = []
        for (batch, sample) in images.enumerated() {
            var row = h[batch]
            for image in sample {
                let codes = MLXArray(image.types.map(\.rawValue))
                let end = image.start + image.types.count
                var span = row[image.start ..< end]
                let isImage = codes .== DeepSeekV41ImageSpan.Code.image.rawValue
                let order = cumsum(isImage.asType(.int32), axis: 0) - 1
                let embeddings = encodeImage(
                    image.patches, height: image.height, width: image.width)
                let gathered = take(
                    embeddings, clip(order, min: 0, max: embeddings.dim(0) - 1), axis: 0)
                span = MLX.where(
                    isImage.expandedDimensions(axis: 1), gathered.asType(dtype), span)
                for (code, mark) in marks {
                    span = MLX.where(
                        (codes .== code.rawValue).expandedDimensions(axis: 1),
                        mark.asType(dtype), span)
                }
                row = concatenated([row[..<image.start], span, row[end...]], axis: 0)
            }
            rows.append(row)
        }
        return stacked(rows, axis: 0).asType(dtype)
    }

    public func inputEmbeddings(
        inputIds: MLXArray, images: [[DeepSeekV41ImageSpan]]? = nil
    ) -> MLXArray {
        let embeddings = languageModel.embedTokens(inputIds)
        guard let images, inputIds.dim(1) != 1 else { return embeddings }
        return mergeImageEmbeddings(images, into: embeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        var embeddings: MLXArray?
        if let image = input.image {
            guard tokens.dim(0) == 1, let frames = image.frames, !frames.isEmpty else {
                throw VLMError.processing(
                    "DeepSeek image prompts require one batch row and a patch grid per image")
            }
            let ids = tokens.asArray(Int32.self)
            var spans: [DeepSeekV41ImageSpan] = []
            var position = 0
            var patchOffset = 0
            let ratio = config.visionConfiguration.downsampleRatio
            for grid in frames {
                while position < ids.count && ids[position] != config.imageTokenId { position += 1 }
                let types = DeepSeekV41Processor.tokenTypes(
                    height: (grid.h + ratio - 1) / ratio, width: (grid.w + ratio - 1) / ratio)
                guard position + types.count <= ids.count,
                    ids[position ..< position + types.count].allSatisfy({
                        $0 == config.imageTokenId
                    }),
                    patchOffset + grid.product <= image.pixels.dim(0)
                else {
                    throw VLMError.processing("DeepSeek image span and patch counts do not match")
                }
                spans.append(
                    .init(
                        start: position,
                        patches: image.pixels[patchOffset ..< patchOffset + grid.product],
                        height: grid.h, width: grid.w, types: types))
                position += types.count
                patchOffset += grid.product
            }
            guard patchOffset == image.pixels.dim(0),
                !ids[position...].contains(Int32(config.imageTokenId))
            else {
                throw VLMError.processing("Unused DeepSeek image patches or placeholders")
            }
            embeddings = inputEmbeddings(inputIds: tokens, images: [spans])
        }
        let entry = cache.first as? DeepSeekV41Cache
        let total = tokens.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel(
                tokens[0..., range], inputsEmbeds: embeddings.map { $0[0..., range] }, cache: entry)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)
        return .logits(
            .init(
                logits: languageModel(
                    tokens[0..., processed...],
                    inputsEmbeds: embeddings.map { $0[0..., processed...] }, cache: entry)))
    }

    /// Repack source bytes without rounding, expanding block scales over rows.
    static func packSourceWeight(_ weight: MLXArray, scales: MLXArray) throws
        -> (packed: MLXArray, scales: MLXArray, mode: QuantizationMode)
    {
        guard weight.ndim == 2, scales.ndim == 2, scales.dtype == .uint8 else {
            throw DeepSeekV41Error.unexpectedSourceQuantization(
                "expected a matrix of FP8/FP4 bytes and E8M0 scales")
        }
        guard weight.dtype == .uint8 || weight.dtype == .int8 else {
            throw DeepSeekV41Error.unexpectedSourceQuantization(
                "expected FP8/FP4 bytes, got \(weight.dtype)")
        }
        let (rows, columns) = (weight.dim(0), weight.dim(1))
        var scales = scales
        let mode: QuantizationMode
        if columns % 16 == 0, scales.dim(0) == rows, scales.dim(1) == columns / 16 {
            mode = .mxfp4
        } else if columns % 32 == 0, scales.dim(-1) == columns / 32 {
            mode = .mxfp8
            if scales.dim(0) == (rows + 31) / 32 {
                scales = repeated(scales, count: 32, axis: 0)[0 ..< rows]
            } else if scales.dim(0) != rows {
                throw DeepSeekV41Error.unexpectedSourceQuantization(
                    "invalid FP8 scale rows \(scales.shape) for \(weight.shape)")
            }
        } else {
            throw DeepSeekV41Error.unexpectedSourceQuantization(
                "invalid scale shape \(scales.shape) for \(weight.shape)")
        }
        return (weight.view(dtype: .uint32), scales, mode)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transform(_ key: String) -> String {
            if key.hasPrefix("language_model.") { return key }
            if key.hasPrefix("model.") || key.hasPrefix("lm_head.") {
                return "language_model.\(key)"
            }
            for (prefix, replacement) in [
                ("embed.", "language_model.embed_tokens."),
                ("head.", "language_model.head."),
                ("norm.", "language_model.norm."),
            ] where key.hasPrefix(prefix) {
                return replacement + key.dropFirst(prefix.count)
            }
            if key.hasPrefix("layers.") { return "language_model.\(key)" }
            return key
        }

        var out: [String: MLXArray] = [:]
        for (key, value) in weights where !key.hasPrefix("mtp.") {
            out[transform(key)] = value
        }

        for key in out.keys where key.hasSuffix(".scale") {
            let weightKey = String(key.dropLast(".scale".count)) + ".weight"
            guard let packedSource = out[weightKey], let scale = out.removeValue(forKey: key)
            else { continue }
            do {
                let (packed, scales, mode) = try DeepSeekV41.packSourceWeight(
                    packedSource, scales: scale)
                if weightKey.hasSuffix(".engram.embed.weight") {
                    out[weightKey] = packed
                    out[key + "s"] = scales
                    continue
                }
                out[weightKey] = dequantized(
                    packed, scales: scales, biases: nil, groupSize: 32,
                    bits: mode == .mxfp4 ? 4 : 8, mode: mode, dtype: .bfloat16)
            } catch {
                preconditionFailure("\(weightKey): \(error.localizedDescription)")
            }
        }

        for layer in 0 ..< config.hiddenLayers {
            do {
                out = try DeepSeekV41Language.sanitizeMoEWeights(
                    out, ffnPrefix: "language_model.layers.\(layer).ffn",
                    experts: config.nRoutedExperts)
            } catch {
                preconditionFailure(error.localizedDescription)
            }
            let gate = "language_model.layers.\(layer).ffn.gate.weight"
            if let value = out[gate] { out[gate] = value.asType(.float32) }

            let prefix = "language_model.layers.\(layer).attn.wo_a"
            for name in ["weight", "scales", "biases"] {
                let key = "\(prefix).\(name)"
                if let value = out[key], value.ndim == 2 {
                    out[key] = value.reshaped(config.oGroups, config.oLoraRank, -1)
                }
            }
        }

        let headWeight = "language_model.head.weight"
        if let packed = out[headWeight], let scales = out["language_model.head.scales"],
            let biases = out["language_model.head.biases"]
        {
            let inputDimensions = scales.dim(-1) * 64
            let bits = 32 * packed.dim(-1) / inputDimensions
            out[headWeight] = dequantized(
                packed, scales: scales, biases: biases, groupSize: 64, bits: bits)
            out["language_model.head.scales"] = nil
            out["language_model.head.biases"] = nil
        }
        if let value = out[headWeight] { out[headWeight] = value.asType(.float32) }
        return out
    }
}

public enum DeepSeekV41Error: LocalizedError {
    case unexpectedSourceQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .unexpectedSourceQuantization(let detail):
            "unexpected source quantization: \(detail)"
        }
    }
}

extension DeepSeekV41: LoRAModel {
    public var loraLayers: [Module] { languageModel.layers }
}
