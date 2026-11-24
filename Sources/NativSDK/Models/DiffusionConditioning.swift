import CoreImage
import Foundation
import MLX

final class DiffusionVisionConditioner {
    let model: Qwen3VL
    let processorConfig: Qwen3VLProcessorConfiguration
    init(directory: URL, processorDirectory: URL) throws {
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        model = Qwen3VL(config)
        var checkpoint = try DiffusionCheckpoint(directory: directory)
        checkpoint.weights = model.sanitize(weights: checkpoint.weights)
        try checkpoint.apply(to: model)
        processorConfig = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(contentsOf: processorDirectory.appending(path: "preprocessor_config.json")))
    }
    func encode(
        _ prompt: String, references: [MLXArray], tokenizer: any Tokenizer
    ) throws -> (MLXArray, [Bool]) {
        let editing = !references.isEmpty
        let system = "Comprehend and analyze the provided prompt."
        let prefix = "<|im_start|>system\n\(system)<|im_end|>\n"
        let placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        let imagePrefix = references.indices.map {
            "<image\($0 + 1)>\(placeholder)"
        }.joined(separator: " ")
        let formatted =
            prefix + "<|im_start|>user\n\(imagePrefix)\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        var tokens = tokenizer.encode(text: formatted, addSpecialTokens: true)
        let drop = tokenizer.encode(text: prefix, addSpecialTokens: false).count
        if !editing { tokens = Array(tokens.prefix(drop + 1024)) }
        var pixels = [MLXArray]()
        var grids = [THW]()
        let processor = Qwen3VLProcessor(processorConfig, tokenizer: tokenizer)
        for reference in references {
            var rgb = (reference[0] + 1) / 2
            if rgb.dim(0) == 4 { rgb = rgb[..<3] * rgb[3 ..< 4] + (1 - rgb[3 ..< 4]) }
            let image = try UserInput.Image.array((clip(rgb, min: 0, max: 1) * 255).asType(.uint8))
                .asCIImage()
            var processing = UserInput.Processing()
            processing.minPixels = processorConfig.minPixels
            processing.maxPixels = processorConfig.maxPixels
            let (patches, grid) = try processor.preprocess(images: [image], processing: processing)
            pixels.append(patches)
            grids.append(grid)
        }
        if editing {
            tokens = try QwenVL.replacePaddingTokens(
                in: tokens, frames: grids, paddingToken: "<|image_pad|>",
                mergeSize: processorConfig.mergeSize, tokenizer: tokenizer)
        }
        guard tokens.count > drop else {
            throw ModelExecutionError.invalidOptions(
                "The image prompt is empty after template trimming")
        }
        let input = LMInput(
            text: .init(tokens: MLXArray(tokens).reshaped([1, -1])),
            image: editing ? .init(pixels: concatenated(pixels), frames: grids) : nil)
        let h = try model.imageConditioning(input, applyFinalNorm: false)[
            0..., drop...]
        eval(h)
        return (h, tokens.dropFirst(drop).map { $0 == model.config.imageTokenIndex })
    }
}

func diffusionResizeRGB(_ pixels: MLXArray, height: Int, width: Int, preserveAlpha: Bool = false)
    throws -> MLXArray
{
    guard pixels.ndim == 4, pixels.dim(0) == 1, [3, 4].contains(pixels.dim(1)), height > 0,
        width > 0
    else {
        throw ModelExecutionError.invalidOptions(
            "Expected one normalized RGB or RGBA reference image")
    }
    var input = pixels
    if preserveAlpha && input.dim(1) == 3 {
        input = concatenated([input, MLXArray.ones([1, 1, input.dim(2), input.dim(3)])], axis: 1)
    }
    if !preserveAlpha { input = input[0..., ..<3, 0..., 0...] }
    if input.dim(2) == height && input.dim(3) == width { return input }
    // Resize planes independently so straight alpha is preserved in RGBA model inputs.
    var planes = [MLXArray]()
    for channel in 0 ..< input.dim(1) {
        let gray = broadcast(
            input[0, channel].expandedDimensions(axis: 0), to: [3, input.dim(2), input.dim(3)])
        let image = try UserInput.Image.array(
            (clip((gray + 1) * 127.5, min: 0, max: 255)).asType(.uint8)
        ).asCIImage()
        let resized = image.clampedToExtent().applyingFilter(
            "CILanczosScaleTransform",
            parameters: [
                kCIInputScaleKey: Float(height) / Float(input.dim(2)),
                kCIInputAspectRatioKey: (Float(width) / Float(input.dim(3)))
                    / (Float(height) / Float(input.dim(2))),
            ]
        ).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        let plane =
            MediaProcessing.asMLXArray(resized, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))[
                0, 0
            ].asType(.float32) * 2 - 1
        planes.append(plane)
    }
    return stacked(planes).expandedDimensions(axis: 0)
}
