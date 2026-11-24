import Foundation
import MLX
import MLXNN

struct LLaDAImageSchedule {
    let uniform: Bool, stochastic: Bool
    let shift: Float
    init(_ c: DiffusionConfiguration) throws {
        guard !c.bool("use_dynamic_shifting", false), !c.bool("invert_sigmas", false) else {
            throw ModelExecutionError.invalidOptions("Unsupported LLaDA scheduler configuration")
        }
        uniform = c.bool("use_uniform_sigmas", false)
        stochastic = c.bool("stochastic_sampling", uniform)
        shift = c.float("shift", uniform ? 3 : 1)
        guard shift.isFinite, shift > 0 else {
            throw ModelExecutionError.invalidOptions("LLaDA scheduler shift must be positive")
        }
    }
    func sigmas(_ steps: Int) -> [Float] {
        (0 ... steps).map { i in
            let sigma: Double =
                i == steps
                ? 0
                : uniform
                    ? 1 - Double(i) / Double(steps)
                    : 1
                        - pow(
                            1 - pow(1 - pow(0.001 + 0.999 * Double(i) / Double(steps), 1.17), 0.8),
                            1.1)
            let s = Float(sigma)
            return shift * s / (1 + (shift - 1) * s)
        }
    }
}

func lladaImageRestoreFP8(_ input: [String: MLXArray], config: DiffusionConfiguration) throws
    -> [String: MLXArray]
{
    var weights = input
    let expert = config.child("llada_fp8_experts")
    let block = config.child("quantization_config")
    if expert.bool("enabled", false)
        && (expert.string("format") != "e4m3fn"
            || expert.string("weight_granularity") != "per_expert_2d_block"
            || expert.ints("weight_block_size", []) != [128, 128])
    {
        throw ModelExecutionError.invalidOptions("Unsupported LLaDA FP8 expert configuration")
    }
    for key in input.keys {
        let isBlock = key.hasSuffix(".weight_scale_inv")
        let isExpert = key.contains(".mlp.experts.") && key.hasSuffix("_scale")
        guard isBlock || isExpert else { continue }
        guard
            isBlock
                ? block.string("quant_method") == "fp8"
                    && block.ints("weight_block_size", []) == [128, 128]
                : expert.bool("enabled", false)
        else {
            throw ModelExecutionError.invalidOptions(
                "FP8 scales require matching quantization metadata")
        }
        let name = String(key.dropLast(isBlock ? 10 : 6))
        guard var value = weights.removeValue(forKey: name),
            var scale = weights.removeValue(forKey: key)
        else { throw ModelExecutionError.invalidOptions("Missing FP8 tensor for \(key)") }
        let shape = value.shape
        if value.ndim == 3 {
            guard shape[1] % 128 == 0,
                scale.shape == [shape[0], shape[1] / 128, (shape[2] + 127) / 128]
            else { throw ModelExecutionError.invalidOptions("Invalid FP8 expert scales") }
            value = value.reshaped([-1, shape[2]])
            scale = scale.reshaped([-1, scale.dim(-1)])
        }
        guard value.ndim == 2,
            scale.shape == [(value.dim(0) + 127) / 128, (value.dim(1) + 127) / 128]
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid FP8 block scale dimensions for \(name)")
        }
        let restored = MiMoV2.dequantizeFP8Block(value, scale: scale).reshaped(shape).asType(
            .bfloat16)
        eval(restored)
        weights[name] = restored
    }
    guard !weights.values.contains(where: { $0.dtype == .uint8 }) else {
        throw ModelExecutionError.invalidOptions("FP8 tensors are missing block scales")
    }
    return weights
}

func lladaImageVQTokens(
    prompt: String, height: Int, width: Int, tokenizer: any Tokenizer,
    encoder: LLaDAImageTextEncoder, head: Linear, codebookSize: Int
) throws -> MLXArray {
    let scale = max(Double(max(height, width)) / 512, 1)
    let rows = Int(Double(height) / scale) / 16
    let columns = Int(Double(width) / scale) / 16
    guard min(rows, columns) > 0 else {
        throw ModelExecutionError.invalidOptions("Image aspect ratio leaves no VQ tokens")
    }
    let system =
        "<role>SYSTEM</role> You are a text-to-image generation assistant. <role>HUMAN</role>"
    func encode(_ s: String) -> [Int] { tokenizer.encode(text: s, addSpecialTokens: true) }
    let image = Array(
        encode("<|image|><|reserved_token_\(rows)|><|reserved_token_\(columns)|><boi><|/image|>")
            .dropLast())
    let conditional = encode(system + prompt + "<role>ASSISTANT</role>") + image
    let unconditional = encode(system + "<uncondition><role>ASSISTANT</role>") + image
    return try lladaImageSampleVQ(
        conditional: conditional, unconditional: unconditional,
        count: rows * columns, codebookSize: codebookSize
    ) { input, mask, positions in
        let hidden = encoder.encode(input, mask: mask, positions: positions)
        return head(hidden[0..., (hidden.dim(1) - 32)..., 0...])
    }
}

/// Block diffusion over image-codebook tokens; prompt preparation and model execution
/// remain separate so prefix masks and token commitment can be verified independently.
func lladaImageSampleVQ(
    conditional: [Int], unconditional: [Int], count: Int, codebookSize: Int,
    imageTokenOffset: Int = 157184, maskToken: Int = 156895, endToken: Int = 156902,
    predict: (MLXArray, MLXArray, MLXArray) throws -> MLXArray
) throws -> MLXArray {
    guard count > 0, codebookSize > 0 else {
        throw ModelExecutionError.invalidOptions("Invalid image token count or codebook size")
    }
    let promptLength = conditional.count
    let pad = promptLength - unconditional.count
    let blockLength = 32
    let steps = min(8, count)
    guard pad >= 0 else {
        throw ModelExecutionError.invalidOptions(
            "The unconditional VQ prompt is longer than the prompt")
    }
    let blocks = (promptLength + count + 31) / 32
    let length = blocks * 32
    let positions = arange(length)
    let ids = positions.floorDivide(32)
    let attention = (ids.expandedDimensions(axis: 1) .>= ids).reshaped([1, 1, length, length])
    let uncondAttention = attention .&& (positions .>= pad).reshaped([1, 1, 1, length])
    let uncondPositions = maximum(positions - pad, 0)
    var tokens = Array(repeating: maskToken, count: length)
    tokens.replaceSubrange(0 ..< promptLength, with: conditional)
    let prefix = Array(repeating: maskToken, count: pad) + unconditional
    for block in (promptLength / blockLength) ..< blocks {
        let end = (block + 1) * blockLength
        let start = end - blockLength
        for step in 0 ..< steps {
            try Task.checkCancellation()
            let active = (start ..< end).filter { tokens[$0] == maskToken }
            if active.isEmpty { break }
            let current = Array(tokens[..<end])
            let unc = prefix + Array(tokens[promptLength ..< end])
            let input = MLXArray(current + unc, [2, end])
            let pos = stacked([positions[..<end], uncondPositions[..<end]])
            let mask = concatenated(
                [
                    attention[0..., 0..., ..<end, ..<end],
                    uncondAttention[0..., 0..., ..<end, ..<end],
                ], axis: 0)
            let logits = try predict(input, mask, pos)
            guard logits.ndim == 3, logits.dim(0) == 2, logits.dim(1) == blockLength else {
                throw ModelExecutionError.invalidOptions(
                    "Image token predictions require conditional/unconditional blocks")
            }
            let guided = logits[1] + 2 * (logits[0] - logits[1])
            let predicted = argMax(guided, axis: -1)
            let probability = takeAlong(
                softmax(guided, axis: -1), predicted.expandedDimensions(axis: -1), axis: -1
            ).flattened()
            let confidence = probability.asArray(Float.self)
            let values = predicted.asArray(Int.self)
            let transfer = blockLength / steps + (step < blockLength % steps ? 1 : 0)
            var selected = active.filter { confidence[$0 - start] > 0.95 }
            if selected.count < transfer {
                selected = Array(
                    active.sorted { confidence[$0 - start] > confidence[$1 - start] }.prefix(
                        min(transfer, active.count)))
            }
            for index in selected { tokens[index] = values[index - start] }
            if let stop = tokens[promptLength ..< end].firstIndex(of: endToken),
                !tokens[promptLength ..< stop].contains(maskToken)
            {
                guard stop - promptLength >= count else {
                    throw ModelExecutionError.invalidOptions(
                        "The text model ended before producing enough image tokens")
                }
                break
            }
        }
    }
    let result = tokens[promptLength ..< (promptLength + count)].map { $0 - imageTokenOffset }
    guard result.allSatisfy({ $0 >= 0 && $0 < codebookSize }) else {
        throw ModelExecutionError.invalidOptions(
            "The text model generated tokens outside the SigVQ codebook")
    }
    return MLXArray(result, [1, count])
}
