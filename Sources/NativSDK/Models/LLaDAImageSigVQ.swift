import Foundation
import MLX
import MLXNN

final class LLaDASigVQAttention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    let heads: Int
    init(_ c: DiffusionConfiguration) {
        let d = c.int("hidden_size", 1152)
        heads = c.int("num_attention_heads", 16)
        qkv = Linear(d, 3 * d, bias: c.bool("attention_bias", true))
        proj = Linear(d, d, bias: c.bool("attention_bias", true))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(qkv(x), parts: 3, axis: -1)
        return proj(lladaImageAttention(parts[0], parts[1], parts[2], heads: heads))
    }
}
final class LLaDASigVQBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var attn: LLaDASigVQAttention
    @ModuleInfo var mlp: LLaDAImageMLP
    init(_ c: DiffusionConfiguration) {
        let d = c.int("hidden_size", 1152)
        let eps = c.float("norm_eps", 1e-6)
        norm1 = LayerNorm(dimensions: d, eps: eps)
        norm2 = LayerNorm(dimensions: d, eps: eps)
        attn = LLaDASigVQAttention(c)
        mlp = LLaDAImageMLP(d, c.int("intermediate_size", 4304), exact: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x))
        return h + mlp(norm2(h))
    }
}
final class LLaDASigVQ: Module {
    @ModuleInfo var patch: Conv2d?
    @ModuleInfo var positions: Embedding?
    @ModuleInfo var blocks: [LLaDASigVQBlock]
    @ModuleInfo var quantConv: Conv2d?
    @ModuleInfo var codebook: Embedding?
    @ModuleInfo(key: "prior_token_embedding") var embedding: Embedding
    @ModuleInfo(key: "prior_projector") var projector: [Linear]
    let patchSize: Int
    init(_ c: DiffusionConfiguration, includeEncoder: Bool) {
        patchSize = c.int("patch_size", 16)
        let d = c.int("hidden_size", 1152)
        let size = c.int("image_size", 512) / patchSize
        let codes = c.int("codebook_size", 16384)
        let codeDim = c.int("codebook_embed_dim", 32)
        let semantic = c.int("semantic_embed_dim", 4096)
        patch =
            includeEncoder
            ? Conv2d(
                inputChannels: c.int("in_channels", 3), outputChannels: d,
                kernelSize: .init(patchSize), stride: .init(patchSize)) : nil
        positions = includeEncoder ? Embedding(embeddingCount: size * size, dimensions: d) : nil
        blocks =
            includeEncoder
            ? (0 ..< c.int("num_hidden_layers", 27)).map { _ in LLaDASigVQBlock(c) } : []
        quantConv =
            includeEncoder ? Conv2d(inputChannels: d, outputChannels: codeDim, kernelSize: 1) : nil
        codebook = includeEncoder ? Embedding(embeddingCount: codes, dimensions: codeDim) : nil
        _embedding.wrappedValue = Embedding(embeddingCount: codes, dimensions: semantic)
        _projector.wrappedValue = [Linear(semantic, semantic), Linear(semantic, semantic)]
        super.init()
    }
    static func resizePositions(_ embedding: MLXArray, height: Int, width: Int) -> MLXArray {
        let size = Int(sqrt(Double(embedding.dim(0))))
        let grid = embedding.asType(.float32)
        let y = clip(
            (arange(height).asType(.float32) + 0.5) * Float(size) / Float(height) - 0.5, min: 0,
            max: Float(size - 1))
        let x = clip(
            (arange(width).asType(.float32) + 0.5) * Float(size) / Float(width) - 0.5, min: 0,
            max: Float(size - 1))
        let y0 = y.asType(.int32)
        let x0 = x.asType(.int32)
        let y1 = minimum(y0 + 1, size - 1)
        let x1 = minimum(x0 + 1, size - 1)
        func gather(_ rows: MLXArray, _ cols: MLXArray) -> MLXArray {
            take(grid, (rows.expandedDimensions(axis: 1) * size + cols).flattened(), axis: 0)
                .reshaped([height, width, -1])
        }
        let dy = (y - y0).reshaped([height, 1, 1])
        let dx = (x - x0).reshaped([1, width, 1])
        let top = gather(y0, x0) * (1 - dx) + gather(y0, x1) * dx
        let bottom = gather(y1, x0) * (1 - dx) + gather(y1, x1) * dx
        return ((1 - dy) * top + dy * bottom).reshaped([1, height * width, -1])
    }
    func encode(_ pixels: MLXArray) throws -> MLXArray {
        guard let patch, let positions, let quantConv, let codebook, pixels.ndim == 4,
            pixels.dim(-1) == 3,
            pixels.dim(1) % patchSize == 0, pixels.dim(2) % patchSize == 0
        else {
            throw ModelExecutionError.invalidOptions(
                "SigVQ requires its encoder and RGB dimensions divisible by the patch size")
        }
        var h = patch(pixels)
        let b = h.dim(0)
        let height = h.dim(1)
        let width = h.dim(2)
        let d = h.dim(3)
        h =
            h.reshaped([b, height * width, d])
            + Self.resizePositions(positions.weight, height: height, width: width).asType(h.dtype)
        for block in blocks { h = block(h) }
        h = quantConv(h.reshaped([b, height, width, d])).reshaped([b, height * width, -1])
        func normalized(_ x: MLXArray) -> MLXArray {
            x
                / maximum(
                    sqrt(sum(square(x.asType(.float32)), axis: -1, keepDims: true)).asType(x.dtype),
                    1e-12)
        }
        h = normalized(h)
        let c = normalized(codebook.weight)
        return argMin(
            sum(square(h), axis: -1, keepDims: true) + sum(square(c), axis: -1) - 2
                * matmul(h, c.T), axis: -1)
    }
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        projector[1](silu(projector[0](embedding(ids))))
    }
    static func sanitize(_ weights: [String: MLXArray], includeEncoder: Bool) -> [String: MLXArray]
    {
        var result = [String: MLXArray]()
        for (key, value) in weights where includeEncoder || key.hasPrefix("prior_") {
            let key = key.replacingOccurrences(of: "visual.patch_embed.proj.", with: "patch.")
                .replacingOccurrences(
                    of: "visual.embeddings.position_embedding.", with: "positions."
                )
                .replacingOccurrences(of: "visual.blocks.", with: "blocks.").replacingOccurrences(
                    of: "vqmodel.quant_conv.", with: "quantConv."
                )
                .replacingOccurrences(of: "vqmodel.quantize.embedding.", with: "codebook.")
                .replacingOccurrences(of: "prior_projector.net.0.proj.", with: "prior_projector.0.")
                .replacingOccurrences(of: "prior_projector.net.2.", with: "prior_projector.1.")
            result[key] = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
        }
        return result
    }
}
