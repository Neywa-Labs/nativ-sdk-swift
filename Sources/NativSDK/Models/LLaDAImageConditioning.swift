import Foundation
import MLX
import MLXNN

func lladaImageAttention(
    _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, heads: Int, mask: MLXArray? = nil
) -> MLXArray {
    let b = q.dim(0)
    let n = q.dim(1)
    let d = q.dim(2)
    let width = d / heads
    func splitHeads(_ x: MLXArray) -> MLXArray {
        x.reshaped([b, -1, heads, width]).transposed(0, 2, 1, 3)
    }
    return MLXFast.scaledDotProductAttention(
        queries: splitHeads(q), keys: splitHeads(k), values: splitHeads(v),
        scale: pow(Float(width), -0.5), mask: mask.map { .array($0) } ?? .none
    ).transposed(0, 2, 1, 3).reshaped([b, n, d])
}
final class LLaDAImageMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let exact: Bool
    init(_ dim: Int, _ intermediate: Int, exact: Bool = false) {
        self.exact = exact
        fc1 = Linear(dim, intermediate)
        fc2 = Linear(intermediate, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(exact ? gelu(fc1(x)) : geluApproximate(fc1(x)))
    }
}
final class LLaDAQueryAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var weight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var bias: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    init(_ c: DiffusionConfiguration) {
        let d = c.int("hidden_size", 4096)
        heads = c.int("num_attention_heads", 32)
        _weight.wrappedValue = .zeros([3 * d, d])
        _bias.wrappedValue = .zeros([3 * d])
        _output.wrappedValue = Linear(d, d)
        super.init()
    }
    func callAsFunction(_ queries: MLXArray, context: MLXArray, mask: MLXArray) -> MLXArray {
        let w = split(weight, parts: 3, axis: 0)
        let b = split(bias, parts: 3)
        let q = addMM(b[0], queries, w[0].T)
        let k = addMM(b[1], context, w[1].T)
        let v = addMM(b[2], context, w[2].T)
        return output(
            lladaImageAttention(
                q, k, v, heads: heads, mask: mask.reshaped([mask.dim(0), 1, 1, -1])))
    }
}
final class LLaDAQueryBlock: Module {
    @ModuleInfo(key: "norm_q") var queryNorm: LayerNorm
    @ModuleInfo(key: "norm_k") var keyNorm: LayerNorm
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn") var attention: LLaDAQueryAttention
    @ModuleInfo var mlp: LLaDAImageMLP
    init(_ c: DiffusionConfiguration) {
        let d = c.int("hidden_size", 4096)
        let eps = c.float("norm_eps", 1e-6)
        _queryNorm.wrappedValue = LayerNorm(dimensions: d, eps: eps, affine: false)
        _keyNorm.wrappedValue = LayerNorm(dimensions: d, eps: eps, affine: false)
        norm1 = LayerNorm(dimensions: d, eps: eps, affine: false)
        _attention.wrappedValue = LLaDAQueryAttention(c)
        mlp = LLaDAImageMLP(d, c.int("intermediate_size", 16384))
        super.init()
    }
    func callAsFunction(_ queries: MLXArray, context: MLXArray, mask: MLXArray) -> MLXArray {
        var q = queryNorm(queries)
        q = q + attention(q, context: keyNorm(context), mask: mask)
        q = norm1(q)
        return q + mlp(q)
    }
}
final class LLaDAQueryFormer: Module {
    @ParameterInfo(key: "meta_queries") var queries: MLXArray
    @ModuleInfo(key: "query_blocks") var blocks: [LLaDAQueryBlock]
    init(_ c: DiffusionConfiguration) {
        _queries.wrappedValue = .zeros([c.int("num_queries", 256), c.int("hidden_size", 4096)])
        _blocks.wrappedValue = (0 ..< c.int("num_hidden_layers", 2)).map { _ in LLaDAQueryBlock(c) }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var q = broadcast(queries, to: [x.dim(0)] + queries.shape)
        let mask = MLXArray.ones([x.dim(0), x.dim(1)], dtype: .bool)
        for block in blocks { q = block(q, context: x, mask: mask) }
        return q
    }
}
final class LLaDAProjectionAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int, norm: DiffusionRMSNorm
    init(_ c: DiffusionConfiguration) {
        let d = c.int("hidden_size", 4096)
        heads = c.int("num_attention_heads", 32)
        norm = DiffusionRMSNorm(c.float("norm_eps", 1e-6))
        _q.wrappedValue = Linear(d, d)
        _k.wrappedValue = Linear(d, d)
        _v.wrappedValue = Linear(d, d)
        _output.wrappedValue = Linear(d, d)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = [x.dim(0), x.dim(1), heads, -1]
        return output(
            lladaImageAttention(
                norm(q(x).reshaped(shape)).reshaped(x.shape),
                norm(k(x).reshaped(shape)).reshaped(x.shape), v(x), heads: heads))
    }
}
final class LLaDAProjectionBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: LLaDAProjectionAttention
    @ModuleInfo var mlp: LLaDAImageMLP
    let norm: DiffusionRMSNorm
    init(_ c: DiffusionConfiguration) {
        norm = DiffusionRMSNorm(c.float("norm_eps", 1e-6))
        _attention.wrappedValue = LLaDAProjectionAttention(c)
        mlp = LLaDAImageMLP(c.int("hidden_size", 4096), c.int("intermediate_size", 16384))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm(x))
        return h + mlp(norm(h))
    }
}
final class LLaDATextProjection: Module {
    @ModuleInfo var layers: [LLaDAProjectionBlock]
    @ModuleInfo var projector: Linear
    init(_ c: DiffusionConfiguration) {
        layers = (0 ..< c.int("num_hidden_layers", 2)).map { _ in LLaDAProjectionBlock(c) }
        projector = Linear(c.int("hidden_size", 4096), c.int("projection_dim", 2560))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projector(layers.reduce(x) { $1($0) }) }
}
