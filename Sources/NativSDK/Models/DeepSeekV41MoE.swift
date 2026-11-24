import Foundation
import MLX
import MLXNN

enum DeepSeekV41Language {
    /// The limit arrives as an array rather than a capture: `compile` freezes a
    /// captured Swift scalar at its first call.
    private static let compiledLimitedSwiGLU: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile { gate, up, limit in
            silu(minimum(gate.asType(.float32), limit))
                * clip(up.asType(.float32), min: -limit, max: limit)
        }

    private static let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile { gate, up in
            silu(gate.asType(.float32)) * up.asType(.float32)
        }

    /// The reference keeps the activation and the routing multiplication in FP32.
    static func swiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
        guard limit > 0 else { return compiledSwiGLU(gate, up) }
        return compiledLimitedSwiGLU(gate, up, MLXArray(limit))
    }

    /// MoE gating with a separate correction bias for image-span tokens.
    ///
    /// The bias steers expert selection only; the routing weights come from the unbiased
    /// scores. Selection is the shared noaux_tc top-k — only the bias source is
    /// V4.1-specific.
    final class MoEGate: Module {
        let topK: Int
        let experts: Int
        let normalizeTopK: Bool
        let routedScalingFactor: Float

        @ParameterInfo(key: "weight") var weight: MLXArray
        @ParameterInfo(key: "bias") var bias: MLXArray
        @ParameterInfo(key: "bias_vl") var biasImage: MLXArray

        init(
            _ config: DeepSeekV41Configuration, experts: Int? = nil,
            numExpertsPerTok: Int? = nil
        ) {
            self.topK = numExpertsPerTok ?? config.numExpertsPerTok
            self.experts = experts ?? config.nRoutedExperts
            self.normalizeTopK = config.normTopkProb
            self.routedScalingFactor = config.routedScalingFactor
            _weight.wrappedValue = MLXArray.zeros([self.experts, config.hiddenSize])
            _bias.wrappedValue = MLXArray.zeros([self.experts], dtype: .float32)
            _biasImage.wrappedValue = MLXArray.zeros([self.experts], dtype: .float32)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, imageMask: MLXArray? = nil)
            -> (indices: MLXArray, weights: MLXArray)
        {
            let scores = sqrt(softplus(matmul(x.asType(.float32), weight.T)))
            var bias = self.bias
            if let imageMask {
                bias = MLX.where(imageMask[.ellipsis, .newAxis], biasImage, self.bias)
            }
            let indices = argPartition(-(scores + bias), kth: topK - 1, axis: -1)[
                .ellipsis, ..<topK
            ].asType(.int32)
            var weights = takeAlong(scores, indices, axis: -1)
            if normalizeTopK, topK > 1 {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            return (indices, weights * routedScalingFactor)
        }
    }

    final class MLP: Module, UnaryLayer {
        let limit: Float

        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: DeepSeekV41Configuration, intermediateSize: Int? = nil) {
            let hidden = intermediateSize ?? config.moeIntermediateSize
            self.limit = config.swigluLimit
            _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let dtype = x.dtype
            let quantized = DeepSeekV41FakeQuant.fp8UE8M0(x)
            let activated = DeepSeekV41Language.swiGLU(
                gate: gateProj(quantized), up: upProj(quantized), limit: limit)
            return downProj(DeepSeekV41FakeQuant.fp8UE8M0(activated.asType(dtype)))
        }
    }

    /// Top-k routed experts, with the routing weights applied before the BF16 cast and
    /// FP8 rounding of the down-projection's input.
    final class SwitchMLP: Module {
        let limit: Float

        @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
        @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
        @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

        init(inputDimensions: Int, hiddenDimensions: Int, experts: Int, limit: Float) {
            self.limit = limit
            _gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDimensions, outputDims: hiddenDimensions,
                numExperts: experts, bias: false)
            _upProj.wrappedValue = SwitchLinear(
                inputDims: inputDimensions, outputDims: hiddenDimensions,
                numExperts: experts, bias: false)
            _downProj.wrappedValue = SwitchLinear(
                inputDims: hiddenDimensions, outputDims: inputDimensions,
                numExperts: experts, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray)
            -> MLXArray
        {
            let shape = indices.shape
            var inputs = expandedDimensions(x, axes: [-2, -3])
            var indices = indices
            var weights = weights
            let doSort = indices.size >= 64
            var order = MLXArray()
            if doSort {
                order = argSort(indices.flattened())
                inputs =
                    inputs.flattened(start: 0, end: -3)[order.floorDivide(shape[shape.count - 1])]
                indices = indices.flattened()[order]
                weights = weights.flattened()[order]
            }
            let up = upProj(inputs, indices, sortedIndices: doSort)
            let gate = gateProj(inputs, indices, sortedIndices: doSort)
            let activated = DeepSeekV41FakeQuant.fp8UE8M0(
                (DeepSeekV41Language.swiGLU(gate: gate, up: up, limit: limit)
                    * weights[.ellipsis, .newAxis, .newAxis]).asType(up.dtype))
            var y = downProj(activated, indices, sortedIndices: doSort)
            if doSort {
                y = unflatten(y[argSort(order)], axis: 0, shape: shape)
            }
            return MLX.squeezed(y, axis: -2)
        }
    }

    /// Top-k routed experts plus one shared expert every token goes through.
    final class MoE: Module {
        @ModuleInfo(key: "gate") var gate: MoEGate
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchMLP
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP

        init(
            _ config: DeepSeekV41Configuration, moeIntermediateSize: Int? = nil,
            experts: Int? = nil, numExpertsPerTok: Int? = nil
        ) {
            let intermediate = moeIntermediateSize ?? config.moeIntermediateSize
            let routed = experts ?? config.nRoutedExperts
            _gate.wrappedValue = MoEGate(
                config, experts: experts, numExpertsPerTok: numExpertsPerTok)
            _switchMLP.wrappedValue = SwitchMLP(
                inputDimensions: config.hiddenSize, hiddenDimensions: intermediate,
                experts: routed, limit: config.swigluLimit)
            _sharedExperts.wrappedValue = MLP(config, intermediateSize: intermediate)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, imageMask: MLXArray? = nil) -> MLXArray {
            let (indices, scores) = gate(x, imageMask: imageMask)
            var y = switchMLP(
                DeepSeekV41FakeQuant.fp8UE8M0(x), indices: indices, weights: scores)
            y = y.asType(.float32).sum(axis: -2)
            return (y + sharedExperts(x).asType(.float32)).asType(x.dtype)
        }
    }

    /// Rename shared experts and stack per-expert tensors for the switch layers.
    ///
    /// Shared `w1/w2/w3` become `gate/down/up_proj`; routed `experts.{e}.w1/w2/w3` stack
    /// over experts onto `switch_mlp`. Checkpoints converted by the reference port
    /// already store the routed experts stacked under `experts.{gate,down,up}_proj`;
    /// those only need the move onto `switch_mlp`.
    static func sanitizeMoEWeights(
        _ weights: [String: MLXArray], ffnPrefix: String, experts: Int
    ) throws -> [String: MLXArray] {
        let remap = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("\(ffnPrefix).shared_experts.") {
                for (old, new) in remap {
                    key = key.replacingOccurrences(
                        of: ".shared_experts.\(old).", with: ".shared_experts.\(new).")
                }
            }
            out[key] = value
        }

        let prefix = "\(ffnPrefix).experts"
        for destination in ["gate_proj", "down_proj", "up_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let stacked = "\(prefix).\(destination).\(suffix)"
                if let value = out.removeValue(forKey: stacked) {
                    out["\(ffnPrefix).switch_mlp.\(destination).\(suffix)"] = value
                }
            }
        }
        for (source, destination) in remap {
            for suffix in ["weight", "scales", "biases"] {
                guard out["\(prefix).0.\(source).\(suffix)"] != nil else { continue }
                let absent = (0 ..< experts).filter {
                    out["\(prefix).\($0).\(source).\(suffix)"] == nil
                }
                if !absent.isEmpty {
                    throw DeepSeekV41WeightError.incompleteExperts(
                        "\(prefix).{expert}.\(source).\(suffix)", experts - absent.count,
                        experts, absent)
                }
                let parts = (0 ..< experts).map {
                    out.removeValue(forKey: "\(prefix).\($0).\(source).\(suffix)")!
                }
                out["\(ffnPrefix).switch_mlp.\(destination).\(suffix)"] = stacked(parts)
            }
        }
        return out
    }
}

public enum DeepSeekV41WeightError: LocalizedError {
    case incompleteExperts(String, Int, Int, [Int])

    public var errorDescription: String? {
        switch self {
        case .incompleteExperts(let key, let present, let total, let absent):
            let shown = absent.prefix(8).map(String.init).joined(separator: ", ")
            let more = absent.count > 8 ? " (+\(absent.count - 8) more)" : ""
            return
                "\(key): checkpoint provides \(present) of \(total) experts; missing \(shown)\(more)"
        }
    }
}
