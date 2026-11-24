import MLX
import MLXNN

class SwiGLUMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(inputDimensions: Int, hiddenDimensions: Int, bias: Bool = false) {
        self._gateProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: bias)
        self._upProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: bias)
        self._downProj.wrappedValue = Linear(hiddenDimensions, inputDimensions, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(compiledSiluProduct(gateProj(x), upProj(x)))
    }
}

class TanhGELUMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let activation = GELU(approximation: .tanh)

    init(hiddenSize: Int, intermediateSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(activation(fc1(x)))
    }
}
