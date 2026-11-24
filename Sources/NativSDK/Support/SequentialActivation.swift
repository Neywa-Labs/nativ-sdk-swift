import MLX
import MLXNN

/// Parameterless activations still need a Module slot to preserve sparse checkpoint
/// indices during parameter filtering and quantization of a Sequential array.
final class SequentialActivation: Module, UnaryLayer {
    private let _activation: UnaryLayer
    init(_ activation: UnaryLayer) {
        self._activation = activation
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { _activation(x) }
}
