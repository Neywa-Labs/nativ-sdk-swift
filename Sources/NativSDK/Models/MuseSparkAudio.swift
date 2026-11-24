import MLX
import MLXNN

enum MuseSparkAudio {
    final class PatchEmbedding: Module {
        @ModuleInfo var proj: Linear
        let norm: RMSNorm
        let patchSize: Int

        init(_ config: MuseSparkConfiguration.Audio, index: Int) {
            let channels = index == 0 ? 1 : config.hiddenSizes[index - 1]
            let width = config.hiddenSizes[index]
            patchSize = config.patchSizes[index]
            _proj.wrappedValue = Linear(channels * patchSize, width)
            norm = RMSNorm(dimensions: width, eps: config.epsilon)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, length, channels) = x.shape3
            return norm(proj(x.reshaped(batch, length / patchSize, patchSize * channels)))
        }
    }

    final class MLP: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(width: Int, intermediate: Int) {
            _fc1.wrappedValue = Linear(width, intermediate)
            _fc2.wrappedValue = Linear(intermediate, width)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
    }

    final class Block: Module {
        @ModuleInfo var conv: Conv1d
        let norm: RMSNorm
        let mlp: MLP
        let pad: Int

        init(_ config: MuseSparkConfiguration.Audio, width: Int) {
            _conv.wrappedValue = Conv1d(
                inputChannels: width, outputChannels: width, kernelSize: config.kernelSize,
                groups: width)
            norm = RMSNorm(dimensions: width, eps: config.epsilon)
            mlp = MLP(width: width, intermediate: Int(Float(width) * config.mlpRatio))
            pad = config.kernelSize - 1
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x + mlp(norm(conv(padded(x, widths: [0, [pad, 0], 0]))))
        }
    }

    final class Stage: Module {
        @ModuleInfo(key: "patch_embedder") var embedder: PatchEmbedding
        @ModuleInfo(key: "conv_blocks") var blocks: [Block]

        init(_ config: MuseSparkConfiguration.Audio, index: Int) {
            _embedder.wrappedValue = PatchEmbedding(config, index: index)
            _blocks.wrappedValue = (0 ..< config.depths[index]).map { _ in
                Block(config, width: config.hiddenSizes[index])
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            blocks.reduce(embedder(x)) { $1($0) }
        }
    }

    final class Tower: Module {
        let layers: [Stage]
        let norm: RMSNorm
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "post_proj_norm") var post: RMSNorm
        let config: MuseSparkConfiguration.Audio

        init(_ config: MuseSparkConfiguration.Audio) {
            self.config = config
            layers = config.hiddenSizes.indices.map { Stage(config, index: $0) }
            let width = config.hiddenSizes.last!
            norm = RMSNorm(dimensions: width, eps: config.epsilon)
            _proj.wrappedValue = Linear(width, config.outputSize)
            _post.wrappedValue = RMSNorm(dimensions: config.outputSize, eps: config.epsilon)
        }

        func callAsFunction(_ waves: MLXArray, mask: MLXArray?) throws -> MLXArray {
            guard waves.ndim == 2, waves.dim(0) > 0, waves.dim(1) > 0,
                waves.dim(1).isMultiple(of: config.samplesPerToken)
            else {
                throw VLMError.processing(
                    "Muse Spark requires padded mono waveforms in (batch, samples) layout.")
            }
            let dtype = layers[0].embedder.proj.weight.dtype
            let x = (dtype.isFloatingPoint ? waves.asType(dtype) : waves).expandedDimensions(
                axis: -1)
            let features = post(proj(norm(layers.reduce(x) { $1($0) })))
            guard let mask else { return features.reshaped(-1, features.dim(-1)) }
            guard mask.shape == waves.shape else {
                throw VLMError.processing(
                    "Muse Spark audio padding mask does not match the waveforms.")
            }
            let lengths = mask.asType(.bool).asType(.int32).sum(axis: -1).asArray(Int.self)
            guard lengths.allSatisfy({ $0 > 0 && $0 <= waves.dim(1) }) else {
                throw VLMError.processing("Muse Spark requires nonempty audio clips.")
            }
            let stride = config.samplesPerToken
            return concatenated(
                lengths.enumerated().map { index, length in
                    features[index, ..<((length + stride - 1) / stride), 0...]
                }, axis: 0)
        }
    }
}
