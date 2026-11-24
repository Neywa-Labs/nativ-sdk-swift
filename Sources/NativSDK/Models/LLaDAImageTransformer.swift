import Foundation
import MLX
import MLXNN

final class LLaDAImageTransformer: Module {
    let config: ZImageTransformerConfiguration
    @ModuleInfo(key: "x_embedder") var image: Linear
    @ModuleInfo(key: "t_embedder") var time: ZImageTimestepEmbedder
    @ModuleInfo(key: "cap_embedder") var caption: [Module]
    @ModuleInfo(key: "semantic_embedder") var semantic: [Module]
    @ModuleInfo(key: "sigvq_embedder") var sigvq: [Module]
    @ModuleInfo(key: "noise_refiner") var noiseRefiners: [ZImageBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiners: [ZImageBlock]
    @ModuleInfo(key: "sigvq_refiner") var semanticRefiners: [ZImageBlock]
    @ModuleInfo var layers: [ZImageBlock]
    @ModuleInfo(key: "final_layer") var final: ZImageFinalLayer
    @ParameterInfo(key: "x_pad_token") var imagePad: MLXArray
    @ParameterInfo(key: "cap_pad_token") var captionPad: MLXArray
    @ParameterInfo(key: "sigvq_pad_token") var semanticPad: MLXArray
    init(_ c: DiffusionConfiguration) throws {
        var values = c.values
        values["all_patch_size"] = c.ints("all_patch_size", [1])
        values["in_channels"] = c.int("in_channels", 128)
        guard c.ints("all_patch_size", [1]) == [1], c.ints("all_f_patch_size", [1]) == [1] else {
            throw ModelExecutionError.invalidOptions("LLaDA image uses patch size one")
        }
        config = try .init(.init(values))
        let cfg = config
        _image.wrappedValue = Linear(cfg.channels, cfg.dim)
        _time.wrappedValue = ZImageTimestepEmbedder(cfg.adaln)
        func projection(_ input: Int) -> [Module] {
            [DiffusionRMSNorm(cfg.eps), Linear(input, cfg.dim)]
        }
        _caption.wrappedValue = projection(cfg.textDim)
        _semantic.wrappedValue = projection(c.int("semantic_feat_dim", 4096))
        _sigvq.wrappedValue = projection(c.int("semantic_feat_dim", 4096))
        func block(_ modulated: Bool) -> ZImageBlock {
            ZImageBlock(
                cfg, modulated: modulated, parameterFree: true, qkNorm: c.bool("qk_norm", true))
        }
        _noiseRefiners.wrappedValue = (0 ..< cfg.refiners).map { _ in block(true) }
        _contextRefiners.wrappedValue = (0 ..< cfg.refiners).map { _ in block(false) }
        _semanticRefiners.wrappedValue = (0 ..< cfg.refiners).map { _ in block(false) }
        _layers.wrappedValue = (0 ..< cfg.depth).map { _ in block(true) }
        _final.wrappedValue = ZImageFinalLayer(cfg)
        _imagePad.wrappedValue = MLXArray.zeros([1, cfg.dim])
        _captionPad.wrappedValue = MLXArray.zeros([1, cfg.dim])
        _semanticPad.wrappedValue = MLXArray.zeros([1, cfg.dim])
        super.init()
    }
    private func project(_ x: MLXArray, _ modules: [Module]) -> MLXArray {
        (modules[1] as! Linear)((modules[0] as! DiffusionRMSNorm)(x))
    }
    private func pad(_ x: MLXArray, token: MLXArray) -> MLXArray {
        let count = (32 - x.dim(1) % 32) % 32
        return count == 0
            ? x : concatenated([x, broadcast(token, to: [x.dim(0), count, x.dim(-1)])], axis: 1)
    }
    private func textPositions(_ length: Int, start: Int = 1) -> MLXArray {
        let positions = stacked(
            [
                arange(start, start + length).asType(.float32), MLXArray.zeros([length]),
                MLXArray.zeros([length]),
            ], axis: -1
        ).expandedDimensions(axis: 0)
        return pad(positions, token: MLXArray.zeros([1, 3]))
    }
    private func imagePositions(_ h: Int, _ w: Int, start: Int) -> MLXArray {
        pad(diffusionPositions(height: h, width: w, offset: start), token: MLXArray.zeros([1, 3]))
    }
    private func angles(_ positions: MLXArray) -> MLXArray {
        diffusionRotaryAngles(positions, sections: config.sections, theta: config.theta)
    }
    func callAsFunction(
        _ x: MLXArray, timestep: MLXArray, caption features: MLXArray, semantics: MLXArray? = nil,
        source: MLXArray? = nil
    ) throws -> MLXArray {
        if let source {
            guard source.shape == x.shape, let semantics else {
                throw ModelExecutionError.invalidOptions(
                    "LLaDA editing requires matching source latents and SigVQ features")
            }
            return edit(
                x, timestep: timestep, features: features, semantics: semantics, source: source)
        }
        let b = x.dim(0)
        let h = x.dim(2)
        let w = x.dim(3)
        let n = h * w
        var images = pad(image(x.transposed(0, 2, 3, 1).reshaped([b, n, -1])), token: imagePad)
        var captions = pad(project(features, caption), token: captionPad)
        var capPositions = textPositions(features.dim(1))
        if let semantics, semantics.dim(1) > 0 {
            capPositions = concatenated(
                [capPositions, textPositions(semantics.dim(1), start: captions.dim(1) + 1)], axis: 1
            )
            captions = concatenated(
                [captions, pad(project(semantics, semantic), token: captionPad)], axis: 1)
        }
        let ia = angles(imagePositions(h, w, start: captions.dim(1) + 1))
        let ca = angles(capPositions)
        let t = time(timestep * config.timeScale)
        for block in noiseRefiners { images = block(images, angles: ia, time: t) }
        for block in contextRefiners { captions = block(captions, angles: ca) }
        var hidden = concatenated([images, captions], axis: 1)
        let a = concatenated([ia, ca], axis: 1)
        for block in layers { hidden = block(hidden, angles: a, time: t) }
        return final(hidden[0..., ..<n], time: t).reshaped([b, h, w, config.channels]).transposed(
            0, 3, 1, 2)
    }
    private func edit(
        _ x: MLXArray, timestep: MLXArray, features: MLXArray, semantics: MLXArray, source: MLXArray
    ) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(2)
        let w = x.dim(3)
        let n = h * w
        let capLength = features.dim(1)
        let t = time(abs(timestep) * config.timeScale)
        let clean = time(MLXArray.zeros(timestep.shape, dtype: timestep.dtype))
        let cap = pad(project(features, caption), token: captionPad)
        var captions = concatenated([cap, cap], axis: 1)
        let cp = concatenated(
            [textPositions(capLength), textPositions(capLength, start: capLength + 3)], axis: 1)
        let cn = concatenated(
            [
                MLXArray.zeros([b, cap.dim(1)], dtype: .bool),
                MLXArray.ones([b, cap.dim(1)], dtype: .bool),
            ], axis: 1)
        let imageTokens = [source, x].map {
            pad(image($0.transposed(0, 2, 3, 1).reshaped([b, n, -1])), token: imagePad)
        }
        let sourceLength = imageTokens[0].dim(1)
        var images = concatenated(imageTokens, axis: 1)
        let ip = concatenated(
            [
                imagePositions(h, w, start: capLength + 1),
                imagePositions(h, w, start: 2 * capLength + 3),
            ], axis: 1)
        let imageNoise = concatenated(
            [
                MLXArray.zeros([b, sourceLength], dtype: .bool),
                MLXArray.ones([b, sourceLength], dtype: .bool),
            ], axis: 1)
        for block in noiseRefiners {
            images = block(
                images, angles: angles(ip), time: t, noiseMask: imageNoise, cleanTime: clean)
        }
        for block in contextRefiners { captions = block(captions, angles: angles(cp)) }
        var tokens = [captions, images]
        var positions = [cp, ip]
        var masks = [cn, imageNoise]
        if semantics.dim(1) > 0 {
            var features = pad(project(semantics, sigvq), token: semanticPad)
            let positionsPart = textPositions(
                semantics.dim(1), start: captions.dim(1) + images.dim(1) + 1)
            for block in semanticRefiners {
                features = block(features, angles: angles(positionsPart))
            }
            tokens.append(features)
            positions.append(positionsPart)
            masks.append(MLXArray.zeros([b, features.dim(1)], dtype: .bool))
        }
        var hidden = concatenated(tokens, axis: 1)
        let a = angles(concatenated(positions, axis: 1))
        let noiseMask = concatenated(masks, axis: 1)
        for block in layers {
            hidden = block(hidden, angles: a, time: t, noiseMask: noiseMask, cleanTime: clean)
        }
        let start = captions.dim(1) + sourceLength
        return final(hidden[0..., start ..< (start + n)], time: t).reshaped([
            b, h, w, config.channels,
        ]).transposed(0, 3, 1, 2)
    }
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            let key = key.replacingOccurrences(of: "all_x_embedder.1-1.", with: "x_embedder.")
                .replacingOccurrences(of: "all_final_layer.1-1.", with: "final_layer.")
                .replacingOccurrences(
                    of: "final_layer.adaLN_modulation.1.", with: "final_layer.adaLN_modulation.0."
                )
                .replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.linear1.")
                .replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.linear2.")
            if key.contains(".attention.to_qkv.") {
                for (name, tensor) in zip(
                    ["to_q", "to_k", "to_v"], split(value, parts: 3, axis: 0))
                { result[key.replacingOccurrences(of: ".to_qkv.", with: ".\(name).")] = tensor }
            } else if key.contains(".feed_forward.w13.") {
                for (name, tensor) in zip(["w1", "w3"], split(value, parts: 2, axis: 0)) {
                    result[key.replacingOccurrences(of: ".w13.", with: ".\(name).")] = tensor
                }
            } else {
                result[key] = value
            }
        }
        return result
    }
}
