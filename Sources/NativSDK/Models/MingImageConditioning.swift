import Foundation
import MLX
import MLXNN

final class MingImageGate: Module {
    @ModuleInfo(key: "gate_proj") var projection: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?
    let config: BailingMoeConfiguration
    init(_ c: BailingMoeConfiguration) {
        config = c
        _projection.wrappedValue = Linear(c.hiddenSize, c.numExperts, bias: false)
        _expertBias.wrappedValue =
            c.moeRouterEnableExpertBias ? MLXArray.zeros([c.numExperts]) : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = projection(x)
        let scores = sigmoid(logits.asType(.float32))
        var choice = scores + (expertBias ?? MLXArray(Float(0)))
        if config.nGroup > 1 {
            let grouped = choice.reshaped([x.dim(0), x.dim(1), config.nGroup, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let count = config.nGroup - config.topkGroup
            if count > 0 {
                let indices = argPartition(groupScores, kth: count - 1, axis: -2)[
                    .ellipsis, ..<count, 0...]
                choice = putAlong(grouped, indices, values: MLXArray(Float(0)), axis: -2).reshaped(
                    scores.shape)
            }
        }
        let indices = argPartition(-choice, kth: config.numExpertsPerToken - 1, axis: -1)[
            .ellipsis, ..<config.numExpertsPerToken]
        var selected = takeAlong(scores, indices, axis: -1)
        if config.numExpertsPerToken > 1 && config.normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, (selected * config.routedScalingFactor).asType(logits.dtype))
    }
}

final class MingImageMoE: Module {
    @ModuleInfo var gate: MingImageGate
    @ModuleInfo(key: "image_gate") var imageGate: MingImageGate
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var shared: BailingMoeMLP
    init(_ c: BailingMoeConfiguration) {
        _gate.wrappedValue = MingImageGate(c)
        _imageGate.wrappedValue = MingImageGate(c)
        _experts.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize, numExperts: c.numExperts,
            bias: false)
        _shared.wrappedValue = BailingMoeMLP(
            c, hiddenDims: c.moeIntermediateSize * c.numSharedExperts)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, imageMask: MLXArray) -> MLXArray {
        let (textIndices, textWeights) = gate(x)
        let (imageIndices, imageWeights) = imageGate(x)
        let select = imageMask.expandedDimensions(axis: -1)
        let indices = MLX.where(select, imageIndices, textIndices)
        let weights = MLX.where(select, imageWeights, textWeights)
        let output = experts(x, indices)
        return (output * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(output.dtype)
            + shared(x)
    }
}

final class MingImageDecoderLayer: Module {
    @ModuleInfo var attention: BailingMoeAttention
    @ModuleInfo var mlp: Module
    @ModuleInfo(key: "input_layernorm") var norm1: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var norm2: RMSNorm
    init(_ c: BailingMoeConfiguration, index: Int) {
        _attention.wrappedValue = BailingMoeAttention(c)
        _mlp.wrappedValue = index >= c.firstKDenseReplace ? MingImageMoE(c) : BailingMoeMLP(c)
        _norm1.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _norm2.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, imageMask: MLXArray
    ) -> MLXArray {
        let h = x + attention(norm1(x), mask: mask, cache: nil)
        let normalized = norm2(h)
        return h
            + ((mlp as? MingImageMoE).map { $0(normalized, imageMask: imageMask) }
                ?? (mlp as! BailingMoeMLP)(normalized))
    }
}

final class MingImageBackbone: Module {
    @ModuleInfo(key: "word_embeddings") var embedding: Embedding
    @ModuleInfo var layers: [MingImageDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    init(_ c: BailingMoeConfiguration) {
        _embedding.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.hiddenLayers).map { MingImageDecoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, imageMask: MLXArray) -> [MLXArray] {
        let mask = createAttentionMask(h: x, cache: Optional<KVCache>.none)
        var h = x
        var states = [MLXArray]()
        for layer in layers {
            states.append(h)
            h = layer(h, mask: mask, imageMask: imageMask)
        }
        states.append(norm(h))
        return states
    }
}

final class MingImageConnector: Module, UnaryLayer {
    @ModuleInfo var layers: [Qwen2TransformerBlock]
    @ModuleInfo var norm: RMSNorm
    init(_ c: Qwen2Configuration) {
        _layers.wrappedValue = (0 ..< c.hiddenLayers).map { _ in Qwen2TransformerBlock(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h, mask: .none, cache: nil) }
        return norm(h)
    }
}

final class MingImageTextEncoder: Module {
    @ModuleInfo var mllm: MingImageBackbone
    @ModuleInfo var connector: MingImageConnector
    @ParameterInfo(key: "query_tokens") var queries: MLXArray
    @ModuleInfo(key: "proj_in") var input: Linear
    @ModuleInfo(key: "proj_out") var output: Linear
    @ModuleInfo(key: "directvlm_norm") var directNorm: RMSNorm
    @ModuleInfo(key: "directvlm_proj") var direct: Linear
    let config: BailingMoeConfiguration
    let selectedLayers: [Int], queryCount: Int, patchID: Int, startID: Int, endID: Int
    init(
        backbone b: DiffusionConfiguration, connector c: DiffusionConfiguration,
        bridge: DiffusionConfiguration
    ) throws {
        var values = b.values
        let defaults: [String: Any] = [
            "model_type": "bailing_moe_v2", "max_position_embeddings": 32768,
            "norm_topk_prob": true,
            "rms_norm_eps": 1e-6, "rope_theta": 600000, "first_k_dense_replace": 1,
            "use_bias": false, "use_qkv_bias": false,
            "use_qk_norm": true, "tie_word_embeddings": false, "partial_rotary_factor": 0.5,
            "moe_router_enable_expert_bias": b.bool("use_expert_bias", true),
            "routed_scaling_factor": b.float("moe_router_topk_scaling_factor", 2.5),
            "score_function": "sigmoid", "n_group": 8, "topk_group": 4,
        ]
        for (key, value) in defaults where values[key] == nil { values[key] = value }
        config = try JSONDecoder().decode(
            BailingMoeConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        var connectorValues = c.values
        connectorValues["vocab_size"] = 1
        let connectorDefaults: [String: Any] = [
            "hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12,
            "num_key_value_heads": 2, "intermediate_size": 8960, "rope_theta": 1_000_000,
            "rms_norm_eps": 1e-6,
        ]
        for (key, value) in connectorDefaults where connectorValues[key] == nil {
            connectorValues[key] = value
        }
        let cc = try JSONDecoder().decode(
            Qwen2Configuration.self, from: JSONSerialization.data(withJSONObject: connectorValues))
        let scale = bridge.ints("img_gen_scales", [16])[0]
        queryCount = scale * scale
        selectedLayers = bridge.ints("selected_hidden_states_layers", [5, 12, 20])
        patchID = b.int("image_patch_token", 157157)
        startID = b.int("image_start_token", 157158)
        endID = b.int("image_end_token", 157159)
        let layerCount = config.hiddenLayers
        guard selectedLayers.allSatisfy({ $0 >= 0 && $0 <= layerCount }) else {
            throw ModelExecutionError.invalidOptions(
                "Ming selected hidden layers exceed encoder depth")
        }
        _mllm.wrappedValue = MingImageBackbone(config)
        _connector.wrappedValue = MingImageConnector(cc)
        _queries.wrappedValue = MLXArray.zeros([queryCount, config.hiddenSize])
        _input.wrappedValue = Linear(config.hiddenSize, cc.hiddenSize)
        _output.wrappedValue = Linear(cc.hiddenSize, bridge.int("diffusion_c_input_dim", 2560))
        _directNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize * selectedLayers.count)
        _direct.wrappedValue = Linear(
            config.hiddenSize * selectedLayers.count, bridge.int("diffusion_inner_dim", 3840))
        super.init()
    }
    func encode(_ prompt: String, tokenizer: any Tokenizer) throws -> (MLXArray, MLXArray) {
        var ids = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], tools: nil,
            additionalContext: ["add_generation_prompt": true])
        ids += [startID] + Array(repeating: patchID, count: queryCount) + [endID]
        return try encode(tokenIDs: ids)
    }
    func encode(tokenIDs ids: [Int]) throws -> (MLXArray, MLXArray) {
        let textPositions = ids.indices.filter { ![patchID, startID, endID].contains(ids[$0]) }
        guard let queryStart = ids.firstIndex(of: patchID), queryStart + queryCount <= ids.count
        else {
            throw ModelExecutionError.invalidOptions("Ming prompt is missing image query tokens")
        }
        let embeddings = mllm.embedding(MLXArray(ids).reshaped([1, -1]))
        let x = concatenated(
            [
                embeddings[0..., ..<queryStart],
                queries.expandedDimensions(axis: 0).asType(embeddings.dtype),
                embeddings[0..., (queryStart + queryCount)...],
            ], axis: 1)
        let states = mllm(x, imageMask: MLXArray(ids.map { $0 == patchID }).reshaped([1, -1]))
        let cap = output(
            connector(input(states.last![0..., queryStart ..< (queryStart + queryCount)])))
        let hidden = concatenated(selectedLayers.map { states[$0] }, axis: -1)
        let direct = direct(directNorm(take(hidden, MLXArray(textPositions), axis: 1)))
        eval(cap, direct)
        return (cap.asType(.bfloat16), direct.asType(.bfloat16))
    }
    static func load(directory: URL) throws -> MingImageTextEncoder {
        let model = try MingImageTextEncoder(
            backbone: .init(contentsOf: directory.appending(path: "mllm/config.json")).child(
                "llm_config"),
            connector: .init(contentsOf: directory.appending(path: "connector/config.json")),
            bridge: .init(contentsOf: directory.appending(path: "mlp/config.json")))
        let nativeDirectory = directory.appending(path: "text_encoder")
        if FileManager.default.fileExists(
            atPath: nativeDirectory.appending(path: "config.json").path),
            try DiffusionConfiguration(contentsOf: nativeDirectory.appending(path: "config.json"))
                .bool("mlx_format", false)
        {
            try DiffusionCheckpoint(directory: nativeDirectory).apply(to: model)
            return model
        }
        var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "mllm"))
        var weights = [String: MLXArray]()
        for (key, value) in checkpoint.weights
        where key.hasPrefix("model.model.") && !key.contains(".mlp.audio_gate.") {
            let key = ("mllm." + key.dropFirst(12)).replacingOccurrences(
                of: ".attention.q_norm.", with: ".attention.query_layernorm."
            )
            .replacingOccurrences(of: ".attention.k_norm.", with: ".attention.key_layernorm.")
            .replacingOccurrences(of: ".mlp.gate.weight", with: ".mlp.gate.gate_proj.weight")
            .replacingOccurrences(
                of: ".mlp.image_gate.weight", with: ".mlp.image_gate.gate_proj.weight")
            weights[key] = value
        }
        for layer in model.config.firstKDenseReplace ..< model.config.hiddenLayers {
            let prefix = "mllm.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"]
            where weights["\(prefix).experts.0.\(projection).weight"] != nil {
                var experts = [MLXArray]()
                for expert in 0 ..< model.config.numExperts {
                    guard
                        let tensor = weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).weight")
                    else {
                        throw ModelExecutionError.invalidOptions("Incomplete Ming expert weights")
                    }
                    experts.append(tensor)
                }
                weights["\(prefix).switch_mlp.\(projection).weight"] = stacked(experts)
            }
        }
        for (key, value) in try DiffusionCheckpoint(
            directory: directory.appending(path: "connector")
        ).weights {
            if key.hasPrefix("model.layers.") || key == "model.norm.weight" {
                weights["connector." + key.dropFirst(6)] = value
            }
        }
        for (key, value) in try DiffusionCheckpoint(directory: directory.appending(path: "mlp"))
            .weights
        {
            if key.hasPrefix("query_tokens_dict.") {
                weights["query_tokens"] = value
            } else if key.hasPrefix("proj_in.") || key.hasPrefix("proj_out.") {
                weights[key] = value
            } else if key == "proj_directvlm.0.weight" {
                weights["directvlm_norm.weight"] = value
            } else if key.hasPrefix("proj_directvlm.1.") {
                weights["directvlm_proj." + key.dropFirst(16)] = value
            }
        }
        checkpoint.weights = weights
        try checkpoint.apply(to: model)
        return model
    }
}
