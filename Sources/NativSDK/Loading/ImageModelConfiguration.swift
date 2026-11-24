import Foundation

/// Pipeline selection depends only on checkpoint contents, never its repository or directory name.
enum ImagePipelineKind: String {
    case bonsai
    case ernieImage = "ernie_image"
    case flux2, flux3, ideogram4
    case lladaImage = "llada_image"
    case mageFlow = "mage_flow"
    case mingImage = "ming_image"
    case mossImage = "moss_image"
    case minimaxH3Image = "minimax_h3_image"
    case qwenImage = "qwen_image"
    case zImage = "z_image"

    static func detect(in directory: URL) throws -> Self {
        // Bonsai shares Flux's class name, but has a distinct packed checkpoint layout.
        if FileManager.default.fileExists(
            atPath: directory.appending(path: "transformer-packed-mflux/config.json").path)
        {
            return .bonsai
        }
        let identifiers: [String: Self] = [
            "Flux2KleinPipeline": .flux2, "Flux2KleinKVPipeline": .flux2,
            "Flux2Transformer2DModel": .flux2,
            "Ideogram4Pipeline": .ideogram4, "Ideogram4Transformer2DModel": .ideogram4,
            "ZImagePipeline": .zImage, "ZImageTransformer2DModel": .zImage,
            "QwenImagePipeline": .qwenImage, "QwenImageEditPipeline": .qwenImage,
            "QwenImageEditPlusPipeline": .qwenImage, "QwenImageTransformer2DModel": .qwenImage,
            "QwenImage21Pipeline": .qwenImage, "QwenImage21Transformer2DModel": .qwenImage,
            "ErnieImagePipeline": .ernieImage,
            "ErnieImageTransformer2DModel": .ernieImage,
            "LLaDAImagePipeline": .lladaImage, "LLaDAImageTransformer2DModel": .lladaImage,
            "MageFlowPipeline": .mageFlow, "MageFlow": .mageFlow,
            "MiniMaxH3ImagePipeline": .minimaxH3Image,
            "MiniMaxH3ImageTransformer2DModel": .minimaxH3Image,
            "MiniMaxH3ImageTransformer": .minimaxH3Image,
            "MOSSImagePipeline": .mossImage, "MOSSImageTransformer": .mossImage,
            "MingImagePipeline": .mingImage, "MingImageTransformer2DModel": .mingImage,
        ]
        var matches = Set<String>()
        for path in [
            "model_index.json", "config.json", "transformer/config.json",
            "transformer_diffusers/config.json",
            "mlx_ming_image.json", "mlx_mage_flow.json", "mlx_ernie_image.json",
            "mlx_qwen_image.json", "mlx_z_image.json",
        ] {
            guard let config = try ImageCheckpointMetadata.optional(path, in: directory) else {
                continue
            }
            if path == "config.json", Flux3Configuration.matches(config) {
                matches.insert(Self.flux3.rawValue)
            }
            for key in ["_class_name", "model_type"] {
                let identifier = config.string(key)
                if let kind = identifiers[identifier] ?? Self(rawValue: identifier) {
                    matches.insert(kind.rawValue)
                }
            }
        }
        if matches.isEmpty {
            if FileManager.default.fileExists(
                atPath: directory.appending(path: "queryformer/config.json").path)
            {
                matches.insert(Self.lladaImage.rawValue)
            }
            if FileManager.default.fileExists(
                atPath: directory.appending(path: "mlp/config.json").path)
            {
                matches.insert(Self.mingImage.rawValue)
            }
        }
        guard matches.count == 1, let name = matches.first, let kind = Self(rawValue: name) else {
            throw ModelExecutionError.invalidOptions(
                "Cannot identify the image pipeline in \(directory.path). Supply a supported _class_name or model_type in checkpoint metadata; conflicting identifiers: \(matches.sorted())."
            )
        }
        return kind
    }
}

enum ImageCheckpointMetadata {
    static func optional(_ path: String, in directory: URL) throws -> DiffusionConfiguration? {
        let url = directory.appending(path: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try DiffusionConfiguration(contentsOf: url)
    }

    static func required(_ path: String, in directory: URL) throws -> DiffusionConfiguration {
        guard let config = try optional(path, in: directory) else {
            throw ModelExecutionError.invalidOptions(
                "Missing image component configuration: \(directory.appending(path: path).path)")
        }
        return config
    }

    static func requireDimensions(_ keys: [String], in config: DiffusionConfiguration, path: String)
        throws
    {
        for key in keys where config.int(key, 0) <= 0 {
            throw ModelExecutionError.invalidOptions("Missing or invalid \(key) in \(path)")
        }
    }

    static func fluxBehavior(in directory: URL) throws -> (
        steps: Int, guidance: Float, referenceCache: Bool
    ) {
        let index = try required("model_index.json", in: directory)
        guard let distilled = index.values["is_distilled"] as? Bool else {
            throw ModelExecutionError.invalidOptions(
                "Missing is_distilled in model_index.json; Flux2 base and distilled checkpoints require different sampling defaults."
            )
        }
        let referenceCache = index.bool(
            "uses_reference_kv_cache", index.string("_class_name") == "Flux2KleinKVPipeline")
        return (distilled ? 4 : 50, distilled ? 1 : 4, referenceCache)
    }
}

extension Flux2TransformerConfiguration {
    static func read(from directory: URL, path: String = "transformer/config.json") throws -> Self {
        let c = try ImageCheckpointMetadata.required(path, in: directory)
        try ImageCheckpointMetadata.requireDimensions(
            [
                "in_channels", "num_layers", "num_single_layers", "attention_head_dim",
                "num_attention_heads", "joint_attention_dim",
            ], in: c, path: path)
        return Self(
            patchSize: c.int("patch_size", 1), inChannels: c.int("in_channels", 128),
            outChannels: c.values["out_channels"] as? Int,
            numLayers: c.int("num_layers", 5), numSingleLayers: c.int("num_single_layers", 20),
            attentionHeadDim: c.int("attention_head_dim", 128),
            numAttentionHeads: c.int("num_attention_heads", 24),
            jointAttentionDim: c.int("joint_attention_dim", 7680),
            timestepGuidanceChannels: c.int("timestep_guidance_channels", 256),
            mlpRatio: c.float("mlp_ratio", 3),
            axesDimsRoPE: c.ints("axes_dims_rope", [32, 32, 32, 32]),
            ropeTheta: c.float("rope_theta", 2000),
            guidanceEmbeds: c.bool("guidance_embeds", false))
    }
}

extension Flux2TextEncoderConfiguration {
    static func read(from directory: URL, path: String = "text_encoder/config.json") throws -> Self
    {
        let c = try ImageCheckpointMetadata.required(path, in: directory)
        try ImageCheckpointMetadata.requireDimensions(
            [
                "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "intermediate_size",
            ], in: c, path: path)
        return Self(diffusion: c)
    }
}

extension Flux2VAEConfiguration {
    static func read(
        from directory: URL, includeEncoder: Bool = false, batchNormEps: Float? = nil
    ) throws -> Self {
        let c = try ImageCheckpointMetadata.required("vae/config.json", in: directory)
        let channels = c.ints("block_out_channels", [])
        let decoderChannels = c.ints("decoder_block_out_channels", channels)
        guard !channels.isEmpty, channels.allSatisfy({ $0 > 0 }), channels.count < 16,
            decoderChannels.count == channels.count, decoderChannels.allSatisfy({ $0 > 0 })
        else {
            throw ModelExecutionError.invalidOptions(
                "Missing or invalid VAE block channels in vae/config.json")
        }
        try ImageCheckpointMetadata.requireDimensions(
            ["latent_channels", "layers_per_block", "norm_num_groups"], in: c,
            path: "vae/config.json")
        return Self(
            decoderBlockOutChannels: decoderChannels, encoderBlockOutChannels: channels,
            includeEncoder: includeEncoder,
            latentChannels: c.int("latent_channels", 32), spatialScale: 1 << (channels.count - 1),
            layersPerBlock: c.int("layers_per_block", 2), normGroups: c.int("norm_num_groups", 32),
            eps: c.float("norm_eps", 1e-6),
            batchNormEps: batchNormEps ?? c.float("batch_norm_eps", 1e-4),
            midBlockAddAttention: c.bool("mid_block_add_attention", true))
    }
}
