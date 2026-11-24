import Foundation
import MLX

struct ErnieImageTextConfiguration: Decodable {
    let text: Ministral3Configuration

    private enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text =
            try container.decodeIfPresent(Ministral3Configuration.self, forKey: .textConfig)
            ?? Ministral3Configuration(from: decoder)
        guard text.hiddenSize > 0, text.vocabularySize > 0, text.intermediateSize > 0,
            text.attentionHeads > 0, text.kvHeads > 0,
            text.attentionHeads.isMultiple(of: text.kvHeads), text.resolvedHeadDimensions > 0,
            text.hiddenLayers > 0,
            text.resolvedLayerTypes.count == text.hiddenLayers,
            text.resolvedLayerTypes.allSatisfy({ $0 == "full_attention" })
        else {
            throw ModelExecutionError.invalidOptions(
                "Unsupported ERNIE-Image text encoder configuration")
        }
    }

    static func read(in directory: URL) throws -> Ministral3Configuration {
        try JSONDecoder().decode(
            Self.self, from: Data(contentsOf: directory.appending(path: "config.json"))
        ).text
    }

    static func load(
        from directory: URL, configuration: Ministral3Configuration
    ) throws -> Ministral3ModelInner {
        try Task.checkCancellation()
        let model = Ministral3ModelInner(configuration)
        var checkpoint = try DiffusionCheckpoint(directory: directory)
        var weights = [String: MLXArray]()
        for (name, value) in checkpoint.weights {
            var key = name
            if let prefix = ["language_model.model.", "model."].first(where: name.hasPrefix) {
                key = String(name.dropFirst(prefix.count))
            }
            guard !key.contains("rotary_emb.inv_freq"),
                ["embed_tokens.", "layers.", "norm."].contains(where: key.hasPrefix)
            else { continue }
            weights[key] = value.dtype.isFloatingPoint ? value.asType(.bfloat16) : value
        }
        checkpoint.weights = weights
        try checkpoint.apply(to: model)
        return model
    }
}

struct ErnieImagePromptExpansionAssets: Sendable {
    let weights: URL
    let tokenizer: URL

    init(directory: URL, bundledOnly: Bool = false) {
        let bundled = directory.appending(path: "pe")
        if bundledOnly || FileManager.default.fileExists(atPath: bundled.path) {
            weights = bundled
            let tokenizerDirectory = directory.appending(path: "pe_tokenizer")
            tokenizer =
                FileManager.default.fileExists(atPath: tokenizerDirectory.path)
                ? tokenizerDirectory : bundled
        } else {
            weights = directory
            tokenizer = directory
        }
    }

    var isAvailable: Bool {
        FileManager.default.fileExists(atPath: weights.appending(path: "config.json").path)
            && FileManager.default.fileExists(
                atPath: tokenizer.appending(path: "tokenizer.json").path)
            && ((try? safetensorWeightURLs(in: weights.resolvingSymlinksInPath()).isEmpty) == false)
    }
}

enum ErnieImagePromptExpansion {
    static let bundledPatterns = ["pe/*", "pe_tokenizer/*"]
    static let overridePatterns =
        bundledPatterns + [
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja", "chat_template.json",
            "model*.safetensors", "weights*.safetensors", "model.safetensors.index.json",
        ]

    static func expand(
        _ prompt: String, assets: ErnieImagePromptExpansionAssets,
        tokenizerLoader: any TokenizerLoader,
        width: Int, height: Int, seed: UInt64?, policy: ModelExecutionPolicy
    ) async throws -> String {
        guard assets.isAvailable else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image prompt enhancement requires the PE checkpoint and its tokenizer")
        }
        let tokenizer = try await tokenizerLoader.load(from: assets.tokenizer)
        let tokenizerConfig = try ImageCheckpointMetadata.required(
            "tokenizer_config.json", in: assets.tokenizer)
        let generationConfig = try ImageCheckpointMetadata.optional(
            "generation_config.json", in: assets.weights)
        let maxNewTokens = min(2048, tokenizerConfig.int("model_max_length", 2048))
        let additional = tokenizerConfig.values["additional_special_tokens"] as? [String] ?? []
        var eos = Set(additional.compactMap(tokenizer.convertTokenToId))
        if let id = tokenizer.eosTokenId { eos.insert(id) }
        if let value = generationConfig?.values["eos_token_id"] {
            if let id = value as? Int { eos.insert(id) }
            if let ids = value as? [Int] { eos.formUnion(ids) }
        }
        let stopTokens = eos
        let json = try inputJSON(prompt, width: width, height: height)
        let tokens = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": json]], tools: nil,
            additionalContext: ["add_generation_prompt": false])
        guard !tokens.isEmpty, maxNewTokens > 0 else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image prompt enhancement requires tokens")
        }
        return try await runModelWorker(policy: policy) {
            defer { MLX.Memory.clearCache() }
            let configuration = try ErnieImageTextConfiguration.read(in: assets.weights)
            let model = try ErnieImageTextConfiguration.load(
                from: assets.weights, configuration: configuration)
            let caches: [KVCache] = model.layers.map { _ in KVCacheSimple() }
            let sampler = TopPSampler(temperature: 0.6, topP: 0.95, seed: seed)
            var input = MLXArray(tokens).reshaped(1, -1)
            var generated = [Int]()
            for _ in 0 ..< maxNewTokens {
                try Task.checkCancellation()
                let hidden = model.encode(
                    input, cache: caches, penultimate: false, useQueryTemperature: false)
                let logits = model.embedTokens.asLinear(hidden[0..., -1, 0...])
                let token = sampler.sample(logits: logits)
                let id = token.item(Int.self)
                if stopTokens.contains(id) { break }
                generated.append(id)
                input = token.reshaped(1, 1)
            }
            let expanded = tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !expanded.isEmpty else {
                throw ModelExecutionError.invalidOptions(
                    "ERNIE-Image prompt enhancement produced an empty caption")
            }
            return expanded
        }
    }

    static func inputJSON(_ prompt: String, width: Int, height: Int) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        let quoted = String(decoding: try encoder.encode(prompt), as: UTF8.self)
        return "{\"prompt\": \(quoted), \"width\": \(width), \"height\": \(height)}"
    }
}
