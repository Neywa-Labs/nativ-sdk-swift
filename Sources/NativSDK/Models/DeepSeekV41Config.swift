import Foundation

public struct DeepSeekV41Configuration: Decodable, Sendable, ModelConfigurationValidating {
    /// The text fields as they appear at the top level, and as they appear under a
    /// nested `text_config`. The top level wins, which is the precedence upstream's
    /// from_dict applies.
    let flat: TextFields
    let nested: TextFields?

    public var modelType: String { flat._modelType ?? nested?._modelType ?? "deepseek_v41" }
    public var vocabularySize: Int { flat._vocabularySize ?? nested?._vocabularySize ?? 129_280 }
    public var hiddenSize: Int { flat._hiddenSize ?? nested?._hiddenSize ?? 5120 }
    public var moeIntermediateSize: Int {
        flat._moeIntermediateSize ?? nested?._moeIntermediateSize ?? 2304
    }
    public var hiddenLayers: Int { flat._hiddenLayers ?? nested?._hiddenLayers ?? 40 }
    public var attentionHeads: Int { flat._attentionHeads ?? nested?._attentionHeads ?? 64 }
    public var kvHeads: Int { flat._kvHeads ?? nested?._kvHeads ?? 1 }
    public var headDim: Int { flat._headDim ?? nested?._headDim ?? 512 }
    public var qkRopeHeadDim: Int { flat._qkRopeHeadDim ?? nested?._qkRopeHeadDim ?? 64 }
    public var qLoraRank: Int { flat._qLoraRank ?? nested?._qLoraRank ?? 1280 }
    public var oLoraRank: Int { flat._oLoraRank ?? nested?._oLoraRank ?? 1024 }
    public var oGroups: Int { flat._oGroups ?? nested?._oGroups ?? 8 }
    public var swigluLimit: Float { flat._swigluLimit ?? nested?._swigluLimit ?? 10 }
    public var rmsNormEps: Float { flat._rmsNormEps ?? nested?._rmsNormEps ?? 1e-20 }
    public var maxPositionEmbeddings: Int {
        flat._maxPositionEmbeddings ?? nested?._maxPositionEmbeddings ?? 1_048_576
    }
    public var ropeTheta: Float { flat._ropeTheta ?? nested?._ropeTheta ?? 10000 }
    public var nRoutedExperts: Int { flat._nRoutedExperts ?? nested?._nRoutedExperts ?? 384 }
    public var nSharedExperts: Int { flat._nSharedExperts ?? nested?._nSharedExperts ?? 1 }
    public var numExpertsPerTok: Int { flat._numExpertsPerTok ?? nested?._numExpertsPerTok ?? 6 }
    public var scoringFunc: String { flat._scoringFunc ?? nested?._scoringFunc ?? "sqrtsoftplus" }
    public var topkMethod: String { flat._topkMethod ?? nested?._topkMethod ?? "noaux_tc" }
    public var normTopkProb: Bool { flat._normTopkProb ?? nested?._normTopkProb ?? true }
    public var routedScalingFactor: Float {
        flat._routedScalingFactor ?? nested?._routedScalingFactor ?? 1.5
    }
    public var slidingWindow: Int { flat._slidingWindow ?? nested?._slidingWindow ?? 128 }
    public var compressRopeTheta: Float {
        flat._compressRopeTheta ?? nested?._compressRopeTheta ?? 160_000
    }
    public var kvSourceLayerIds: [Int] {
        flat._kvSourceLayerIds ?? nested?._kvSourceLayerIds ?? [2, 8, 14, 20]
    }
    public var indexSourceLayerIds: [Int] {
        flat._indexSourceLayerIds ?? nested?._indexSourceLayerIds
            ?? [2, 8, 14, 20, 24, 28, 32, 36]
    }
    public var indexHeads: Int { flat._indexHeads ?? nested?._indexHeads ?? 32 }
    public var indexHeadDim: Int { flat._indexHeadDim ?? nested?._indexHeadDim ?? 128 }
    public var indexTopK: Int { flat._indexTopK ?? nested?._indexTopK ?? 512 }
    public var candidateSourceLayerId: Int {
        flat._candidateSourceLayerId ?? nested?._candidateSourceLayerId ?? 20
    }
    public var candidateTopKBlocks: Int {
        flat._candidateTopKBlocks ?? nested?._candidateTopKBlocks ?? 2048
    }
    public var candidateBlockSize: Int {
        flat._candidateBlockSize ?? nested?._candidateBlockSize ?? 8
    }
    public var hcMultiplier: Int { flat._hcMultiplier ?? nested?._hcMultiplier ?? 4 }
    public var hcSinkhornIterations: Int {
        flat._hcSinkhornIterations ?? nested?._hcSinkhornIterations ?? 20
    }
    public var hcEps: Float { flat._hcEps ?? nested?._hcEps ?? 1e-6 }
    public var engramLayerIds: [Int] { flat._engramLayerIds ?? nested?._engramLayerIds ?? [1, 14] }
    public var engramNumEmbeddings: [Int] {
        flat._engramNumEmbeddings ?? nested?._engramNumEmbeddings
            ?? [384_006_168, 384_016_682]
    }
    public var engramMaxNgramSize: Int {
        flat._engramMaxNgramSize ?? nested?._engramMaxNgramSize ?? 4
    }
    public var engramVocabularySize: Int {
        flat._engramVocabularySize ?? nested?._engramVocabularySize ?? 16_000_000
    }
    public var engramHeads: Int { flat._engramHeads ?? nested?._engramHeads ?? 8 }
    public var engramHeadDim: Int { flat._engramHeadDim ?? nested?._engramHeadDim ?? 256 }
    public var engramPadTokenId: Int { flat._engramPadTokenId ?? nested?._engramPadTokenId ?? 2 }
    public var engramCompressedVocabularySize: Int {
        flat._engramCompressedVocabularySize ?? nested?._engramCompressedVocabularySize
            ?? 99092
    }
    public var imageTokenId: Int { flat._imageTokenId ?? nested?._imageTokenId ?? 129_264 }
    public var tieWordEmbeddings: Bool {
        flat._tieWordEmbeddings ?? nested?._tieWordEmbeddings ?? false
    }

    /// One ratio per backbone layer. Absent, it is the released pattern; a native
    /// checkpoint also carries ratios for prediction layers this port does not run, so
    /// the list is trimmed to the backbone.
    public var ropeScaling: [String: StringOrNumber]? {
        flat.ropeScaling ?? nested?.ropeScaling
    }
    public var eosTokenId: IntOrIntArray? { flat.eosTokenId ?? nested?.eosTokenId }

    public var compressRatios: [Int] {
        let declared = declaredCompressRatios
        let ratios =
            declared.isEmpty
            ? [0, 0] + Array(repeating: 2, count: 18) + Array(repeating: 1, count: 20)
            : declared
        return Array(ratios.prefix(hiddenLayers))
    }

    public let visionConfiguration: VisionConfiguration

    public struct VisionConfiguration: Decodable, Sendable {
        private let _numLayers: Int?
        private let _hiddenSize: Int?
        private let _numHeads: Int?
        private let _intermediateSize: Int?
        private let _patchSize: Int?
        private let _ropeTheta: Float?
        private let _downsampleRatio: Int?
        private let _minPixels: Int?
        private let _maxImageTokens: Int?
        public let maxWidthHeightRatio: Float?

        public var numLayers: Int { _numLayers ?? 32 }
        public var hiddenSize: Int { _hiddenSize ?? 1024 }
        public var numHeads: Int { _numHeads ?? 16 }
        public var intermediateSize: Int { _intermediateSize ?? 2816 }
        public var patchSize: Int { _patchSize ?? 14 }
        public var ropeTheta: Float { _ropeTheta ?? 10000 }
        public var downsampleRatio: Int { _downsampleRatio ?? 3 }
        public var minPixels: Int { _minPixels ?? 295_936 }
        public var maxImageTokens: Int { _maxImageTokens ?? 1024 }

        /// Upstream flattens `vision_config` onto the model config under a `vision_`
        /// prefix, renaming two keys on the way; both spellings are accepted here.
        enum CodingKeys: String, CodingKey {
            case _numLayers = "num_layers"
            case _hiddenSize = "hidden_size"
            case _numHeads = "num_heads"
            case _intermediateSize = "intermediate_size"
            case _patchSize = "patch_size"
            case _ropeTheta = "rope_theta"
            case _downsampleRatio = "downsample_ratio"
            case _minPixels = "min_pixels"
            case _maxImageTokens = "max_image_tokens"
            case maxWidthHeightRatio = "max_wh_ratio"
        }

        enum NestedCodingKeys: String, CodingKey {
            case numLayers = "num_hidden_layers"
            case numHeads = "num_attention_heads"
        }

        public init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let nested = try decoder.container(keyedBy: NestedCodingKeys.self)
            _numLayers =
                try container.decodeIfPresent(Int.self, forKey: ._numLayers)
                ?? nested.decodeIfPresent(Int.self, forKey: .numLayers)
            _numHeads =
                try container.decodeIfPresent(Int.self, forKey: ._numHeads)
                ?? nested.decodeIfPresent(Int.self, forKey: .numHeads)
            _hiddenSize = try container.decodeIfPresent(Int.self, forKey: ._hiddenSize)
            _intermediateSize = try container.decodeIfPresent(
                Int.self, forKey: ._intermediateSize)
            _patchSize = try container.decodeIfPresent(Int.self, forKey: ._patchSize)
            _ropeTheta = try container.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _downsampleRatio = try container.decodeIfPresent(
                Int.self, forKey: ._downsampleRatio)
            _minPixels = try container.decodeIfPresent(Int.self, forKey: ._minPixels)
            _maxImageTokens = try container.decodeIfPresent(
                Int.self, forKey: ._maxImageTokens)
            maxWidthHeightRatio = try container.decodeIfPresent(
                Float.self, forKey: .maxWidthHeightRatio)
        }
    }

    var declaredCompressRatios: [Int] {
        flat._compressRatios ?? nested?._compressRatios ?? []
    }

    enum OuterCodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
    }

    enum CodingKeys: String, CodingKey {
        case _modelType = "model_type"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _moeIntermediateSize = "moe_intermediate_size"
        case _hiddenLayers = "num_hidden_layers"
        case _attentionHeads = "num_attention_heads"
        case _kvHeads = "num_key_value_heads"
        case _headDim = "head_dim"
        case _qkRopeHeadDim = "qk_rope_head_dim"
        case _qLoraRank = "q_lora_rank"
        case _oLoraRank = "o_lora_rank"
        case _oGroups = "o_groups"
        case _swigluLimit = "swiglu_limit"
        case _rmsNormEps = "rms_norm_eps"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _ropeTheta = "rope_theta"
        case _nRoutedExperts = "n_routed_experts"
        case _nSharedExperts = "n_shared_experts"
        case _numExpertsPerTok = "num_experts_per_tok"
        case _scoringFunc = "scoring_func"
        case _topkMethod = "topk_method"
        case _normTopkProb = "norm_topk_prob"
        case _routedScalingFactor = "routed_scaling_factor"
        case _slidingWindow = "sliding_window"
        case _compressRatios = "compress_ratios"
        case _compressRopeTheta = "compress_rope_theta"
        case _kvSourceLayerIds = "kv_source_layer_ids"
        case _indexSourceLayerIds = "index_source_layer_ids"
        case _indexHeads = "index_n_heads"
        case _indexHeadDim = "index_head_dim"
        case _indexTopK = "index_topk"
        case _candidateSourceLayerId = "candidate_source_layer_id"
        case _candidateTopKBlocks = "candidate_topk_blocks"
        case _candidateBlockSize = "candidate_block_size"
        case _hcMultiplier = "hc_mult"
        case _hcSinkhornIterations = "hc_sinkhorn_iters"
        case _hcEps = "hc_eps"
        case _engramLayerIds = "engram_layer_ids"
        case _engramNumEmbeddings = "engram_num_embeddings"
        case _engramMaxNgramSize = "engram_max_ngram_size"
        case _engramVocabularySize = "engram_vocab_size"
        case _engramHeads = "engram_n_heads"
        case _engramHeadDim = "engram_head_dim"
        case _engramPadTokenId = "engram_pad_token_id"
        case _engramCompressedVocabularySize = "engram_compressed_vocab_size"
        case _imageTokenId = "image_token_id"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case eosTokenId = "eos_token_id"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: OuterCodingKeys.self)
        nested = try container.decodeIfPresent(TextFields.self, forKey: .textConfiguration)
        flat = try TextFields(from: decoder)
        visionConfiguration =
            try container.decodeIfPresent(
                VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
    }

    public func validateModelConfiguration() throws {
        let declared = declaredCompressRatios
        if !declared.isEmpty, declared.count < hiddenLayers {
            throw DeepSeekV41ConfigurationError.compressRatiosTooShort(
                declared.count, hiddenLayers)
        }
        guard engramLayerIds.count == engramNumEmbeddings.count else {
            throw DeepSeekV41ConfigurationError.engramTableCount(
                engramLayerIds.count, engramNumEmbeddings.count)
        }
        guard qkRopeHeadDim <= headDim, qkRopeHeadDim % 2 == 0 else {
            throw DeepSeekV41ConfigurationError.ropeWiderThanHead(qkRopeHeadDim, headDim)
        }
        for (name, width) in [
            ("hidden_size", hiddenSize), ("q_lora_rank", qLoraRank),
            ("head_dim", headDim), ("index_head_dim", indexHeadDim),
            ("moe_intermediate_size", moeIntermediateSize),
            ("o_groups * o_lora_rank", oGroups * oLoraRank),
        ] where width % 32 != 0 {
            throw DeepSeekV41ConfigurationError.notBlockAligned(name, width)
        }
        try DeepSeekV41EngramLayout(
            layerIds: engramLayerIds, maxNgramSize: engramMaxNgramSize,
            heads: engramHeads, headDim: engramHeadDim,
            vocabularySize: engramVocabularySize, numEmbeddings: engramNumEmbeddings
        )?.validate()
    }

    /// The text fields, decodable from either the top level or a nested `text_config`.
    /// Every one is optional so the two can be layered.
    struct TextFields: Decodable {
        let _modelType: String?
        let _vocabularySize: Int?
        let _hiddenSize: Int?
        let _moeIntermediateSize: Int?
        let _hiddenLayers: Int?
        let _attentionHeads: Int?
        let _kvHeads: Int?
        let _headDim: Int?
        let _qkRopeHeadDim: Int?
        let _qLoraRank: Int?
        let _oLoraRank: Int?
        let _oGroups: Int?
        let _swigluLimit: Float?
        let _rmsNormEps: Float?
        let _maxPositionEmbeddings: Int?
        let _ropeTheta: Float?
        let _nRoutedExperts: Int?
        let _nSharedExperts: Int?
        let _numExpertsPerTok: Int?
        let _scoringFunc: String?
        let _topkMethod: String?
        let _normTopkProb: Bool?
        let _routedScalingFactor: Float?
        let _slidingWindow: Int?
        let _compressRatios: [Int]?
        let _compressRopeTheta: Float?
        let _kvSourceLayerIds: [Int]?
        let _indexSourceLayerIds: [Int]?
        let _indexHeads: Int?
        let _indexHeadDim: Int?
        let _indexTopK: Int?
        let _candidateSourceLayerId: Int?
        let _candidateTopKBlocks: Int?
        let _candidateBlockSize: Int?
        let _hcMultiplier: Int?
        let _hcSinkhornIterations: Int?
        let _hcEps: Float?
        let _engramLayerIds: [Int]?
        let _engramNumEmbeddings: [Int]?
        let _engramMaxNgramSize: Int?
        let _engramVocabularySize: Int?
        let _engramHeads: Int?
        let _engramHeadDim: Int?
        let _engramPadTokenId: Int?
        let _engramCompressedVocabularySize: Int?
        let _imageTokenId: Int?
        let _tieWordEmbeddings: Bool?
        let ropeScaling: [String: StringOrNumber]?
        let eosTokenId: IntOrIntArray?

        typealias CodingKeys = DeepSeekV41Configuration.CodingKeys
    }
}

public enum DeepSeekV41ConfigurationError: LocalizedError {
    case compressRatiosTooShort(Int, Int)
    case engramTableCount(Int, Int)
    case notBlockAligned(String, Int)
    case ropeWiderThanHead(Int, Int)

    public var errorDescription: String? {
        switch self {
        case .compressRatiosTooShort(let given, let layers):
            "`compress_ratios` must cover every backbone layer, got \(given) for \(layers)"
        case .engramTableCount(let layers, let tables):
            "`engram_layer_ids` has \(layers) entries but `engram_num_embeddings` has \(tables)"
        case .notBlockAligned(let name, let width):
            "`\(name)` is \(width), which the 32-wide quantization blocks do not divide"
        case .ropeWiderThanHead(let rope, let head):
            "`qk_rope_head_dim` is \(rope), which must be even and at most `head_dim` \(head)"
        }
    }
}
