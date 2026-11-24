// Copyright © 2024 Apple Inc.

import Foundation

/// Immutable built-in architecture and processor constructors, with local overrides for tests.
struct ModelRegistry: Sendable {
    typealias ModelConstructor = @Sendable (Data) throws -> any LanguageModel
    typealias ProcessorConstructor =
        @Sendable (Data, any Tokenizer) throws -> any UserInputProcessor
    typealias DecisionConstructor = @Sendable (Data) throws -> any DecisionModel

    static let `default` = ModelRegistry()
    let models: [String: ModelConstructor]
    let processors: [String: ProcessorConstructor]
    let decisionModels: [String: DecisionConstructor]
    private let overriddenModelTypes: Set<String>

    init(
        models: [String: ModelConstructor] = [:], processors: [String: ProcessorConstructor] = [:],
        decisionModels: [String: DecisionConstructor] = [:]
    ) {
        self.models = Self.builtInModels.merging(models) { _, override in override }
        self.processors = Self.builtInProcessors.merging(processors) { _, override in override }
        self.overriddenModelTypes = Set(models.keys)
        let builtInDecisions: [String: DecisionConstructor] = [
            "lfm3_vl": { data in
                LFM3DecisionModel(
                    try JSONDecoder.json5().decode(LFM3DecisionConfiguration.self, from: data))
            },
            "lfm3_omni": { data in
                LFM3DecisionModel(
                    try JSONDecoder.json5().decode(LFM3DecisionConfiguration.self, from: data))
            },
            "decider2": { data in
                DeciderModel(try JSONDecoder.json5().decode(DeciderConfiguration.self, from: data))
            },
            "laya": { data in
                LayaModel(try JSONDecoder.json5().decode(LayaConfiguration.self, from: data))
            },
        ]
        self.decisionModels = builtInDecisions.merging(decisionModels) { _, override in override }
    }

    func createDecisionModel(configuration: Data, modelType: String) throws -> any DecisionModel {
        guard let constructor = decisionModels[modelType] else {
            throw ModelFactoryError.unsupportedModelType("\(modelType) (decision)")
        }
        return try constructor(configuration)
    }

    func contains(_ modelType: String) -> Bool { models[modelType] != nil }

    func createModel(
        configuration: Data, modelType: String, weightNames: Set<String>? = nil
    ) throws -> any LanguageModel {
        if MiniCPMOCRJevConfiguration.matches(configuration) {
            throw ModelExecutionError.unsupportedOperation(
                "MiniCPM-OCR Jev checkpoints use NativOCRScorer.score")
        }
        if LFM3DecisionModel.isDecision(configuration) {
            throw ModelExecutionError.unsupportedOperation(
                "d1 checkpoints use NativDecisionModel.predict, not text or speech generation")
        }
        if !overriddenModelTypes.contains(modelType), let weightNames, !weightNames.isEmpty,
            let textModel = Self.textOnlyModels[modelType],
            !weightNames.contains(where: Self.isMultimodalWeight)
        {
            return try textModel(configuration)
        }
        if modelType == "ernie5_5_moe", !overriddenModelTypes.contains(modelType), let weightNames {
            let config = try Self.decodeConfiguration(
                Ernie55Configuration.self, from: configuration)
            return Ernie55Model(
                config,
                retainEcho: Ernie55Checkpoint.containsEcho(weightNames, layers: config.layers))
        }
        guard let create = models[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try create(configuration)
    }

    static func needsWeightInspection(_ modelType: String) -> Bool {
        textOnlyModels[modelType] != nil || MuseSparkConfiguration.matches(modelType)
            || modelType == "ernie5_5_moe"
    }

    private static func isMultimodalWeight(_ name: String) -> Bool {
        name.split(separator: ".").contains {
            $0.contains("vision") || $0.contains("visual") || $0.contains("audio")
                || $0 == "multi_modal_projector" || $0 == "mm_projector"
        }
    }

    private static let textOnlyModels: [String: ModelConstructor] = [
        "mistral_large4": model(MistralLarge4Configuration.self, MistralLarge4TextModel.init),
        "gemma3": model(Gemma3TextModelConfiguration.self, Gemma3TextModel.init),
        "gemma3n": model(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
        "gemma4": model(Gemma4ModelConfiguration.self, Gemma4Model.init),
        "gemma4_unified": model(Gemma4ModelConfiguration.self, Gemma4Model.init),
        "mistral3": { data in
            let probe = try decodeConfiguration(Mistral4RoutingProbe.self, from: data)
            if probe.textConfig?.modelType == "mistral4" {
                return Mistral4(try decodeConfiguration(Mistral4Configuration.self, from: data))
            }
            return Mistral3TextModel(
                try decodeConfiguration(Mistral3TextConfiguration.self, from: data))
        },
        "qwen3_5": model(Qwen35ModelConfiguration.self, Qwen35Model.init),
        "qwen3_5_moe": model(Qwen35ModelConfiguration.self, Qwen35MoEModel.init),
    ]

    func createProcessor(
        configuration: Data, processorType: String, tokenizer: any Tokenizer
    ) throws -> any UserInputProcessor {
        guard let create = processors[processorType] else {
            throw ModelFactoryError.unsupportedProcessorType(processorType)
        }
        return try create(configuration, tokenizer)
    }

    static func decodeConfiguration<C: Decodable>(_ type: C.Type, from data: Data) throws -> C {
        let configuration = try JSONDecoder.json5().decode(type, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return configuration
    }

    static func model<C: Decodable & SendableMetatype, M: LanguageModel>(
        _ configuration: C.Type, _ construct: @escaping @Sendable (C) throws -> M
    ) -> ModelConstructor {
        { data in try construct(try decodeConfiguration(configuration, from: data)) }
    }

    static func processor<C: Decodable & SendableMetatype, P: UserInputProcessor>(
        _ configuration: C.Type, _ construct: @escaping @Sendable (C, any Tokenizer) -> P
    ) -> ProcessorConstructor {
        { data, tokenizer in
            construct(try decodeConfiguration(configuration, from: data), tokenizer)
        }
    }

    struct ProcessorPolicy: Sendable {
        let override: String?
        let fallback: String?
        let supportsMultimodalInput: Bool

        init(
            override: String? = nil, fallback: String? = nil, supportsMultimodalInput: Bool = false
        ) {
            self.override = override
            self.fallback = fallback
            self.supportsMultimodalInput = supportsMultimodalInput
        }
    }

    private struct TextRoutingProbe: Decodable {
        let architectures: [String]
        let isMultimodal: Bool

        private struct Tower: Decodable {
            let isPresent: Bool
            private struct Key: CodingKey {
                let stringValue: String
                var intValue: Int? { nil }
                init?(stringValue: String) { self.stringValue = stringValue }
                init?(intValue: Int) { return nil }
            }
            init(from decoder: any Decoder) throws {
                isPresent = !(try decoder.container(keyedBy: Key.self)).allKeys.isEmpty
            }
        }
        enum CodingKeys: String, CodingKey {
            case architectures
            case vision = "vision_config"
            case audio = "audio_config"
        }
        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
            let vision = try c.decodeIfPresent(Tower.self, forKey: .vision)?.isPresent ?? false
            let audio = try c.decodeIfPresent(Tower.self, forKey: .audio)?.isPresent ?? false
            isMultimodal =
                !architectures.contains { $0.hasSuffix("ForCausalLM") }
                && (vision || audio)
        }
    }

    private struct Gemma3nRoutingProbe: Decodable {
        let isMultimodal: Bool
        enum CodingKeys: String, CodingKey {
            case architectures
            case visionConfiguration = "vision_config"
        }
        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let architectures =
                try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
            if architectures.contains("Gemma3nForConditionalGeneration") {
                isMultimodal = true
            } else if architectures.contains("Gemma3nForCausalLM") {
                isMultimodal = false
            } else {
                isMultimodal =
                    try container.contains(.visionConfiguration)
                    && !container.decodeNil(forKey: .visionConfiguration)
            }
        }
    }

    private static let builtInModels: [String: ModelConstructor] = [
        "acereason": model(Qwen2Configuration.self, Qwen2Model.init),
        "afm7": model(AFM7Configuration.self, AFM7Model.init),
        "afmoe": model(AfMoEConfiguration.self, AfMoEModel.init),
        "apertus": model(ApertusConfiguration.self, ApertusModel.init),
        "aya_vision": model(AyaVisionConfiguration.self, AyaVision.init),
        "baichuan_m1": model(BaichuanM1Configuration.self, BaichuanM1Model.init),
        "bailing_hybrid": model(BailingHybridConfiguration.self, BailingHybrid.init),
        "bailing_moe": model(BailingMoeConfiguration.self, BailingMoeModel.init),
        "bailing_moe_linear": model(BailingMoeLinearConfiguration.self, BailingMoeLinearModel.init),
        "bitnet": model(BitnetConfiguration.self, BitnetModel.init),
        "bunny-llama": model(BunnyConfiguration.self, BunnyLlama.init),
        "cohere": model(CohereConfiguration.self, CohereModel.init),
        "cohere2": model(Cohere2Configuration.self, Cohere2Model.init),
        "cohere2_moe": model(Cohere2MoEConfiguration.self, Cohere2MoE.init),
        "cohere2_vision": model(AyaVisionConfiguration.self, AyaVision.init),
        "cohere2moe": model(Cohere2MoEConfiguration.self, Cohere2MoE.init),
        "cohere_compass": model(CohereCompassConfiguration.self, CohereCompass.init),
        "dbrx": model(DBRXConfiguration.self, DBRXModel.init),
        "deepseek": model(DeepseekConfiguration.self, DeepseekModel.init),
        "deepseek_v2": model(DeepseekV2Configuration.self, DeepseekV2Model.init),
        "deepseek_v3": model(DeepseekV3Configuration.self, DeepseekV3Model.init),
        "deepseek_v32": model(DeepseekV32Configuration.self, DeepseekV32Model.init),
        "deepseek_v4": model(DeepSeekV4Configuration.self, DeepSeekV4.init),
        "deepseek_v41": model(DeepSeekV41Configuration.self, DeepSeekV41.init),
        "deepseek_vl_v2": model(DeepSeekVL2Configuration.self, DeepSeekVL2.init),
        "deepseekocr": model(DeepSeekOCRConfiguration.self, DeepSeekOCR.init),
        "deepseekocr_2": model(DeepSeekOCRConfiguration.self, DeepSeekOCR2.init),
        "unlimited-ocr": model(DeepSeekOCRConfiguration.self, DeepSeekOCR.init),
        "unlimited_ocr": model(DeepSeekOCRConfiguration.self, DeepSeekOCR.init),
        "diffusion_gemma": model(DiffusionGemmaConfiguration.self, DiffusionGemma.init),
        "dots1": model(Dots1Configuration.self, Dots1Model.init),
        "dots_ocr": model(DotsOCRConfiguration.self, DotsOCR.init),
        "ernie4_5": model(Ernie45Configuration.self, Ernie45Model.init),
        "ernie4_5_moe": model(Ernie45MoEConfiguration.self, Ernie45MoEModel.init),
        "ernie4_5_moe_vl": model(Ernie4_5VLConfiguration.self, Ernie4_5VL.init),
        "ernie5_5_moe": model(Ernie55Configuration.self, Ernie55Model.init),
        "exaone": model(ExaoneConfiguration.self, ExaoneModel.init),
        "exaone4": model(Exaone4Configuration.self, Exaone4Model.init),
        "exaone_moe": model(ExaoneMoEConfiguration.self, ExaoneMoEModel.init),
        "falcon-perception": model(FalconPerceptionConfiguration.self, FalconPerception.init),
        "falcon_h1": model(FalconH1Configuration.self, FalconH1Model.init),
        "falcon_mamba": model(MambaConfiguration.self, MambaModel.init),
        "falcon_ocr": model(FalconOCRConfiguration.self, FalconOCR.init),
        "falcon_perception": model(FalconPerceptionConfiguration.self, FalconPerception.init),
        "fastvlm": model(FastVLMConfiguration.self, FastVLM.init),
        "florence2": model(FlorenceConfiguration.self, Florence.init),
        "flint": model(FlintConfiguration.self, Flint.init),
        "gemma": model(GemmaConfiguration.self, GemmaModel.init),
        "gemma2": model(Gemma2Configuration.self, Gemma2Model.init),
        "gemma3": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Gemma3(try decodeConfiguration(Gemma3Configuration.self, from: data))
            }
            return Gemma3TextModel(
                try decodeConfiguration(Gemma3TextModelConfiguration.self, from: data))
        },
        "gemma3_text": model(Gemma3TextModelConfiguration.self, Gemma3TextModel.init),
        "gemma3n": { data in
            let probe = try decodeConfiguration(Gemma3nRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Gemma3n(try decodeConfiguration(Gemma3nConfiguration.self, from: data))
            }
            return Gemma3nTextModel(
                config: try decodeConfiguration(Gemma3nTextConfiguration.self, from: data))
        },
        "gemma3n_text": model(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
        "gemma4": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Gemma4(try decodeConfiguration(Gemma4Configuration.self, from: data))
            }
            return Gemma4Model(try decodeConfiguration(Gemma4ModelConfiguration.self, from: data))
        },
        "gemma4_text": model(Gemma4TextModelConfiguration.self, Gemma4TextModel.init),
        "gemma4_unified": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Gemma4UnifiedAudioModel(
                    try decodeConfiguration(Gemma4UnifiedAudioModelConfiguration.self, from: data))
            }
            return Gemma4Model(try decodeConfiguration(Gemma4ModelConfiguration.self, from: data))
        },
        "glm": model(GLMConfiguration.self, GLMModel.init),
        "glm4": model(GLM4Configuration.self, GLM4Model.init),
        "glm4_moe": model(GLM4MoEConfiguration.self, GLM4MoEModel.init),
        "glm4_moe_lite": model(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
        "glm4v": model(GLM4VConfiguration.self, GLM4V.init),
        "glm4v_moe": model(GLM4VMoEConfiguration.self, GLM4VMoE.init),
        "glm5_next": model(Glm5NextConfiguration.self, Glm5Next.init),
        "glm_moe_dsa": model(GlmMoeDsaConfiguration.self, GlmMoeDsaModel.init),
        "glm_ocr": model(GlmOcrConfiguration.self, GlmOcr.init),
        "gpt2": model(GPT2Configuration.self, GPT2Model.init),
        "gpt_bigcode": model(GPTBigCodeConfiguration.self, GPTBigCodeModel.init),
        "gpt_neox": model(GPTNeoXConfiguration.self, GPTNeoXModel.init),
        "gpt_oss": model(GPTOSSConfiguration.self, GPTOSSModel.init),
        "granite": model(GraniteConfiguration.self, GraniteModel.init),
        "granite-vision": model(GraniteVisionConfiguration.self, GraniteVision.init),
        "granite4-vision": model(Granite4VisionConfiguration.self, Granite4Vision.init),
        "granite4_vision": model(Granite4VisionConfiguration.self, Granite4Vision.init),
        "granite_vision": model(GraniteVisionConfiguration.self, GraniteVision.init),
        "granitemoe": model(GraniteMoEConfiguration.self, GraniteMoEModel.init),
        "granitemoehybrid": model(GraniteMoeHybridConfiguration.self, GraniteMoeHybridModel.init),
        "helium": model(HeliumConfiguration.self, HeliumModel.init),
        "hrm_text": model(HRMTextConfiguration.self, HRMText.init),
        "hunyuan": model(HunyuanMoEConfiguration.self, HunyuanMoEModel.init),
        "hunyuan_v1_dense": model(HunyuanConfiguration.self, HunyuanModel.init),
        "hunyuan_vl": model(HunyuanVLConfiguration.self, HunyuanVL.init),
        "hy_v4": model(HyV4Configuration.self, HyV4Model.init),
        "idefics2": model(Idefics2Configuration.self, Idefics2.init),
        "idefics3": model(Idefics3Configuration.self, Idefics3.init),
        "indic_ocr": { data in Qwen35(try IndicOCR.configuration(from: data)) },
        "inkling": model(InklingTextConfiguration.self, InklingModel.init),
        "inkling2": model(Inkling2Configuration.self, Inkling2.init),
        "internlm2": model(InternLM2Configuration.self, InternLM2Model.init),
        "internlm3": model(Internlm3Configuration.self, Internlm3Model.init),
        "internvl": model(InternVLNextConfiguration.self, InternVLNext.init),
        "internvl_chat": model(InternVLConfiguration.self, InternVL.init),
        "iquestcoder": model(LlamaConfiguration.self, LlamaModel.init),
        "iquestloopcoder": model(IQuestLoopCoderConfiguration.self, IQuestLoopCoderModel.init),
        "jamba": model(JambaConfiguration.self, JambaModel.init),
        "jina_vlm": model(JinaVLMConfiguration.self, JinaVLM.init),
        "joyai_llm_flash": model(DeepseekV3Configuration.self, DeepseekV3Model.init),
        "jvlm": model(JinaVLMConfiguration.self, JinaVLM.init),
        "k2_horizon": model(K2HorizonConfiguration.self, K2HorizonModel.init),
        "kimi_k2": model(DeepseekV3Configuration.self, DeepseekV3Model.init),
        "kimi_k3": model(KimiK3Configuration.self, KimiK3.init),
        "kimi_linear": model(KimiLinearConfiguration.self, KimiLinearModel.init),
        "kimi_vl": model(KimiVLConfiguration.self, KimiVL.init),
        "klear": model(KlearConfiguration.self, KlearModel.init),
        "laguna": model(LagunaConfiguration.self, Laguna.init),
        "lfm2": model(LFM2Configuration.self, LFM2Model.init),
        "lfm3_audio": model(LFM3Configuration.self, LFM3Audio.init),
        "lfm3_omni": model(LFM3Configuration.self, LFM3Omni.init),
        "lfm2-vl": model(LFM2VLConfiguration.self, LFM2VL.init),
        "lfm2_moe": model(LFM2MoEConfiguration.self, LFM2MoEModel.init),
        "lfm2_vl": model(LFM2VLConfiguration.self, LFM2VL.init),
        "lille-130m": model(Lille130mConfiguration.self, Lille130mModel.init),
        "limite": model(LimiteConfiguration.self, LimiteModel.init),
        "llada2_moe": model(LLaDA2Configuration.self, LLaDA2.init),
        "llama": model(LlamaConfiguration.self, LlamaModel.init),
        "llama4": model(Llama4Configuration.self, Llama4.init),
        "llama4_text": model(Llama4TextConfiguration.self, Llama4TextModel.init),
        "llava": model(LLaVAConfiguration.self, LLaVA.init),
        "llava_next": model(LLaVANextConfiguration.self, LLaVANext.init),
        "llava_onevision": model(LLaVAOneVisionConfiguration.self, LLaVAOneVision.init),
        "llava_onevision2": model(MageVLConfiguration.self, MageVL.init),
        "llava_qwen2": model(FastVLMConfiguration.self, FastVLM.init),
        "llmjpvl": model(LLMjpVLConfiguration.self, LLMjpVL.init),
        "locateanything": model(LocateAnythingConfiguration.self, LocateAnything.init),
        "longcat_flash": model(LongcatFlashConfiguration.self, LongcatFlashModel.init),
        "longcat_flash_ngram": model(
            LongcatFlashNgramConfiguration.self, LongcatFlashNgramModel.init),
        "longcat_flash_sparse": model(
            LongcatFlashSparseConfiguration.self, LongcatFlashSparseModel.init),
        "mage_vl": model(MageVLConfiguration.self, MageVL.init),
        "mamba": model(MambaConfiguration.self, MambaModel.init),
        "mamba2": model(Mamba2Configuration.self, Mamba2Model.init),
        "mellum": model(MellumConfiguration.self, MellumModel.init),
        "mimo": model(MiMoConfiguration.self, MiMoModel.init),
        "mimo_v2": model(MiMoV2Configuration.self, MiMoV2.init),
        "mimo_v2_flash": model(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
        "minicpm": model(MiniCPMConfiguration.self, MiniCPMModel.init),
        "minicpm3": model(MiniCPM3Configuration.self, MiniCPM3Model.init),
        "minicpmo": model(MiniCPMOConfiguration.self, MiniCPMO.init),
        "minicpmv4_6": model(MiniCPMVConfiguration.self, MiniCPMV.init),
        "minicpm_ocr": model(MiniCPMVConfiguration.self, MiniCPMV.init),
        "oxr": model(OXRConfiguration.self, OXR.init),
        "minicpmv4_7": model(MiniCPMVConfiguration.self, MiniCPMV.init),
        "minimax": model(MiniMaxConfiguration.self, MiniMaxModel.init),
        "minimax_m2": model(MiniMaxConfiguration.self, MiniMaxModel.init),
        "minimax_m3": { data in
            MiniMaxM3Model(
                try decodeConfiguration(MiniMaxM3Configuration.self, from: data).textConfiguration)
        },
        "minimax_m3_vl": model(MiniMaxM3Configuration.self, MiniMaxM3.init),
        "ministral3": model(Ministral3Configuration.self, Ministral3Model.init),
        "mistral": model(LlamaConfiguration.self, LlamaModel.init),
        "mistral3": { data in
            let probe = try decodeConfiguration(Mistral4RoutingProbe.self, from: data)
            if probe.textConfig?.modelType == "mistral4" {
                return Mistral4(try decodeConfiguration(Mistral4Configuration.self, from: data))
            }
            let routing = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if !routing.isMultimodal {
                return Mistral3TextModel(
                    try decodeConfiguration(Mistral3TextConfiguration.self, from: data))
            }
            return Mistral3VLM(
                try decodeConfiguration(Mistral3VLMConfiguration.self, from: data))
        },
        "mistral4": model(Mistral4Configuration.self, Mistral4.init),
        "mistral_large3": model(MistralLarge3Configuration.self, MistralLarge3.init),
        "mistral_large4": { data in
            let config = try decodeConfiguration(MistralLarge4Configuration.self, from: data)
            if config.vision == nil { return MistralLarge4TextModel(config) }
            return try MistralLarge4(config)
        },
        "mistral_large4_text": model(MistralLarge4Configuration.self, MistralLarge4TextModel.init),
        "mixtral": model(MixtralConfiguration.self, MixtralModel.init),
        "mllama": model(MllamaConfiguration.self, Mllama.init),
        "molmo": model(MolmoConfiguration.self, Molmo.init),
        "molmo2": model(Molmo2Configuration.self, Molmo2.init),
        "molmo_point": model(MolmoPointConfiguration.self, MolmoPoint.init),
        "moondream1": model(MoondreamConfiguration.self, Moondream.init),
        "moondream2": model(MoondreamConfiguration.self, Moondream.init),
        "moondream3": model(Moondream3Configuration.self, Moondream3.init),
        "multi_modality": model(DeepSeekVLConfiguration.self, DeepSeekVL.init),
        "muse_glimmer": model(MuseGlimmerConfiguration.self, MuseGlimmer.init),
        "muse_spark": model(MuseSparkConfiguration.self, MuseSpark.init),
        "muse_spark_open": model(MuseSparkConfiguration.self, MuseSpark.init),
        "nanbeige": model(NanbeigeConfiguration.self, NanbeigeModel.init),
        "nanochat": model(NanoChatConfiguration.self, NanoChatModel.init),
        "nemotron": model(NemotronConfiguration.self, NemotronModel.init),
        "nemotron-nas": model(NemotronNASConfiguration.self, NemotronNASModel.init),
        "nemotron_h": model(NemotronHConfiguration.self, NemotronHModel.init),
        "nemotron_transcribe": model(NemotronTranscribeConfiguration.self, NemotronTranscribe.init),
        "nemotron_h_omni": model(NemotronOmniConfiguration.self, NemotronOmni.init),
        "nemotron_h_nano_omni": model(NemotronOmniConfiguration.self, NemotronOmni.init),
        "nemotron_labs_diffusion": model(
            NemotronLabsDiffusionConfiguration.self, NemotronLabsDiffusion.init),
        "nemotron_nas": model(NemotronNASConfiguration.self, NemotronNASModel.init),
        "nemotron_parse": { data in
            if NemotronParse35Configuration.matches(data) {
                return NemotronParse35(
                    try decodeConfiguration(NemotronParse35Configuration.self, from: data))
            }
            return NemotronParse(
                try decodeConfiguration(NemotronParseConfiguration.self, from: data))
        },
        "nemotron_parse_v3_5": model(NemotronParse35Configuration.self, NemotronParse35.init),
        "nemotronh_nano_omni_reasoning_v3": model(
            NemotronOmniConfiguration.self, NemotronOmni.init),
        "olmo": model(OlmoConfiguration.self, OlmoModel.init),
        "olmo2": model(Olmo2Configuration.self, Olmo2Model.init),
        "olmo3": model(Olmo3Configuration.self, Olmo3Model.init),
        "olmoe": model(OlmoEConfiguration.self, OlmoEModel.init),
        "openelm": model(OpenElmConfiguration.self, OpenELMModel.init),
        "paddleocr_vl": model(PaddleOCRVLConfiguration.self, PaddleOCRVL.init),
        "paligemma": model(PaliGemmaConfiguration.self, PaliGemma.init),
        "phi": model(PhiConfiguration.self, PhiModel.init),
        "phi-msft": model(PhixtralConfiguration.self, PhixtralModel.init),
        "phi3": model(Phi3Configuration.self, Phi3Model.init),
        "phi3_v": model(Phi3VisionConfiguration.self, Phi3Vision.init),
        "phi3small": model(Phi3SmallConfiguration.self, Phi3SmallModel.init),
        "phi4-siglip": model(Phi4SigLipConfiguration.self, Phi4SigLip.init),
        "phi4_siglip": model(Phi4SigLipConfiguration.self, Phi4SigLip.init),
        "phi4mm": model(Phi4MultimodalConfiguration.self, Phi4Multimodal.init),
        "phimoe": model(PhiMoEConfiguration.self, PhiMoEModel.init),
        "phixtral": model(PhixtralConfiguration.self, PhixtralModel.init),
        "pixtral": model(PixtralConfiguration.self, PixtralVLM.init),
        "plamo": model(PlamoConfiguration.self, PlamoModel.init),
        "plamo2": model(Plamo2Configuration.self, Plamo2Model.init),
        "prism_hadamard_qwen35": model(
            PrismHadamardQwen35Configuration.self, PrismHadamardQwen35.init),
        "qwen": model(QwenConfiguration.self, QwenModel.init),
        "qwen2": model(Qwen2Configuration.self, Qwen2Model.init),
        "qwen2_5_vl": model(Qwen25VLConfiguration.self, Qwen25VL.init),
        "qwen2_moe": model(Qwen2MoEConfiguration.self, Qwen2MoEModel.init),
        "qwen2_vl": model(Qwen2VLConfiguration.self, Qwen2VL.init),
        "qwen3": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            let configuration = try decodeConfiguration(Qwen3Configuration.self, from: data)
            if probe.architectures.contains("JinaForRanking") {
                return JinaRerankerModel(configuration)
            }
            return Qwen3Model(configuration)
        },
        "qwen3_5": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Qwen35(try decodeConfiguration(Qwen35Configuration.self, from: data))
            }
            return Qwen35Model(try decodeConfiguration(Qwen35ModelConfiguration.self, from: data))
        },
        "qwen3_5_moe": { data in
            let probe = try decodeConfiguration(TextRoutingProbe.self, from: data)
            if probe.isMultimodal {
                return Qwen35MoE(try decodeConfiguration(Qwen35Configuration.self, from: data))
            }
            return Qwen35MoEModel(
                try decodeConfiguration(Qwen35ModelConfiguration.self, from: data))
        },
        "qwen3_5_moe_text": model(Qwen35ModelConfiguration.self, Qwen35MoEModel.init),
        "qwen3_5_text": model(Qwen35TextConfiguration.self, Qwen35TextModel.init),
        "qwen3_moe": model(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
        "qwen3_next": model(Qwen3NextConfiguration.self, Qwen3NextModel.init),
        "qwen3_omni_moe": model(Qwen3OmniMoEConfiguration.self, Qwen3OmniMoE.init),
        "qwen3_vl": model(Qwen3VLConfiguration.self, Qwen3VL.init),
        "qwen3_vl_moe": model(Qwen3VLMoEConfiguration.self, Qwen3VLMoE.init),
        "qwen4_exp": model(Qwen4ExpConfiguration.self, Qwen4Exp.init),
        "raon1_mini": model(Raon1MiniConfiguration.self, Raon1Mini.init),
        "recurrent_gemma": model(RecurrentGemmaConfiguration.self, RecurrentGemmaModel.init),
        "ricursive": model(RippleConfiguration.self, Ripple.init),
        "rwkv7": model(RWKV7Configuration.self, RWKV7Model.init),
        "seed_oss": model(SeedOSSConfiguration.self, SeedOSSModel.init),
        "smollm3": model(SmolLM3Configuration.self, SmolLM3Model.init),
        "smolvlm": model(Idefics3Configuration.self, Idefics3.init),
        "solar_open": model(GLM4MoEConfiguration.self, GLM4MoEModel.init),
        "spark2_5": model(Spark2_5Configuration.self, Spark2_5Model.init),
        "stablelm": model(StableLMConfiguration.self, StableLMModel.init),
        "starcoder2": model(Starcoder2Configuration.self, Starcoder2Model.init),
        "step3p5": model(Step3p5Configuration.self, Step3p5Model.init),
        "step3p7": model(Step3p7Configuration.self, Step3p7.init),
        "telechat3": model(Telechat3Configuration.self, Telechat3Model.init),
        "youtu_llm": model(YoutuLLMConfiguration.self, YoutuLLMModel.init),
        "youtu_vl": model(YoutuVLConfiguration.self, YoutuVL.init),
        "z1t": model(Z1TConfiguration.self, Z1TModel.init),
        "zaya1_vl": model(Zaya1VLConfiguration.self, Zaya1VL.init),
    ]

    private static let builtInProcessors: [String: ProcessorConstructor] = [
        "NemotronTranscribeProcessor": processor(
            NemotronTranscribeConfiguration.self, NemotronTranscribeProcessor.init),
        "AyaVisionProcessor": processor(
            AyaVisionProcessorConfiguration.self, AyaVisionProcessor.init),
        "BunnyProcessor": processor(BunnyProcessorConfiguration.self, BunnyProcessor.init),
        "DeepseekOCR2Processor": processor(
            DeepSeekOCRProcessorConfiguration.self, DeepSeekOCR2Processor.init),
        "DeepseekOCRProcessor": processor(
            DeepSeekOCRProcessorConfiguration.self, DeepSeekOCRProcessor.init),
        "UnlimitedOCRProcessor": processor(
            DeepSeekOCRProcessorConfiguration.self, UnlimitedOCRProcessor.init),
        "DeepseekV41Processor": processor(DeepSeekV41Configuration.self, DeepSeekV41Processor.init),
        "DeepseekVLV2Processor": processor(
            DeepSeekVL2ProcessorConfiguration.self, DeepSeekVL2Processor.init),
        "DiffusionGemma4Processor": processor(
            DiffusionGemmaProcessorConfiguration.self, DiffusionGemmaProcessor.init),
        "DotsVLProcessor": processor(DotsOCRProcessorConfiguration.self, DotsOCRProcessor.init),
        "Ernie4_5_VLProcessor": processor(
            Ernie4_5VLProcessorConfiguration.self, Ernie4_5VLProcessor.init),
        "Ernie4_5VLProcessor": processor(
            Ernie4_5VLProcessorConfiguration.self, Ernie4_5VLProcessor.init),
        "FalconOCRProcessor": processor(
            FalconOCRProcessorConfiguration.self, FalconOCRProcessor.init),
        "FalconPerceptionProcessor": processor(
            FalconPerceptionProcessorConfiguration.self, FalconPerceptionProcessor.init),
        "FastVLMProcessor": processor(FastVLMProcessorConfiguration.self, FastVLMProcessor.init),
        "Florence2Processor": processor(
            FlorenceProcessorConfiguration.self, FlorenceProcessor.init),
        "Gemma3nProcessor": processor(Gemma3nProcessorConfiguration.self, Gemma3nProcessor.init),
        "Gemma3Processor": processor(Gemma3ProcessorConfiguration.self, Gemma3Processor.init),
        "Gemma4Processor": processor(Gemma4ProcessorConfiguration.self, Gemma4Processor.init),
        "Gemma4UnifiedProcessor": processor(
            Gemma4UnifiedAudioProcessorConfiguration.self, Gemma4UnifiedAudioProcessor.init),
        "Glm46VMoEProcessor": processor(GLM4VProcessorConfiguration.self, GLM4VProcessor.init),
        "Glm46VProcessor": processor(GLM4VProcessorConfiguration.self, GLM4VProcessor.init),
        "Glm4VMoEProcessor": processor(GLM4VProcessorConfiguration.self, GLM4VProcessor.init),
        "Glm4vProcessor": processor(GLM4VProcessorConfiguration.self, GLM4VProcessor.init),
        "Granite4VisionProcessor": processor(
            Granite4VisionProcessorConfiguration.self, Granite4VisionProcessor.init),
        "GraniteVisionProcessor": processor(
            GraniteVisionProcessorConfiguration.self, GraniteVisionProcessor.init),
        "GridImageProcessor": processor(
            GridImageProcessorConfiguration.self, GridImageProcessor.init),
        "Idefics2Processor": processor(Idefics2ProcessorConfiguration.self, Idefics2Processor.init),
        "Idefics3Processor": processor(Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
        "InternVLChatProcessor": processor(
            InternVLProcessorConfiguration.self, InternVLProcessor.init),
        "InternVLProcessor": processor(InternVLProcessorConfiguration.self, InternVLProcessor.init),
        "JinaVLMProcessor": processor(JinaVLMProcessorConfiguration.self, JinaVLMProcessor.init),
        "KimiK3Processor": processor(KimiK3ProcessorConfiguration.self, KimiK3Processor.init),
        "KimiVLProcessor": processor(KimiVLProcessorConfiguration.self, KimiVLProcessor.init),
        "Lfm2VlProcessor": processor(LFM2VLProcessorConfiguration.self, LFM2VLProcessor.init),
        "Llama4Processor": processor(Llama4ProcessorConfiguration.self, Llama4Processor.init),
        "LlavaNextProcessor": processor(
            LLaVANextProcessorConfiguration.self, LLaVANextProcessor.init),
        "LlavaOnevisionProcessor": processor(
            LLaVAOneVisionProcessorConfiguration.self, LLaVAOneVisionProcessor.init),
        "LlavaProcessor": processor(LLaVAProcessorConfiguration.self, LLaVAProcessor.init),
        "LLMjpVLProcessor": processor(LLMjpVLProcessorConfiguration.self, LLMjpVLProcessor.init),
        "LocateAnythingProcessor": processor(
            LocateAnythingProcessorConfiguration.self, LocateAnythingProcessor.init),
        "MageVLProcessor": processor(MageVLProcessorConfiguration.self, MageVLProcessor.init),
        "MiniCPMOProcessor": processor(MiniCPMOProcessorConfiguration.self, MiniCPMOProcessor.init),
        "MiniCPMV4_6Processor": processor(
            MiniCPMVProcessorConfiguration.self, MiniCPMVProcessor.init),
        "MiniCPMV4_7Processor": processor(
            MiniCPMVProcessorConfiguration.self, MiniCPMVProcessor.init),
        "MiniCPMVProcessor": processor(MiniCPMVProcessorConfiguration.self, MiniCPMVProcessor.init),
        "MiniCPMOcrProcessor": processor(
            MiniCPMVProcessorConfiguration.self, MiniCPMVProcessor.init),
        "Mistral3Processor": processor(
            Mistral3VLMProcessorConfiguration.self, Mistral3VLMProcessor.init),
        "MllamaProcessor": processor(MllamaProcessorConfiguration.self, MllamaProcessor.init),
        "Molmo2Processor": processor(Molmo2ProcessorConfiguration.self, Molmo2Processor.init),
        "MolmoProcessor": processor(MolmoProcessorConfiguration.self, MolmoProcessor.init),
        "Moondream3Processor": processor(
            Moondream3ProcessorConfiguration.self, Moondream3Processor.init),
        "MoondreamProcessor": processor(
            MoondreamProcessorConfiguration.self, MoondreamProcessor.init),
        "MuseGlimmerProcessor": processor(
            MuseGlimmerProcessorConfiguration.self, MuseGlimmerProcessor.init),
        "Inkling2Processor": processor(
            Inkling2ProcessorConfiguration.self, Inkling2Processor.init),
        "MuseSparkProcessor": processor(
            MuseSparkProcessorConfiguration.self, MuseSparkProcessor.init),
        "NemotronOmniProcessor": processor(
            NemotronOmniProcessorConfiguration.self, NemotronOmniProcessor.init),
        "NemotronParseProcessor": processor(
            NemotronParseProcessorConfiguration.self, NemotronParseProcessor.init),
        "NemotronParse35Processor": processor(
            NemotronParseProcessorConfiguration.self, NemotronParse35Processor.init),
        "PaddleOCRVLProcessor": processor(
            PaddleOCRVLProcessorConfiguration.self, PaddleOCRVLProcessor.init),
        "OXRProcessor": processor(OXRProcessorConfiguration.self, OXRProcessor.init),
        "PaliGemmaProcessor": processor(
            PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
        "Phi3VProcessor": processor(
            Phi3VisionProcessorConfiguration.self, Phi3VisionProcessor.init),
        "Phi4MMProcessor": processor(
            Phi4MultimodalProcessorConfiguration.self, Phi4MultimodalProcessor.init),
        "Phi4SigLipProcessor": processor(
            Phi4SigLipProcessorConfiguration.self, Phi4SigLipProcessor.init),
        "PixtralProcessor": processor(PixtralProcessorConfiguration.self, PixtralProcessor.init),
        "Qwen2_5_VLProcessor": processor(
            Qwen25VLProcessorConfiguration.self, Qwen25VLProcessor.init),
        "Qwen2VLProcessor": processor(Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
        "Qwen3OmniMoEProcessor": processor(
            Qwen3OmniMoEProcessorConfiguration.self, Qwen3OmniMoEProcessor.init),
        "Lfm3AudioProcessor": processor(LFM3ProcessorConfiguration.self, LFM3Processor.init),
        "Lfm3OmniProcessor": processor(LFM3ProcessorConfiguration.self, LFM3Processor.init),
        "Qwen3OmniMoeProcessor": processor(
            Qwen3OmniMoEProcessorConfiguration.self, Qwen3OmniMoEProcessor.init),
        "Qwen3VLProcessor": processor(Qwen3VLProcessorConfiguration.self, Qwen3VLProcessor.init),
        "SmolVLMProcessor": processor(SmolVLMProcessorConfiguration.self, SmolVLMProcessor.init),
        "Step3VLProcessor": processor(Step3VLProcessorConfiguration.self, Step3VLProcessor.init),
        "TextProcessor": { data, tokenizer in
            let configuration =
                (try? decodeConfiguration(TextProcessorConfiguration.self, from: data))
                ?? TextProcessorConfiguration()
            return TextProcessor(configuration, tokenizer: tokenizer)
        },
        "VLChatProcessor": processor(
            DeepSeekVLProcessorConfiguration.self, DeepSeekVLProcessor.init),
        "YoutuVLProcessor": processor(YoutuVLProcessorConfiguration.self, YoutuVLProcessor.init),
        "Zaya1VLProcessor": processor(Zaya1VLProcessorConfiguration.self, Zaya1VLProcessor.init),
    ]

    static let processorPolicies: [String: ProcessorPolicy] = [
        "nemotron_transcribe": .init(
            override: "NemotronTranscribeProcessor", fallback: "NemotronTranscribeProcessor",
            supportsMultimodalInput: true),
        "bunny-llama": .init(fallback: "BunnyProcessor"),
        "cohere2_moe": .init(fallback: "TextProcessor"),
        "cohere2moe": .init(fallback: "TextProcessor"),
        "cohere_compass": .init(
            override: "GridImageProcessor", fallback: "GridImageProcessor",
            supportsMultimodalInput: true),
        "deepseek_v4": .init(fallback: "TextProcessor"),
        "deepseek_v41": .init(
            override: "DeepseekV41Processor", fallback: "DeepseekV41Processor",
            supportsMultimodalInput: true),
        "deepseek_vl_v2": .init(override: "DeepseekVLV2Processor"),
        "deepseekocr": .init(override: "DeepseekOCRProcessor"),
        "deepseekocr_2": .init(override: "DeepseekOCR2Processor"),
        "unlimited-ocr": .init(fallback: "UnlimitedOCRProcessor", supportsMultimodalInput: true),
        "unlimited_ocr": .init(fallback: "UnlimitedOCRProcessor", supportsMultimodalInput: true),
        "diffusion_gemma": .init(
            override: "DiffusionGemma4Processor", fallback: "TextProcessor",
            supportsMultimodalInput: true),
        "ernie4_5_moe_vl": .init(
            override: "Ernie4_5_VLProcessor", fallback: "Ernie4_5_VLProcessor"),
        "falcon-perception": .init(
            override: "FalconPerceptionProcessor", fallback: "FalconPerceptionProcessor"),
        "falcon_ocr": .init(fallback: "FalconOCRProcessor"),
        "falcon_perception": .init(
            override: "FalconPerceptionProcessor", fallback: "FalconPerceptionProcessor"),
        "gemma3n": .init(override: "Gemma3nProcessor", fallback: "Gemma3nProcessor"),
        "gemma4_unified": .init(override: "Gemma4UnifiedProcessor"),
        "glm4v": .init(override: "Glm4vProcessor", fallback: "Glm4vProcessor"),
        "glm4v_moe": .init(override: "Glm46VMoEProcessor", fallback: "Glm46VMoEProcessor"),
        "glm5_next": .init(
            override: "GridImageProcessor", fallback: "GridImageProcessor",
            supportsMultimodalInput: true),
        "granite-vision": .init(
            override: "GraniteVisionProcessor", fallback: "GraniteVisionProcessor",
            supportsMultimodalInput: true),
        "granite4-vision": .init(
            override: "Granite4VisionProcessor", fallback: "Granite4VisionProcessor",
            supportsMultimodalInput: true),
        "granite4_vision": .init(override: "Granite4VisionProcessor"),
        "granite_vision": .init(override: "GraniteVisionProcessor"),
        "hrm_text": .init(fallback: "TextProcessor"),
        "hunyuan_vl": .init(
            override: "GridImageProcessor", fallback: "GridImageProcessor",
            supportsMultimodalInput: true),
        "indic_ocr": .init(override: "Qwen3VLProcessor", fallback: "Qwen3VLProcessor"),
        "internvl": .init(
            override: "InternVLProcessor", fallback: "InternVLProcessor",
            supportsMultimodalInput: true),
        "jina_vlm": .init(override: "JinaVLMProcessor", fallback: "JinaVLMProcessor"),
        "jvlm": .init(override: "JinaVLMProcessor", fallback: "JinaVLMProcessor"),
        "kimi_k3": .init(
            override: "KimiK3Processor", fallback: "KimiK3Processor", supportsMultimodalInput: true),
        "laguna": .init(fallback: "TextProcessor"),
        "llada2_moe": .init(fallback: "TextProcessor"),
        "llama4": .init(override: "Llama4Processor", fallback: "Llama4Processor"),
        "llava_onevision": .init(
            override: "LlavaOnevisionProcessor", fallback: "LlavaOnevisionProcessor",
            supportsMultimodalInput: true),
        "llava_onevision2": .init(
            override: "MageVLProcessor", fallback: "MageVLProcessor", supportsMultimodalInput: true),
        "llmjpvl": .init(override: "LLMjpVLProcessor", fallback: "LLMjpVLProcessor"),
        "mage_vl": .init(
            override: "MageVLProcessor", fallback: "MageVLProcessor", supportsMultimodalInput: true),
        "mimo_v2": .init(
            override: "Qwen2_5_VLProcessor", fallback: "Qwen2_5_VLProcessor",
            supportsMultimodalInput: true),
        "inkling2": .init(fallback: "Inkling2Processor", supportsMultimodalInput: true),
        "muse_spark": .init(fallback: "MuseSparkProcessor", supportsMultimodalInput: true),
        "muse_spark_open": .init(fallback: "MuseSparkProcessor", supportsMultimodalInput: true),
        "minicpmv4_7": .init(override: "MiniCPMV4_7Processor", fallback: "MiniCPMV4_7Processor"),
        "minicpm_ocr": .init(
            override: "MiniCPMOcrProcessor", fallback: "MiniCPMOcrProcessor",
            supportsMultimodalInput: true),
        "oxr": .init(
            override: "OXRProcessor", fallback: "OXRProcessor", supportsMultimodalInput: true),
        "minicpmo": .init(override: "MiniCPMOProcessor", fallback: "MiniCPMOProcessor"),
        "minimax_m3": .init(fallback: "TextProcessor"),
        "minimax_m3_vl": .init(
            override: "GridImageProcessor", fallback: "GridImageProcessor",
            supportsMultimodalInput: true),
        "mistral3": .init(override: "Mistral3Processor"),
        "mistral_large3": .init(override: "Mistral3Processor", fallback: "Mistral3Processor"),
        "mistral_large4": .init(override: "Mistral3Processor", fallback: "Mistral3Processor"),
        "moondream1": .init(override: "MoondreamProcessor", fallback: "MoondreamProcessor"),
        "moondream2": .init(override: "MoondreamProcessor", fallback: "MoondreamProcessor"),
        "moondream3": .init(override: "Moondream3Processor", fallback: "Moondream3Processor"),
        "nemotron_labs_diffusion": .init(fallback: "TextProcessor"),
        "nemotron_parse": .init(
            override: "NemotronParseProcessor", fallback: "NemotronParseProcessor",
            supportsMultimodalInput: true),
        "nemotron_parse_v3_5": .init(
            override: "NemotronParse35Processor", fallback: "NemotronParse35Processor",
            supportsMultimodalInput: true),
        "nemotron_h_omni": .init(
            override: "NemotronOmniProcessor", fallback: "NemotronOmniProcessor",
            supportsMultimodalInput: true),
        "nemotronh_nano_omni_reasoning_v3": .init(
            override: "NemotronOmniProcessor", fallback: "NemotronOmniProcessor",
            supportsMultimodalInput: true),
        "phi4-siglip": .init(
            override: "Phi4SigLipProcessor", fallback: "Phi4SigLipProcessor",
            supportsMultimodalInput: true),
        "phi4_siglip": .init(
            override: "Phi4SigLipProcessor", fallback: "Phi4SigLipProcessor",
            supportsMultimodalInput: true),
        "qwen3": .init(fallback: "TextProcessor"),
        "qwen3_omni_moe": .init(
            override: "Qwen3OmniMoeProcessor", fallback: "Qwen3OmniMoeProcessor"),
        "lfm3_audio": .init(override: "Lfm3AudioProcessor", fallback: "Lfm3AudioProcessor"),
        "lfm3_omni": .init(override: "Lfm3OmniProcessor", fallback: "Lfm3OmniProcessor"),
        "qwen4_exp": .init(
            override: "Qwen3VLProcessor", fallback: "Qwen3VLProcessor",
            supportsMultimodalInput: true),
        "step3p7": .init(override: "Step3VLProcessor", fallback: "Step3VLProcessor"),
        "youtu_vl": .init(override: "YoutuVLProcessor", fallback: "YoutuVLProcessor"),
        "zaya1_vl": .init(override: "Zaya1VLProcessor", fallback: "Zaya1VLProcessor"),
    ]
}

extension ModelRegistry {
    typealias EmbeddingConstructor =
        @Sendable (Data, EmbeddingPooling?) throws -> any EmbeddingModel
    private static let embeddingModels: [String: EmbeddingConstructor] = [
        "bert": bertEmbedding,
        "embedding_gemma2": embeddingGemma2,
        "embedding_gemma2_text": embeddingGemma2,
        "colidefics3": { data, _ in try ColIdefics3EmbeddingModel(configuration: data) },
        "colqwen2_5": { data, _ in
            try ColQwen25EmbeddingModel(
                decodeConfiguration(Qwen25VLConfiguration.self, from: data),
                embeddingDimension: decodeConfiguration(EmbeddingOptions.self, from: data)
                    .embeddingDimension ?? 128)
        },
        "gemma3_embedding": gemmaEmbedding,
        "gemma3_text": gemmaEmbedding,
        "lfm2": lfmEmbedding,
        "lfm2_colbert": { data, pooling in
            try LFM2EmbeddingModel(
                decodeConfiguration(LFM2Configuration.self, from: data),
                pooling: pooling ?? .cls,
                embeddingDimension: decodeConfiguration(EmbeddingOptions.self, from: data)
                    .embeddingDimension ?? 128)
        },
        "lfm2_embedding": lfmEmbedding,
        "llama_bidirec": { data, _ in
            try LlamaBidirectionalEmbeddingModel(
                decodeConfiguration(LlamaConfiguration.self, from: data))
        },
        "llama_nemotron_vl": nemotronEmbedding,
        "llama_nemotron_vl_embedding": nemotronEmbedding,
        "ministral3": ministralEmbedding,
        "ministral3_embedding": ministralEmbedding,
        "modernbert": { data, pooling in
            try ModernBertEmbeddingModel(
                decodeConfiguration(ModernBertConfiguration.self, from: data),
                pooling: pooling ?? .mean)
        },
        "qwen3": qwenEmbedding,
        "qwen3_embedding": qwenEmbedding,
        "qwen3_vl": qwenVLEmbedding,
        "qwen3_vl_embedding": qwenVLEmbedding,
        "siglip": { data, _ in try SiglipEmbeddingModel(configuration: data) },
        "xlm-roberta": bertEmbedding,
        "xlm_roberta": bertEmbedding,
    ]

    private struct EmbeddingOptions: Decodable {
        let normalize: Bool?
        let embeddingDimension: Int?
        enum CodingKeys: String, CodingKey {
            case normalize
            case embeddingDimension = "embedding_dim"
        }
    }
    private static func qwenEmbedding(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try Qwen3EmbeddingModel(
            decodeConfiguration(Qwen3Configuration.self, from: data), pooling: pooling ?? .lasttoken
        )
    }
    private static func embeddingGemma2(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        guard pooling == nil || pooling == .mean else {
            throw ModelExecutionError.invalidOptions("EmbeddingGemma 2 requires mean pooling")
        }
        return try EmbeddingGemma2Model(
            decodeConfiguration(EmbeddingGemma2Configuration.self, from: data))
    }
    private static func gemmaEmbedding(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try Gemma3EmbeddingModel(
            decodeConfiguration(Gemma3TextModelConfiguration.self, from: data),
            pooling: pooling ?? .mean)
    }
    private static func lfmEmbedding(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try LFM2EmbeddingModel(
            decodeConfiguration(LFM2Configuration.self, from: data), pooling: pooling ?? .cls)
    }
    private static func ministralEmbedding(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try Ministral3EmbeddingModel(
            decodeConfiguration(Ministral3Configuration.self, from: data), pooling: pooling ?? .mean
        )
    }
    private static func bertEmbedding(_ data: Data, _ pooling: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try BertEmbeddingModel(
            decodeConfiguration(BertConfiguration.self, from: data), pooling: pooling ?? .mean)
    }
    private static func qwenVLEmbedding(_ data: Data, _: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try Qwen3VLEmbeddingModel(
            decodeConfiguration(Qwen3VLConfiguration.self, from: data),
            normalize: decodeConfiguration(EmbeddingOptions.self, from: data).normalize ?? true)
    }
    private static func nemotronEmbedding(_ data: Data, _: EmbeddingPooling?) throws
        -> any EmbeddingModel
    {
        try LlamaNemotronEmbeddingModel(
            decodeConfiguration(LlamaNemotronEmbeddingConfiguration.self, from: data))
    }

    func createEmbeddingModel(
        configuration: Data, modelType: String, pooling: EmbeddingPooling? = nil
    )
        throws -> any EmbeddingModel
    {
        guard let constructor = Self.embeddingModels[modelType] else {
            throw ModelFactoryError.unsupportedModelType("\(modelType) (embedding)")
        }
        return try constructor(configuration, pooling)
    }
}
