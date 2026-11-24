// The d1 prompt and set-attention contract are adapted from LiquidAI's
// System One implementation under the LFM Open License v1.0.
// See Documentation/Models/LFM3/D1-LICENSE.
import Foundation
import MLX
import MLXNN

struct LFM3DecisionConfiguration: Decodable, Sendable {
    let text: LFM3TextConfiguration
    let vision: LFM3VisionConfiguration
    let omni: LFM3Configuration?
    let setSize: Int
    let codes: Int
    let maxPixels: Int
    let imageToken: Int
    let videoToken: Int
    let projectorSize: Int
    let projectorBias: Bool
    let projectorEps: Float
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case text = "text_config"
        case vision = "vision_config"
        case attention = "d1_attention"
        case setSize = "d1_set_size"
        case codes = "d1_codes"
        case prompt = "d1_prompt"
        case maxPixels = "d1_vision_max_pixels"
        case imageToken = "image_token_id"
        case videoToken = "video_token_id"
        case projectorSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorEps = "projector_layer_norm_eps"
        case downsample = "downsample_factor"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .modelType)
        let architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        guard ["lfm3_vl", "lfm3_omni"].contains(type), architectures.contains("D1Model"),
            try c.decode(String.self, forKey: .attention) == "set"
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Expected an LFM3 d1 set-attention checkpoint")
        }
        text = try c.decode(LFM3TextConfiguration.self, forKey: .text)
        vision = try c.decode(LFM3VisionConfiguration.self, forKey: .vision)
        omni = type == "lfm3_omni" ? try LFM3Configuration(from: decoder) : nil
        setSize = try c.decodeIfPresent(Int.self, forKey: .setSize) ?? 4
        codes = try c.decodeIfPresent(Int.self, forKey: .codes) ?? 1
        maxPixels = try c.decodeIfPresent(Int.self, forKey: .maxPixels) ?? 1_048_576
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 125030
        videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 125029
        projectorSize = try c.decodeIfPresent(Int.self, forKey: .projectorSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorEps = try c.decodeIfPresent(Float.self, forKey: .projectorEps) ?? 1e-6
        let prompt = try c.decodeIfPresent(Int.self, forKey: .prompt) ?? 1
        let downsample = try c.decodeIfPresent(Int.self, forKey: .downsample) ?? 2
        guard (2 ... 128).contains(setSize), (1 ... 2).contains(codes), (1 ... 2).contains(prompt),
            maxPixels > 0, projectorSize > 0, text.tieEmbeddings, vision.mergeSize == downsample,
            [imageToken, videoToken].allSatisfy({ (0 ..< text.vocabularySize).contains($0) })
        else { throw ModelFactoryError.invalidConfiguration("Invalid d1 configuration") }
    }
}

/// d1 shares LFM3's encoders, but uses bidirectional set attention and a tied
/// option readout. It deliberately does not conform to LanguageModel.
final class LFM3DecisionModel: Module, MultimodalDecisionModel {
    let config: LFM3DecisionConfiguration
    let decisionTypes = Set(DecisionType.allCases)
    var calibration = LFM3DecisionCalibration()
    var specialTokens: [String] = []
    @ModuleInfo(key: "language_model") var language: LFM3Language
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3Vision
    @ModuleInfo(key: "multi_modal_projector") var visionProjector: LFM3Projector
    @ModuleInfo(key: "audio_tower") var audioTower: LFM3AudioEncoder?
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3Projector?
    // Preserve every trained Omni tensor for strict checkpoint loading. These
    // speech-output modules are never used by the decision path.
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding?
    @ModuleInfo(key: "depth_decoder") var depthDecoder: LFM3DepthDecoder?

    init(_ c: LFM3DecisionConfiguration) {
        config = c
        _language.wrappedValue = LFM3Language(c.text)
        _visionTower.wrappedValue = LFM3Vision(c.vision)
        _visionProjector.wrappedValue = LFM3Projector(
            input: c.vision.hiddenSize, hidden: c.projectorSize, output: c.text.hiddenSize,
            merge: c.vision.mergeSize, bias: c.projectorBias, eps: c.projectorEps,
            fp32Activation: false)
        if let omni = c.omni {
            _audioTower.wrappedValue = LFM3AudioEncoder(omni.audio)
            _audioProjector.wrappedValue = LFM3Projector(
                input: omni.audio.hiddenSize, hidden: c.text.hiddenSize, output: c.text.hiddenSize)
            _codebookEmbedding.wrappedValue = Embedding(
                embeddingCount: omni.numCodebooks * omni.codebookSize, dimensions: c.text.hiddenSize
            )
            _depthDecoder.wrappedValue = LFM3DepthDecoder(omni)
        }
    }

    static func isDecision(_ data: Data) -> Bool {
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return (object?["architectures"] as? [String])?.contains("D1Model") == true
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        LFM3Audio.sanitize(weights: weights, tied: config.text.tieEmbeddings)
    }

    static func setMask(groups: [Int], stateLength: Int) -> MLXArray {
        let positions = MLXArray(0 ..< groups.count)
        let group = MLXArray(groups)
        return (positions.expandedDimensions(axis: 1) .< stateLength)
            .|| (positions.expandedDimensions(axis: 0) .< stateLength)
            .|| (group.expandedDimensions(axis: 1) .== group.expandedDimensions(axis: 0))
    }

    func hidden(_ row: LFM3DecisionRow, features: [LFM3DecisionFeatures]) throws -> MLXArray {
        guard row.tokens.allSatisfy({ (0 ..< config.text.vocabularySize).contains($0) }) else {
            throw ModelExecutionError.invalidOptions("d1 input token is out of vocabulary")
        }
        var h = language.model.embedTokens!(MLXArray(row.tokens).expandedDimensions(axis: 0))
        for token in Set(features.map(\.token)) {
            let matching = features.filter { $0.token == token }.map(\.values)
            h = try LFM3Audio.merge(
                h, ids: row.tokens, features: concatenated(matching), token: token)
        }
        let full = Self.setMask(groups: row.groups, stateLength: row.stateLength)
        let sliding = createAttentionMask(h: h, cache: nil, windowSize: config.text.slidingWindow)
        for layer in language.model.layers {
            try Task.checkCancellation()
            h = layer(h, mask: layer.type == "full_attention" ? .array(full) : sliding)
        }
        return language.model.norm(h)[0]
    }

    func encode(_ parts: [LFM3DecisionMediaPart]) throws -> [LFM3DecisionFeatures] {
        try parts.map { part in
            let values: MLXArray
            if let grid = part.grid {
                values = try visionProjector(visionTower(part.values, grids: [grid]))
            } else {
                guard let audioTower, let audioProjector else {
                    throw ModelExecutionError.unsupportedOperation(
                        "d1-2B audio input; use d1-Omni-2B")
                }
                values = audioProjector(audioTower(part.values, lengths: [part.values.dim(-1)]))
            }
            eval(values)
            return .init(token: part.token, values: values)
        }
    }

    func predict(
        state: DecisionState, questions: [NamedDecisionQuestion], tokenizer: any Tokenizer,
        options: DecisionPredictionOptions
    ) throws -> ModelDecisionResult {
        try predict(
            state: state, questions: questions, tokenizer: tokenizer, options: options,
            media: .init())
    }

    func predict(
        state: DecisionState, questions: [NamedDecisionQuestion], tokenizer: any Tokenizer,
        options: DecisionPredictionOptions, media: LFM3DecisionPreparedMedia
    ) throws -> ModelDecisionResult {
        try DecisionValidation.validate(questions, supported: decisionTypes, options: options)
        let prompt = LFM3DecisionPrompt(
            tokenizer: tokenizer, codes: config.codes, specials: specialTokens)
        let layouts = try questions.map { try prompt.layout($0.question) }
        let prefix = try prompt.prefix(state: state, media: media.markup)
        guard options.modelOptions == nil || options.modelOptions is LFM3DecisionOptions else {
            throw ModelExecutionError.invalidOptions("d1 requires LFM3DecisionOptions")
        }
        let settings = options.modelOptions as? LFM3DecisionOptions ?? .init()
        let size = settings.setSize ?? config.setSize
        guard size > 0 else {
            throw ModelExecutionError.invalidOptions("d1 set size must be positive")
        }
        let features = try encode(media.parts)
        var answers: [String: DecisionAnswer] = [:]
        var count = 0
        for start in stride(from: 0, to: questions.count, by: size) {
            let end = min(questions.count, start + size)
            let row = try prompt.pack(
                prefix: prefix, layouts: Array(layouts[start ..< end]),
                maxStateTokens: options.maxStateTokens)
            let h = try hidden(row, features: features)
            for (local, index) in (start ..< end).enumerated() {
                let q = questions[index]
                let layout = layouts[index]
                let probabilities = layout.readouts.enumerated().map { slot, readout in
                    let tokens = Array(Set(readout.flatMap { $0 })).sorted()
                    let logits = matmul(
                        h[row.positions[local][slot]],
                        language.model.embedTokens!(MLXArray(tokens)).T
                    )
                    .asType(.float32).asArray(Float.self)
                    let scores = Dictionary(uniqueKeysWithValues: zip(tokens, logits))
                    let values = readout.map { $0.map { Double(scores[$0]!) }.max()! }
                    let temperature =
                        q.question.type == .multiLabel ? 1 : calibration.temperature(q.question)
                    return LFM3DecisionCalibration.softmax(values.map { $0 / temperature })
                }
                answers[q.name] = calibration.answer(
                    q.question, probabilities: probabilities,
                    scoreReadout: settings.scoreReadout)
            }
            count += row.tokens.count
        }
        return .init(answers: answers, inputTokenCount: count)
    }
}

extension LFM3DecisionModel {
    func preparePrediction(
        state: DecisionState, questions: [NamedDecisionQuestion],
        tokenizer: any Tokenizer, options: DecisionPredictionOptions,
        media: [DecisionMedia], isolation: isolated (any Actor)?
    ) async throws -> any PreparedDecisionRequest {
        let prepared = try await LFM3DecisionMediaProcessor(config: config, tokenizer: tokenizer)
            .prepare(media)
        return LFM3PreparedDecision(
            model: self, state: state, questions: questions,
            tokenizer: tokenizer, options: options, media: prepared)
    }
}
private struct LFM3PreparedDecision: PreparedDecisionRequest {
    let model: LFM3DecisionModel
    let state: DecisionState
    let questions: [NamedDecisionQuestion]
    let tokenizer: any Tokenizer
    let options: DecisionPredictionOptions
    let media: LFM3DecisionPreparedMedia
    func execute() throws -> ModelDecisionResult {
        try model.predict(
            state: state, questions: questions, tokenizer: tokenizer,
            options: options, media: media)
    }
}
