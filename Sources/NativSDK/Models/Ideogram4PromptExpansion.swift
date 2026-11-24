import Foundation

public struct Ideogram4GenerationOptions: Sendable {
    public var autoJSONCaption: Bool
    public var sampler: Ideogram4SamplerPreset
    public var guidanceSchedule: [Float]?
    var legacyPromptExpansionModel: ModelSource?

    @available(*, deprecated, message: "Use ImageGenerationParameters.promptExpansion.")
    public var promptExpansionModel: ModelSource? {
        get { legacyPromptExpansionModel }
        set { legacyPromptExpansionModel = newValue }
    }

    public init(
        autoJSONCaption: Bool = true,
        sampler: Ideogram4SamplerPreset = .v4Default20, guidanceSchedule: [Float]? = nil
    ) {
        self.autoJSONCaption = autoJSONCaption
        self.sampler = sampler
        self.guidanceSchedule = guidanceSchedule
    }

    @available(*, deprecated, message: "Use ImageGenerationParameters.promptExpansion.")
    public init(
        autoJSONCaption: Bool = true, promptExpansionModel: ModelSource?,
        sampler: Ideogram4SamplerPreset = .v4Default20, guidanceSchedule: [Float]? = nil
    ) {
        self.init(
            autoJSONCaption: autoJSONCaption, sampler: sampler, guidanceSchedule: guidanceSchedule)
        legacyPromptExpansionModel = promptExpansionModel
    }
}

extension Ideogram4Prompting {
    static let expansionSchema =
        #"{"type":"object","properties":{"high_level_description":{"type":"string","minLength":1},"style_description":{"anyOf":[{"type":"object","properties":{"aesthetics":{"type":"string","minLength":1},"lighting":{"type":"string","minLength":1},"photo":{"type":"string","minLength":1},"medium":{"type":"string","minLength":1},"color_palette":{"type":"array","items":{"type":"string","pattern":"^#[0-9A-F]{6}$"},"maxItems":16}},"required":["aesthetics","lighting","photo","medium"],"additionalProperties":false},{"type":"object","properties":{"aesthetics":{"type":"string","minLength":1},"lighting":{"type":"string","minLength":1},"medium":{"type":"string","minLength":1},"art_style":{"type":"string","minLength":1},"color_palette":{"type":"array","items":{"type":"string","pattern":"^#[0-9A-F]{6}$"},"maxItems":16}},"required":["aesthetics","lighting","medium","art_style"],"additionalProperties":false}]},"compositional_deconstruction":{"type":"object","properties":{"background":{"type":"string","minLength":1},"elements":{"type":"array","items":{"anyOf":[{"type":"object","properties":{"type":{"type":"string","enum":["obj"]},"bbox":{"type":"array","items":{"type":"integer","minimum":0,"maximum":1000},"minItems":4,"maxItems":4},"desc":{"type":"string","minLength":1},"color_palette":{"type":"array","items":{"type":"string","pattern":"^#[0-9A-F]{6}$"},"maxItems":5}},"required":["type","desc"],"additionalProperties":false},{"type":"object","properties":{"type":{"type":"string","enum":["text"]},"bbox":{"type":"array","items":{"type":"integer","minimum":0,"maximum":1000},"minItems":4,"maxItems":4},"text":{"type":"string"},"desc":{"type":"string","minLength":1},"color_palette":{"type":"array","items":{"type":"string","pattern":"^#[0-9A-F]{6}$"},"maxItems":5}},"required":["type","text","desc"],"additionalProperties":false}]}}},"required":["background","elements"],"additionalProperties":false}},"required":["compositional_deconstruction"],"additionalProperties":false}"#
    static let expansionInstructions = """
        You prepare structured JSON captions for Ideogram 4 image generation. Return
        only JSON matching the provided schema. Preserve the user's intent, requested
        wording, and constraints while making the visual description more specific and
        useful to the image model.

        Always include a concrete high_level_description and
        compositional_deconstruction. Write descriptions as observations of the desired
        image, never as commands or as a copy of the user's request. The background must
        describe the actual scene, not a generic placeholder.

        Use one obj element for each explicitly named visual subject. Use one text
        element for every quoted string or other visible wording the user requests.
        Copy each text field verbatim, including capitalization, punctuation, line
        breaks, and non-ASCII characters. Do not hide requested lettering inside an obj
        description.

        Bounding boxes are optional. Include them only when useful for layout, using
        integer normalized [0, 1000] coordinates as [y_min, x_min, y_max, x_max] with
        y_min < y_max and x_min < x_max. If style_description is included, use exactly
        one of photo or art_style. Use only uppercase #RRGGBB values in color palettes.
        """
    static func expanded(
        _ prompt: String, model source: ModelSource, width: Int, height: Int,
        seed: UInt64?, policy: ModelExecutionPolicy
    ) async throws -> String {
        let model = try await NativModel.loadModel(from: source, policy: policy)
        let input = UserInput(chat: [
            .system(expansionInstructions + "\nJSON schema: " + expansionSchema),
            .user(
                "Convert this prompt into an Ideogram 4 JSON caption. Target dimensions: \(width)x\(height).\n"
                    + prompt),
        ])
        let request = model.generate(
            input: input,
            options: .init(
                sampling: .init(temperature: 0.6, seed: seed), output: .init(maxTokens: 2048)))
        var text = ""
        for try await event in request { if case .chunk(let chunk) = event { text += chunk } }
        _ = try await request.metrics()
        return try validateExpandedCaption(text)
    }
    static func validateExpandedCaption(_ text: String) throws -> String {
        let data = Data(text.trimmingCharacters(in: .whitespacesAndNewlines).utf8)
        let value = try JSONDecoder().decode(JSONValue.self, from: data)
        let schema = try JSONSerialization.jsonObject(with: Data(expansionSchema.utf8))
        guard case .valid = ToolSchemaValidator.validate(value, against: schema) else {
            throw ModelExecutionError.invalidOptions(
                "The prompt expansion model did not return a valid Ideogram caption")
        }
        let object = try JSONSerialization.jsonObject(with: data)
        return String(
            decoding: try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
            as: UTF8.self)
    }
}
