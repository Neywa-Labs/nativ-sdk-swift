import Foundation
import MLX

/// Byte-level grammar for MiniMax's two-field caption object. The thinking
/// span uses the language model's own stopping delimiter.
struct MiniMaxH3CaptionGrammar {
    enum Phase: Hashable {
        case object
        case captionKey(Int)
        case captionColon, captionQuote
        case caption(Bool)
        case escape
        case unicode(Int)
        case comma
        case ratioKey(Int)
        case ratioColon, ratioQuote
        case ratio(String)
        case close, complete
    }

    let ratioKey: [UInt8]
    let ratios: [String]
    private static let captionKey = Array("\"expanded_caption\"".utf8)

    init(ratioKey: String, ratios: [String]) {
        self.ratioKey = Array("\"\(ratioKey)\"".utf8)
        self.ratios = ratios
    }

    private func isSpace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 9 || byte == 10 || byte == 13
    }

    func advancing(_ phase: Phase, bytes: [UInt8]) -> Phase? {
        var next = phase
        for byte in bytes {
            guard let phase = advance(next, byte) else { return nil }
            next = phase
        }
        return next
    }

    private func advance(_ phase: Phase, _ byte: UInt8) -> Phase? {
        func punctuation(_ expected: UInt8, then next: Phase) -> Phase? {
            if isSpace(byte) { return phase }
            return byte == expected ? next : nil
        }
        func key(_ name: [UInt8], index: Int, continued: (Int) -> Phase, then next: Phase) -> Phase?
        {
            if index == 0 && isSpace(byte) { return phase }
            guard byte == name[index] else { return nil }
            return index + 1 == name.count ? next : continued(index + 1)
        }
        switch phase {
        case .object: return punctuation(0x7b, then: .captionKey(0))
        case .captionKey(let index):
            return key(
                Self.captionKey, index: index, continued: Phase.captionKey, then: .captionColon)
        case .captionColon: return punctuation(0x3a, then: .captionQuote)
        case .captionQuote: return punctuation(0x22, then: .caption(false))
        case .caption(let nonempty):
            if byte == 0x22 { return nonempty ? .comma : nil }
            if byte == 0x5c { return .escape }
            guard byte >= 0x20 else { return nil }
            return .caption(nonempty || byte != 0x20)
        case .escape:
            if byte == 0x75 { return .unicode(4) }
            return Self.escapes.contains(byte) ? .caption(true) : nil
        case .unicode(let remaining):
            guard Self.hexadecimal.contains(byte) else { return nil }
            return remaining == 1 ? .caption(true) : .unicode(remaining - 1)
        case .comma: return punctuation(0x2c, then: .ratioKey(0))
        case .ratioKey(let index):
            return key(ratioKey, index: index, continued: Phase.ratioKey, then: .ratioColon)
        case .ratioColon: return punctuation(0x3a, then: .ratioQuote)
        case .ratioQuote: return punctuation(0x22, then: .ratio(""))
        case .ratio(let prefix):
            if byte == 0x22 { return ratios.contains(prefix) ? .close : nil }
            guard (0x30 ... 0x3a).contains(byte) else { return nil }
            let next = prefix + String(UnicodeScalar(byte))
            return ratios.contains(where: { $0.hasPrefix(next) }) ? .ratio(next) : nil
        case .close: return punctuation(0x7d, then: .complete)
        case .complete: return isSpace(byte) ? .complete : nil
        }
    }

    private static let escapes = Set("\"\\/bfnrt".utf8)
    private static let hexadecimal = Set("0123456789abcdefABCDEF".utf8)
}

private final class MiniMaxH3CaptionConstraint: GenerationTokenConstraint {
    private let tokenizer: any Tokenizer
    private let grammar: MiniMaxH3CaptionGrammar
    private let stop: Set<Int>
    private let endThinking: Int
    private var thinking: Bool
    private var phase = MiniMaxH3CaptionGrammar.Phase.object
    private var vocabulary: [[UInt8]?]?
    private var masks: [MiniMaxH3CaptionGrammar.Phase: MLXArray] = [:]

    init(tokenizer: any Tokenizer, grammar: MiniMaxH3CaptionGrammar, stop: Set<Int>, thinking: Bool)
        throws
    {
        guard let end = tokenizer.convertTokenToId("</think>"), !stop.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3 prompt expansion requires a compatible Qwen tokenizer")
        }
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.stop = stop
        self.endThinking = end
        self.thinking = thinking
    }

    private static let byteDecoder: [UnicodeScalar: UInt8] = {
        let literal = Set(Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255))
        var offset = 256
        var map: [UnicodeScalar: UInt8] = [:]
        for byte in 0 ... 255 {
            if literal.contains(byte) {
                map[UnicodeScalar(byte)!] = UInt8(byte)
            } else {
                map[UnicodeScalar(offset)!] = UInt8(byte)
                offset += 1
            }
        }
        return map
    }()

    func apply(to logits: MLXArray) throws -> MLXArray {
        if thinking { return logits }
        let count = logits.dim(-1)
        if vocabulary == nil {
            vocabulary = (0 ..< count).map { id in
                guard !stop.contains(id), id != endThinking,
                    let name = tokenizer.convertIdToToken(id), !name.isEmpty,
                    !name.hasPrefix("<|"), name != "<think>"
                else { return nil }
                let bytes = name.unicodeScalars.compactMap { Self.byteDecoder[$0] }
                return bytes.count == name.unicodeScalars.count ? bytes : nil
            }
        }
        if let mask = masks[phase] { return MLX.where(mask, logits, -Float.infinity) }
        let allowed = vocabulary!.enumerated().map { index, bytes in
            phase == .complete
                ? stop.contains(index)
                : bytes.flatMap { grammar.advancing(phase, bytes: $0) } != nil
        }
        guard allowed.contains(true) else {
            throw ModelExecutionError.invalidOptions(
                "No compatible tokens for the MiniMax-H3 caption grammar")
        }
        let mask = MLXArray(allowed)
        masks[phase] = mask
        return MLX.where(mask, logits, -Float.infinity)
    }

    func sampled(_ token: Int) throws {
        if thinking {
            if token == endThinking { thinking = false }
            return
        }
        if phase == .complete && stop.contains(token) { return }
        guard let bytes = vocabulary?[token], let next = grammar.advancing(phase, bytes: bytes)
        else { throw ModelExecutionError.invalidOptions("Invalid token in the MiniMax-H3 caption") }
        phase = next
    }
}

enum MiniMaxH3ImagePromptExpansion {
    static let defaultModel: ModelSource = .repository("MiniMaxAI/MiniMax-H3-Image-PE")
    static let aspectRatios = [
        "1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16",
    ]

    struct Caption: Sendable {
        let text: String
        let aspectRatio: String
    }

    static func aspectRatio(width: Int, height: Int) -> String {
        var a = width
        var b = height
        while b != 0 { (a, b) = (b, a % b) }
        return "\(width / a):\(height / a)"
    }

    static func input(
        prompt: String, references: [ModelImageData], width: Int, height: Int
    ) throws -> UserInput {
        let context: [String: any Sendable] = [
            "enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "xhigh",
        ]
        if references.isEmpty {
            return UserInput(
                chat: [
                    .system(MiniMaxH3ImagePromptTemplates.system),
                    .user(
                        MiniMaxH3ImagePromptTemplates.textToImage.replacingOccurrences(
                            of: "<<USER_PROMPT>>", with: prompt)),
                ], additionalContext: context)
        }

        let count = references.count
        var content: [[String: String]] = [
            ["type": "text", "text": MiniMaxH3ImagePromptTemplates.edit],
            [
                "type": "text",
                "text":
                    "Reference Image Count:\n\(count) input image(s), labeled <Picture 1> through <Picture \(count)>.",
            ],
        ]
        var images: [UserInput.Image] = []
        for (index, image) in references.enumerated() {
            let h = image.shape[2]
            let w = image.shape[3]
            content.append(
                [
                    "type": "text",
                    "text":
                        "Reference Media:\n- <Picture \(index + 1)> [aspect_ratio \(aspectRatio(width: w, height: h))]",
                ])
            content.append(["type": "image"])
            let pixels = image.tensor.asType(.float32).transposed(0, 2, 3, 1)
            let rgb = clip(round((pixels[.ellipsis, ..<3] + 1) * 127.5), min: 0, max: 255).asType(
                .uint8)
            let scale = min(1, 2048 / Double(max(h, w)))
            let preview = DiffusionRGBResize.resize(
                rgb, height: max(1, Int((Double(h) * scale).rounded())),
                width: max(1, Int((Double(w) * scale).rounded())), lanczos: true)
            images.append(.array(preview[0].transposed(2, 0, 1)))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let encoded = String(decoding: try encoder.encode(prompt), as: UTF8.self)
        let ratio = aspectRatio(width: width, height: height)
        content.append([
            "type": "text",
            "text": """

            User Editing Instruction (JSON string):
            \(encoded)

            User Aspect Ratio Selection:
            \(ratio)

            Apply the instructions above to the supplied inputs and reference images. Output only the required two-field JSON object.
            """,
        ])
        return UserInput(
            messages: [["role": "user", "content": content]], images: images,
            additionalContext: context)
    }

    static func parse(_ text: String, ratioKey: String, ratios: [String]) throws -> Caption {
        guard
            let value = try JSONSerialization.jsonObject(with: Data(text.utf8))
                as? [String: String],
            Set(value.keys) == ["expanded_caption", ratioKey],
            let caption = value["expanded_caption"]?.trimmingCharacters(
                in: .whitespacesAndNewlines),
            !caption.isEmpty, let ratio = value[ratioKey], ratios.contains(ratio)
        else { throw ModelExecutionError.invalidOptions("Invalid MiniMax-H3 expanded caption") }
        return Caption(text: caption, aspectRatio: ratio)
    }

    static func expand(
        _ prompt: String, model source: ModelSource, references: [ModelImageData],
        width: Int, height: Int, seed: UInt64?, policy: ModelExecutionPolicy
    ) async throws -> Caption {
        let model = try await NativModel.loadModel(from: source, policy: policy)
        return try await model.withModel { _, context in
            let input = try Self.input(
                prompt: prompt, references: references, width: width, height: height)
            let prepared = try await context.processor.prepare(input: input)
            let tokens = prepared.text.tokens.flattened()
            let tail = context.tokenizer.decode(
                tokenIds: tokens[max(0, tokens.size - 48)...].asArray(Int.self))
            let thinking = tail.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("<think>")
            let ratioKey = references.isEmpty ? "aspect_ratio" : "target_aspect_ratio"
            let ratios =
                references.isEmpty ? aspectRatios : [aspectRatio(width: width, height: height)]
            let grammar = MiniMaxH3CaptionGrammar(ratioKey: ratioKey, ratios: ratios)
            let constraint = try MiniMaxH3CaptionConstraint(
                tokenizer: context.tokenizer, grammar: grammar,
                stop: StopPolicy.tokenIDs(context: context), thinking: thinking)
            let options = ModelGenerationOptions(
                sampling: .init(temperature: 1, topP: 0.95, topK: 20, seed: seed),
                output: .init(maxTokens: 4096, format: .tokens))
            let iterator = try AutoregressiveIterator(
                input: prepared, model: context.languageModel, options: options,
                constraint: constraint)
            var answer: [Int] = []
            let result = try AutoregressiveEngine.consume(
                iterator, input: prepared, context: context, tools: nil, options: options
            ) { event in
                if case .token(let token) = event { answer.append(token) }
            }
            guard result.stopReason == .stop else {
                throw ModelExecutionError.invalidOptions(
                    "MiniMax-H3 prompt expansion did not finish a caption")
            }
            if thinking {
                guard let end = context.tokenizer.convertTokenToId("</think>"),
                    let boundary = answer.firstIndex(of: end)
                else {
                    throw ModelExecutionError.invalidOptions(
                        "MiniMax-H3 prompt expansion did not finish reasoning")
                }
                answer.removeFirst(boundary + 1)
            }
            return try parse(
                context.tokenizer.decode(tokenIds: answer, skipSpecialTokens: true),
                ratioKey: ratioKey, ratios: ratios)
        }
    }
}
