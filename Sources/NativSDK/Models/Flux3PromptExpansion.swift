import CoreImage
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// The native layout prompt ends with a JSON table on a 0–1000 canvas.
enum Flux3LayoutPrompt {
    typealias Row = [String: JSONValue]

    static func split(_ input: String) -> (caption: String, rows: [JSONValue]?) {
        for index in input.indices.reversed() where input[index] == "[" {
            if let value = try? JSONDecoder().decode(
                [JSONValue].self, from: Data(input[index...].utf8))
            {
                return (input[..<index].trimmingCharacters(in: .whitespacesAndNewlines), value)
            }
        }
        return (input.trimmingCharacters(in: .whitespacesAndNewlines), nil)
    }

    static func detect(_ input: String) -> (caption: String, rows: [Row])? {
        let (caption, values) = split(input)
        guard let values, !values.isEmpty else { return nil }
        let rows = values.compactMap { value -> Row? in
            guard case .object(let row) = value, id(row) != nil, coordinates(row) != nil else {
                return nil
            }
            return row
        }
        guard rows.count == values.count else { return nil }
        let header = "ORIGINAL PROMPT:\n"
        if caption.hasPrefix(header),
            let separator = caption.range(of: "\n\nLOCKED BOUNDING-BOX ANNOTATIONS "),
            let start = caption.index(
                caption.startIndex, offsetBy: header.count, limitedBy: separator.lowerBound)
        {
            return (String(caption[start ..< separator.lowerBound]), rows)
        }
        return (caption, rows)
    }

    static func isFinished(_ input: String) -> Bool {
        guard let (caption, rows) = detect(input) else { return false }
        return rows.allSatisfy { row in
            id(row).map { caption.contains("<\($0)>") } ?? false
        }
    }

    static func lock(_ caption: String, rows: [Row]) throws -> String {
        """
        ORIGINAL PROMPT:
        \(caption)

        LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, provenance value, null, and coordinate exactly:
        \(try serialize(rows))
        """
    }

    static func reconcile(_ output: String, with locked: [Row]) throws -> String {
        var (caption, values) = split(output)
        let rows = (values ?? []).compactMap { value -> Row? in
            guard case .object(let row) = value, id(row) != nil else { return nil }
            return row
        }
        let lockedIDs = Set(locked.compactMap(id))
        var byID: [String: Row] = [:]
        for row in rows {
            if let name = id(row) { byID[name] = row }
        }
        var claimed: Set<String> = []
        var result: [Row] = []
        for lock in locked {
            guard let name = id(lock), let box = coordinates(lock) else { continue }
            var match = byID[name]
            if match == nil,
                let similar = rows.first(where: { row in
                    guard let candidate = id(row), !lockedIDs.contains(candidate),
                        !claimed.contains(candidate), let candidateBox = coordinates(row)
                    else { return false }
                    return intersectionOverUnion(box, candidateBox) >= 0.9
                }),
                let oldName = id(similar)
            {
                caption = caption.replacingOccurrences(of: "<\(oldName)>", with: "<\(name)>")
                claimed.insert(oldName)
                match = similar
            }
            var row = lock
            if case .string(let desc) = match?["desc"] {
                let desc = desc.trimmingCharacters(in: .whitespacesAndNewlines)
                if !desc.isEmpty { row["desc"] = .string(desc) }
            }
            result.append(row)
        }

        var used = lockedIDs.union(claimed)
        for row in rows {
            guard let name = id(row), !used.contains(name), caption.contains("<\(name)>"),
                let valid = added(row, id: name)
            else { continue }
            result.append(valid)
            used.insert(name)
        }
        return "\(caption) \(try serialize(result))"
    }

    private static func serialize(_ rows: [Row]) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(rows), as: UTF8.self)
    }

    private static func id(_ row: Row) -> String? {
        guard case .string(let name) = row["id"],
            name.wholeMatch(of: /[A-Za-z][A-Za-z0-9_]*/) != nil
        else { return nil }
        return name
    }

    private static func coordinates(_ row: Row) -> [Double]? {
        box(row["bbox"]) ?? box(row["tgt_bbox"]) ?? box(row["src_bbox"])
    }

    private static func box(_ value: JSONValue?) -> [Double]? {
        guard case .array(let values) = value, values.count == 4 else { return nil }
        let axes = values.compactMap { value -> Double? in
            switch value {
            case .int(let n): return Double(n)
            case .double(let n) where n.isFinite: return n
            default: return nil
            }
        }
        guard axes.count == 4, axes[0] >= 0, axes[1] >= 0, axes[2] > axes[0], axes[3] > axes[1],
            axes[2] <= 1000, axes[3] <= 1000, max(axes[2], axes[3]) > 1
        else { return nil }
        return axes
    }

    private static func added(_ row: Row, id: String) -> Row? {
        guard case .string(let raw) = row["desc"] else { return nil }
        let desc = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !desc.isEmpty else { return nil }
        if let boundingBox = row["bbox"], box(boundingBox) != nil {
            return ["id": .string(id), "desc": .string(desc), "bbox": boundingBox]
        }
        guard case .string(let origin) = row["from"],
            origin == "target" || origin.wholeMatch(of: /ref_image_[0-9]+/) != nil
        else { return nil }
        let source = row["src_bbox"] ?? .null
        let target = row["tgt_bbox"] ?? .null
        guard source != .null || target != .null, (origin == "target") == (source == .null),
            source == .null || box(source) != nil, target == .null || box(target) != nil
        else { return nil }
        return [
            "id": .string(id), "from": .string(origin), "desc": .string(desc),
            "src_bbox": source, "tgt_bbox": target,
        ]
    }

    private static func intersectionOverUnion(_ a: [Double], _ b: [Double]) -> Double {
        let intersection =
            max(0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        let union =
            (a[2] - a[0]) * (a[3] - a[1])
            + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union
    }
}

enum Flux3PromptExpansion {
    static let defaultModel: ModelSource = .repository("black-forest-labs/flux-3-image-klein-pup")
    private static let maxTokens = 3584

    static func referenceSize(width: Int, height: Int, count: Int) throws -> (Int, Int) {
        guard width > 0, height > 0, (1 ... 10).contains(count),
            Double(max(width, height)) / Double(min(width, height)) <= 200
        else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 prompt expansion requires up to 10 references with aspect ratios within 200:1"
            )
        }
        let cap = count == 1 ? 1024 * 1024 : count == 2 ? 724 * 724 : 512 * 512
        func rounded(_ side: Int) -> Int {
            max(32, Int((Double(side) / 32).rounded(.toNearestOrEven)) * 32)
        }
        func floored(_ side: Double) -> Int {
            max(32, Int(side / 32) * 32)
        }
        var (w, h) = (rounded(width), rounded(height))
        if w > cap / h {
            let ratio = sqrt(Double(width) * Double(height) / Double(cap))
            w = floored(Double(width) / ratio)
            h = floored(Double(height) / ratio)
        }
        if w > cap / h {
            if w == 32 { h = floored(Double(cap / w)) } else { w = floored(Double(cap / h)) }
        }
        return (w, h)
    }

    private static func reference(_ image: ModelImageData, count: Int) throws -> UserInput.Image {
        let (w, h) = try referenceSize(width: image.shape[3], height: image.shape[2], count: count)
        let rgb = DiffusionRGBResize.rgbBytes(
            image.tensor.asType(.float32), flattenAlpha: true)
        let pixels = DiffusionRGBResize.resize(rgb, height: h, width: w, lanczos: true)[0]
        let cgImage = try ModelImageData(tensor: pixels).cgImage()
        let jpeg = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                jpeg, UTType.jpeg.identifier as CFString, 1, nil)
        else {
            throw ModelExecutionError.invalidOptions("Cannot create a FLUX.3 reference preview")
        }
        CGImageDestinationAddImage(
            destination, cgImage,
            [
                kCGImageDestinationLossyCompressionQuality: 0.92
            ] as CFDictionary)
        guard CGImageDestinationFinalize(destination), let preview = CIImage(data: jpeg as Data)
        else {
            throw ModelExecutionError.invalidOptions("Cannot encode a FLUX.3 reference preview")
        }
        return .ciImage(preview)
    }

    static func expand(
        _ prompt: String, model source: ModelSource, references: [ModelImageData],
        seed: UInt64?, policy: ModelExecutionPolicy
    ) async throws -> String {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelExecutionError.invalidOptions("FLUX.3 requires a nonempty prompt")
        }
        let layout = Flux3LayoutPrompt.detect(prompt)
        let editing = !references.isEmpty
        let task = (editing ? "i2i" : "t2i") + (layout == nil ? "" : "_bbox")
        let instruction =
            try layout.map { try Flux3LayoutPrompt.lock($0.caption, rows: $0.rows) } ?? prompt
        let model = try await NativModel.loadModel(from: source, policy: policy)
        let promptFile = model.assets.modelDirectory.appending(path: "system_prompts.json")
        let templates =
            if FileManager.default.fileExists(atPath: promptFile.path) {
                try JSONDecoder().decode([String: String].self, from: Data(contentsOf: promptFile))
            } else { Flux3PromptTemplates.defaults }
        guard let system = templates[task] else {
            throw ModelExecutionError.invalidOptions(
                "Missing FLUX.3 upsampler task prompt: \(task)")
        }
        let images = try references.map { try reference($0, count: references.count) }
        let content: any Sendable
        if editing {
            let request =
                layout == nil
                ? """
                Reference images follow in order (the first image, the second image, ...).
                User request: \(instruction)
                Output the caption.
                """
                : """
                Reference images follow in attachment order: ref_image_0, ref_image_1, ...
                User editing instruction: \(instruction)
                Decide the edit, imagine the target image, and output the annotation record.
                """
            content = images.map { _ in ["type": "image"] } + [["type": "text", "text": request]]
        } else {
            content = instruction
        }
        let input = UserInput(
            messages: [
                ["role": "system", "content": system], ["role": "user", "content": content],
            ],
            images: images, additionalContext: ["enable_thinking": false])
        let generation = model.generate(
            input: input,
            options: .init(
                sampling: .init(
                    temperature: 0, topP: 0.95, topK: 20, seed: seed,
                    presence: editing ? nil : .init(1.5, contextSize: maxTokens)),
                output: .init(maxTokens: maxTokens, separateReasoning: false)))
        var caption = ""
        for try await event in generation {
            if case .chunk(let text) = event { caption += text }
        }
        let metrics = try await generation.metrics()
        let text = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard metrics.stopReason == .stop, !text.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 prompt expansion did not finish a caption")
        }
        return try layout.map { try Flux3LayoutPrompt.reconcile(text, with: $0.rows) } ?? text
    }
}
