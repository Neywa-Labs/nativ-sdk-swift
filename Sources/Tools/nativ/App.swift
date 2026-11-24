import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import HuggingFace
@preconcurrency import MLX
import NativSDK

@main
enum App {
    static func main() async {
        do {
            let command = try CommandLineOptions.parse()
            switch command {
            case .help:
                Usage.printRoot()
            case .generate(let options):
                try await runGenerate(options)
            case .image(let options):
                try await runImage(options)
            case .chat(let options):
                try await runChat(options)
            case .predict(let options):
                try await runPredict(options)
            case .convert(let options):
                let result = try ModelConversion.convert(
                    source: resolveURL(options.source),
                    destination: resolveURL(options.destination),
                    quantization: options.quantization,
                    recipe: options.recipe,
                    progress: {
                        print($0)
                        fflush(stdout)
                    })
                print(
                    String(
                        format: "Saved %d shards (%.2f GB), quantized %d matrices to %@",
                        result.shardCount, Double(result.tensorBytes) / 1e9,
                        result.quantizedMatrixCount, resolveURL(options.destination).path))
            }
        } catch let error as CLIError {
            if case .help = error {
                error.printUsage()
                return
            }
            fputs("Error: \(error.description)\n\n", stderr)
            error.printUsage()
            exit(1)
        } catch ModelExecutionError.invalidOptions(let message) {
            fputs("Error: \(message)\n", stderr)
            exit(1)
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func runChat(_ options: ChatOptions) async throws {
        if let cacheLimitMB = options.cacheLimitMB {
            Memory.cacheLimit = cacheLimitMB * 1024 * 1024
        }

        let source = try resolveModelSource(options.model, revision: options.revision)
        print("Loading \(source.name)")

        let model = try await NativModel.loadModel(
            from: source,
            progress: printProgress,
        )

        var sessionConfiguration = PromptCacheReuseConfiguration()
        sessionConfiguration.reuseEnabled = !options.noReuse
        sessionConfiguration.checkpointInterval = options.checkpointInterval
        sessionConfiguration.maxCheckpoints = options.maxCheckpoints

        let parameters = ModelGenerationOptions(
            prefill: .init(stepSize: options.prefillStepSize),
            sampling: .init(
                temperature: options.temperature, topP: options.topP,
                topK: options.topK, seed: options.seed),
            output: .init(maxTokens: options.maxTokens)
        )

        let session = ModelSession(
            model: model, instructions: options.system,
            options: parameters, cacheReuse: sessionConfiguration)

        print("Chat ready. /exit quits, /reset clears the conversation.")
        var turn = 0
        while true {
            print("> ", terminator: "")
            guard let line = readLine() else {
                break
            }
            let text = line.trimmingCharacters(in: .whitespaces)
            if text.isEmpty {
                continue
            }
            if text == "/exit" || text == "/quit" {
                break
            }
            if text == "/reset" {
                try await session.reset()
                print("(conversation reset)")
                continue
            }

            turn += 1
            let turnStart = CFAbsoluteTimeGetCurrent()
            var turnStats: ModelCacheStatistics?
            let events = try await session.respond(to: text)
            for try await event in events {
                switch event {
                case .chunk(let chunk):
                    print(chunk, terminator: "")
                    fflush(stdout)
                case .completion(let summary):
                    turnStats = summary.cache
                default: break
                }
            }
            print("")

            if let stats = turnStats {
                let wallSeconds = CFAbsoluteTimeGetCurrent() - turnStart
                print(
                    String(
                        format: "[stats] kind=%@ reused=%d prefilled=%d gen=%d "
                            + "ttft=%.2fs prefill=%.2fs tps=%.1f",
                        stats.kind.rawValue,
                        stats.reusedTokenCount,
                        stats.prefilledTokenCount,
                        stats.generatedTokenCount,
                        stats.firstTokenSeconds,
                        stats.prefillSeconds,
                        stats.tokensPerSecond,
                    ),
                )
                if options.statsJSON {
                    emitChatStatsJSON(stats, turn: turn, wallSeconds: wallSeconds)
                }
            }
        }
        await session.close()
    }

    /// One line-delimited JSON object per turn on stderr, so stdout stays pure reply
    /// text and benchmarks can capture `2> stats.jsonl`.
    private static func emitChatStatsJSON(
        _ stats: ModelCacheStatistics, turn: Int, wallSeconds: Double,
    ) {
        struct TurnStats: Encodable {
            let turn: Int
            let kind: String
            let promptTokens: Int
            let reusedTokens: Int
            let prefilledTokens: Int
            let generatedTokens: Int
            let usedCheckpoint: Bool
            let trimmedTokens: Int
            let ttftS: Double
            let prefillS: Double
            let generateS: Double
            let tokPerS: Double
            let stop: String
            let wallS: Double
        }
        let payload = TurnStats(
            turn: turn,
            kind: stats.kind.rawValue,
            promptTokens: stats.promptTokenCount,
            reusedTokens: stats.reusedTokenCount,
            prefilledTokens: stats.prefilledTokenCount,
            generatedTokens: stats.generatedTokenCount,
            usedCheckpoint: stats.usedCheckpoint,
            trimmedTokens: stats.trimmedTokenCount,
            ttftS: stats.firstTokenSeconds,
            prefillS: stats.prefillSeconds,
            generateS: stats.generateSeconds,
            tokPerS: stats.tokensPerSecond,
            stop: stats.stopReason,
            wallS: wallSeconds,
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        if let data = try? encoder.encode(payload),
            let json = String(data: data, encoding: .utf8)
        {
            FileHandle.standardError.write(Data((json + "\n").utf8))
        }
    }

    private static func runPredict(_ options: PredictOptions) async throws {
        let model = try await NativDecisionModel.loadModel(
            from: resolveModelSource(options.model, revision: options.revision))
        let result = try await model.predict(
            state: options.state, questions: options.questions, options: options.prediction,
            media: options.media
        ).result()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        print(String(decoding: try encoder.encode(result), as: UTF8.self))
    }

    private static func runGenerate(_ options: GenerateOptions) async throws {
        if let cacheLimitMB = options.cacheLimitMB {
            Memory.cacheLimit = cacheLimitMB * 1024 * 1024
        }

        try options.validateFiles()

        let source = try resolveModelSource(options.model, revision: options.revision)
        let prompt = options.prompt
        print("Loading \(source.name)")

        let input = UserInput(
            chat: [
                .user(
                    prompt,
                    images: options.imagePaths.map { .url(resolveURL($0)) },
                    videos: options.videoPaths.map { .url(resolveURL($0)) },
                    audios: options.audioPaths.map { .url(resolveURL($0)) },
                )
            ],
            processing: options.processing
        )

        if let draft = options.draftModel {
            let draftSource = try resolveModelSource(
                draft, revision: options.draftRevision, optionName: "--draft-model"
            )
            print("Loading draft \(draftSource.name)")
            let model = try await NativDraftModel.loadModel(
                target: source, draft: draftSource, progress: printProgress
            )
            let parameters = ModelGenerationOptions(
                prefill: .init(stepSize: options.prefillStepSize),
                sampling: options.sampling,
                output: .init(maxTokens: options.maxTokens), mode: options.generationMode
            )
            print("Prompt: \(prompt)")
            printMediaSummary(
                images: options.imagePaths, videos: options.videoPaths, audios: options.audioPaths)
            print("Output:")
            let start = CFAbsoluteTimeGetCurrent()
            try await consumeGeneration(
                model.generate(
                    input: input, options: parameters,
                    speculation: options.draftTokens.map { .init(draftTokenCount: $0) } ?? .init()
                ),
                benchmarkJSON: options.benchmarkJSON
            )
            if options.benchmark {
                print(
                    String(format: "Generation elapsed: %.2fs", CFAbsoluteTimeGetCurrent() - start))
            }
            return
        }

        let model = try await NativModel.loadModel(
            from: source,
            expertStreaming: options.expertCacheMB.map {
                .init(cacheSizeBytes: $0 * 1024 * 1024, expertBatchSize: options.expertBatchSize ?? 8)
            },
            progress: printProgress
        )

        let prepareStart = CFAbsoluteTimeGetCurrent()
        let preparedInput = try await model.prepare(input: input)
        let prepareElapsed = CFAbsoluteTimeGetCurrent() - prepareStart
        preparedInput.inspect {
            printPreparedInput($0, elapsed: prepareElapsed, debugTokens: options.debugInput)
        }

        if options.prepareOnly {
            return
        }

        let parameters = ModelGenerationOptions(
            prefill: .init(stepSize: options.prefillStepSize),
            sampling: options.sampling,
            output: .init(maxTokens: options.maxTokens)
        )

        print("Prompt: \(prompt)")
        printMediaSummary(
            images: options.imagePaths, videos: options.videoPaths, audios: options.audioPaths)

        let generationStart = CFAbsoluteTimeGetCurrent()
        if let outputAudioPath = options.outputAudioPath {
            let isLFM3 = try await model.withModel { _, context in context.model is LFM3Audio }
            let audioOptions: any AudioGenerationOptions
            if isLFM3 {
                var audio = options.lfm3Audio
                audio.maxTokens = options.talkerMaxTokens
                audioOptions = audio
            } else {
                audioOptions = ModelAudioOptions(
                    speaker: options.speaker, maxTokens: options.talkerMaxTokens,
                    chunkSize: options.chunkSize, leftContextSize: options.leftContextSize)
            }
            let result = try await model.generateAudio(
                prepared: preparedInput, options: parameters, audio: audioOptions
            ).result()
            let outputURL = resolveOutputURL(outputAudioPath)
            try AudioIO.writeWAV(
                waveform: result.waveform,
                sampleRate: result.sampleRate,
                to: outputURL,
            )
            print("Text: \(result.text)")
            print("Waveform shape: \(result.shape)")
            print("Wrote WAV: \(outputURL.path)")
            printGenerationInfo(result.summary)
        } else {
            print("Output:")
            var generationOptions = parameters
            generationOptions.mode = options.generationMode
            let usesDiffusion = try await model.withModel { _, context in
                try ModelGenerationStrategy.resolve(
                    options.generationMode, model: context.languageModel)
                    != .autoregressive
            }
            if usesDiffusion {
                generationOptions.prefill = .init()
                generationOptions.diffusion = .init(
                    maxTokens: options.maxTokens, blockLength: options.diffusionBlockLength,
                    steps: options.diffusionSteps, threshold: options.diffusionThreshold)
            }
            let stream = model.generate(prepared: preparedInput, options: generationOptions)

            try await consumeGeneration(stream, benchmarkJSON: options.benchmarkJSON)
        }

        if options.benchmark {
            let elapsed = CFAbsoluteTimeGetCurrent() - generationStart
            print(String(format: "Generation elapsed: %.2fs", elapsed))
        }
    }

    private static func consumeGeneration(
        _ stream: ModelGeneration, benchmarkJSON: Bool
    ) async throws {
        var output = ""
        var reasoning = ""
        for try await event in stream {
            switch event {
            case .chunk(let chunk):
                output += chunk
                print(chunk, terminator: "")
                fflush(stdout)
            case .completion(let info):
                printGenerationInfo(info, leadingNewline: true)
                if benchmarkJSON {
                    emitBenchmarkRecord(info, output: output, reasoning: reasoning)
                }
            case .reasoning(let text):
                reasoning += text
                print(text, terminator: "")
            case .token(let token):
                print(token)
            case .protocolError(let message):
                fputs("\n\(message)\n", stderr)
            case .toolCall(let toolCall):
                print("\nTool call: \(toolCall)")
            case .rejectedToolCall(let rejection):
                print("\nRejected tool call: \(rejection.reason)")
            }
        }

        if output.isEmpty {
            print("")
        }
    }

    private static func runImage(_ options: ImageOptions) async throws {
        if let cacheLimitMB = options.cacheLimitMB {
            Memory.cacheLimit = cacheLimitMB * 1024 * 1024
        }

        try options.validateFiles()

        guard
            options.task == .edit
                ? !options.referenceImagePaths.isEmpty : options.referenceImagePaths.isEmpty
        else {
            throw CLIError.invalidValue("--task", "editing requires reference images")
        }
        guard let modelSource = options.model else { throw CLIError.missingValue("--model") }
        let model = try await NativImageModel.loadModel(
            from: resolveModelSource(modelSource, revision: options.revision),
            progress: printProgress)
        let references = try options.referenceImagePaths.map { path in
            let array = try ImageIOHelpers.loadFlux2ReferenceImage(
                from: resolveURL(path), width: nil, height: nil, preserveAlpha: true)
            return try ModelImageData(tensor: array)
        }
        var parameters = options.parameters
        parameters.maxPromptLength = options.maxLength
        parameters.bucketedPrompt = options.bucketedPrompt
        parameters.steps = options.steps
        parameters.guidance = options.guidance
        parameters.width = options.width
        parameters.height = options.height
        parameters.seed = options.seed
        if options.encodeOnly {
            let shapes = try await model.promptShapes(
                prompt: options.prompt, parameters: parameters)
            print("Prompt feature shapes: \(shapes)")
            return
        }
        let result = try await model.generate(
            prompt: options.prompt, referenceImages: references, parameters: parameters
        ).result()
        try result.image.savePNG(to: resolveOutputURL(options.outputPath))
        print(
            "Saved \(result.width)x\(result.height) image to \(options.outputPath) (seed \(result.seed), \(result.steps) steps)"
        )
    }

    private static func printProgress(_ progress: Progress) {
        guard progress.totalUnitCount > 0 else {
            return
        }
        let percent = Int((progress.fractionCompleted * 100).rounded())
        print("Download \(percent)%")
    }

    private static func printPreparedInput(
        _ input: LMInput,
        elapsed: TimeInterval,
        debugTokens: Bool,
    ) {
        print(String(format: "Prepared input in %.2fs", elapsed))
        print("Token shape: \(input.text.tokens.shape)")
        if let mask = input.text.mask {
            print("Token mask shape: \(mask.shape)")
        }
        if let image = input.image {
            print("Image pixels shape: \(image.pixels.shape)")
            if let frames = image.frames {
                print("Image frames: \(frames)")
            }
        }
        if let video = input.video {
            print("Video pixels shape: \(video.pixels.shape)")
            if let frames = video.frames {
                print("Video frames: \(frames)")
            }
        }
        if let audio = input.audio {
            print("Audio features shape: \(audio.features.shape)")
            if let mask = audio.mask {
                print("Audio mask shape: \(mask.shape)")
            }
        }
        if debugTokens {
            let tokens = input.text.tokens.asArray(Int.self)
            print("Token count: \(tokens.count)")
            print("Token prefix: \(Array(tokens.prefix(32)))")
            print("Token suffix: \(Array(tokens.suffix(32)))")
        }
    }

    private static func printMediaSummary(images: [String], videos: [String], audios: [String]) {
        if !images.isEmpty {
            print("Images: \(images.map { resolveURL($0).path }.joined(separator: ", "))")
        }
        if !videos.isEmpty {
            print("Videos: \(videos.map { resolveURL($0).path }.joined(separator: ", "))")
        }
        if !audios.isEmpty {
            print("Audio inputs: \(audios.map { resolveURL($0).path }.joined(separator: ", "))")
        }
        if images.isEmpty, videos.isEmpty, audios.isEmpty {
            print("Media: none")
        }
    }

    /// Writes a prefixed JSON record to stderr, separating content from reasoning counts.
    private static func emitBenchmarkRecord(
        _ info: ModelGenerationSummary, output: String, reasoning: String
    ) {
        var record: [String: Any] = [
            "prompt_tokens": info.promptTokenCount,
            "generation_tokens": info.generatedTokenCount,
            "prompt_seconds": info.promptSeconds,
            "generate_seconds": info.generationSeconds,
            "prompt_tps": info.promptSeconds > 0
                ? Double(info.promptTokenCount) / info.promptSeconds : 0,
            "generation_tps": info.generationSeconds > 0
                ? Double(info.generatedTokenCount) / info.generationSeconds : 0,
            "peak_memory_bytes": Memory.peakMemory,
            "stop_reason": info.stopReason.rawValue,
            "output_chars": output.count,
            "reasoning_chars": reasoning.count,
        ]
        if let stats = info.speculative {
            record["speculative"] = [
                "strategy": stats.strategy.rawValue,
                "proposed_tokens": stats.proposedTokenCount,
                "accepted_tokens": stats.acceptedTokenCount,
                "verification_rounds": stats.verificationCount,
                "acceptance_rate": stats.proposedTokenCount > 0
                    ? Double(stats.acceptedTokenCount) / Double(stats.proposedTokenCount) : 0,
            ]
        }
        guard
            let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
            let text = String(data: data, encoding: .utf8)
        else { return }
        FileHandle.standardError.write(Data(("BENCH_JSON " + text + "\n").utf8))
    }

    private static func printGenerationInfo(
        _ info: ModelGenerationSummary?,
        leadingNewline: Bool = false,
    ) {
        guard let info else {
            return
        }
        let prefix = leadingNewline ? "\n\n" : ""
        print(
            "\(prefix)Generated \(info.generatedTokenCount) tokens at "
                + "\((Double(info.generatedTokenCount) / max(info.generationSeconds, 0.000001)).formatted(.number.precision(.fractionLength(2)))) tokens/s",
        )
        if let stats = info.speculative {
            print(
                "Draft: accepted \(stats.acceptedTokenCount)/\(stats.proposedTokenCount) "
                    + "proposed tokens in \(stats.verificationCount) verification rounds")
        }
    }
}

private enum ToolCommand {
    case help
    case generate(GenerateOptions)
    case image(ImageOptions)
    case chat(ChatOptions)
    case predict(PredictOptions)
    case convert(ConvertOptions)
}

private enum CommandLineOptions {
    static func parse() throws -> ToolCommand {
        var arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else {
            return .help
        }
        arguments.removeFirst()

        switch command {
        case "--help", "-h", "help":
            return .help
        case "generate":
            if Self.outputModality(in: arguments) == "image" {
                return try .image(ImageOptions.parse(arguments))
            }
            return try .generate(GenerateOptions.parse(arguments))
        case "generate_image", "generate-image", "image":
            return try .image(ImageOptions.parse(arguments))
        case "chat":
            return try .chat(ChatOptions.parse(arguments))
        case "predict":
            return try .predict(PredictOptions.parse(arguments))
        case "convert":
            return try .convert(ConvertOptions.parse(arguments))
        default:
            throw CLIError.unknownCommand(command)
        }
    }

    private static func outputModality(in arguments: [String]) -> String? {
        for index in arguments.indices where arguments[index] == "--output-modality" {
            let valueIndex = arguments.index(after: index)
            guard valueIndex < arguments.endIndex else {
                return nil
            }
            return arguments[valueIndex].lowercased()
        }
        return nil
    }
}

private struct ConvertOptions {
    var source = ""
    var destination = ""
    var quantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
    var recipe: QuantizationRecipe?

    static func parse(_ arguments: [String]) throws -> Self {
        var options = Self()
        var paths: [String] = []
        var bits: Int?
        var groupSize: Int?
        var mode = QuantizationMode.affine
        var explicitQuantization = false
        var iterator = ArgumentIterator(arguments)
        while let argument = iterator.next() {
            switch argument {
            case "--help", "-h": throw CLIError.help(.convert)
            case "--bits":
                bits = try iterator.intValue(for: argument)
                explicitQuantization = true
            case "--group-size":
                groupSize = try iterator.intValue(for: argument)
                explicitQuantization = true
            case "--mode":
                let value = try iterator.value(for: argument)
                guard let parsed = QuantizationMode(rawValue: value) else {
                    throw CLIError.invalidValue(argument, value)
                }
                mode = parsed
                explicitQuantization = true
            case "--quant-predicate":
                let value = try iterator.value(for: argument)
                guard options.recipe == nil, let recipe = QuantizationRecipe(rawValue: value) else {
                    throw CLIError.invalidValue(argument, value)
                }
                options.recipe = recipe
            default:
                guard !argument.hasPrefix("-"), !argument.isEmpty, paths.count < 2 else {
                    throw CLIError.unknownOption(argument, command: .convert)
                }
                paths.append(argument)
            }
        }
        guard paths.count == 2 else {
            throw CLIError.missingOption("<source> <destination>", command: .convert)
        }
        if let recipe = options.recipe, recipe != .mixed, explicitQuantization {
            throw CLIError.invalidValue(
                "--quant-predicate",
                "named mixed-bit recipes fix bits, group size and mode; omit those options")
        }
        options.source = paths[0]
        options.destination = paths[1]
        options.quantization = .init(
            groupSize: groupSize ?? (mode == .affine ? 64 : mode == .nvfp4 ? 16 : 32),
            bits: bits ?? (mode == .mxfp8 ? 8 : 4), mode: mode)
        return options
    }
}

private struct PredictOptions {
    var model = ""
    var revision: String?
    var state: DecisionState = .text("")
    var questions: [String: DecisionQuestion] = [:]
    var prediction = DecisionPredictionOptions()
    var lfm3 = LFM3DecisionOptions()
    var hasLFM3Options = false
    var media: [DecisionMedia] = []

    static func parse(_ arguments: [String]) throws -> Self {
        var options = Self()
        var state: String?
        var questionsPath: String?
        var jsonState = false
        var iterator = ArgumentIterator(arguments)
        while let argument = iterator.next() {
            switch argument {
            case "--help", "-h": throw CLIError.help(.predict)
            case "--model": options.model = try iterator.value(for: argument)
            case "--revision": options.revision = try iterator.value(for: argument)
            case "--state":
                guard state == nil else {
                    throw CLIError.invalidValue(argument, "provide state once")
                }
                state = try iterator.value(for: argument)
            case "--state-file":
                guard state == nil else {
                    throw CLIError.invalidValue(argument, "provide state once")
                }
                state = try String(
                    contentsOf: resolveURL(iterator.value(for: argument)), encoding: .utf8)
            case "--image":
                options.media.append(.image(try resolveURL(iterator.value(for: argument))))
            case "--audio":
                options.media.append(.audio(try resolveURL(iterator.value(for: argument))))
            case "--video":
                options.media.append(.video(try resolveURL(iterator.value(for: argument))))
            case "--set-size":
                options.lfm3.setSize = try iterator.intValue(for: argument)
                options.hasLFM3Options = true
            case "--score-readout":
                let value = try iterator.value(for: argument)
                guard let readout = LFM3DecisionOptions.ScoreReadout(rawValue: value) else {
                    throw CLIError.invalidValue(argument, "expected median, mean, or argmax")
                }
                options.lfm3.scoreReadout = readout
                options.hasLFM3Options = true
            case "--questions-file": questionsPath = try iterator.value(for: argument)
            case "--json-state": jsonState = true
            case "--packed": options.prediction.independent = false
            case "--isolated": options.prediction.isolated = true
            case "--no-isolated": options.prediction.isolated = false
            case "--max-state-tokens":
                options.prediction.maxStateTokens = try iterator.intValue(for: argument)
            default: throw CLIError.unknownOption(argument, command: .predict)
            }
        }
        guard !options.model.isEmpty else {
            throw CLIError.missingOption("--model", command: .predict)
        }
        guard state != nil || !options.media.isEmpty else {
            throw CLIError.missingOption("--state", command: .predict)
        }
        guard let questionsPath else {
            throw CLIError.missingOption("--questions-file", command: .predict)
        }
        guard options.prediction.maxStateTokens > 0 else {
            throw CLIError.invalidValue("--max-state-tokens", "must be positive")
        }
        if let state {
            options.state =
                jsonState
                ? .json(try JSONDecoder().decode(JSONValue.self, from: Data(state.utf8)))
                : .text(state)
        } else {
            options.state = .json(.null)
        }
        if options.hasLFM3Options { options.prediction.modelOptions = options.lfm3 }
        let data = try Data(contentsOf: resolveURL(questionsPath))
        let specs = try JSONDecoder().decode([String: JSONValue].self, from: data)
        for (name, value) in specs {
            guard case .object(let spec) = value,
                case .string(let instructions) = spec["instructions"], !instructions.isEmpty
            else {
                throw CLIError.invalidValue("--questions-file", "question needs instructions")
            }
            let type: DecisionType
            if let raw = spec["type"] {
                guard case .string(let string) = raw, let parsed = DecisionType(rawValue: string)
                else {
                    throw CLIError.invalidValue("--questions-file", "unknown decision type")
                }
                type = parsed
            } else {
                type = .choice
            }
            var criteria: [DecisionCriterion] = []
            if case .array(let values) = spec["criteria"] {
                guard type != .bool && type != .noul else {
                    throw CLIError.invalidValue(
                        "--questions-file", "Boolean criteria must be a mapping")
                }
                for (index, value) in values.enumerated() {
                    if type == .score {
                        criteria.append(.init(String(index), description: try text(value)))
                    } else if case .string(let label) = value {
                        criteria.append(.init(label))
                    } else {
                        throw CLIError.invalidValue("--questions-file", "labels must be strings")
                    }
                }
            } else if case .object(let values) = spec["criteria"] {
                guard type != .score else {
                    throw CLIError.invalidValue(
                        "--questions-file", "score criteria must be an ordered list")
                }
                criteria = try values.keys.sorted().map { key in
                    .init(key, description: values[key] == .null ? nil : try text(values[key]!))
                }
            } else if spec["criteria"] != nil && spec["criteria"] != .null {
                throw CLIError.invalidValue("--questions-file", "invalid criteria")
            }
            var question = DecisionQuestion(
                type: type, instructions: instructions, criteria: criteria)
            if let isolated = spec["isolated"] {
                guard case .bool(let value) = isolated else {
                    throw CLIError.invalidValue("--questions-file", "isolated must be Boolean")
                }
                question.isolated = value
            }
            if let threshold = spec["threshold"] {
                switch threshold {
                case .double(let value): question.threshold = value
                case .int(let value): question.threshold = Double(value)
                default:
                    throw CLIError.invalidValue("--questions-file", "threshold must be numeric")
                }
            }
            options.questions[name] = question
        }
        guard !options.questions.isEmpty else {
            throw CLIError.invalidValue(
                "--questions-file", "at least one named question is required")
        }
        return options
    }
    private static func text(_ value: JSONValue) throws -> String {
        if case .string(let string) = value { return string }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

private struct GenerateOptions {
    var benchmarkJSON = false
    var model = ""
    var revision: String?
    var draftModel: String?
    var draftRevision: String?
    var draftTokens: Int?
    var prompt = ""
    var imagePaths: [String] = []
    var videoPaths: [String] = []
    var audioPaths: [String] = []
    var outputAudioPath: String?
    var outputModality = "text"
    var lfm3Audio = LFM3AudioOptions()
    var speaker = "ethan"
    var talkerMaxTokens = 2048
    var chunkSize = 300
    var leftContextSize = 25
    var maxTokens = 128
    var temperature: Float = 0
    var topP: Float = 1
    var topK = 0
    var seed: UInt64?
    var repetitionPenalty: Float?
    var repetitionContextSize = 20
    var sampling: ModelSamplingOptions {
        .init(
            temperature: temperature, topP: topP, topK: topK, seed: seed,
            repetition: repetitionPenalty.map { .init($0, contextSize: repetitionContextSize) })
    }
    var prefillStepSize = ModelPrefillOptions.defaultStepSize
    var generationMode: GenerationMode = .automatic
    var diffusionBlockLength = 32
    var diffusionSteps = 32
    var diffusionThreshold: Float = 0.95
    var debugInput = false
    var prepareOnly = false
    var benchmark = false

    var expertCacheMB: Int?
    var expertBatchSize: Int?
    var cacheLimitMB: Int?
    var processing = UserInput.Processing()

    static func parse(_ arguments: [String]) throws -> GenerateOptions {
        var options = GenerateOptions()
        var iterator = ArgumentIterator(arguments)

        while let argument = iterator.next() {
            switch argument {
            case "--help", "-h":
                throw CLIError.help(.generate)
            case "--model":
                options.model = try iterator.value(for: argument)
            case "--revision":
                options.revision = try iterator.value(for: argument)
            case "--draft-model":
                options.draftModel = try iterator.value(for: argument)
            case "--draft-revision":
                options.draftRevision = try iterator.value(for: argument)
            case "--draft-tokens":
                options.draftTokens = try iterator.intValue(for: argument)
            case "--prompt":
                options.prompt = try iterator.value(for: argument)
            case "--prompt-file":
                options.prompt = try String(
                    contentsOf: resolveURL(try iterator.value(for: argument)),
                    encoding: .utf8)
            case "--json":
                options.benchmarkJSON = true
            case "--output-modality":
                let value = try iterator.value(for: argument).lowercased()
                guard value == "text" else {
                    throw CLIError.invalidValue(argument, value)
                }
                options.outputModality = value
            case "--image":
                try options.imagePaths.append(iterator.value(for: argument))
            case "--video":
                try options.videoPaths.append(iterator.value(for: argument))
            case "--audio":
                try options.audioPaths.append(iterator.value(for: argument))
            case "--output-audio":
                options.outputAudioPath = try iterator.value(for: argument)
            case "--audio-mode":
                let raw = try iterator.value(for: argument)
                guard let mode = LFM3AudioOptions.Mode(rawValue: raw) else {
                    throw CLIError.invalidValue(argument, raw)
                }
                options.lfm3Audio.mode = mode
            case "--audio-temperature":
                options.lfm3Audio.temperature = try iterator.floatValue(for: argument)
            case "--audio-top-k":
                options.lfm3Audio.topK = try iterator.intValue(for: argument)
            case "--audio-repetition-penalty":
                options.lfm3Audio.repetitionPenalty = try iterator.floatValue(for: argument)
            case "--speaker":
                options.speaker = try iterator.value(for: argument)
            case "--talker-max-tokens":
                options.talkerMaxTokens = try iterator.intValue(for: argument)
            case "--chunk-size":
                options.chunkSize = try iterator.intValue(for: argument)
            case "--left-context-size":
                options.leftContextSize = try iterator.intValue(for: argument)
            case "--repetition-penalty":
                options.repetitionPenalty = try iterator.floatValue(for: argument)
            case "--repetition-context-size":
                options.repetitionContextSize = try iterator.intValue(for: argument)
            case "--max-tokens":
                options.maxTokens = try iterator.intValue(for: argument)
            case "--temperature":
                options.temperature = try iterator.floatValue(for: argument)
            case "--top-p":
                options.topP = try iterator.floatValue(for: argument)
            case "--top-k":
                options.topK = try iterator.intValue(for: argument)
            case "--seed":
                let raw = try iterator.value(for: argument)
                guard let seed = UInt64(raw) else {
                    throw CLIError.invalidValue(argument, raw)
                }
                options.seed = seed
            case "--prefill-step-size":
                options.prefillStepSize = try iterator.intValue(for: argument)
            case "--generation-mode":
                let value = try iterator.value(for: argument)
                guard let mode = GenerationMode(cliValue: value) else {
                    throw CLIError.invalidValue(argument, value)
                }
                options.generationMode = mode
            case "--diffusion-block-length":
                options.diffusionBlockLength = try iterator.intValue(for: argument)
            case "--diffusion-steps":
                options.diffusionSteps = try iterator.intValue(for: argument)
            case "--diffusion-threshold":
                options.diffusionThreshold = try iterator.floatValue(for: argument)
            case "--min-pixels":
                options.processing.minPixels = try iterator.intValue(for: argument)
            case "--max-pixels":
                options.processing.maxPixels = try iterator.intValue(for: argument)
            case "--debug-input":
                options.debugInput = true
            case "--prepare-only":
                options.prepareOnly = true
            case "--benchmark":
                options.benchmark = true

            case "--expert-batch-size":
                options.expertBatchSize = try iterator.intValue(for: argument)
            case "--expert-cache-mb":
                options.expertCacheMB = try iterator.intValue(for: argument)
            case "--cache-limit-mb":
                options.cacheLimitMB = try iterator.intValue(for: argument)
            default:
                if !argument.hasPrefix("-"), options.prompt.isEmpty {
                    options.prompt = argument
                } else {
                    throw CLIError.unknownOption(argument, command: .generate)
                }
            }
        }

        guard !options.model.isEmpty else {
            throw CLIError.missingOption("--model", command: .generate)
        }
        guard !options.prompt.isEmpty else {
            throw CLIError.missingOption("--prompt", command: .generate)
        }

        if let batchSize = options.expertBatchSize {
            guard (1 ... 256).contains(batchSize), options.expertCacheMB != nil else {
                throw CLIError.invalidValue(
                    "--expert-batch-size", "requires --expert-cache-mb and a value between 1 and 256")
            }
        }
        if let budget = options.expertCacheMB {
            guard budget > 0, budget <= 4096 else {
                throw CLIError.invalidValue("--expert-cache-mb", "must be between 1 and 4096")
            }
            guard options.draftModel == nil, options.outputModality == "text",
                options.outputAudioPath == nil
            else {
                throw CLIError.invalidValue(
                    "--expert-cache-mb", "expert streaming supports text output without a draft model")
            }
        }
        let hasDraftOptions =
            arguments.contains("--draft-revision")
            || arguments.contains("--draft-tokens")
        if options.draftModel == nil && hasDraftOptions {
            throw CLIError.missingOption("--draft-model", command: .generate)
        }
        if let draft = options.draftModel {
            guard !draft.isEmpty else {
                throw CLIError.invalidValue("--draft-model", "must name a repository or directory")
            }
            guard (1 ... 16).contains(options.draftTokens ?? 4) else {
                throw CLIError.invalidValue("--draft-tokens", "must be between 1 and 16")
            }
            guard
                options.videoPaths.isEmpty
                    && options.audioPaths.isEmpty && options.outputAudioPath == nil
            else {
                throw CLIError.invalidValue(
                    "--draft-model",
                    "drafting supports text output; image input requires an LFM2-VL, Qwen3.5 or Gemma4 target, and video/audio are unsupported"
                )
            }
            guard options.generationMode != .diffusion else {
                throw CLIError.invalidValue(
                    "--generation-mode", "drafting requires autoregressive decoding")
            }
            guard !options.prepareOnly && !options.debugInput else {
                throw CLIError.invalidValue(
                    "--draft-model",
                    "--prepare-only and --debug-input require ordinary model loading")
            }
        }

        if let penalty = options.repetitionPenalty, !penalty.isFinite || penalty <= 0 {
            throw CLIError.invalidValue("--repetition-penalty", "must be finite and positive")
        }
        guard options.repetitionContextSize >= 0 else {
            throw CLIError.invalidValue("--repetition-context-size", "must be nonnegative")
        }
        return options
    }

    func validateFiles() throws {
        try validateInputFiles(imagePaths + videoPaths + audioPaths)
    }
}

private struct ChatOptions {
    var model = ""
    var revision: String?
    var system: String?
    var maxTokens = 512
    var temperature: Float = 0
    var topP: Float = 1
    var topK = 0
    var seed: UInt64?
    var prefillStepSize = ModelPrefillOptions.defaultStepSize

    var cacheLimitMB: Int?
    var statsJSON = false
    var noReuse = false
    var checkpointInterval: Int? = 256
    var maxCheckpoints = 4

    static func parse(_ arguments: [String]) throws -> ChatOptions {
        var options = ChatOptions()
        var iterator = ArgumentIterator(arguments)

        while let argument = iterator.next() {
            switch argument {
            case "--help", "-h":
                throw CLIError.help(.chat)
            case "--model":
                options.model = try iterator.value(for: argument)
            case "--revision":
                options.revision = try iterator.value(for: argument)
            case "--system":
                options.system = try iterator.value(for: argument)
            case "--max-tokens":
                options.maxTokens = try iterator.intValue(for: argument)
            case "--temperature":
                options.temperature = try iterator.floatValue(for: argument)
            case "--top-p":
                options.topP = try iterator.floatValue(for: argument)
            case "--top-k":
                options.topK = try iterator.intValue(for: argument)
            case "--seed":
                let raw = try iterator.value(for: argument)
                guard let seed = UInt64(raw) else {
                    throw CLIError.invalidValue(argument, raw)
                }
                options.seed = seed
            case "--prefill-step-size":
                options.prefillStepSize = try iterator.intValue(for: argument)

            case "--cache-limit-mb":
                options.cacheLimitMB = try iterator.intValue(for: argument)
            case "--stats-json":
                options.statsJSON = true
            case "--no-reuse":
                options.noReuse = true
            case "--checkpoint-interval":
                let value = try iterator.intValue(for: argument)
                options.checkpointInterval = value > 0 ? value : nil
            case "--max-checkpoints":
                options.maxCheckpoints = try iterator.intValue(for: argument)
            default:
                throw CLIError.unknownOption(argument, command: .chat)
            }
        }

        guard !options.model.isEmpty else {
            throw CLIError.missingOption("--model", command: .chat)
        }

        return options
    }
}

private struct ImageOptions {
    enum Task: String {
        case generate
        case edit
    }

    var parameters = ImageGenerationParameters()
    var model: String?
    var revision: String?
    var task = Task.generate
    var prompt = ""
    var referenceImagePaths: [String] = []
    var outputPath = "image.png"
    var width = 1024
    var height = 1024
    var steps: Int?
    var guidance: Float?
    var seed: UInt64?
    var maxLength = 512
    var bucketedPrompt = true
    var encodeOnly = false

    var cacheLimitMB: Int?

    static func parse(_ arguments: [String]) throws -> ImageOptions {
        var options = ImageOptions()
        var iterator = ArgumentIterator(arguments)

        while let argument = iterator.next() {
            switch argument {
            case "--help", "-h":
                throw CLIError.help(.image)
            case "--prompt-expansion-model", "--h3-prompt-expansion-model":
                let source = try resolveModelSource(
                    iterator.value(for: argument), revision: nil, optionName: argument)
                options.parameters.promptExpansion = .model(source)
            case "--upsample-prompt":
                if options.parameters.promptExpansion == .disabled {
                    options.parameters.promptExpansion = .defaultModel
                }
            case "--no-json-caption": options.parameters.ideogram4.autoJSONCaption = false
            case "--negative-prompt":
                options.parameters.negativePrompt = try iterator.value(for: argument)
            case "--strength": options.parameters.strength = try iterator.floatValue(for: argument)
            case "--cfg-truncation":
                options.parameters.cfgTruncation = try iterator.floatValue(for: argument)
            case "--no-reference-cache": options.parameters.useReferenceCache = false
            case "--reference-resolution":
                options.parameters.referenceResolution = try iterator.intValue(for: argument)
            case "--h3-canvas-area":
                options.parameters.minimaxH3Image.minimumCanvasArea = try iterator.intValue(
                    for: argument)
            case "--image-token-prior": options.parameters.useImageTokenPrior = true
            case "--stochastic-sampling": options.parameters.stochasticSampling = true
            case "--deterministic-sampling": options.parameters.stochasticSampling = false
            case "--model":
                options.model = try iterator.value(for: argument)
            case "--revision":
                options.revision = try iterator.value(for: argument)
            case "--output-modality":
                let value = try iterator.value(for: argument).lowercased()
                guard value == "image" else {
                    throw CLIError.invalidValue(argument, value)
                }
            case "--task":
                let value = try iterator.value(for: argument).lowercased()
                guard let task = Task(rawValue: value) else {
                    throw CLIError.invalidValue(argument, value)
                }
                options.task = task
            case "--prompt":
                options.prompt = try iterator.value(for: argument)
            case "--image":
                try options.referenceImagePaths.append(iterator.value(for: argument))
            case "--reference-image":
                try options.referenceImagePaths.append(iterator.value(for: argument))
                options.task = .edit
            case "--output":
                options.outputPath = try iterator.value(for: argument)
            case "--size":
                let size = try iterator.value(for: argument)
                let dimensions = try parseImageSize(size)
                options.width = dimensions.width
                options.height = dimensions.height
            case "--width":
                options.width = try iterator.intValue(for: argument)
            case "--height":
                options.height = try iterator.intValue(for: argument)
            case "--steps":
                options.steps = try iterator.intValue(for: argument)
            case "--guidance":
                options.guidance = try iterator.floatValue(for: argument)
            case "--seed":
                let raw = try iterator.value(for: argument)
                guard let seed = UInt64(raw) else {
                    throw CLIError.invalidValue(argument, raw)
                }
                options.seed = seed
            case "--max-length":
                options.maxLength = try iterator.intValue(for: argument)
            case "--no-bucketed-prompt":
                options.bucketedPrompt = false
            case "--encode-only":
                options.encodeOnly = true

            case "--cache-limit-mb":
                options.cacheLimitMB = try iterator.intValue(for: argument)
            default:
                if !argument.hasPrefix("-"), options.prompt.isEmpty {
                    options.prompt = argument
                } else {
                    throw CLIError.unknownOption(argument, command: .image)
                }
            }
        }

        guard options.model?.isEmpty == false else {
            throw CLIError.missingOption("--model", command: .image)
        }
        guard !options.prompt.isEmpty else {
            throw CLIError.missingOption("--prompt", command: .image)
        }

        return options
    }

    func validateFiles() throws {
        try validateInputFiles(referenceImagePaths)
    }
}

private struct ArgumentIterator {
    private var arguments: [String]
    private var index = 0

    init(_ arguments: [String]) {
        self.arguments = arguments
    }

    mutating func next() -> String? {
        guard index < arguments.count else {
            return nil
        }
        defer { index += 1 }
        return arguments[index]
    }

    mutating func value(for option: String) throws -> String {
        guard let value = next() else {
            throw CLIError.missingValue(option)
        }
        return value
    }

    mutating func intValue(for option: String) throws -> Int {
        let raw = try value(for: option)
        guard let value = Int(raw) else {
            throw CLIError.invalidValue(option, raw)
        }
        return value
    }

    mutating func floatValue(for option: String) throws -> Float {
        let raw = try value(for: option)
        guard let value = Float(raw) else {
            throw CLIError.invalidValue(option, raw)
        }
        return value
    }
}

private enum CLICommand {
    case root
    case generate
    case image
    case chat
    case predict
    case convert
}

private enum CLIError: Error, CustomStringConvertible {
    case help(CLICommand)
    case unknownCommand(String)
    case unknownOption(String, command: CLICommand)
    case missingValue(String)
    case invalidValue(String, String)
    case missingOption(String, command: CLICommand)
    case inputFileNotFound(String)

    var description: String {
        switch self {
        case .help:
            "Help requested"
        case .unknownCommand(let command):
            "Unknown command \(command)"
        case .unknownOption(let option, _):
            "Unknown option \(option)"
        case .missingValue(let option):
            "Missing value for \(option)"
        case .invalidValue(let option, let value):
            "Invalid value for \(option): \(value)"
        case .missingOption(let option, _):
            "Missing required option \(option)"
        case .inputFileNotFound(let path):
            "Input file not found: \(path)"
        }
    }

    func printUsage() {
        switch self {
        case .help(let command):
            Usage.print(command)
        case .unknownOption(_, let command), .missingOption(_, let command):
            Usage.print(command)
        case .unknownCommand, .missingValue, .invalidValue, .inputFileNotFound:
            Usage.printRoot()
        }
    }
}

private enum Usage {
    static func print(_ command: CLICommand) {
        switch command {
        case .root:
            printRoot()
        case .generate:
            printGenerate()
        case .image:
            printImage()
        case .chat:
            printChat()
        case .predict:
            printPredict()
        case .convert:
            printConvert()
        }
    }

    static func printRoot() {
        Swift.print(
            """
            Usage:
              swift run nativ <command> [options]

            Commands:
              generate        Run text, VL, video, audio-input, or audio-output generation.
              generate_image  Run image generation; equivalent to generate --output-modality image.
              image           Alias for generate_image.
              chat            Interactive multi-turn chat with prompt-cache reuse.
              predict         Predict named decisions with calibrated probabilities.
              convert         Quantize a local MLX checkpoint using its model policy or a mixed recipe.

            Run `swift run nativ <command> --help` for command options.
            """,
        )
    }

    static func printConvert() {
        Swift.print(
            """
            Usage:
              swift run nativ convert <source> <destination> [options]

            Quantize an existing local MLX checkpoint on an Apple silicon Mac.
            By default, the model's predicate can skip layers, use the options below, or
            specify per-layer settings. Vision/audio is excluded and shape checks still apply.
            Explicit recipes override the model's default policy. Packed weights are preserved.

            Options:
              --bits <int>              Default bit width (affine: 4).
              --group-size <int>        Default quantization group size (affine: 64).
              --mode <mode>             affine (default), mxfp4, mxfp8, or nvfp4.
              --quant-predicate <name>  mixed_2_6, mixed_3_4, mixed_3_5, mixed_3_6,
                                        mixed_3_8, mixed_4_6, mixed_4_8, or mixed.

            `mixed` selects the model's expert streaming recipe. Other eligible layers
            use the options above.
            LFM2 MoE uses MXFP4 experts and affine 8-bit routers. MiniCPM-V 4.7 preserves
            routed experts and uses affine 8-bit routers and vision/projector linears.
            Named mixed-bit recipes use affine group size 64 and cannot be combined with
            --bits, --group-size or --mode. They give embeddings, the output head and
            selected value/down projections more bits, following Python's layer schedule.
            Conversion needs additional memory and disk space. The destination must not exist.
            Transformers layouts, GGUF, AWQ and GPTQ inputs are not supported.
            """)
    }

    static func printPredict() {
        Swift.print(
            """
            Usage:
              swift run nativ predict --model <repo|path> --state <text> --questions-file <path>

            Options:
              --state-file <path>       Read state text from a UTF-8 file instead.
              --json-state              Parse the state as structured JSON.
              --packed                  Evaluate questions together, in alphabetical name order.
              --isolated / --no-isolated Override the checkpoint's score-level isolation.
              --max-state-tokens <int>   Limit the context prefix (default: 32768).
              --image/--audio/--video <path>  Add ordered d1 media; audio requires d1-Omni.
              --set-size <int>          d1 questions per pass (checkpoint default: 16).
              --score-readout <mode>    d1 median, mean, or argmax (default: median).
              --revision <revision>     Select a Hugging Face revision.

            Questions use the upstream JSON schema: type, instructions, and criteria.
            Choice criteria are labels or label-to-description mappings; score levels are ordered lists.
            Results are JSON on stdout. No answer tokens are generated.
            """)
    }

    static func printGenerate() {
        Swift.print(
            """
            Usage:
              swift run nativ generate --model <repo|path> --prompt <text> [media options]
              swift run nativ generate --output-modality image --model <repo|path> --prompt <text> --output <path>

            Media:
              --image <path>                Add an image input. Can be repeated.
              --video <path>                Add a video input. Can be repeated.
              --audio <path>                Add an audio input. Can be repeated.

            Prompt:
              --prompt <text>              Prompt text.
              --prompt-file <path>         Read UTF-8 prompt text from a file.

            Generation:
              --output-modality <mode>      text or image. Default: text.
              --draft-model <repo|path>     DFlash/DSpark support sampling; Qwen MTP and independent drafts are greedy.
              --draft-revision <rev>        Select the draft checkpoint revision.
              --draft-tokens <n>            Proposals per round. Default: 4; DFlash/DSpark: 7; Laguna DFlash: 15; Qwen MTP: 1.
              --max-tokens <n>              Default: 128.
              --temperature <f>             Default: 0.
              --top-p <f>                   Default: 1.
              --top-k <n>                   Default: 0.
              --seed <n>
              --generation-mode <mode>      auto, ar, or diffusion.
              --repetition-penalty <f>       Optional text repetition penalty.
              --repetition-context-size <n> Tokens considered; 0 uses full history. Default: 20.
              --prefill-step-size <n>       Default: \(ModelPrefillOptions.defaultStepSize).
              --diffusion-block-length <n>  Default: 32.
              --diffusion-steps <n>         Default: 32.
              --diffusion-threshold <f>     Default: 0.95.

            Audio output (Qwen3-Omni and LFM3):
              --output-audio <path>         Generate speech/audio and write WAV.
              --audio-mode <mode>          LFM3: automatic, text, audio, interleaved.
              --audio-temperature <f>      LFM3 speech sampling temperature.
              --audio-top-k <n>            LFM3 speech top-k sampling.
              --audio-repetition-penalty <f> LFM3 speech repetition penalty.
              --speaker <name>             Qwen3-Omni speaker. Default: ethan.
              --talker-max-tokens <n>       Default: 2048.

            Loading and diagnostics:
              --revision <rev>

              --prepare-only
              --debug-input
              --benchmark
              --json                       Emit a BENCH_JSON record to stderr after generation.
              --min-pixels <n>
              --max-pixels <n>
              --expert-cache-mb <n>        Stream MoE experts from disk with a cache of n MiB.
              --expert-batch-size <n>      Maximum experts per evaluation (default: 8; byte capped).
              --cache-limit-mb <n>
            """,
        )
    }

    static func printChat() {
        Swift.print(
            """
            Usage:
              swift run nativ chat --model <repo|path> [options]

            Interactive multi-turn chat (text only) with prompt-cache reuse: each turn
            re-feeds only the new suffix tokens instead of re-prefilling the whole
            conversation. /exit quits, /reset clears the conversation.

            Options:
              --system <text>               System prompt for the conversation.
              --max-tokens <n>              Default: 512.
              --temperature <f>             Default: 0.
              --top-p <f>                   Default: 1.
              --top-k <n>                   Default: 0.
              --seed <n>
              --prefill-step-size <n>       Default: \(ModelPrefillOptions.defaultStepSize).
              --no-reuse                    Disable prompt-cache reuse (cold baseline).
              --checkpoint-interval <n>     Snapshot every n generated tokens; 0 disables. Default: 256.
              --max-checkpoints <n>         Default: 4.
              --stats-json                  Emit one JSON stats object per turn on stderr.
              --revision <rev>

              --cache-limit-mb <n>
            """,
        )
    }

    static func printImage() {
        Swift.print(
            """
            Usage:
              swift run nativ generate_image --model <repo|path> --prompt <text> --output <path> [options]
              swift run nativ generate --output-modality image --model <repo|path> --prompt <text> --output <path> [options]

            Model:
              --model <repo|path>           Required repository ID or local directory.
              --revision <rev>

            Image conditioning:
              --negative-prompt <text>      Unconditional caption for CFG.
              --strength <f>               Img2img strength in (0, 1].
              --cfg-truncation <f>          Z-Image CFG cutoff in [0, 1].
              --no-reference-cache          Recompute reference attention each step.
              --reference-resolution <n>    Qwen reference area resolution; default 1024.
              --h3-canvas-area <n>          Minimum MiniMax-H3-Image canvas area in pixels.
              --upsample-prompt             Use the image pipeline's default prompt expander.
              --prompt-expansion-model <repo|path>
                                            Use a compatible prompt expander from a repository or directory.
              --image-token-prior           Use LLaDA's discrete image-token prior.
              --stochastic-sampling         Enable LLaDA stochastic sampling.
              --deterministic-sampling      Disable LLaDA stochastic sampling.
              --no-json-caption             Disable Ideogram's minimal caption wrapper.
              --task <generate|edit>        Default: generate.
              --image <path>                Reference image for --task edit. Can be repeated.
              --reference-image <path>      Alias for --image that also selects edit mode.

            Generation:
              --output-modality image       Accepted for mlx-vlm-style generate invocations.
              --size <WIDTHxHEIGHT>         mlx-vlm-style size option.
              --width <n>                   Default: 1024.
              --height <n>                  Default: 1024.
              --steps <n>                   Default: model-specific.
              --guidance <f>                Default: model-specific.
              --seed <n>
              --max-length <n>              Prompt tokenizer max length. Default: 512.
              --no-bucketed-prompt
              --encode-only                 Load model and encode the prompt without denoising.

            Loading and diagnostics:

              --cache-limit-mb <n>
            """,
        )
    }
}

private enum ImageIOHelpers {
    static func loadFlux2ReferenceImage(
        from url: URL, width: Int?, height: Int?, preserveAlpha: Bool = false
    ) throws -> MLXArray {
        guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true])
        else {
            throw CLIError.inputFileNotFound(url.path)
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [
            .workingColorSpace: colorSpace, .outputColorSpace: colorSpace,
        ])
        let width = width ?? Int(image.extent.width)
        let height = height ?? Int(image.extent.height)
        let extent = image.extent
        let translated = image.transformed(
            by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y),
        )
        let scaled = translated.transformed(
            by: CGAffineTransform(
                scaleX: CGFloat(width) / max(extent.width, 1),
                y: CGFloat(height) / max(extent.height, 1),
            ),
        )
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { buffer in
            context.render(
                scaled.cropped(to: bounds).unpremultiplyingAlpha(),
                toBitmap: buffer.baseAddress!,
                rowBytes: width * 4,
                bounds: bounds,
                format: .RGBA8,
                colorSpace: colorSpace,
            )
        }

        var nchw = [Float]()
        nchw.reserveCapacity(3 * width * height)
        for channel in 0 ..< (preserveAlpha ? 4 : 3) {
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let index = (y * width + x) * 4 + channel
                    nchw.append(Float(rgba[index]) / 127.5 - 1.0)
                }
            }
        }
        return MLXArray(nchw, [1, preserveAlpha ? 4 : 3, height, width]).asType(
            Flux2Precision.dtype)
    }
}

private enum AudioIO {
    static func writeWAV(waveform: MLXArray, sampleRate: Int, to url: URL) throws {
        let samples = waveform.asArray(Float.self)
        let pcm = samples.map { sample -> Int16 in
            let clipped = min(max(sample, -1), 1)
            return Int16(clipped * Float(Int16.max))
        }

        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        data.appendLittleEndian(UInt32(36 + pcm.count * MemoryLayout<Int16>.size))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.appendLittleEndian(UInt32(16))
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt32(sampleRate))
        data.appendLittleEndian(UInt32(sampleRate * MemoryLayout<Int16>.size))
        data.appendLittleEndian(UInt16(MemoryLayout<Int16>.size))
        data.appendLittleEndian(UInt16(16))
        data.append(contentsOf: "data".utf8)
        data.appendLittleEndian(UInt32(pcm.count * MemoryLayout<Int16>.size))
        for sample in pcm {
            data.appendLittleEndian(UInt16(bitPattern: sample))
        }

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
        )
        try data.write(to: url, options: .atomic)
    }
}

extension Data {
    fileprivate mutating func appendLittleEndian(_ value: some FixedWidthInteger) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
}

extension GenerationMode {
    fileprivate init?(cliValue: String) {
        switch cliValue {
        case "auto", "automatic":
            self = .automatic
        case "ar", "autoregressive":
            self = .autoregressive
        case "diffusion":
            self = .diffusion
        default:
            return nil
        }
    }
}

private func validateInputFiles(_ paths: [String]) throws {
    for path in paths {
        let url = resolveURL(path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError.inputFileNotFound(url.path)
        }
    }
}

private func parseImageSize(_ raw: String) throws -> (width: Int, height: Int) {
    let normalized = raw.lowercased().replacingOccurrences(of: "x", with: "x")
    let parts = normalized.split(separator: "x", maxSplits: 1).map(String.init)
    guard parts.count == 2,
        let width = Int(parts[0]),
        let height = Int(parts[1])
    else {
        throw CLIError.invalidValue("--size", raw)
    }
    return (width, height)
}

private func resolveURL(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
}

private func resolveOutputURL(_ path: String) -> URL {
    let expanded = (path as NSString).expandingTildeInPath
    if expanded.hasPrefix("/") {
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: expanded)
        .standardizedFileURL
}

private func resolveModelSource(
    _ model: String, revision: String?, optionName: String = "--model"
) throws -> ModelSource {
    let localURL = resolveURL(model)
    var isDirectory = ObjCBool(false)
    if FileManager.default.fileExists(atPath: localURL.path, isDirectory: &isDirectory) {
        guard isDirectory.boolValue else {
            throw CLIError.invalidValue(optionName, "expected a directory: \(model)")
        }
        return .directory(localURL)
    }
    return .repository(model, revision: revision)
}
