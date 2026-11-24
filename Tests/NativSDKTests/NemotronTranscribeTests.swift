import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class NemotronTranscribeTests: XCTestCase {
    func testNativeTaskPrompts() throws {
        XCTAssertTrue(
            try NemotronTranscribeTask.transcription.prompt().contains("do not translate"))
        XCTAssertThrowsError(try NemotronTranscribeTask.translation.prompt())
        XCTAssertEqual(
            try NemotronTranscribeTask.translation.prompt(language: "German"),
            "Translate the speech in this audio into German.")
        XCTAssertTrue(try NemotronTranscribeTask.speakerAttribution.prompt().contains("<spk:7>"))
        XCTAssertEqual(try NemotronTranscribeTask.audioQuestion.prompt(), "<|audio|>")
    }

    private func tinyConfiguration() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "nemotron-transcribe-config", withExtension: "json",
                subdirectory: "Fixtures"))
        return try Data(contentsOf: url)
    }

    func testConvertedAudioModelMatchesResidentWithStreamingAndChunkedPrefill() throws {
        try checkConvertedAudioModel(recipe: nil)
    }

    func testThreeBitExpertRecipeMatchesResidentWithStreamingAndChunkedPrefill() throws {
        try checkConvertedAudioModel(recipe: .mixed)
    }

    func testMixedRecipePreservesBoundaryExpertsAndOutputProjections() throws {
        try checkConvertedAudioModel(recipe: .mixed, boundaryLayers: true)
    }

    private func checkConvertedAudioModel(
        recipe: QuantizationRecipe?, boundaryLayers: Bool = false
    ) throws {
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer {
            for url in [source, destination] where FileManager.default.fileExists(atPath: url.path)
            {
                try? FileManager.default.removeItem(at: url)
            }
        }
        var data = try tinyConfiguration()
        if boundaryLayers {
            var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            var language = try XCTUnwrap(json["llm_config"] as? [String: Any])
            language["num_hidden_layers"] = 16
            language["hybrid_override_pattern"] = "M" + String(repeating: "E", count: 14) + "*"
            json["llm_config"] = language
            data = try JSONSerialization.data(withJSONObject: json)
        }
        try data.write(to: source.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(NemotronTranscribeConfiguration.self, from: data)
        MLXRandom.seed(47)
        let initial = NemotronTranscribe(config)
        let original = Dictionary(
            uniqueKeysWithValues: initial.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            })
        try save(
            arrays: original, metadata: ["format": "mlx"],
            url: source.appendingPathComponent("model.safetensors"))
        let conversion = try ModelConversion.convert(
            source: source, destination: destination,
            quantization: .init(groupSize: 32, bits: 4), recipe: recipe)
        XCTAssertGreaterThan(conversion.quantizedMatrixCount, 0)
        let converted = try loadArrays(
            url: destination.appendingPathComponent("model-00001.safetensors"))
        XCTAssertTrue(
            converted.keys.contains { $0.contains(".switch_mlp.") && $0.hasSuffix(".scales") })
        for (name, value) in original where name.hasPrefix("perception.") {
            XCTAssertEqual(converted[name]?.dtype, value.dtype)
            XCTAssertTrue(arrayEqual(converted[name]!, value).item(Bool.self), name)
        }
        let metadata = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: destination.appendingPathComponent("config.json")))
        for name in converted.keys where name.hasSuffix(".scales") {
            let path = String(name.dropLast(".scales".count))
            let layer = path.split(separator: ".").compactMap { Int($0) }.first ?? -1
            let expert =
                path.contains(".switch_mlp.") && path.hasSuffix(".fc1")
                && (!boundaryLayers || (4 ..< 12).contains(layer))
            let parameters = metadata.perLayerQuantization?.quantization(layer: path)
            XCTAssertEqual(parameters?.bits, recipe == .mixed && expert ? 3 : 4, path)
            XCTAssertEqual(parameters?.groupSize, recipe == .mixed && expert ? 64 : 32, path)
        }
        let resident = NemotronTranscribe(config)
        let streamed = NemotronTranscribe(config)
        try CheckpointLoader.load(
            modelDirectory: destination, model: resident,
            perLayerQuantization: metadata.perLayerQuantization)
        try CheckpointLoader.load(
            modelDirectory: destination, model: streamed,
            perLayerQuantization: metadata.perLayerQuantization,
            expertStreaming: .init(
                cacheSizeBytes: 64 * 1024, prefetch: false, sharedExpertBuffers: true))
        let waveform =
            sin(MLXArray(0 ..< 16000).asType(.float32) * Float(440 * 2 * Double.pi / 16000)) * 0.1
        // This recording crosses the tiny encoder's chunk boundary. Releasing a
        // speech layer must preserve both later chunks and later requests.
        XCTAssertNil(resident.perception.encoder.speech.reloadLayerWeights)
        XCTAssertNotNil(streamed.perception.encoder.speech.reloadLayerWeights)
        let referenceAudio = try resident.perception(waveform)
        eval(referenceAudio)
        for _ in 0 ..< 2 {
            let streamedAudio = try streamed.perception(waveform)
            XCTAssertTrue(arrayEqual(referenceAudio, streamedAudio).item(Bool.self))
        }
        let audioCount =
            (waveform.size / config.mel.hopLength + config.speech.factor - 1) / config.speech.factor
        let ids = [1, 2] + Array(repeating: 0, count: audioCount) + [3]
        let mask = MLXArray(ids.indices.map { $0 >= 2 && $0 < 2 + audioCount })
        let input = LMInput(
            text: .init(tokens: MLXArray(ids)), audio: .init(features: waveform, mask: mask))
        var outputs: [[Int]] = []
        for (model, step) in [(resident, 256), (resident, 4), (streamed, 4), (streamed, 256)] {
            let iterator = try AutoregressiveIterator(
                input: input, model: model,
                options: .init(
                    prefill: .init(stepSize: step), sampling: .init(temperature: 0),
                    output: .init(maxTokens: 4)))
            outputs.append(Array(iterator))
            XCTAssertNil(iterator.failure)
        }
        XCTAssertEqual(outputs[0], outputs[1])
        XCTAssertEqual(outputs[0], outputs[2])
        XCTAssertEqual(outputs[0], outputs[3])
        let cache = try streamed.newCache(configuration: nil)
        let logits = streamed(MLXArray([1, 2, 3], [1, 3]), cache: cache)
        eval(logits, cache)
        XCTAssertEqual(logits.dtype, .bfloat16)
        XCTAssertEqual(cache[0].state[1].dtype, .float32)
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
    }

    // Real encoder differential check against the pinned MLX Audio implementation.
    // Fixtures are generated locally; this test never downloads checkpoints.
    func testLocalPretrainedPerceptionAndPrompts() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["NATIV_TRANSCRIBE_SOURCE"],
            let fixturePath = env["NATIV_TRANSCRIBE_FIXTURES"]
        else {
            throw XCTSkip("Set NATIV_TRANSCRIBE_SOURCE and NATIV_TRANSCRIBE_FIXTURES")
        }
        let source = URL(fileURLWithPath: modelPath)
        let fixtures = URL(fileURLWithPath: fixturePath)
        let config = try JSONDecoder().decode(
            NemotronTranscribeConfiguration.self,
            from: Data(contentsOf: source.appendingPathComponent("config.json")))
        let perception = NemotronSpeechPerception(config)
        let weights = try loadArrays(url: fixtures.appendingPathComponent("perception.safetensors"))
        XCTAssertEqual(Set(weights.keys), Set(perception.parameters().flattened().map(\.0)))
        try perception.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 64 * 1024 * 1024
        defer { Memory.cacheLimit = previousLimit }
        for name in ["announcement", "conversation", "spoken_request", "rich_speech", "long"] {
            let reference = try loadArrays(
                url: fixtures.appendingPathComponent(name + ".safetensors"))
            let audio = try XCTUnwrap(reference["audio"])
            let mel = perception.preprocessor(audio)
            let normalized = NemotronParallelSpeechEncoder.normalize(mel)
            let encoded = try perception.encoder(mel)
            let projected = perception.proj(encoded)
            for (stage, actual) in [
                ("mel", mel), ("normalized", normalized), ("encoded", encoded),
                ("projected", projected),
            ] {
                guard let expected = reference[stage] else { continue }
                XCTAssertEqual(actual.shape, expected.shape)
                let difference = abs(actual.asType(.float32) - expected.asType(.float32))
                let maxError = difference.max().item(Float.self)
                let meanError = difference.mean().item(Float.self)
                print("TRANSCRIBE_PARITY \(name) \(stage) max=\(maxError) mean=\(meanError)")
                XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
                if stage == "mel" || stage == "normalized" {
                    // FFT implementations differ by a few ULPs at near-silent bins.
                    XCTAssertLessThan(maxError, 0.001)
                    XCTAssertLessThan(meanError, 0.000001)
                } else {
                    // BF16 rounding accumulates across 32 encoder layers. Check relative
                    // RMS as well as the real task outputs; bitwise equality is not expected.
                    let relativeRMS = sqrt(
                        difference.square().mean() / expected.asType(.float32).square().mean()
                    ).item(Float.self)
                    print("TRANSCRIBE_RELATIVE_RMS \(name) \(stage) \(relativeRMS)")
                    XCTAssertLessThan(relativeRMS, 2.0 / 128)  // Two BF16 relative precision units.
                }
            }
        }
        let tokenizer = try await DefaultTokenizerLoader().load(from: source)
        let processor = NemotronTranscribeProcessor(config, tokenizer: tokenizer)
        let reference = try loadArrays(
            url: fixtures.appendingPathComponent("announcement.safetensors"))
        let frameCount = reference["audio"]!.size / config.mel.hopLength
        let audioTokenCount = (frameCount + config.speech.factor - 1) / config.speech.factor
        let records = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: fixtures.appendingPathComponent("prompts.json")))
                as? [[String: Any]])
        for record in records {
            let ids = try XCTUnwrap(record["ids"] as? [Int])
            let tag = try XCTUnwrap(tokenizer.convertTokenToId("<|audio|>"))
            let index = try XCTUnwrap(ids.firstIndex(of: tag))
            let expected =
                Array(ids[..<index]) + Array(repeating: 0, count: audioTokenCount)
                + Array(ids[(index + 1)...])
            let input = UserInput(
                prompt: record["prompt"] as! String, audios: [.array(reference["audio"]!)])
            let prepared = try await processor.prepare(input: input)
            XCTAssertEqual(prepared.text.tokens.asArray(Int.self), expected)
        }
    }

    func testLocalPretrainedTasksWithExpertStreaming() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["NATIV_TRANSCRIBE_MODEL"],
            let fixturePath = env["NATIV_TRANSCRIBE_FIXTURES"]
        else {
            throw XCTSkip("Set NATIV_TRANSCRIBE_MODEL and NATIV_TRANSCRIBE_FIXTURES")
        }
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 64 * 1024 * 1024
        defer { Memory.cacheLimit = previousLimit }
        let model = try await NativModel.loadModel(
            from: .directory(URL(fileURLWithPath: modelPath)),
            expertStreaming: .init(
                cacheSizeBytes: 1536 * 1024 * 1024, sharedExpertBuffers: true))
        let cases: [(String, String, String, [String: any Sendable])] = [
            ("transcription", "announcement", "", ["task": "asr"]),
            ("translation", "announcement", "", ["task": "translate", "language": "German"]),
            ("speakers", "conversation", "", ["task": "saa"]),
            (
                "question", "announcement",
                "What time is the design review, where is it, and what should people bring?",
                ["task": "qa"]
            ),
        ]
        for (name, recording, prompt, context) in cases {
            let request = model.generate(
                input: UserInput(
                    prompt: prompt,
                    audios: [
                        .url(
                            URL(fileURLWithPath: fixturePath).appendingPathComponent(
                                recording + ".wav"))
                    ],
                    additionalContext: context),
                options: .init(
                    prefill: .init(stepSize: 256),
                    sampling: .init(temperature: 0), output: .init(maxTokens: 256)))
            let response = try await request.response()
            let metrics = try await request.metrics()
            print(
                "TRANSCRIBE_LES \(name) text=\(response.debugDescription) promptSeconds=\(metrics.promptSeconds) decodeSeconds=\(metrics.generationSeconds) generated=\(metrics.generatedTokenCount) peak=\(Memory.peakMemory)"
            )
            XCTAssertEqual(metrics.stopReason, .stop)
            XCTAssertFalse(response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if name == "transcription" {
                XCTAssertTrue(response.lowercased().contains("prototype"))
            }
            if name == "translation" { XCTAssertTrue(response.lowercased().contains("prototyp")) }
            if name == "speakers" {
                XCTAssertTrue(response.contains("<spk:0>"))
                XCTAssertTrue(response.contains("<spk:1>"))
                XCTAssertTrue(
                    response.trimmingCharacters(in: .whitespacesAndNewlines)
                        .hasSuffix("test results."),
                    "The fixture contains no speech after test results")
            }
            if name == "question" { XCTAssertTrue(response.lowercased().contains("prototype")) }
        }
    }
}
