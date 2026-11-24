import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class LFM3Tests: XCTestCase {
    private var directory: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures/LFM3")
    }
    // CPU gives strict float32 parity on every Apple GPU generation. Opt into
    // GPU parity with LFM3_TEST_GPU=1 and MLX_ENABLE_TF32=0 before process startup.
    private func referenceMath(_ body: () throws -> Void) rethrows {
        let gpu = ProcessInfo.processInfo.environment["LFM3_TEST_GPU"] == "1"
        try Stream.withNewDefaultStream(device: gpu ? .gpu : .cpu, body)
    }
    private func configuration() throws -> LFM3Configuration {
        try JSONDecoder().decode(
            LFM3Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
    }
    private func fixture() throws -> (LFM3Omni, [String: MLXArray]) {
        let model = LFM3Omni(try configuration())
        let weights = try loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        try model.update(
            parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)),
            verify: [.all])
        model.train(false)
        return (model, try loadArrays(url: directory.appendingPathComponent("golden.safetensors")))
    }
    private func close(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        guard actual.shape == expected.shape else { return }
        XCTAssertLessThanOrEqual(
            abs(actual - expected).max().item(Float.self), tolerance, file: file, line: line)
    }
    private func input(_ g: [String: MLXArray]) -> LMInput {
        LMInput(
            text: .init(tokens: g["ids"]!),
            image: .init(pixels: g["pixels"]!, frames: [THW(1, 4, 4)]),
            audio: .init(features: g["features"]!, mask: g["mask"]!))
    }

    func testModelOwnedAudioOptions() throws {
        let model = LFM3Omni(try configuration())
        XCTAssertNoThrow(try AudioGenerationEngine.validate(.init(), audio: nil, model: model))
        XCTAssertNoThrow(
            try AudioGenerationEngine.validate(
                .init(), audio: LFM3AudioOptions(mode: .interleaved), model: model))
        XCTAssertThrowsError(
            try AudioGenerationEngine.validate(
                .init(), audio: ModelAudioOptions(), model: model))
        XCTAssertThrowsError(
            try AudioGenerationEngine.validate(
                .init(), audio: LFM3AudioOptions(maxTokens: 0), model: model))
    }

    func testPublicCheckpointConfigurations() throws {
        for name in ["production-audio", "production-omni"] {
            let data = try Data(contentsOf: directory.appendingPathComponent(name + ".json"))
            let c = try JSONDecoder().decode(LFM3Configuration.self, from: data)
            XCTAssertEqual(c.text.hiddenSize, 2048)
            XCTAssertEqual(c.numCodebooks, 12)
            XCTAssertEqual(c.schedule, [4, 4, 4])
            XCTAssertEqual(c.audio.text.hiddenLayers, 17)
            XCTAssertEqual(c.vision != nil, name == "production-omni")
        }
    }

    func testRegistryAndLegacyAudioLayout() throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        XCTAssertTrue(
            try ModelRegistry.default.createModel(configuration: data, modelType: "lfm3_omni")
                is LFM3Omni)
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        json["model_type"] = "lfm3_audio"
        json["lfm"] = json.removeValue(forKey: "text_config")
        json["encoder"] = json.removeValue(forKey: "audio_encoder_config")
        json["depthformer"] = json.removeValue(forKey: "depth_config")
        json["vision_config"] = [:] as [String: Any]
        let legacy = try JSONSerialization.data(withJSONObject: json)
        let model = try ModelRegistry.default.createModel(
            configuration: legacy, modelType: "lfm3_audio")
        XCTAssertTrue(model is LFM3Audio)
        XCTAssertEqual(model.toolCallFormat, .xmlFunction)
    }

    func testPythonMultimodalEmbeddingsAndBackbone() throws {
        try referenceMath {

            let (model, g) = try fixture()
            let embeddings = try model.inputEmbeddings(input(g))
            close(embeddings, g["embeddings"]!)
            let h = model.language.model(embeddings)
            close(h, g["hidden"]!, tolerance: 4e-5)
            // Placeholder logits are -infinity; compare the text vocabulary slice.
            close(
                model.textLogits(h)[0..., 0..., ..<80], g["logits"]![0..., 0..., ..<80],
                tolerance: 4e-5)
        }
    }

    func testMixedCacheMatchesFullPrefillAcrossSlidingWindow() throws {
        try referenceMath {

            let (model, g) = try fixture()
            let embeddings = g["embeddings"]!
            let cache = try model.newCache(configuration: nil)
            XCTAssertTrue(cache[0] is MambaCache)
            XCTAssertTrue(cache[2] is RotatingKVCache)
            var pieces = [MLXArray]()
            for i in 0 ..< embeddings.dim(1) {
                pieces.append(
                    model.language.model(embeddings[0..., i ..< (i + 1), 0...], cache: cache))
            }
            close(concatenated(pieces, axis: 1), g["hidden"]!, tolerance: 4e-5)
            XCTAssertEqual(cache.map(\.offset), [0, 12, 12])
        }
    }

    func testChunkedMultimodalPrepareMatchesPython() throws {
        try referenceMath {

            let (model, g) = try fixture()
            let cache = try model.newCache(configuration: nil)
            let result = try model.prepare(
                input(g), cache: cache, state: nil, prefill: .init(stepSize: 3))
            guard case .logits(let output) = result else { return XCTFail("Expected logits") }
            close(output.logits[0..., -1, ..<80], g["logits"]![0..., -1, ..<80], tolerance: 4e-5)
            XCTAssertEqual(cache.map(\.offset), [0, 12, 12])
        }
    }

    func testPythonAudioFrontendAndPadding() throws {
        let (_, g) = try fixture()
        close(try LFM3AudioFeatures.logMel(g["wave"]!), g["log_mel"]!, tolerance: 2e-4)
        XCTAssertEqual(
            [1, 399, 400, 401, 560, 561].map(LFM3AudioFeatures.frameCount), [1, 1, 1, 2, 2, 3])
        close(
            try LFM3AudioFeatures.logMel(.zeros([400])),
            .full([128, 1], values: MLXArray(Float(-2))))
        XCTAssertThrowsError(try LFM3AudioFeatures.logMel(.zeros([0])))
    }

    func testPythonDepthDecoderAndMaskGITFrame() throws {
        try referenceMath {

            let (model, g) = try fixture()
            close(
                model.depthDecoder(
                    g["depth_hidden"]!, codes: g["depth_codes"]!, masked: g["depth_masked"]!),
                g["depth_logits"]!)
            let frame = try model.depthDecoder.sampleFrame(
                g["depth_hidden"]!, temperature: 0, topK: 0, penalty: 1, history: [],
                order: [2, 0, 1])
            let raw = g["depth_frame"]!
            let expected = which(any(raw .== 8, axis: -1, keepDims: true), 8, raw)
            close(frame.reshaped(1, -1), expected)
            XCTAssertThrowsError(
                try model.depthDecoder.sampleFrame(
                    g["depth_hidden"]!, temperature: 0, topK: 0, penalty: 1, history: [],
                    order: [0, 0, 1]))
        }
    }

    func testMissingAndExtraMediaAreRejected() throws {
        let (model, g) = try fixture()
        XCTAssertThrowsError(try model.inputEmbeddings(.init(tokens: g["ids"]!)))
        XCTAssertThrowsError(
            try model.inputEmbeddings(
                .init(
                    text: .init(tokens: MLXArray([1, 2])),
                    image: .init(pixels: g["pixels"]!, frames: [THW(1, 4, 4)]))))
        XCTAssertThrowsError(try model.visionTower!(g["pixels"]!, grids: [THW(1, 2, 4)]))
        XCTAssertThrowsError(try LFM3Processor.expand([80], token: 80, replacements: []))
        XCTAssertEqual(
            try LFM3Processor.expand(
                [1, 80, 80, 2], token: 80, replacements: [[80, 80], [80, 80, 80]]),
            [1, 80, 80, 80, 80, 80, 2])
    }

    func testQuantizedBackboneModuleReplacement() throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var text = json["text_config"] as! [String: Any]
        text["hidden_size"] = 64
        text["intermediate_size"] = 128
        text["head_dim"] = 32
        json["text_config"] = text
        let model = LFM3Omni(
            try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json)))
        try quantizeByParent(model: model, groupSize: 32, bits: 4) { path, _ in
            path.hasPrefix("language_model.")
        }
        let output = model(MLXArray([1, 2, 3], [1, 3]), cache: nil)
        XCTAssertEqual(output.shape, [1, 3, 96])
        XCTAssertTrue(model.language.model.layers[0].feedForward.w2 is QuantizedLinear)
        XCTAssertTrue(all(isFinite(output[0..., 0..., ..<80])).item(Bool.self))
    }

    func testTextAPIReservesSpeechModesForJointGeneration() throws {
        let (model, golden) = try fixture()
        let ids = MLXArray([model.config.audioStart, model.config.mixedStart])
        let text = model.textLogits(golden["hidden"]!)
        let joint = model.textLogits(golden["hidden"]!, allowSpeech: true)
        XCTAssertTrue(all(text[.ellipsis, ids] .== -Float.infinity).item(Bool.self))
        XCTAssertTrue(all(isFinite(joint[.ellipsis, ids])).item(Bool.self))
    }

    func testLiquidFSQCodebookBoundaries() throws {
        try referenceMath {
            let config = try JSONDecoder().decode(
                LiquidCodecConfiguration.self,
                from: Data(
                    #"{"encoder_dim":4,"quantizer_dim":4,"n_q":1,"stft_n_fft":16,"stft_win_length":16,"stft_hop_length":4}"#
                        .utf8))
            let quantizer = LiquidQuantizer(config)
            try quantizer.output.update(
                parameters: ModuleParameters.unflattened([
                    "weight": eye(4), "bias": MLXArray.zeros([4]),
                ]))
            let output = quantizer(MLXArray([0, 8, 2015], [1, 1, 3]))
            close(
                output,
                MLXArray(
                    [Float(-1), -1, -1, -1, -1, -2.0 / 3, -1, -1, 0.75, 1, 2.0 / 3, 2.0 / 3],
                    [1, 3, 4])
            )
        }
    }

    func testAudioPaddingPreservesEvenAndOddClipLengths() throws {
        try referenceMath {
            let (model, _) = try fixture()
            for count in [8, 9, 16, 17, 24] {
                let features = sin(MLXArray(0 ..< (128 * count)).asType(.float32) * 0.07).reshaped(
                    1, 128, count)
                let paddedFeatures = padded(features, widths: [0, 0, IntOrPair((0, 40 - count))])
                close(
                    model.audioTower(features, lengths: [count]),
                    model.audioTower(paddedFeatures, lengths: [count]))
            }
        }
    }

    func testProcessorResizeRoundingAndVideoBudget() throws {
        let image = try LFM3Processor.targetSize(
            height: 80, width: 112, factor: 32, minPixels: 1, maxPixels: 100_000)
        XCTAssertEqual([image.0, image.1], [64, 128])
        // Odd frame counts use the rounded temporal count for the budget check,
        // but the actual frame count for the scale factor, as in the reference.
        let video = try LFM3Processor.targetSize(
            height: 320, width: 320, factor: 32, frames: 5, temporal: 2, minPixels: 1,
            maxPixels: 450_000)
        XCTAssertEqual([video.0, video.1], [320, 320])
        let small = try LFM3Processor.targetSize(
            height: 8, width: 16, factor: 32, minPixels: 1024, maxPixels: 4096)
        XCTAssertEqual([small.0, small.1], [32, 64])
    }

    func testCheckpointLiquidCodecParity() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let checkpoint = environment["LFM3_CODEC_CHECKPOINT"],
            let reference = environment["LFM3_CODEC_REFERENCE"]
        else {
            throw XCTSkip(
                "Set LFM3_CODEC_CHECKPOINT and LFM3_CODEC_REFERENCE for real codec parity")
        }
        try referenceMath {
            let codec = try LiquidCodec.load(
                from: URL(fileURLWithPath: checkpoint), dtype: .float32)
            let golden = try loadArrays(url: URL(fileURLWithPath: reference))
            close(try codec.decode(golden["codes"]!), golden["waveform"]!, tolerance: 5e-4)
        }
    }

    func testLiquidISTFTOverlapAndTrim() throws {
        let config = try JSONDecoder().decode(
            LiquidCodecConfiguration.self,
            from: Data(#"{"stft_n_fft":16,"stft_win_length":16,"stft_hop_length":4}"#.utf8))
        let istft = LiquidISTFT(config)
        // All-ones rectangular window and DC-only spectra produce a constant
        // signal after dividing by the overlap envelope, including both edges.
        try istft.update(parameters: ModuleParameters.unflattened(["window": MLXArray.ones([16])]))
        let spectrum = MLXArray.zeros([1, 5, 9], dtype: .complex64)
        spectrum[0..., 0..., 0] = MLXArray(real: 16, imaginary: 0)
        close(istft(spectrum), MLXArray.ones([1, 20]))
    }

    func testInvalidScheduleAndMissingVisionAreRejected() throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        json["maskgit_schedule"] = [1]
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json)))
        json["maskgit_schedule"] = [1, 2]
        json.removeValue(forKey: "vision_config")
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json)))
    }

    func testSanitizationIsIdempotent() throws {
        let (model, _) = try fixture()
        let w = [
            "encoder.downsample.conv_in.weight": MLXArray.zeros([4, 1, 3, 3]),
            "lfm.model.layers.0.conv.conv.weight": MLXArray.zeros([16, 1, 3]),
            "depth_embeddings.0.to_logits.weight": MLXArray.zeros([9, 16]),
        ]
        let first = model.sanitize(weights: w)
        let second = model.sanitize(weights: model.sanitize(weights: w))
        XCTAssertEqual(Set(first.keys), Set(second.keys))
        XCTAssertEqual(first["audio_tower.downsample.conv_in.weight"]?.shape, [4, 3, 3, 1])
        XCTAssertEqual(first["language_model.model.layers.0.conv.conv.weight"]?.shape, [16, 3, 1])
        for key in first.keys { close(first[key]!, second[key]!) }
    }
}
