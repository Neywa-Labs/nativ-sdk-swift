import AVFoundation
import CoreImage
import Foundation
import MLX
import XCTest

@testable import NativSDK

private struct MuseSparkTestTokenizer: Tokenizer {
    let ids = [
        "<|patch|>": 318, "<|video|>": 317, "<|audio|>": 316,
        "<|image_start|>": 100, "<|image_end|>": 101, "<|vid_start|>": 102,
        "<|vid_frame_separator|>": 103, "<|vid_end|>": 104,
        "<|soft_audio_start|>": 105, "<|soft_audio_end|>": 106,
    ]
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        switch text {
        case "Time: 0.00s": [80]
        case "Time: 1.25s": [125]
        default: [99]
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.description }
    func convertTokenToId(_ token: String) -> Int? { ids[token] }
    func convertIdToToken(_ id: Int) -> String? { ids.first(where: { $0.value == id })?.key }
    func applyChatTemplate(
        messages: [Message], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            (message["content"] as? [[String: String]] ?? []).map { part in
                switch part["type"] {
                case "image": 318
                case "video": 317
                case "audio": 316
                default: 99
                }
            }
        }
    }
}

final class MuseSparkTests: XCTestCase {
    private var configuration: Data {
        Data(
            """
            {
              "model_type":"muse_spark_open",
              "text_config":{
                "hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "num_hidden_layers":2,"vocab_size":320,"rms_norm_eps":1e-5,"post_norm_eps":1e-8,
                "rope_parameters":{"rope_theta":500000},
                "layer_rope_theta":[500000,0],"layer_types":["sliding_attention","full_attention"],
                "sliding_window":8,"final_logit_softcapping":20,"output_multiplier":0.1768,
                "num_local_experts":4,"num_experts_per_tok":2,"moe_hidden_size":32,"moe_intermediate_size":32
              },
              "vision_config":{
                "hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                "patch_size":2,"patch_temporal":2,"merge_size":2,"pos_emb_height":2,"pos_emb_width":2
              },
              "audio_config":{
                "patch_sizes":[2,3],"hidden_sizes":[8,16],"depths":[1,1],"num_hidden_layers":2,
                "mlp_ratio":2,"conv_kernel_size":3,"out_hidden_size":32
              },
              "image_token_id":318,"video_token_id":317,"audio_token_id":316,"projector_hidden_size":12
            }
            """.utf8)
    }

    func testMixedMediaPromptOrderAndPadding() async throws {
        let config = try JSONDecoder().decode(
            MuseSparkProcessorConfiguration.self, from: configuration)
        let processor = MuseSparkProcessor(config, tokenizer: MuseSparkTestTokenizer())
        func frame(_ color: CIColor) -> UserInput.Image {
            .ciImage(CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 12, height: 8)))
        }
        let clip = UserInput.Video.frames([
            .init(image: frame(.red), timeStamp: .zero),
            .init(image: frame(.green), timeStamp: CMTime(value: 1, timescale: 2)),
            .init(image: frame(.blue), timeStamp: CMTime(value: 5, timescale: 4)),
        ])
        let input = UserInput(chat: [
            .user("Listen", audios: [.array(.ones([5]))]),
            .user("Watch", videos: [clip]),
            .user("Compare", images: [frame(.white)], audios: [.array(.ones([8]))]),
        ])
        let processed = try await processor.prepare(input: input)
        let image = try XCTUnwrap(processed.image)
        let video = try XCTUnwrap(processed.video)
        let audio = try XCTUnwrap(processed.audio)
        XCTAssertEqual(image.pixels.shape, [24, 24])
        XCTAssertEqual(video.pixels.shape, [48, 24])
        XCTAssertEqual(video.frames?.map(\.t), [2])
        XCTAssertEqual(audio.features.shape, [2, 12])
        XCTAssertEqual(audio.mask?.sum(axis: -1).asArray(Int.self), [5, 8])
        let group = Array(repeating: 317, count: 6)
        XCTAssertEqual(
            processed.text.tokens.asArray(Int.self),
            [105, 316, 106, 99] + [102, 80] + group + [103, 125] + group + [104, 99]
                + [100] + Array(repeating: 318, count: 6) + [101, 105, 316, 316, 106, 99])

        let model = MuseSpark(config.model)
        XCTAssertEqual(
            try model.inputEmbeddings(processed).shape,
            [1, processed.text.tokens.dim(1), 32])
        let a = video.pixels[24..., ..<12]
        let b = video.pixels[24..., 12...]
        XCTAssertTrue(all(a .== b).item(Bool.self))
    }

    func testLoaderUsesPresenceOfAudioWeights() async throws {
        let config = try JSONDecoder().decode(MuseSparkConfiguration.self, from: configuration)
        let original = MuseSpark(config)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try configuration.write(to: directory.appendingPathComponent("config.json"))
        let weights = Dictionary(
            uniqueKeysWithValues: original.parameters().flattened()
                .filter { !$0.0.hasPrefix("audio_tower.") })
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NativModel.loadModel(
            from: .directory(directory), using: TextFixtureTokenizerLoader())
        try await loaded.withModel { _, context in
            let model = try XCTUnwrap(context.model as? MuseSpark)
            XCTAssertNotNil(model.vision)
            XCTAssertNil(model.audio)
            XCTAssertTrue(context.processor is MuseSparkProcessor)
            let caches = try model.newCache(configuration: nil)
            XCTAssertTrue(caches[0] is RotatingKVCache)
            XCTAssertTrue(caches[1] is KVCacheSimple)
            let logits = model(MLXArray([4, 11, 9], [1, 3]), cache: caches)
            XCTAssertEqual(logits.shape, [1, 3, 320])
            XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        }
    }
}
