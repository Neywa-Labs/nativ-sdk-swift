import CoreImage
import CoreMedia
import MLX
import XCTest

@testable import NativSDK

private struct MiniCPMVTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|im_start|>": 248_045,
        "<|im_end|>": 248_046,
        "<|image_pad|>": 248_056,
        "<|video_pad|>": 248_057,
        "<unk>": 248_077,
        "<image>": 248_078,
        "</image>": 248_079,
        "<slice>": 248_088,
        "</slice>": 248_089,
        "<image_id>": 248_090,
        "</image_id>": 248_091,
    ]

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|im_end|>"
    }

    var unknownToken: String? {
        "<unk>"
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var result: [Int] = addSpecialTokens ? [248_045] : []
        var index = text.startIndex
        let specials = tokenIDs.keys.sorted { $0.count > $1.count }

        while index < text.endIndex {
            if let match = specials.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[match]
            {
                result.append(id)
                index = text.index(index, offsetBy: match.count)
            } else {
                let scalar = text[index].unicodeScalars.first?.value ?? 0
                result.append(Int(scalar) % 1000 + 10)
                index = text.index(after: index)
            }
        }
        return result
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { convertIdToToken($0) ?? String($0) }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenIDs[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenIDs.first { $0.value == id }?.key
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "Describe.", addSpecialTokens: false)
    }
}

final class MiniCPMVTests: XCTestCase {
    func testProcessorBuildsMiniCPMPlaceholderSpansAndPackedPixels() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MiniCPMVProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MiniCPMV4_6Processor",
                  "max_slice_nums": 9,
                  "scale_resolution": 56,
                  "patch_size": 14,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "slice_mode": true,
                  "use_image_id": true,
                  "downsample_mode": "16x"
                }
                """.data(using: .utf8)),
        )
        let processor = MiniCPMVProcessor(processorConfig, tokenizer: MiniCPMVTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 248_056 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [9408])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertNil(prepared.video?.frames)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testProcessorBuildsVideoInputFromFramePrimitive() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MiniCPMVProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MiniCPMV4_6Processor",
                  "max_slice_nums": 9,
                  "max_num_frames": 4,
                  "scale_resolution": 56,
                  "patch_size": 14,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "slice_mode": true,
                  "use_image_id": true,
                  "downsample_mode": "16x"
                }
                """.data(using: .utf8)),
        )
        let processor = MiniCPMVProcessor(processorConfig, tokenizer: MiniCPMVTestTokenizer())
        let frames = [
            UserInput.VideoFrame(
                image: .ciImage(
                    CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))),
                timeStamp: CMTime(value: 0, timescale: 1),
            ),
            UserInput.VideoFrame(
                image: .ciImage(
                    CIImage(color: .green).cropped(
                        to: CGRect(x: 0, y: 0, width: 28, height: 28))),
                timeStamp: CMTime(value: 1, timescale: 1),
            ),
        ]
        let input = UserInput(chat: [.user("Describe.", videos: [.frames(frames)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertNotNil(prepared.image)
        XCTAssertEqual(tokens.count(where: { $0 == 248_057 }), 2)
        XCTAssertEqual(prepared.image?.pixels.shape, [18816])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 2])
        XCTAssertNil(prepared.video?.frames)
        XCTAssertEqual(prepared.image?.frames?.count, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSanitizeSkipsMLXMetadataAndMapsReferenceKeys() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPMVConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "minicpmv4_6",
                  "text_config": {"tie_word_embeddings": true},
                  "vision_config": {}
                }
                """.data(using: .utf8)),
        )
        let model = MiniCPMV(config)
        let currentWeights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([8, 4])
        ]

        let skipped = model.sanitize(weights: currentWeights, metadata: ["format": "mlx"])
        XCTAssertNotNil(skipped["language_model.model.embed_tokens.weight"])

        let sanitized = model.sanitize(weights: [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            "language_model.model.layers.0.input_layernorm.weight": MLXArray.zeros([4]),
            "language_model.lm_head.weight": MLXArray.zeros([8, 4]),
            "model.vision_tower.vit_merger.linear_1.weight": MLXArray.zeros([4, 4]),
            "vpm.embeddings.patch_embedding.weight": MLXArray.zeros([4, 3, 2, 2]),
            "llm.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([2, 3, 4]),
            "unused.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["language_model.model.layers.0.input_layernorm.weight"])
        XCTAssertNil(sanitized["language_model.model.model.layers.0.input_layernorm.weight"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNotNil(sanitized["vit_merger.linear_1.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.embeddings.patch_embedding.weight"]?.shape, [4, 2, 2, 3])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"]?.shape,
            [2, 4, 3],
        )
        XCTAssertNil(sanitized["unused.weight"])
    }
}
