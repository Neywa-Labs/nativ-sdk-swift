import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct DeepSeekOCRTestTokenizer: Tokenizer {
    let imageTokenID = 128_815

    var bosToken: String? {
        "<｜begin▁of▁sentence｜>"
    }

    var eosToken: String? {
        "<｜end▁of▁sentence｜>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<image>") {
                output.append(imageTokenID)
                index = text.index(index, offsetBy: "<image>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            id == imageTokenID ? "<image>" : String(UnicodeScalar(id)!)
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": imageTokenID
        case bosToken: 0
        case eosToken: 1
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case imageTokenID: "<image>"
        case 0: bosToken
        case 1: eosToken
        default: nil
        }
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "<image>\nExtract.", addSpecialTokens: false)
    }
}

final class DeepSeekOCRTests: XCTestCase {
    func testConfigurationDecodesDeepSeekOCRShape() throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekOCRConfiguration.self,
            from: Self.configJSON(modelType: "deepseekocr", visionName: "deeplip_b_l"),
        )

        XCTAssertEqual(config.modelType, "deepseekocr")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.textConfiguration.firstKDenseReplace, 1)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertFalse(config.visionConfiguration.isVersion2)
        XCTAssertEqual(config.projectorConfiguration.inputDimension, 16)
    }

    func testConfigurationDecodesDeepSeekOCR2Shape() throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekOCRConfiguration.self,
            from: Self.configJSON(modelType: "deepseekocr_2", visionName: "deepencoderv2"),
        )

        XCTAssertEqual(config.modelType, "deepseekocr_2")
        XCTAssertTrue(config.visionConfiguration.isVersion2)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.projectorConfiguration.inputDimension, 8)
    }

    func testProcessorBuildsDeepSeekOCRGlobalTokenRun() async throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekOCRProcessorConfiguration.self,
            from: Self.processorJSON(localSize: 32, baseSize: 64),
        )
        let processor = DeepSeekOCRProcessor(config, tokenizer: DeepSeekOCRTestTokenizer())
        let input = UserInput(chat: [
            .user("Read.", images: [.ciImage(Self.image(width: 8, height: 8))])
        ])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 64, 64])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 3, 32, 32])
        XCTAssertEqual(prepared.image?.frames?.first?.h, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 1)
        XCTAssertEqual(prepared.text.mask?.asArray(Int8.self).count(where: { $0 == 1 }), 3)
    }

    func testProcessorBuildsDeepSeekOCR2LocalAndGlobalTokenRun() async throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekOCRProcessorConfiguration.self,
            from: Self.processorJSON(localSize: 48, baseSize: 64, minPatches: 1, maxPatches: 1),
        )
        let processor = DeepSeekOCR2Processor(config, tokenizer: DeepSeekOCRTestTokenizer())
        let input = UserInput(chat: [
            .user("Read.", images: [.ciImage(Self.image(width: 8, height: 8))])
        ])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 64, 64])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 3, 48, 48])
        XCTAssertEqual(prepared.text.mask?.asArray(Int8.self).count(where: { $0 == 1 }), 401)
    }

    func testSanitizeMapsDeepSeekOCRKeysAndTransposesConvs() throws {
        let model = try DeepSeekOCR(
            Self.smallConfig(modelType: "deepseekocr", visionName: "deeplip_b_l"))
        let weights: [String: MLXArray] = [
            "model.sam_model.patch_embed.proj.weight": MLXArray.zeros([8, 3, 16, 16]),
            "model.view_seperator": MLXArray.zeros([8]),
            "lm_head.weight": MLXArray.zeros([16, 8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["sam_model.patch_embed.proj.weight"]?.shape, [8, 16, 16, 3])
        XCTAssertNotNil(sanitized["view_separator"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
    }

    func testSanitizeMapsDeepSeekOCR2QwenKeys() throws {
        let model = try DeepSeekOCR2(
            Self.smallConfig(modelType: "deepseekocr_2", visionName: "deepencoderv2"),
        )
        let weights: [String: MLXArray] = [
            "model.qwen2_model.model.model.layers.0.self_attn.q_proj.weight":
                MLXArray.zeros([8, 8]),
            "model.qwen2_model.query_768.weight": MLXArray.zeros([144, 8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["vision_model.qwen2_encoder.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["vision_model.qwen2_encoder.query_768"])
    }

    func testUnlimitedOCRProcessorPlacesOrderedImagesAndPreservesPromptSuffix() async throws {
        let tokenizer = DeepSeekOCRTestTokenizer()
        for paged in [false, true] {
            let config = try JSONDecoder().decode(
                DeepSeekOCRProcessorConfiguration.self,
                from: Data(
                    """
                    {"image_size":64,"base_size":128,"page_layout":\(paged),"max_crops":4}
                    """.utf8))
            let processor = UnlimitedOCRProcessor(config, tokenizer: tokenizer)
            let images: [UserInput.Image] = [
                .ciImage(Self.image(width: 20, height: 20)),
                .ciImage(Self.image(width: 128, height: 64)),
            ]
            let input = UserInput(chat: [.user("Parse.", images: images)])
            let result = try await processor.prepare(input: input)
            XCTAssertEqual(result.image?.pixels.shape, [2, 3, 128, 128])
            XCTAssertEqual(result.image?.frames?.map(\.h), [1, 2])
            XCTAssertEqual(result.video?.pixels.shape, [2, 3, 64, 64])
            XCTAssertEqual(result.text.mask?.asArray(Int8.self).count(where: { $0 == 1 }), 17)
            let ids = result.text.tokens.asArray(Int.self)
            XCTAssertEqual(ids.first, 0)
            let prompt = tokenizer.decode(
                tokenIds: Array(ids.dropFirst()), skipSpecialTokens: false)
            let first = String(repeating: "<image>", count: 7)
            let second = String(repeating: "<image>", count: 10)
            XCTAssertEqual(
                prompt,
                paged
                    ? "<PAGE>" + first + "<PAGE>" + second + "Parse."
                    : first + second + "Parse.")

            let base = try await processor.prepare(
                input: UserInput(
                    chat: [.user("Parse.", images: images)], additionalContext: ["cropping": false])
            )
            XCTAssertNil(base.video)
            XCTAssertEqual(base.text.mask?.asArray(Int8.self).count(where: { $0 == 1 }), 14)
        }
    }

    func testUnlimitedOCRReferenceCacheRetainsPromptAcrossRingWraps() {
        let cache = UnlimitedOCRReferenceCache(window: 2)
        cache.promptLength = 3
        let first = MLXArray([1, 2], [1, 1, 2, 1])
        _ = cache.update(keys: first, values: first)
        for position in 3 ... 8 {
            let value = MLXArray([position], [1, 1, 1, 1])
            _ = cache.update(keys: value, values: value)
        }
        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(cache.state[0].asArray(Int.self), [1, 2, 3, 8, 7])
        let copy = cache.copy()
        let token = MLXArray([9], [1, 1, 1, 1])
        _ = copy.update(keys: token, values: token)
        XCTAssertEqual(copy.state[0].asArray(Int.self), [1, 2, 3, 8, 9])
        XCTAssertEqual(cache.state[0].asArray(Int.self), [1, 2, 3, 8, 7])
    }

    func testUnlimitedOCR2RoutesByPageAndPreservesStateThroughChunkedPrefill() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            MLXRandom.seed(47)
            let config = Data(
                """
                {
                  "model_type":"unlimited-ocr","image_token_index":112,
                  "language_config":{
                    "vocab_size":128,"hidden_size":32,"intermediate_size":64,
                    "num_hidden_layers":2,"num_attention_heads":2,"first_k_dense_replace":2
                  },
                  "vision_config":{"hidden_size":8,"layers":1,"intermediate_size":16,
                      "num_attention_heads":2,
                      "width":{"sam_vit_b":{"width":8,"layers":1,"heads":2,"global_attn_indexes":[]}}},
                  "projector_config":{"input_dim":1032,"n_embed":32},
                  "sparse_attention_config":{
                    "training_stage":"dash","use_sparse_attention":true,"dash_stage0":"head_cls",
                    "sparse_block_size":4,"sparse_local_block":2,"sparse_index_dim":16,
                    "dash_topk_blocks":2,"sparse_main_rope_mode":"page2d","sparse_rope_page_pairs":1,
                    "sparse_rope_page_mod":2,"image_token_id":112,"page_token_id":113
                  }
                }
                """.utf8)
            let model = try XCTUnwrap(
                ModelRegistry.default.createModel(configuration: config, modelType: "unlimited-ocr")
                    as? DeepSeekOCR)
            let dash = try XCTUnwrap(model.config.dashAttention)
            let ids: [Int32] =
                [0, 113] + Array(repeating: 112, count: 10) + [113]
                + Array(repeating: 112, count: 11) + [3, 18, 30, 8, 19, 20, 12]
            let pages = UnlimitedOCRPromptPages(ids: ids, parameters: dash)
            XCTAssertEqual(pages.coordinates?.page.asArray(Int32.self)[12], 2)
            XCTAssertEqual(pages.coordinates?.page.asArray(Int32.self)[24], 0)
            XCTAssertEqual(
                pages.prefillCandidates(range: 19 ..< 20, blocks: 6).asArray(Bool.self),
                [false, false, false, true, true, true])
            let idsArray = MLXArray(ids, [1, ids.count])
            let input = LMInput(text: .init(tokens: idsArray))
            let fullCache = try model.newCache(configuration: nil)
            let chunkCache = try model.newCache(configuration: nil)
            let full = try model.prepare(
                input, cache: fullCache, state: nil,
                prefill: .init(chunking: .unchunked))
            let chunked = try model.prepare(
                input, cache: chunkCache, state: nil,
                prefill: .init(stepSize: 5, chunking: .remainder))
            guard case .logits(let fullOutput) = full, case .logits(let chunkedOutput) = chunked
            else {
                return XCTFail("Unlimited-OCR prefill must produce logits")
            }
            XCTAssertLessThan(
                max(
                    abs(
                        fullOutput.logits[0..., (-1)..., 0...]
                            - chunkedOutput.logits[0..., (-1)..., 0...])
                )
                .item(Float.self), 1e-4)
            for id in [5, 113, 6, 19, 23, 113, 7, 2] {
                let token = MLXArray([id], [1, 1])
                let a = model(token, cache: fullCache)
                let b = model(token, cache: chunkCache)
                XCTAssertLessThan(max(abs(a - b)).item(Float.self), 1e-4)
            }
            let cache = try XCTUnwrap(chunkCache.first as? UnlimitedOCRDASHCache)
            XCTAssertEqual(cache.offset, ids.count + 8)
            XCTAssertEqual(cache.summaryCount, (ids.count + 8) / dash.blockSize)
            XCTAssertEqual(cache.responsePosition?.markers.item(Int32.self), 4)
        }
    }

    private static func image(width: Int, height: Int) -> CIImage {
        CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private static func processorJSON(
        localSize: Int,
        baseSize: Int,
        minPatches: Int? = nil,
        maxPatches: Int? = nil,
    )
        -> Data
    {
        let patchFields =
            if let minPatches, let maxPatches {
                #","min_patches": \#(minPatches), "max_patches": \#(maxPatches)"#
            } else {
                ""
            }
        return """
            {
              "processor_class": "DeepseekVLV2Processor",
              "image_token": "<image>",
              "patch_size": 16,
              "downsample_ratio": 4,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "normalize": true,
              "image_size": \(localSize),
              "base_size": \(baseSize)\(patchFields)
            }
            """.data(using: .utf8)!
    }

    private static func smallConfig(modelType: String, visionName: String) throws
        -> DeepSeekOCRConfiguration
    {
        try JSONDecoder.json5().decode(
            DeepSeekOCRConfiguration.self,
            from: configJSON(modelType: modelType, visionName: visionName),
        )
    }

    private static func configJSON(modelType: String, visionName: String) -> Data {
        let projectorInput = visionName == "deepencoderv2" ? 8 : 16
        return """
            {
              "model_type": "\(modelType)",
              "image_token_index": 128815,
              "language_config": {
                "model_type": "deepseek_v2",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1
              },
              "vision_config": {
                "model_type": "vision",
                "model_name": "\(visionName)",
                "image_size": 64,
                "width": {
                  "clip-l-14-224": {
                    "width": 8,
                    "layers": 1,
                    "heads": 2,
                    "patch_size": 4,
                    "image_size": 8
                  },
                  "qwen2-0-5b": {
                    "dim": 8,
                    "layers": 1,
                    "heads": 2
                  },
                  "sam_vit_b": {
                    "width": 8,
                    "layers": 1,
                    "heads": 2,
                    "patch_size": 16,
                    "global_attn_indexes": [0]
                  }
                }
              },
              "projector_config": {
                "projector_type": "linear",
                "input_dim": \(projectorInput),
                "n_embed": 8
              }
            }
            """.data(using: .utf8)!
    }
}
