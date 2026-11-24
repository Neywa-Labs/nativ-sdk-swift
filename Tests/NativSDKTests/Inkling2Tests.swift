import CoreImage
import Foundation
import MLX
import XCTest

@testable import NativSDK

final class Inkling2Tests: XCTestCase {
    func testRelativeAttentionRetainsChronologicalOrderAfterSlidingCacheWrap() throws {
        let data = Data(
            """
            {
              "text_config":{
                "hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "num_hidden_layers":4,"vocab_size":96,"unpadded_vocab_size":90,
                "layer_types":["hybrid_sliding","hybrid_full","hybrid_sliding","hybrid_full"],
                "sliding_window_size":4,"rel_extent":5,"d_rel":4,"log_scaling_n_floor":3,
                "n_layers_per_attn_res_block":2,"n_routed_experts":4,"num_experts_per_tok":2,
                "intermediate_size":32
              }
            }
            """.utf8)
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: data)
        let model = withRandomState(MLXRandom.RandomState(seed: 323)) { Inkling2(config) }
        // Nonconstant relative logits make key order observable at every distance.
        let weights = model.parameters().mapValues { name, weight in
            if name.hasSuffix("rel_proj") {
                return MLXArray((0 ..< weight.size).map { Float($0) / Float(weight.size) })
                    .reshaped(weight.shape)
            }
            return weight
        }
        try model.update(parameters: weights, verify: [.all])

        let tokens = MLXArray([4, 9, 11, 82, 13, 3, 53, 23, 8, 27, 41, 15], [1, 12])
        let full = model(tokens, cache: nil)
        let caches = try model.newCache(configuration: nil)
        let ranges = [0 ..< 3, 3 ..< 4, 4 ..< 6, 6 ..< 7, 7 ..< 8, 8 ..< 12]
        var decoded: [MLXArray] = []
        for range in ranges {
            decoded.append(model(tokens[0..., range], cache: caches))
            eval(caches)
        }
        let chunked = concatenated(decoded, axis: 1)
        XCTAssertEqual(chunked.shape, [1, 12, 90])
        XCTAssertTrue(all(isFinite(chunked)).item(Bool.self))
        // Multi-token and singleton GPU matmuls can use different TF32 paths.
        XCTAssertLessThan((full - chunked).abs().max().item(Float.self), 8e-3)
        let positions = try caches.map { try XCTUnwrap($0 as? CacheList)[0].offset }
        XCTAssertEqual(positions, Array(repeating: 12, count: 4))
    }
}

private struct Inkling2Tokenizer: Tokenizer {
    let pieces = [
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
        "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>",
        "think", "answer", "lookup", #"{"name":"lookup","args":{"query":"Paris"}}"#,
        #"{"name":"other","args":{}}"#, "<|unused_200054|>",
    ]
    var bosToken: String? { nil }
    var eosToken: String? { "<|content_model_end_sampling|>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        convertTokenToId(text).map { [$0] } ?? []
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap(convertIdToToken).joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces.indices.contains(id) ? pieces[id] : nil }
    func applyChatTemplate(
        messages: [Message], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            (message["content"] as? [[String: String]] ?? []).map { $0["type"] == "image" ? 11 : 7 }
        }
    }
}

extension Inkling2Tests {
    func testAssistantFramesRouteReasoningTextAndNamedToolCalls() throws {
        let decoder = ModelOutputDecoder(
            tokenizer: Inkling2Tokenizer(), format: .inkling2,
            tools: [["function": ["name": "lookup"]]])
        let events = [
            0, 1, 6, 4,
            0, 2, 7, 4,
            0, 8, 3, 9, 4,
            0, 8, 3, 10, 4,
            5,
        ].flatMap(decoder.push)
        XCTAssertEqual(events.count, 5)
        XCTAssertEqual(events.first, .reasoning("think"))
        XCTAssertEqual(events.dropFirst().first, .response("answer"))
        guard case .toolCall(let call)? = events.dropFirst(2).first,
            case .rejectedToolCall(let rejected)? = events.dropFirst(3).first
        else { return XCTFail("Expected an accepted call and a rejected call") }
        XCTAssertEqual(call.function.name, "lookup")
        XCTAssertEqual(call.function.arguments["query"], .string("Paris"))
        XCTAssertEqual(rejected.reason, .malformedSyntax)
        XCTAssertEqual(events.last, .stop)
    }

    func testPartialImageTilesExpandEachPlaceholderAndPreserveNegativePadding() async throws {
        let data = Data(
            """
            {
              "image_token_id":11,
              "text_config":{
                "hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "num_hidden_layers":1,"vocab_size":96,"n_routed_experts":4,
                "num_experts_per_tok":2,"intermediate_size":32
              },
              "vision_config":{
                "native_patch_size":2,"adapter_merge_size":2,"hidden_size":16,
                "num_hidden_layers":1,"num_attention_heads":2,"projection_hidden_dim":16
              }
            }
            """.utf8)
        let config = try JSONDecoder().decode(Inkling2ProcessorConfiguration.self, from: data)
        let processor = Inkling2Processor(config, tokenizer: Inkling2Tokenizer())
        let first = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 7, height: 5))
        let second = CIImage(color: .clear).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let input = UserInput(chat: [
            .user("Describe", images: [.ciImage(first), .ciImage(second)])
        ])
        let prepared = try await processor.prepare(input: input)
        let pixels = try XCTUnwrap(prepared.image?.pixels)
        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [11, 11, 11, 11, 11, 7])
        XCTAssertEqual(prepared.image?.frames?.map(\.h), [2, 1])
        XCTAssertEqual(prepared.image?.frames?.map(\.w), [2, 1])
        XCTAssertEqual(pixels.shape, [5, 4, 4, 3])
        XCTAssertEqual(pixels[0, 0, 0, 0].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(pixels[3, 3, 3, 0].item(Float.self), -1 - 2 / 255, accuracy: 1e-6)
        XCTAssertEqual(pixels[4, 0, 0, 0].item(Float.self), 0.6, accuracy: 1e-6)
    }
}
