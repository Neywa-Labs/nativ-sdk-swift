import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct ImageMarkerTokenizer: Tokenizer {
    let marker: String
    let imageID: Int
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let pieces = text.components(separatedBy: marker)
        return pieces.enumerated().flatMap { index, piece in
            (index == 0 ? [] : [imageID]) + piece.utf8.map { Int($0) + 256 }
        }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { $0 == imageID ? marker : String(UnicodeScalar($0 - 256)!) }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { token == marker ? imageID : nil }
    func convertIdToToken(_ id: Int) -> String? { id == imageID ? marker : nil }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let prompt = messages.map { message in
            if let text = message["content"] as? String { return text }
            return (message["content"] as? [[String: String]] ?? []).map {
                $0["type"] == "image" ? marker : ($0["text"] ?? "")
            }.joined()
        }.joined(separator: "\n")
        return encode(text: prompt, addSpecialTokens: false)
    }
}

final class AdditionalModelIntegrationTests: XCTestCase {
    func testRGBResizeMatchesPillowAntialiasedBilinear() {
        let source =
            ((MLXArray(0 ..< 105) * 7) % 256).asType(.float32)
            .reshaped(5, 7, 3).transposed(2, 0, 1)[.newAxis] / 255
        let expected = MLXArray([
            95, 137, 136, 133, 147, 114, 102, 138, 98, 140, 154, 116, 109, 140, 105, 147, 117, 123,
        ]).asType(.float32).reshaped(1, 3, 2, 3)
        XCTAssertLessThan(
            abs(resizeRGBBilinear(source, size: (2, 3)) * 255 - expected).max().item(Float.self),
            0.01)
        let enlarged = resizeRGBBilinear(source, size: (8, 9)) * 255
        XCTAssertLessThan(
            abs(enlarged[0, 0, 0] - MLXArray([0, 14, 30, 47, 63, 79, 96, 112, 126])).max().item(
                Float.self), 0.01)
    }

    func testPatchProcessorsPreserveLayoutAndExpandImageSpans() async throws {
        let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        let kimiData = Data(
            #"{"vision_config":{"patch_size":2,"merge_kernel_size":[2,2]},"in_patch_limit":16,"patch_limit_on_one_side":4}"#
                .utf8)
        let kimi = KimiK3Processor(
            try JSONDecoder().decode(KimiK3ProcessorConfiguration.self, from: kimiData),
            tokenizer: ImageMarkerTokenizer(marker: "<|media_pad|>", imageID: 90))
        let kimiInput = try await kimi.prepare(
            input: UserInput(prompt: "read", images: [.ciImage(red)]))
        XCTAssertEqual(kimiInput.text.tokens.asArray(Int.self).filter { $0 == 90 }.count, 4)
        XCTAssertEqual(kimiInput.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4]])
        XCTAssertEqual(
            try XCTUnwrap(kimiInput.image).pixels[0, 0, 0, 0].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(kimiInput.image?.pixels[0, 1, 0, 0].item(Float.self), -1)

        let phiData = Data(#"{"vision_config":{"patch_size":2},"max_num_patches":6}"#.utf8)
        let phi = Phi4SigLipProcessor(
            try JSONDecoder().decode(Phi4SigLipProcessorConfiguration.self, from: phiData),
            tokenizer: ImageMarkerTokenizer(marker: "<image>", imageID: 90))
        let phiInput = try await phi.prepare(
            input: UserInput(prompt: "read", images: [.ciImage(red)]))
        XCTAssertEqual(phiInput.text.tokens.asArray(Int.self).filter { $0 == -200 }.count, 1)
        let patches = try XCTUnwrap(phiInput.image?.pixels)
        XCTAssertEqual(patches.shape, [1, 6, 12])
        XCTAssertEqual(patches[0, 0, ..<3].asArray(Float.self), [1, -1, -1])
        XCTAssertEqual(abs(patches[0, 4...]).max().item(Float.self), 0)

        let nemoData = Data(
            #"{"final_size":[12,16],"image_mean":[0,0,0],"image_std":[1,1,1]}"#.utf8)
        let nemo = NemotronParseProcessor(
            try JSONDecoder().decode(NemotronParseProcessorConfiguration.self, from: nemoData),
            tokenizer: ImageMarkerTokenizer(marker: "<image>", imageID: 90))
        let document = try await nemo.prepare(
            input: UserInput(prompt: "read", images: [.ciImage(red)]))
        let canvas = try XCTUnwrap(document.image?.pixels)
        XCTAssertEqual(canvas[0, 1, 0, 0].item(Float.self), 1)
        XCTAssertEqual(canvas[0, 1, 2, 4].item(Float.self), 0)
        XCTAssertEqual(canvas[0, 1, 9, 11].item(Float.self), 0)
        XCTAssertEqual(canvas[0, 1, 10, 12].item(Float.self), 1)
    }

    func testSparseCachesCopyTheirIndexStateAndRewindTogether() throws {
        let keys = MLXArray(0 ..< 12).asType(.float32).reshaped(1, 1, 3, 4)
        let qsa = QSAKVCache()
        _ = qsa.update(keys: keys, values: keys + 1)
        try qsa.updateIndexer(
            keys: keys.reshaped(1, 3, 4), positionIDs: MLXArray(0 ..< 3)[.newAxis])
        qsa.indexBlockKeys = MLXArray.ones([1, 1, 4])
        qsa.indexBlockRatio = 2
        let copy = try XCTUnwrap(qsa.copy() as? QSAKVCache)
        XCTAssertEqual(qsa.trim(2), 2)
        try qsa.updateIndexer(
            keys: MLXArray.zeros([1, 1, 4]), positionIDs: MLXArray([1])[.newAxis])
        XCTAssertEqual(copy.offset, 3)
        XCTAssertEqual(copy.indexKeys?.dim(1), 3)
        XCTAssertEqual(copy.indexBlockRatio, 2)
        XCTAssertEqual(copy.indexKeys?[0, 2, 3].item(Float.self), 11)
        XCTAssertNil(qsa.indexBlockKeys)
        XCTAssertTrue(
            KVCacheLeaf(path: [], cache: qsa).supports(
                .init(strategy: .fullPrecision), protectedPaths: []) == true)

        let mini = MiniMaxM3Text.IndexKVCache()
        _ = mini.update(keys: keys, values: keys)
        _ = mini.updateIndexAndFetch(keys + 100)
        let saved = try XCTUnwrap(mini.copy() as? MiniMaxM3Text.IndexKVCache)
        XCTAssertEqual(mini.trim(2), 2)
        _ = mini.update(keys: keys[.ellipsis, ..<1, 0...], values: keys[.ellipsis, ..<1, 0...])
        let changed = mini.updateIndexAndFetch(MLXArray.zeros([1, 1, 1, 4]))
        XCTAssertEqual(changed.dim(2), 2)
        XCTAssertEqual(changed[0, 0, 1, 0].item(Float.self), 0)
        let restored = saved.updateIndexAndFetch(MLXArray.ones([1, 1, 1, 4]))
        XCTAssertEqual(saved.indexOffset, 4)
        XCTAssertEqual(restored[0, 0, 2, 3].item(Float.self), 111)
        XCTAssertTrue((QSAKVCache().copy() as? QSAKVCache)?.innerState().isEmpty == true)
        XCTAssertTrue(
            (MiniMaxM3Text.IndexKVCache().copy() as? MiniMaxM3Text.IndexKVCache)?.innerState()
                .isEmpty == true)
    }

    func testGridProcessorsComposeGeometryAndExpandOnlyImageTokens() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Sidecar defaults conflict with the actual checkpoint geometry.
        try Data(
            #"{"patch_size":14,"image_mean":[0,0,0],"image_std":[1,1,1],"min_pixels":64,"max_pixels":64,"min_image_tokens":1,"max_image_tokens":1}"#
                .utf8
        ).write(to: directory.appendingPathComponent("preprocessor_config.json"))
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        for (name, marker, expected) in [
            ("cohere_compass", "<|IMAGE_PAD|>", 1), ("hunyuan_vl", "<image>", 4),
            ("glm5_next", "<|image|>", 1), ("minimax_m3_vl", "]<]image[>[", 1),
        ] {
            let data = Data(
                """
                {"model_type":"\(name)","image_token_id":90,"vision_config":{"patch_size":4,"spatial_merge_size":2,"temporal_patch_size":1}}
                """.utf8)
            let (composed, type) = try await ModelLoader().loadProcessorConfig(
                from: directory, modelType: name, configData: data)
            let tokenizer = ImageMarkerTokenizer(marker: marker, imageID: 90)
            let processor = try ModelRegistry.default.createProcessor(
                configuration: composed, processorType: type, tokenizer: tokenizer)
            let input = try await processor.prepare(
                input: UserInput(chat: [
                    .user("read", images: [.ciImage(image)]), .assistant("answer"),
                    .user("follow up"),
                ]))
            let ids = input.text.tokens.asArray(Int.self)
            XCTAssertEqual(ids.filter { $0 == 90 }.count, expected, name)
            XCTAssertTrue(
                tokenizer.decode(tokenIds: ids, skipSpecialTokens: false).contains("follow up"),
                name)
            let pixels = try XCTUnwrap(input.image?.pixels)
            XCTAssertEqual(pixels.shape, [4, 48], name)
            XCTAssertEqual(pixels[0, 0].item(Float.self), 1, accuracy: 1e-4, name)
            XCTAssertEqual(pixels[0, 16].item(Float.self), 0, accuracy: 1e-4, name)
        }
    }

    func testMixedPrecisionReplacementSkipsDensePrefixAndKeepsTupleSiblings() throws {
        final class Block: Module {
            @ModuleInfo var projection: Linear
            override init() { _projection.wrappedValue = Linear(64, 64, bias: false) }
        }
        final class Model: Module {
            @ModuleInfo var layers: [Block]
            @ModuleInfo var merger: (Linear, GELU, Linear)
            override init() {
                _layers.wrappedValue = [Block(), Block()]
                _merger.wrappedValue = (Linear(64, 64), GELU(), Linear(64, 64))
                super.init()
            }
        }
        let model = Model()
        let before = model.layers[0].projection.weight[.ellipsis]
        try quantizeByParent(
            model: model, groupSize: 32, bits: 8,
            filter: { path, _ in path == "layers.1.projection" || path.hasPrefix("merger.") })
        XCTAssertTrue(model.layers[1].projection is QuantizedLinear)
        XCTAssertFalse(model.layers[0].projection is QuantizedLinear)
        XCTAssertEqual(abs(model.layers[0].projection.weight - before).max().item(Float.self), 0)
        XCTAssertTrue(model.merger.0 is QuantizedLinear)
        XCTAssertTrue(model.merger.2 is QuantizedLinear)
        let output = model.merger.2(model.merger.1(model.merger.0(MLXArray.ones([1, 64]))))
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
