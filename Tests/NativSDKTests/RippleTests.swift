import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class RippleTests: XCTestCase {
    private let configuration = Data(
        """
        {"model_type":"ricursive","hidden_size":64,"intermediate_size":128,"num_hidden_layers":5,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":128,
         "sliding_window":4,"use_valres":true,"embed_scale":8,"ffn_clamp_threshold":0.7}
        """.utf8)

    func testSourceLayoutAndHybridCacheContinuation() async throws {
        let model = try ModelRegistry.default.createModel(
            configuration: configuration, modelType: "ricursive")
        let caches = try model.newCache(configuration: nil)
        XCTAssertEqual(caches.map(\.maxSize), [4, 4, 4, 4, nil])
        XCTAssertTrue(caches[4] is KVCacheSimple)
        try await assertTextCheckpoint(configuration: configuration, modelType: "ricursive")
    }

    func testWrappedCheckpointKeepsPerLayerQuantization() async throws {
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(configuration: configuration, modelType: "ricursive")
                as? Ripple)
        try quantizeByParent(model: model) { path, module in
            guard module is Linear || module is Embedding else { return nil }
            return (32, path == "model.embed_tokens" ? 8 : 4, .affine)
        }
        let ids = [1, 5, 3, 7, 2, 9]
        let expected = model(MLXArray(ids, [1, ids.count]), cache: nil).asArray(Float.self)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(
            arrays: Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map {
                    ("language_model." + $0.0, $0.1)
                }), url: directory.appendingPathComponent("model.safetensors"))
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration) as? [String: Any])
        config["quantization"] = [
            "group_size": 32, "bits": 4,
            "language_model.model.embed_tokens": ["group_size": 32, "bits": 8],
        ]
        config["eos_token_id"] = [126, 127]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))
        let loaded = try await NativModel.loadModel(
            from: .directory(directory), using: TextFixtureTokenizerLoader())
        XCTAssertEqual(loaded.metadata.eosTokenIds, [126, 127])
        let actual = try await loaded.withModel { _, context in
            let model = try XCTUnwrap(context.model as? Ripple)
            return model(MLXArray(ids, [1, ids.count]), cache: nil).asArray(Float.self)
        }
        XCTAssertEqual(actual, expected)
    }
}
