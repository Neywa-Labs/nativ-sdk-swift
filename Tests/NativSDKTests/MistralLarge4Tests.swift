import Foundation
import MLX
import XCTest

@testable import NativSDK

final class MistralLarge4Tests: XCTestCase {
    private let native = """
        {
          "dim": 8, "hidden_dim": 16, "n_layers": 4, "n_heads": 2, "norm_eps": 0.00001,
          "vocab_size": 32, "kv_lora_rank": 4, "qk_rope_head_dim": 2, "qk_nope_head_dim": 2,
          "v_head_dim": 2, "nope": "rope,rope,nope,nope", "lerope": true,
          "kv_sharing": {"source_layers_str": "0,0,2,2"},
          "dsa": {"topk": 2, "head_dim": 4, "n_heads": 2, "sharing": "full,shared"},
          "moe": {"num_experts": 4, "num_experts_per_tok": 2, "expert_hidden_dim": 8}
        }
        """

    func testNativeConfigurationPreservesSharedCacheSchedules() throws {
        let normalized = try XCTUnwrap(
            MistralLarge4Configuration.normalizeNative(Data(native.utf8)))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: normalized)
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(configuration: normalized, modelType: base.modelType)
                as? MistralLarge4TextModel)
        XCTAssertEqual(model.config.text.rotary, [true, true, false, false])
        XCTAssertEqual(model.config.text.kvSources, [0, 0, 2, 2])
        XCTAssertEqual(model.config.text.fullIndexers, [true, false, true, false])
        let cache = try model.newCache(configuration: nil)
        eval(model(MLXArray([1, 6, 9, 3]), cache: cache))
        XCTAssertEqual(cache.map(\.offset), [4, 4, 4, 4])
        XCTAssertEqual(cache.map { $0.state.count }, [2, 0, 2, 0])
        let copy = cache.map { $0.copy() }
        XCTAssertEqual(copy.map { $0.trim(2) }, [2, 2, 2, 2])
        eval(model(MLXArray([9, 3]), cache: copy))
        XCTAssertEqual(copy.map(\.offset), [4, 4, 4, 4])
        XCTAssertEqual(copy.map { $0.state.count }, [2, 0, 2, 0])
    }

    func testKVReadersMustShareTheSourceRotaryLayout() throws {
        let raw = Data(native.replacingOccurrences(of: "0,0,2,2", with: "0,0,0,2").utf8)
        let normalized = try XCTUnwrap(MistralLarge4Configuration.normalizeNative(raw))
        XCTAssertThrowsError(
            try JSONDecoder().decode(MistralLarge4Configuration.self, from: normalized))
    }
}
