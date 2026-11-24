import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// deepseek_v32's sparse latent attention plus three things of its own: a learned
/// per-head attention sink, a per-head sigmoid gate on the attention output, and a
/// residual stream carrying `hc_mult` parallel copies.
final class HyV4Tests: XCTestCase {
    private func configuration(overrides: [String: Any] = [:]) throws -> HyV4Configuration {
        try JSONDecoder().decode(
            HyV4Configuration.self,
            from: ModelTestConfigurations.configuration("hy_v4", overrides: overrides))
    }

    /// Absent from the config, the pattern is derived: the first two layers and then every
    /// fourth run their own indexer, the rest reuse the preceding selection.
    func testIndexerPatternIsDerivedWhenAbsent() throws {
        XCTAssertEqual(try configuration().indexerTypes, ["full", "full", "shared", "shared"])
        XCTAssertEqual(
            try configuration(overrides: ["num_hidden_layers": 10]).indexerTypes,
            [
                "full", "full", "shared", "shared", "shared", "full", "shared", "shared",
                "shared", "full",
            ])
        let declared = try configuration(
            overrides: ["indexer_types": ["full", "shared", "full", "shared"]])
        XCTAssertEqual(declared.indexerTypes, ["full", "shared", "full", "shared"])
    }

    /// A pattern that does not describe this stack is refused at decode, so no model can
    /// be built from one.
    func testBadIndexerPatternIsRefused() throws {
        XCTAssertThrowsError(try configuration(overrides: ["indexer_types": ["full"]]))
        XCTAssertThrowsError(
            try configuration(
                overrides: ["indexer_types": ["full", "full", "sparse", "shared"]]))
    }

    /// Only a "full" layer carries an indexer; a "shared" one reuses the selection, so it
    /// publishes no indexer weights at all.
    func testOnlyFullLayersCarryAnIndexer() throws {
        let model = HyV4Model(try configuration())
        let keys = Set(model.parameters().flattened().map(\.0))
        for layer in 0 ..< 2 {
            XCTAssertTrue(
                keys.contains("model.layers.\(layer).self_attn.indexer.wq_b.weight"),
                "layer \(layer) is full and should have one")
        }
        for layer in 2 ..< 4 {
            XCTAssertFalse(
                keys.contains { $0.hasPrefix("model.layers.\(layer).self_attn.indexer") },
                "layer \(layer) is shared and should not")
        }
    }

    /// The pre-mix collapses the copies and hands back the coefficients that fold a
    /// block's output back in; expand adds no cross-copy term.
    func testHyperConnectionCollapsesAndExpands() throws {
        let configuration = try configuration()
        let pre = HyV4IdentityHyperConnectionPre(configuration)
        let residual = MLXArray.ones([1, 3, 2, 32])
        let (collapsed, post) = pre(residual)
        XCTAssertEqual(collapsed.shape, [1, 3, 32])
        XCTAssertEqual(post.shape, [1, 3, 2])

        let expanded = hyV4HyperConnectionExpand(
            MLXArray.zeros([1, 3, 32]), residual: residual, post: post)
        XCTAssertEqual(expanded.shape, residual.shape)
        XCTAssertEqual(expanded[0, 0, 0, 0].item(Float.self), 1, accuracy: 1e-5)
    }

    /// A checkpoint may be written under the wrapper prefix; the tree is flat.
    func testWrapperPrefixAndRotaryTableAreHandled() throws {
        let model = HyV4Model(try configuration())
        let out = model.sanitize(weights: [
            "language_model.model.norm.weight": MLXArray.ones([32]),
            "model.layers.0.self_attn.indexer.rotary_emb.inv_freq": MLXArray.ones([4]),
        ])
        XCTAssertNotNil(out["model.norm.weight"])
        XCTAssertFalse(out.keys.contains { $0.contains("inv_freq") })
    }
}
