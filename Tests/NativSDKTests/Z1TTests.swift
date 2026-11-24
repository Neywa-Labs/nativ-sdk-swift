import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Z1TTests: XCTestCase {
    func testPublishedCheckpointTreeStrictlyLoadsAndPreservesLogits() throws {
        let source = Self.alive(Z1TModel(try Self.config()))
        var published: [String: MLXArray] = [:]
        for (key, value) in source.parameters().flattened() {
            var exported = key
            for (canonical, original) in [
                ("model.embed_tokens.", "embedding."), ("model.layers.", "blocks."),
                ("model.norm.", "norm."), ("lm_head.", "clf."),
            ] where key.hasPrefix(canonical) {
                exported = original + key.dropFirst(canonical.count)
                break
            }
            if key == "model.pe" { exported = "pe.pe" }
            published[exported] = value
        }
        let target = Z1TModel(try Self.config())
        try target.update(
            parameters: ModuleParameters.unflattened(target.sanitize(weights: published)),
            verify: .all)
        let input = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let expected = source(input, cache: nil)
        let actual = target(input, cache: nil)
        eval(expected, actual)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-6)
        XCTAssertEqual(
            Set(target.sanitize(weights: published).keys),
            Set(source.parameters().flattened().map(\.0)))
    }

    private static func config(
        fanIn: Int? = nil, tanhLinear: Bool = false
    ) throws -> Z1TConfiguration {
        try JSONDecoder.json5().decode(
            Z1TConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "z1t",
                overrides: [
                    "linear_fan_in": fanIn as Any? ?? NSNull(), "tanh_linear": tanhLinear,
                ]))
    }

    private static func alive(_ model: Z1TModel) -> Z1TModel {
        var updates: [(String, MLXArray)] = []
        for (index, (key, value)) in model.parameters().flattened().enumerated() {
            if key.hasSuffix("indices") { continue }
            eval(value)
            if abs(value.asType(.float32)).max().item(Float.self) == 0 {
                updates.append(
                    (
                        key,
                        MLXRandom.normal(value.shape, key: MLXRandom.key(700 + UInt64(index)))
                            * 0.2
                    ))
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        return model
    }

    /// The O(1) decode path and the flat forward have to agree: the streaming state is the
    /// pool's running sums plus the convolution's tail, and an empty cache is the same
    /// thing as the flat forward's left padding.
    func testDecodeStepMatchesTheFlatForward() throws {
        let config = try Self.config()
        let model = Self.alive(Z1TModel(config))
        let tokens = MLXArray(0 ..< 7).reshaped(1, 7)

        let flat = model(tokens, cache: nil)

        let cache = try model.newCache(configuration: nil)
        var stepwise: [MLXArray] = []
        for t in 0 ..< 7 {
            stepwise.append(model(tokens[0..., t ..< (t + 1)], cache: cache))
        }
        let streamed = concatenated(stepwise, axis: 1)
        eval(flat, streamed)

        XCTAssertGreaterThan(abs(flat).max().item(Float.self), 1e-3)
        XCTAssertLessThan(
            abs(flat - streamed).max().item(Float.self), 2e-3,
            "the streaming state does not reproduce the flat forward")
    }

    /// A chunked prefill has to agree too, which is the path that carries the convolution
    /// window across a boundary rather than one position at a time.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let config = try Self.config()
        let model = Self.alive(Z1TModel(config))
        let tokens = MLXArray(0 ..< 9).reshaped(1, 9)

        let flat = model(tokens, cache: nil)

        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 2e-3,
            "the convolution window or the pool did not cross the chunk boundary")
    }

    /// The convolution stencil is `exp(w) - 1`, so an all-zero `w_conv` contributes
    /// nothing and the block reduces to the pool alone.
    func testConvolutionStencilIsExpMinusOne() throws {
        let config = try Self.config()
        let block = Z1TAFTConv(config)
        block.update(
            parameters: ModuleParameters.unflattened([
                ("qkv_proj.weight", MLXRandom.normal([2 * 16 + 4, 16], key: MLXRandom.key(3))),
                ("out_proj.weight", MLXRandom.normal([16, 16], key: MLXRandom.key(4))),
            ]))
        let x = MLXRandom.normal([1, 5, 16], key: MLXRandom.key(5))

        let zeroed = block(x, cache: nil)
        block.update(
            parameters: ModuleParameters.unflattened([
                ("w_conv", MLXArray.ones([4, 3]))
            ]))
        let contributing = block(x, cache: nil)
        eval(zeroed, contributing)

        XCTAssertGreaterThan(
            abs(zeroed - contributing).max().item(Float.self), 1e-4,
            "a nonzero w_conv changed nothing, so exp(w) - 1 is not being applied")
    }

    /// A `nil` fan-in is dense; a fan-in below the input width switches to the sparse
    /// linear, whose fan-in table is a loaded parameter rather than a trained one.
    func testFanInSelectsBetweenDenseAndSparse() throws {
        let dense = Z1TAFTConv(try Self.config(fanIn: nil))
        XCTAssertTrue(dense.qkvProj is Linear)

        let sparse = Z1TAFTConv(try Self.config(fanIn: 4))
        let projection = try XCTUnwrap(sparse.qkvProj as? Z1TSparseLinear)
        XCTAssertEqual(projection.weight.shape, [2 * 16 + 4, 4])
        XCTAssertEqual(projection.indices.shape, [2 * 16 + 4, 4])
        XCTAssertEqual(projection.indices.dtype, .int32)

        // A fan-in at or above the width is dense, not a sparse table of every column.
        XCTAssertTrue(Z1TAFTConv(try Self.config(fanIn: 16)).qkvProj is Linear)
    }

    /// The sparse linear gathers `weight[o, k] * x[indices[o, k]]`, which a hand-built
    /// table pins exactly.
    func testSparseLinearGathersItsOwnColumns() {
        let linear = Z1TSparseLinear(
            inputDimensions: 4, outputDimensions: 2, fanIn: 2)
        linear.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(converting: [1.0, 10.0, 100.0, 1000.0]).reshaped(2, 2)),
                ("indices", MLXArray([Int32(0), 3, 1, 2]).reshaped(2, 2)),
                ("bias", MLXArray(converting: [0.5, -0.5])),
            ]))
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 4)
        let out = linear(x)
        eval(out)

        // row 0: 1*x[0] + 10*x[3] + 0.5 = 1 + 40 + 0.5
        // row 1: 100*x[1] + 1000*x[2] - 0.5 = 200 + 3000 - 0.5
        XCTAssertEqual(out.reshaped(2).asArray(Float.self), [41.5, 3199.5])
    }

    /// Positions come from a table that is a parameter, and reading past its end
    /// zero-extends rather than trapping.
    func testPositionTableZeroExtendsPastItsEnd() throws {
        let model = Z1TModel(try Self.config())
        let dtype = DType.float32

        let inside = model.model.positions(offset: 0, length: 4, dtype: dtype)
        eval(inside)
        XCTAssertEqual(inside.shape, [4, 16])
        XCTAssertGreaterThan(abs(inside).max().item(Float.self), 0)

        // The table is 12 long, so this straddles the end.
        let straddling = model.model.positions(offset: 10, length: 5, dtype: dtype)
        eval(straddling)
        XCTAssertEqual(straddling.shape, [5, 16])
        XCTAssertEqual(
            abs(straddling[2...]).max().item(Float.self), 0, accuracy: 1e-7,
            "past the table the rows must be zero")

        let beyond = model.model.positions(offset: 40, length: 3, dtype: dtype)
        eval(beyond)
        XCTAssertEqual(beyond.shape, [3, 16])
        XCTAssertEqual(abs(beyond).max().item(Float.self), 0, accuracy: 1e-7)
    }

    /// `tanh_linear` wraps the internal projections and leaves the head alone.
    func testTanhLinearWrapsTheBodyNotTheHead() throws {
        let plain = Self.alive(Z1TModel(try Self.config(tanhLinear: false)))
        let wrapped = Z1TModel(try Self.config(tanhLinear: true))
        wrapped.update(
            parameters: ModuleParameters.unflattened(plain.parameters().flattened()))

        let tokens = MLXArray(0 ..< 5).reshaped(1, 5)
        let a = plain(tokens, cache: nil)
        let b = wrapped(tokens, cache: nil)
        eval(a, b)
        XCTAssertGreaterThan(abs(a - b).max().item(Float.self), 1e-4)

        // The head is exempt: scaling its weight up scales the logits up, which a tanh
        // on the head would bound instead.
        wrapped.update(
            parameters: ModuleParameters.unflattened([
                ("lm_head.weight", MLXArray.ones([32, 16]) * 50)
            ]))
        let loud = wrapped(tokens, cache: nil)
        eval(loud)
        XCTAssertGreaterThan(
            abs(loud).max().item(Float.self), 1.0,
            "the head is being squashed, so tanh_linear is not exempting it")
    }

}
