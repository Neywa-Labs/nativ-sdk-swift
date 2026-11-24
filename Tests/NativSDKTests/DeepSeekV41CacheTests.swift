import Foundation
import MLX
import XCTest

@testable import NativSDK

/// Shapes and values captured from upstream mlx-vlm's DeepseekV41Cache.
final class DeepSeekV41CacheTests: XCTestCase {
    private func rows(_ count: Int, _ width: Int, offset: Float = 0) -> MLXArray {
        (MLXArray(0 ..< (count * width)).asType(.float32) + offset)
            .reshaped(1, count, width)
    }

    /// Buffers grow a block at a time but publish only the populated prefix, so a
    /// consumer never reads the slack.
    func testAppendPublishesOnlyThePopulatedPrefix() {
        let cache = DeepSeekV41Cache(layers: 3)
        XCTAssertEqual(cache.step, 256)

        XCTAssertEqual(cache.appendCompressed(layer: 1, values: rows(2, 4)).shape, [1, 2, 4])
        let grown = cache.appendCompressed(layer: 1, values: rows(3, 4, offset: 100))
        XCTAssertEqual(grown.shape, [1, 5, 4])
        XCTAssertEqual(grown[0, 2].asArray(Float.self), [100, 101, 102, 103])
    }

    /// Crossing a block boundary reallocates and copies the populated prefix across.
    func testAppendGrowsPastOneBlock() {
        let cache = DeepSeekV41Cache(layers: 3)
        _ = cache.appendCompressed(layer: 1, values: rows(2, 4))
        _ = cache.appendCompressed(layer: 1, values: rows(3, 4, offset: 100))
        XCTAssertEqual(
            cache.compressedCapacity(layer: 1), 256,
            "five rows still take a whole block")
        let big = cache.appendCompressed(
            layer: 1, values: MLXArray.zeros([1, 300, 4], dtype: .float32))
        XCTAssertEqual(big.shape, [1, 305, 4])
        XCTAssertEqual(cache.compressedCapacity(layer: 1), 512)
        XCTAssertEqual(
            big[0, 2].asArray(Float.self), [100, 101, 102, 103],
            "the rows written before the reallocation must survive it")
    }

    /// The two growing slots keep independent buffers.
    func testCompressedAndIndexKeysAreSeparate() {
        let cache = DeepSeekV41Cache(layers: 2)
        _ = cache.appendCompressed(layer: 0, values: rows(2, 4))
        _ = cache.appendIndexKeys(layer: 0, values: rows(5, 4))
        XCTAssertEqual(cache.compress[0]?.dim(1), 2)
        XCTAssertEqual(cache.keys[0]?.dim(1), 5)
    }

    /// Five per-layer slots plus Engram plus the four handoffs: 5*3 + 1 + 4 = 20.
    func testStateRoundTripsThroughSerialization() {
        let cache = DeepSeekV41Cache(layers: 3)
        _ = cache.appendIndexKeys(layer: 0, values: rows(2, 4))
        cache.window[2] = rows(5, 4)
        cache.compressorState.kvState[1] = rows(3, 4)
        cache.offset = 7
        cache.engram = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        cache.candidates = MLXArray([true, false]).reshaped(1, 2)

        let state = cache.state
        XCTAssertEqual(state.count, 20)
        XCTAssertEqual(cache.metaState, ["7"])

        let restored = DeepSeekV41Cache()
        restored.state = state
        restored.metaState = cache.metaState
        XCTAssertEqual(restored.offset, 7)
        XCTAssertEqual(restored.layers, 3)
        XCTAssertEqual(restored.keys[0]?.shape, [1, 2, 4])
        XCTAssertEqual(restored.window[2]?.shape, [1, 5, 4])
        XCTAssertNil(restored.window[0], "an empty stand-in restores as unset")
        XCTAssertEqual(restored.compressorState.kvState[1]?.shape, [1, 3, 4])
        XCTAssertEqual(restored.engram?.asArray(Int32.self), [1, 2, 3])
        XCTAssertEqual(restored.candidates?.asArray(Bool.self), [true, false])
        XCTAssertNotNil(restored.candidates)
        XCTAssertNil(restored.indexKeys, "a handoff that was unset stays unset")
    }

    /// A restored cache can keep appending to the slot it restored, which it cannot do
    /// if the buffer was not adopted along with the published prefix.
    func testRestoredCacheKeepsAppending() {
        let cache = DeepSeekV41Cache(layers: 2)
        _ = cache.appendCompressed(layer: 0, values: rows(2, 4))
        let restored = DeepSeekV41Cache()
        restored.state = cache.state
        let grown = restored.appendCompressed(layer: 0, values: rows(1, 4, offset: 50))
        XCTAssertEqual(grown.shape, [1, 3, 4])
        XCTAssertEqual(grown[0, 0].asArray(Float.self), [0, 1, 2, 3])
        XCTAssertEqual(grown[0, 2].asArray(Float.self), [50, 51, 52, 53])
    }

    /// The handoff is cleared at initialization, not per step: a step that completes no
    /// compression group leaves the consumers reading what the source last published.
    func testHandoffSurvivesAStepAndClearsOnReset() {
        let cache = DeepSeekV41Cache(layers: 2)
        cache.compressKV = rows(2, 4)
        cache.advance(1)
        XCTAssertNotNil(cache.compressKV, "advancing a step must not clear the handoff")
        XCTAssertEqual(cache.offset, 1)
        cache.resetHandoff()
        XCTAssertNil(cache.compressKV)
    }

    /// A copy is independent: appending to one must not reach the other.
    func testCopyIsIndependent() throws {
        let cache = DeepSeekV41Cache(layers: 2)
        _ = cache.appendCompressed(layer: 0, values: rows(2, 4))
        cache.offset = 5
        let copy = try XCTUnwrap(cache.copy() as? DeepSeekV41Cache)
        XCTAssertEqual(copy.offset, 5)
        XCTAssertEqual(copy.compressedCapacity(layer: 0), 256)
        _ = copy.appendCompressed(layer: 0, values: rows(1, 4, offset: 99))
        XCTAssertEqual(copy.compress[0]?.dim(1), 3)
        XCTAssertEqual(cache.compress[0]?.dim(1), 2)
        XCTAssertEqual(copy.compress[0]?[0, 2].asArray(Float.self), [99, 100, 101, 102])
        XCTAssertEqual(
            cache.appendCompressed(layer: 0, values: rows(1, 4, offset: 7))[0, 2]
                .asArray(Float.self), [7, 8, 9, 10],
            "the original's own append must not see the copy's row")
        XCTAssertEqual(
            copy.compress[0]?[0, 2].asArray(Float.self), [99, 100, 101, 102],
            "the copy's row was overwritten through a shared buffer")
    }
}
