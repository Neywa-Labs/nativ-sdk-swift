import Foundation
import MLX
import XCTest

@testable import NativSDK

final class ExpertSlotCacheTests: XCTestCase {
    typealias Key = ExpertWeightStore.Key
    private func fixture() throws -> (
        ExpertSlotCache, ExpertWeightStore.Layer, [URL: FileHandle], Int
    ) {
        let width = Int(getpagesize()) / MemoryLayout<Float>.size
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let values = (1 ... 4).flatMap { Array(repeating: Float($0), count: width) }
        try values.withUnsafeBytes { try Data($0).write(to: url) }
        let handle = try FileHandle(forReadingFrom: url)
        addTeardownBlock {
            try handle.close()
            try FileManager.default.removeItem(at: url)
        }
        let tensor = ExpertWeightStore.Tensor(
            url: url, offset: 0, shape: [1, width], dtype: .float32,
            bytesPerExpert: Int(getpagesize()))
        let layer = ExpertWeightStore.Layer(tensors: ["weight": tensor], quantization: [:])
        let cache = try ExpertSlotCache(layers: [layer], budget: 3 * Int(getpagesize()))
        XCTAssertEqual(cache.count, 2)
        XCTAssertLessThanOrEqual(cache.byteCount, 3 * Int(getpagesize()))
        return (cache, layer, [url: handle], width)
    }
    func testPinnedSlotsCannotBeOverwrittenAndGPUSeesReusedWeights() throws {
        let (cache, layer, handles, width) = try fixture()
        let keys = [Key(layer: 0, expert: 0), Key(layer: 0, expert: 1)]
        try cache.begin(keys)
        for key in keys {
            try cache.read(key, tensors: layer.tensors, handles: handles, speculative: false)
        }
        let slots = try cache.selected(keys)
        let before = cache.arrays["weight"]![MLXArray(slots)] * 2
        eval(before)
        XCTAssertEqual(
            before.asArray(Float.self),
            Array(repeating: 2, count: width) + Array(repeating: 4, count: width))
        let replacement = Key(layer: 0, expert: 2)
        XCTAssertNil(
            try cache.read(replacement, tensors: layer.tensors, handles: handles, speculative: true)
        )
        try cache.begin([replacement, keys[1]])
        try cache.read(replacement, tensors: layer.tensors, handles: handles, speculative: false)
        let after =
            cache.arrays["weight"]![MLXArray(try cache.selected([replacement, keys[1]]))] * 2
        eval(after)
        XCTAssertEqual(
            after.asArray(Float.self),
            Array(repeating: 6, count: width) + Array(repeating: 4, count: width))
        XCTAssertEqual(
            before.asArray(Float.self),
            Array(repeating: 2, count: width) + Array(repeating: 4, count: width))
    }
    private final class Results: @unchecked Sendable {
        let lock = NSLock()
        var slots = [Int]()
        var errors = [String]()
        func append(_ result: Result<Int?, Error>) {
            lock.lock()
            defer { lock.unlock() }
            switch result {
            case .success(let value): if let value { slots.append(value) }
            case .failure(let error): errors.append(String(describing: error))
            }
        }
    }
    func testConcurrentMissesShareASlotAndFailedReadsReleaseIt() throws {
        let (cache, layer, handles, _) = try fixture()
        let key = Key(layer: 0, expert: 1)
        try cache.begin([key])
        let results = Results()
        DispatchQueue.concurrentPerform(iterations: 16) { _ in
            results.append(
                Result {
                    try cache.read(
                        key, tensors: layer.tensors, handles: handles, speculative: false)
                })
        }
        XCTAssertTrue(results.errors.isEmpty, results.errors.joined(separator: ","))
        XCTAssertEqual(results.slots.count, 16)
        XCTAssertEqual(Set(results.slots).count, 1)
        let bad = Key(layer: 0, expert: 4)
        try cache.begin([key, bad])
        XCTAssertThrowsError(
            try cache.read(bad, tensors: layer.tensors, handles: handles, speculative: false))
        let good = Key(layer: 0, expert: 0)
        try cache.begin([key, good])
        XCTAssertNotNil(
            try cache.read(good, tensors: layer.tensors, handles: handles, speculative: false))
        XCTAssertEqual(try cache.selected([key, good]).count, 2)
    }

    func testDemandBatchWaitsForUnrelatedSpeculativeRead() throws {
        let bytes = 4 * 1024 * 1024
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data(count: 6 * bytes).write(to: url)
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
            try? FileManager.default.removeItem(at: url)
        }
        let tensor = ExpertWeightStore.Tensor(
            url: url, offset: 0, shape: [bytes / 4],
            dtype: .float32, bytesPerExpert: bytes)
        let layer = ExpertWeightStore.Layer(tensors: ["weight": tensor], quantization: [:])
        let cache = try ExpertSlotCache(layers: [layer], budget: 2 * bytes + Int(getpagesize()))
        XCTAssertEqual(cache.count, 2)
        let results = Results()
        for attempt in 0 ..< 32 {
            let first = (attempt % 2) * 2
            let selected = [Key(layer: 0, expert: first), Key(layer: 0, expert: first + 1)]
            try cache.begin(selected)
            let keys = [Key(layer: 0, expert: 4 + attempt % 2)] + selected
            DispatchQueue.concurrentPerform(iterations: 3) { index in
                results.append(
                    Result {
                        try cache.read(
                            keys[index], tensors: layer.tensors, handles: [url: handle],
                            speculative: index == 0)
                    })
            }
        }
        XCTAssertTrue(results.errors.isEmpty, results.errors.joined(separator: ","))
    }

    func testCompletedGPUWorkReleasesSlotsForSpeculativeReads() throws {
        let (cache, layer, handles, _) = try fixture()
        let keys = [Key(layer: 0, expert: 0), Key(layer: 0, expert: 1)]
        try cache.begin(keys)
        for key in keys {
            try cache.read(key, tensors: layer.tensors, handles: handles, speculative: false)
        }
        let output = cache.arrays["weight"]![MLXArray(try cache.selected(keys))] * 2
        asyncEval(output)
        let replacement = Key(layer: 0, expert: 2)
        XCTAssertNil(
            try cache.read(replacement, tensors: layer.tensors, handles: handles, speculative: true)
        )
        eval(output)
        let expected = output.asArray(Float.self)
        cache.releasePins()
        XCTAssertNotNil(
            try cache.read(replacement, tensors: layer.tensors, handles: handles, speculative: true)
        )
        XCTAssertEqual(output.asArray(Float.self), expected)
    }
    func testContiguousRecordReadPreservesProjectionBuffersAndRejectsTruncation() throws {
        let page = Int(getpagesize())
        let width = page / MemoryLayout<Float>.size
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let values = (1 ... 8).flatMap { Array(repeating: Float($0), count: width) }
        try values.withUnsafeBytes { try Data($0).write(to: url) }
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
            try? FileManager.default.removeItem(at: url)
        }
        let layer = ExpertWeightStore.Layer(
            tensors: [
                "a": .init(
                    storage: .separate(
                        (0 ..< 4).map { .init(url: url, offset: UInt64(2 * $0 * page)) }),
                    shape: [1, width], dtype: .float32, bytesPerExpert: page),
                "b": .init(
                    storage: .separate(
                        (0 ..< 4).map { .init(url: url, offset: UInt64((2 * $0 + 1) * page)) }),
                    shape: [1, width], dtype: .float32, bytesPerExpert: page),
            ], quantization: [:])
        let cache = try ExpertSlotCache(layers: [layer], budget: 6 * page)
        let keys = [Key(layer: 0, expert: 1), Key(layer: 0, expert: 3)]
        try cache.begin(keys)
        for key in keys {
            try cache.read(key, tensors: layer.tensors, handles: [url: handle], speculative: false)
        }
        let indices = MLXArray(try cache.selected(keys))
        let a = cache.arrays["a"]![indices] * 1
        let b = cache.arrays["b"]![indices] * 1
        eval(a, b)
        XCTAssertEqual(
            a.asArray(Float.self),
            Array(repeating: 3, count: width) + Array(repeating: 7, count: width))
        XCTAssertEqual(
            b.asArray(Float.self),
            Array(repeating: 4, count: width) + Array(repeating: 8, count: width))
        let writable = try FileHandle(forWritingTo: url)
        try writable.truncate(atOffset: UInt64(page))
        try writable.close()
        let missing = Key(layer: 0, expert: 0)
        try cache.begin([missing])
        XCTAssertThrowsError(
            try cache.read(
                missing, tensors: layer.tensors, handles: [url: handle], speculative: false))
        XCTAssertTrue(cache.ready([missing]).isEmpty)
    }

    func testMixedLayerSizesFallBackWithoutExceedingCacheBudget() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.safetensors")
        var weights: [String: MLXArray] = [:]
        for layer in 0 ..< 2 {
            let dtype: DType = layer == 0 ? .float32 : .float16
            let prefix = "language_model.model.layers.\(layer).mlp.experts."
            weights[prefix + "gate_up_proj"] = MLXArray.ones([32, 64, 32], dtype: dtype)
            weights[prefix + "down_proj"] = MLXArray.ones([32, 32, 32], dtype: dtype)
        }
        try save(arrays: weights, url: file)
        let budget = 192 * 1024
        let store = try ExpertWeightStore(
            urls: [file],
            layout: .init(
                layerCount: 2, expertCount: 32, expertsPerToken: 8, hiddenSize: 32,
                intermediateSize: 32, expertModulePaths: ["mlp.experts"],
                quantizedModulePath: "mlp.switch_mlp", activation: .silu, canonicalWeightKey: { $0 }
            ),
            options: .init(cacheSizeBytes: budget, prefetch: false, sharedExpertBuffers: true),
            quantization: nil, perLayerQuantization: nil)
        for layer in [1, 0, 1, 0] {
            let dtype: DType = layer == 0 ? .float32 : .float16
            for first in stride(from: 0, to: 32, by: 8) {
                let result = try store.apply(
                    MLXArray.ones([8, 32], dtype: dtype), layer: layer,
                    experts: Array(first ..< first + 8), expertIndices: Array(0 ..< 8))
                eval(result)
                XCTAssertLessThanOrEqual(store.residentBytes, budget)
            }
        }
    }

    func testDecodeUsesAvailableSharedSlotsBeyondPrefillByteCap() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.safetensors")
        let values = MLXArray((1 ... 16).map { Float($0) / 512 }).reshaped(16, 1, 1)
        let weights = broadcast(values, to: [16, 512, 512])
        let prefix = "language_model.model.layers.0.experts."
        try save(
            arrays: Dictionary(
                uniqueKeysWithValues: ["gate_proj", "up_proj", "down_proj"].map {
                    (prefix + $0 + ".weight", weights)
                }), url: file)
        let layout = ExpertStreamingLayout(
            layerCount: 1, expertCount: 16, expertsPerToken: 8, hiddenSize: 512,
            intermediateSize: 512, expertModulePaths: ["experts"],
            quantizedModulePath: "experts", activation: .gelu, canonicalWeightKey: { $0 })
        let reference = try ExpertWeightStore(
            urls: [file], layout: layout, options: .init(cacheSizeBytes: 32 * 1024 * 1024),
            quantization: nil, perLayerQuantization: nil)
        // Each expert is 3 MiB. Prefill fits five; decode fits six or eight
        // depending on the existing shared cache, including its page padding.
        for (mib, expectedBatch) in [(20, 6), (32, 8)] {
            let budget = mib * 1024 * 1024
            let store = try ExpertWeightStore(
                urls: [file], layout: layout,
                options: .init(cacheSizeBytes: budget, prefetch: false, sharedExpertBuffers: true),
                quantization: nil, perLayerQuantization: nil)
            for (length, first) in [(2, 0), (1, 0), (1, 8), (1, 0)] {
                let input = MLXArray.ones([1, length, 512])
                let indices = broadcast(MLXArray(Array(first ..< first + 8)), to: [1, length, 8])
                let expected = reference.selectedExperts(input, indices: indices, layer: 0)
                let actual = store.selectedExperts(input, indices: indices, layer: 0)
                eval(expected, actual)
                try store.checkError()
                XCTAssertTrue(allClose(expected, actual, rtol: 1e-5, atol: 1e-5).item(Bool.self))
                XCTAssertEqual(store.batchSize(layer: 0), 5)
                XCTAssertEqual(store.batchSize(layer: 0, singleToken: true), expectedBatch)
                XCTAssertLessThanOrEqual(store.residentBytes, budget)
            }
            // Independent router inputs do not themselves wait on the previous
            // output. Reusing slots must still preserve an unevaluated result.
            let input = MLXArray.ones([1, 1, 512])
            let firstIndices = MLXArray(Array(0 ..< 8), [1, 1, 8])
            let nextIndices = MLXArray(Array(8 ..< 16), [1, 1, 8])
            let first = store.selectedExperts(input, indices: firstIndices, layer: 0)
            let next = store.selectedExperts(input, indices: nextIndices, layer: 0)
            let expectedFirst = reference.selectedExperts(input, indices: firstIndices, layer: 0)
            let expectedNext = reference.selectedExperts(input, indices: nextIndices, layer: 0)
            XCTAssertTrue(allClose(first, expectedFirst, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            XCTAssertTrue(allClose(next, expectedNext, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            try store.checkError()
        }
    }

}
