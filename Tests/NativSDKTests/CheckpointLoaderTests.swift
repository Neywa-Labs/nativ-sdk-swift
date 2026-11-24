import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class CheckpointModel: Module, LanguageModel, AdditionalWeightFilesProviding {
    @ModuleInfo var proj = Linear(64, 64, bias: false)
    @ModuleInfo var other = Linear(64, 64, bias: false)
    var additionalWeightFiles: [String] = []
    var prepared = false
    var failPreparation = false
    var metadataSeen: [String: String] = [:]

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        metadataSeen = metadata
        return Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                (
                    metadata["format"] == "test"
                        ? key.replacingOccurrences(of: "checkpoint.", with: "") : key, value
                )
            })
    }

    func prepare() throws {
        prepared = true
        if failPreparation { throw CocoaError(.fileReadUnknown) }
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { other(proj(inputs)) }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [] }
}

final class CheckpointLoaderTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url.resolvingSymlinksInPath()
    }

    private func file(_ name: String, in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: url)
        return url
    }

    private func index(_ map: [String: String], in directory: URL) throws {
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    func testIndexedFilesDeduplicateSortAndIncludeExplicitNestedAndExtraFiles() throws {
        let dir = try directory()
        let a = try file("model-a.safetensors", in: dir)
        let b = try file("nested/model-b.safetensors", in: dir)
        let extra = try file("projector.safetensors", in: dir)
        _ = try file("unrelated.safetensors", in: dir)
        try index(
            [
                "a": a.lastPathComponent, "a2": a.lastPathComponent,
                "b": "nested/model-b.safetensors",
            ], in: dir)
        XCTAssertEqual(
            try safetensorWeightURLs(
                in: dir,
                additionalFiles: [
                    "model-a.safetensors", "projector.safetensors", "missing.safetensors",
                ]), [a, b, extra])
    }

    func testStaleIndexAndConventionalFallbacks() throws {
        let dir = try directory()
        let arbitrary = try file("aux.safetensors", in: dir)
        _ = try file("nested/ignored.safetensors", in: dir)
        XCTAssertEqual(
            try safetensorWeightURLs(in: dir).map(\.lastPathComponent),
            [arbitrary.lastPathComponent])
        let weight = try file("weight-a.safetensors", in: dir)
        XCTAssertEqual(
            try safetensorWeightURLs(in: dir).map(\.lastPathComponent), [weight.lastPathComponent])
        let model = try file("model-a.safetensors", in: dir)
        try index(["weight": "missing.safetensors"], in: dir)
        XCTAssertEqual(
            try safetensorWeightURLs(in: dir).map(\.lastPathComponent), [model.lastPathComponent])
        try Data("not json".utf8).write(
            to: dir.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertThrowsError(try safetensorWeightURLs(in: dir))
    }

    func testConcurrentReaderPreservesFilePrecedenceAndFirstNonemptyMetadata() throws {
        let dir = try directory()
        let first = dir.appendingPathComponent("model-a.safetensors")
        let second = dir.appendingPathComponent("model-b.safetensors")
        try MLX.save(arrays: ["a": MLXArray([1, 2]), "shared": MLXArray([3])], url: first)
        try MLX.save(
            arrays: ["b": MLXArray([4]), "shared": MLXArray([5])], metadata: ["source": "second"],
            url: second)
        let result = try loadWeightArrays(urls: [first, second])
        XCTAssertEqual(result.weights["shared"]?.asArray(Int.self), [5])
        XCTAssertEqual(result.weights["a"]?.asArray(Int.self), [1, 2])
        XCTAssertEqual(result.metadata, ["source": "second"])
        let spans = try safetensorSpansInFileOrder(url: first)
        XCTAssertEqual(Set(spans.map(\.name)), ["a", "shared"])
        XCTAssertEqual(spans.reduce(0) { $0 + $1.byteCount }, 12)
        // Grouping must partition the input without duplication, including zero-sized tensors.
        for counts: [Int64] in [[0, 0], [1, 20, 2, 0, 30], [1024], []] {
            let groups = contiguousLoadGroups(byteCounts: counts, groupCount: 4)
            XCTAssertEqual(groups.flatMap(Array.init), Array(counts.indices))
            XCTAssertTrue(groups.allSatisfy { !$0.isEmpty })
        }
    }

    func testPlainAndMixedQuantizedCheckpointsRoundTripParametersAndLogits() throws {
        for quantized in [false, true] {
            let dir = try directory()
            let source = CheckpointModel()
            let perLayer = BaseConfiguration.PerLayerQuantization(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "proj": .quantize(.init(groupSize: 32, bits: 8)), "other": .skip,
                ])
            if quantized {
                quantize(model: source) { path, _ in perLayer.quantization(layer: path)?.asTuple }
            }
            let arrays = Dictionary(
                uniqueKeysWithValues: source.parameters().flattened().map {
                    ("checkpoint." + $0.0, $0.1)
                })
            try MLX.save(
                arrays: arrays, metadata: ["format": "test"],
                url: dir.appendingPathComponent("model.safetensors"))
            let local = CheckpointModel()
            try CheckpointLoader.load(
                modelDirectory: dir, model: local,
                quantization: quantized ? .init(groupSize: 32, bits: 4) : nil,
                perLayerQuantization: quantized ? perLayer : nil)
            XCTAssertTrue(local.prepared)
            XCTAssertEqual(local.metadataSeen, ["format": "test"])
            XCTAssertEqual(local.proj is QuantizedLinear, quantized)
            XCTAssertFalse(local.other is QuantizedLinear)
            let expectedParameters = Dictionary(
                uniqueKeysWithValues: source.parameters().flattened())
            for (key, actual) in local.parameters().flattened() {
                let expected = try XCTUnwrap(expectedParameters[key])
                XCTAssertEqual(actual.dtype, expected.dtype)
                XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), key)
            }
            let input = MLXArray(0 ..< 64).asType(.float32).reshaped(1, 64) / 64
            XCTAssertTrue(
                allClose(local(input, cache: nil), source(input, cache: nil)).item(Bool.self))
        }
    }

    func testAdditionalWeightsSanitizationAndPreparationFailurePolicy() throws {
        let dir = try directory()
        let source = CheckpointModel()
        try MLX.save(
            arrays: ["proj.weight": source.proj.weight],
            url: dir.appendingPathComponent("model.safetensors"))
        try MLX.save(
            arrays: ["other.weight": source.other.weight],
            url: dir.appendingPathComponent("extra.safetensors"))
        let loaded = CheckpointModel()
        loaded.additionalWeightFiles = ["extra.safetensors"]
        loaded.failPreparation = true
        let report = try CheckpointLoader.load(modelDirectory: dir, model: loaded)
        XCTAssertNotNil(report.optimizationError)
        XCTAssertTrue(loaded.prepared)
        XCTAssertTrue(arrayEqual(loaded.other.weight, source.other.weight).item(Bool.self))
    }

    func testMissingAndUnexpectedParametersFailBeforePreparation() throws {
        let dir = try directory()
        let model = CheckpointModel()
        XCTAssertThrowsError(try CheckpointLoader.load(modelDirectory: dir, model: model))
        XCTAssertFalse(model.prepared)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        weights["unexpected"] = MLXArray([1])
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try CheckpointLoader.load(modelDirectory: dir, model: model))
        XCTAssertFalse(model.prepared)
    }
}
