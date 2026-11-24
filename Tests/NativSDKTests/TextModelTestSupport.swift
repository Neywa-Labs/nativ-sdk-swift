import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Checks numerical continuation, including a nonempty prefill and several decode steps.
func assertTextModelCacheEquivalence(
    _ model: any LanguageModel, tokens: [Int] = [1, 3, 4, 2, 5, 7],
    vocabularySize: Int? = nil, file: StaticString = #filePath, line: UInt = #line
) throws {
    model.train(false)
    let weights = model.parameters().mapValues { array in
        array.ndim > 1 && array.dtype == .float32 ? array * 0.1 : array
    }
    try model.update(parameters: weights, verify: .all)
    eval(model)
    let full = model(
        MLXArray(tokens, [1, tokens.count]), cache: try model.newCache(configuration: nil))
    eval(full)
    if let vocabularySize {
        XCTAssertEqual(full.shape, [1, tokens.count, vocabularySize], file: file, line: line)
    }
    XCTAssertTrue(full.asArray(Float.self).allSatisfy(\.isFinite), file: file, line: line)
    let cache = try model.newCache(configuration: nil)
    let prefill = model(MLXArray(Array(tokens.prefix(3)), [1, 3]), cache: cache)
    eval(prefill, cache)
    for index in 3 ..< tokens.count {
        let output = model(MLXArray([tokens[index]], [1, 1]), cache: cache)
        eval(output, cache)
        XCTAssertTrue(
            allClose(full[0..., index ..< (index + 1), 0...], output, rtol: 2e-3, atol: 1e-3).item(
                Bool.self),
            "Cached continuation differs at token \(index) for \(type(of: model)); max error \(abs(full[0..., index..<(index + 1), 0...] - output).max().item(Float.self))",
            file: file, line: line)
    }
}

/// Exercises registry dispatch, checkpoint sanitation, processor selection, and generation.
func assertTextCheckpoint(
    configuration: Data, modelType: String, quantized: Bool = false,
    file: StaticString = #filePath, line: UInt = #line
) async throws {
    MLXRandom.seed(42)
    let model = try ModelRegistry.default.createModel(
        configuration: configuration, modelType: modelType)
    try assertTextModelCacheEquivalence(model, file: file, line: line)
    if quantized {
        quantize(model: model) { _, module in
            guard
                let weight = module.parameters().flattened().first(where: { $0.0 == "weight" })?.1,
                weight.dim(-1).isMultiple(of: 32)
            else { return nil }
            return (groupSize: 32, bits: 4, mode: .affine)
        }
    }
    let input = MLXArray([1, 3, 4], [1, 3])
    let expected = model(input, cache: try model.newCache(configuration: nil)).asArray(Float.self)
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try MLX.save(
        arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
        url: directory.appendingPathComponent("model.safetensors"))
    var config = try XCTUnwrap(JSONSerialization.jsonObject(with: configuration) as? [String: Any])
    config["model_type"] = modelType
    if quantized { config["quantization"] = ["group_size": 32, "bits": 4] }
    try JSONSerialization.data(withJSONObject: config).write(
        to: directory.appendingPathComponent("config.json"))
    let loadedModel = try await NativModel.loadModel(
        from: .directory(directory), using: TextFixtureTokenizerLoader())
    let actual = try await loadedModel.withModel { _, context in
        XCTAssertTrue(context.processor is TextProcessor, file: file, line: line)
        return try context.languageModel(
            MLXArray([1, 3, 4], [1, 3]),
            cache: try context.languageModel.newCache(configuration: nil)
        ).asArray(Float.self)
    }
    XCTAssertEqual(actual, expected, file: file, line: line)
    let request = loadedModel.generate(
        input: UserInput(prompt: "hello"),
        options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 2)))
    for try await _ in request {}
    let summary = try await request.metrics()
    XCTAssertEqual(summary.generatedTokenCount, 2, file: file, line: line)
}
