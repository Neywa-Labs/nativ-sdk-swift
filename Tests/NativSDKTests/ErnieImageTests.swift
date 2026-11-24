import Foundation
import MLX
import XCTest

@testable import NativSDK

final class ErnieImageTests: XCTestCase {
    func testStrengthSelectsScheduleSuffixAndPreservesFloat32Update() {
        XCTAssertEqual(ErnieImageSchedule.editStart(steps: 8, strength: 0.6), 3)
        XCTAssertEqual(ErnieImageSchedule.editStart(steps: 5, strength: 0.5), 3)
        XCTAssertEqual(ErnieImageSchedule.editStart(steps: 5, strength: 0.7), 1)
        XCTAssertEqual(ErnieImageSchedule.editStart(steps: 8, strength: 0.001), 7)
        XCTAssertEqual(ErnieImageSchedule.editStart(steps: 8, strength: 1), 0)
        let schedule = ErnieImageSchedule(steps: 4, shift: 4, trainTimesteps: 1000)
        let sigmas: [Float] = [1, 12 / 13, 4 / 5, 4 / 7, 0]
        for (actual, expected) in zip(schedule.sigmas.asArray(Float.self), sigmas) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        let latent = MLXArray(1.0, dtype: .bfloat16)
        let velocity = MLXArray(0.75, dtype: .bfloat16)
        let output = schedule.step(velocity, at: 2, latents: latent)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(output.item(Float.self), 0.828125)
    }

    func testNativeTurboConfigIdentifiesRenamedLocalCheckpoint() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let transformer = directory.appending(path: "transformer")
        try FileManager.default.createDirectory(at: transformer, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"_class_name":"ErnieImageTransformer2DModel"}"#.utf8)
            .write(to: transformer.appending(path: "config.json"))
        XCTAssertEqual(try ImagePipelineKind.detect(in: directory), .ernieImage)

        let source = ModelSource.directory(directory)
        let base = try ErnieImageVariant.resolve(
            in: directory, source: source, transformer: .init())
        let turbo = try ErnieImageVariant.resolve(
            in: directory, source: source, transformer: .init(["lora_rank": 4, "use_lora": false]))
        XCTAssertEqual(base.steps, 50)
        XCTAssertEqual(base.guidance(editing: true), 4)
        XCTAssertEqual(turbo.steps, 8)
        XCTAssertEqual(turbo.guidance(editing: false), 1)
        XCTAssertEqual(turbo.guidance(editing: true), 3)

        try Data(#"{"model_type":"ernie_image","variant":"ernie-image-turbo"}"#.utf8)
            .write(to: directory.appending(path: "mlx_ernie_image.json"))
        XCTAssertEqual(
            try ErnieImageVariant.resolve(in: directory, source: source, transformer: .init()),
            .turbo)
    }

    func testEnhancerInputPreservesUnicodeAndOutputDimensions() throws {
        let prompt = "青い猫 / \"雨\"\n"
        let json = try ErnieImagePromptExpansion.inputJSON(prompt, width: 1024, height: 768)
        XCTAssertEqual(
            json, "{\"prompt\": \"青い猫 / \\\"雨\\\"\\n\", \"width\": 1024, \"height\": 768}")
    }
}
