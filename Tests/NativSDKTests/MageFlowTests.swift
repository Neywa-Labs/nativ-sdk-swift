import Foundation
import MLX
import XCTest

@testable import NativSDK

final class MageFlowTests: XCTestCase {
    func testAllSixVariantDefaultsAndSidecarPrecedence() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let names = [
            "Mage-Flow", "Mage-Flow-Base", "Mage-Flow-Turbo",
            "Mage-Flow-Edit", "Mage-Flow-Edit-Base", "Mage-Flow-Edit-Turbo",
        ]
        for (index, name) in names.enumerated() {
            let repo = "mage-flow-community/" + name
            let variant = try MageFlowVariant.resolve(directory: root, source: .repository(repo))
            XCTAssertEqual(variant.repository, repo)
            XCTAssertEqual(variant.editing, index >= 3)
            XCTAssertEqual(variant.steps, [20, 30, 4, 30, 30, 4][index])
            XCTAssertEqual(variant.guidance, index % 3 == 2 ? 1 : 5)
        }
        try Data(#"{"model_type":"mage_flow","variant":"mage-flow-edit-turbo"}"#.utf8)
            .write(to: root.appending(path: "mlx_mage_flow.json"))
        XCTAssertEqual(try ImagePipelineKind.detect(in: root), .mageFlow)
        XCTAssertEqual(
            try MageFlowVariant.resolve(
                directory: root, source: .repository("microsoft/Mage-Flow")), .editTurbo)
    }

    func testReferencePromptAndLatentsShareAttachmentOrder() {
        let caption = MageFlowConditioner.formattedPrompt("Blend the objects.", references: 2)
        let placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        XCTAssertTrue(
            caption.contains("Image 1:\u{20}\(placeholder)Image 2:\u{20}\(placeholder)Blend"))
        let references = MLXArray(0 ..< 12).reshaped(2, 1, 3, 2)
        let tokens = MageFlowSampling.referenceTokens(references)
        XCTAssertEqual(tokens.shape, [1, 6, 2])
        XCTAssertEqual(tokens.asArray(Int.self), Array(0 ..< 12))
    }

    func testReferenceResizeAcceptsThinInputsAndRoundsTiesToEven() throws {
        let (height, width) = try MageFlowConditioner.visionSize(
            height: 16, width: 96, factor: 32, minPixels: 65_536, maxPixels: 16_777_216)
        XCTAssertEqual(height, 128)
        XCTAssertEqual(width, 640)
        let (_, tie) = try MageFlowConditioner.visionSize(
            height: 320, width: 336, factor: 32, minPixels: 65_536, maxPixels: 16_777_216)
        XCTAssertEqual(tie, 320)
    }
}
