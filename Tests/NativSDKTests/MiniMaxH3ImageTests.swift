import Foundation
import MLX
import XCTest

@testable import NativSDK

final class MiniMaxH3ImageTests: XCTestCase {
    func testLayoutKeepsOrderedReferencesAndSeparatesTheirTimestep() {
        let layout = MiniMaxH3ImagePacking.layout(
            text: [1, 0, 1], grids: [(2, 4), (4, 2), (4, 4)], patch: [1, 2, 2])
        XCTAssertEqual(layout.referenceCount, 4)
        XCTAssertEqual(layout.targetStart, 7)
        XCTAssertEqual(
            layout.positions[0..., 0].asArray(Float.self), [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5])
        XCTAssertEqual(
            layout.modulationRows.asArray(Int32.self), [1, 0, 1, 3, 3, 3, 3, 0, 0, 0, 0])
        XCTAssertEqual(layout.timeRows.asArray(Int32.self), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    }

    func testPatchPackingPreservesChannelAndPixelOrder() {
        let input = MLXArray(0 ..< 32).reshaped(1, 2, 8, 2)
        let patch = [1, 2, 2]
        let packed = MiniMaxH3ImagePacking.pack(input, patch: patch)
        XCTAssertEqual(packed[0].asArray(Int.self), [0, 2, 16, 18, 1, 3, 17, 19])
        let unpacked = MiniMaxH3ImagePacking.unpack(
            packed, height: 2, width: 8, channels: 2, patch: patch)
        XCTAssertTrue(all(input .== unpacked).item(Bool.self))
    }

    func testTilePlansDistributeAlignedOverlapAcrossRectangularCanvases() {
        let plan = MiniMaxH3ImageVAE.tilePlan(512, tile: 256, overlap: 64, alignment: 16)
        XCTAssertEqual(plan.ranges, [0 ..< 256, 128 ..< 384, 256 ..< 512])
        XCTAssertEqual(plan.overlap, [128, 128])
        let belowThreshold = MiniMaxH3ImageVAE.tilePlan(192, tile: 256, overlap: 64, alignment: 16)
        XCTAssertEqual(belowThreshold.ranges, [0 ..< 192])
        XCTAssertTrue(belowThreshold.overlap.isEmpty)
    }
}

extension MiniMaxH3ImageTests {
    func testPromptExpanderDecodesNestedQwenProcessorSettings() throws {
        let data = Data(
            """
            {
              "processor_class": "Qwen3VLProcessor",
              "image_processor": {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "image_processor_type": "Qwen2VLImageProcessor",
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}
              }
            }
            """.utf8)
        let config = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.mergeSize, 2)
        XCTAssertEqual(config.minPixels, 65_536)
        XCTAssertEqual(config.maxPixels, 16_777_216)
    }

    func testCaptionGrammarHandlesTokenBoundariesAndEscapedContent() throws {
        let grammar = MiniMaxH3CaptionGrammar(
            ratioKey: "aspect_ratio", ratios: ["16:9", "1:1"])
        var phase = MiniMaxH3CaptionGrammar.Phase.object
        let tokens = [
            #" {"expanded_cap"#, #"tion": "A \"café\" with \u27"#, #"28", "#,
            #""aspect_ratio": "16:"#,
            #"9"}"#,
        ]
        for token in tokens {
            phase = try XCTUnwrap(grammar.advancing(phase, bytes: Array(token.utf8)))
        }
        XCTAssertEqual(phase, .complete)
        XCTAssertNil(grammar.advancing(phase, bytes: Array("extra".utf8)))
        let result = try MiniMaxH3ImagePromptExpansion.parse(
            tokens.joined(), ratioKey: "aspect_ratio", ratios: ["16:9"])
        XCTAssertEqual(result.text, #"A "café" with ✨"#)
        XCTAssertEqual(result.aspectRatio, "16:9")
    }

    func testEditingGrammarLocksTheCanvasRatioAndRejectsExtraProperties() {
        let grammar = MiniMaxH3CaptionGrammar(ratioKey: "target_aspect_ratio", ratios: ["3:2"])
        let malformed = [
            #"{"expanded_caption":"","target_aspect_ratio":"3:2"}"#,
            #"{"expanded_caption":"x","target_aspect_ratio":"2:3"}"#,
            #"{"expanded_caption":"x","aspect_ratio":"3:2"}"#,
            #"{"expanded_caption":"x","target_aspect_ratio":"3:2","extra":true}"#,
        ]
        for answer in malformed {
            XCTAssertNil(grammar.advancing(.object, bytes: Array(answer.utf8)), answer)
        }
    }
}
