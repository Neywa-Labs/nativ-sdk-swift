import Foundation
import MLX
import XCTest

@testable import NativSDK

final class Flux3Tests: XCTestCase {
    func testBaseAndDistilledSamplerDefaults() throws {
        for flowMap in [true, false] {
            var metadata: [String: Any] = [
                "model": "flux-3-image-klein",
                "dit": [
                    "in_channels": ["image": 128, "image_cond": 128, "text": 20480],
                    "flow_map": flowMap,
                ],
            ]
            let config = try Flux3Configuration(.init(metadata))
            XCTAssertEqual(config.steps, flowMap ? 16 : 50)
            XCTAssertEqual(config.guidance, flowMap ? 1 : 3.5)
            metadata["quantization"] = ["format": "NVFP4"]
            XCTAssertThrowsError(try Flux3Configuration(.init(metadata)))
        }
    }

    func testOrderedReferencesHaveDistinctPositions() throws {
        let ids = Flux3ImageSampling.imageIDs(height: 32, width: 32, reference: 2, scale: 0.5)
        XCTAssertEqual(ids.shape, [1, 4, 4])
        XCTAssertEqual(
            ids.asArray(Float.self),
            [
                0, 0, 0, -3, 0, 0, 0.5, -3, 0, 0.5, 0, -3, 0, 0.5, 0.5, -3,
            ])
        let (width, height) = try Flux3PromptExpansion.referenceSize(
            width: 3000, height: 1500, count: 2)
        XCTAssertEqual(width, 992)
        XCTAssertEqual(height, 480)
    }

    func testLayoutPreservesLockedBoxesAndRejectsInvalidAddedRows() throws {
        let input = #"Place a teapot [{"id":"pot","bbox":[100,200,800,900],"desc":"a teapot"}]"#
        let drop = "A shop with a sign marked [SALE]"
        let finished =
            #"A green teapot <pot> [{"id":"pot","bbox":[100,200,800,900],"desc":"a green teapot"}]"#
        XCTAssertNil(Flux3LayoutPrompt.detect(drop))
        XCTAssertFalse(Flux3LayoutPrompt.isFinished(input))
        XCTAssertTrue(Flux3LayoutPrompt.isFinished(finished))
        let locked = try XCTUnwrap(Flux3LayoutPrompt.detect(input)).rows
        let response =
            #"A green <kettle> with a <cup> [{"id":"kettle","bbox":[101,200,801,900],"desc":"green teapot"},{"id":"cup","bbox":[100,100,300,200],"desc":"small cup"},{"id":"ignored","bbox":[300,400,500,600],"desc":"unreferenced object"}]"#
        let reconciled = try Flux3LayoutPrompt.reconcile(response, with: locked)
        let parsed = try XCTUnwrap(Flux3LayoutPrompt.detect(reconciled))
        XCTAssertEqual(parsed.caption, "A green <pot> with a <cup>")
        XCTAssertEqual(parsed.rows.count, 2)
        XCTAssertEqual(parsed.rows[0]["bbox"], locked[0]["bbox"])
        XCTAssertEqual(parsed.rows[0]["desc"], .string("green teapot"))
    }
}
