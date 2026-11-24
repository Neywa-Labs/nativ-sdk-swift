import Foundation
import XCTest

@testable import NativSDK

final class DeepSeekV41ConfigTests: XCTestCase {
    private func configuration(_ json: String) throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(DeepSeekV41Configuration.self, from: Data(json.utf8))
    }

    /// Upstream's from_dict layers `text_config` under the top level, and the top level
    /// wins where both carry a key.
    func testTextFieldsAreReadFlatOrNested() throws {
        let nested = try configuration(
            #"{"text_config": {"hidden_size": 64, "num_hidden_layers": 4}}"#)
        XCTAssertEqual(nested.hiddenSize, 64)
        XCTAssertEqual(nested.hiddenLayers, 4)

        let flat = try configuration(#"{"hidden_size": 64, "num_hidden_layers": 4}"#)
        XCTAssertEqual(flat.hiddenSize, 64)

        let both = try configuration(
            #"{"hidden_size": 64, "text_config": {"hidden_size": 128}}"#)
        XCTAssertEqual(both.hiddenSize, 64, "the top level wins")
    }

    func testInvalidConfigurationsAreRefused() throws {
        let short = try configuration(
            #"{"num_hidden_layers": 4, "compress_ratios": [0, 0]}"#)
        XCTAssertThrowsError(try short.validateModelConfiguration())

        let mismatched = try configuration(
            #"{"engram_layer_ids": [1, 2, 3], "engram_num_embeddings": [10, 20]}"#)
        XCTAssertThrowsError(try mismatched.validateModelConfiguration())

        XCTAssertNoThrow(try configuration("{}").validateModelConfiguration())
    }

    /// Upstream renames two vision keys as it flattens them, so both spellings reach the
    /// same field.
    func testVisionKeyAliases() throws {
        let renamed = try configuration(
            #"{"vision_config": {"num_hidden_layers": 6, "num_attention_heads": 3}}"#)
        XCTAssertEqual(renamed.visionConfiguration.numLayers, 6)
        XCTAssertEqual(renamed.visionConfiguration.numHeads, 3)

        let flattened = try configuration(#"{"num_layers": 7, "num_heads": 5}"#)
        XCTAssertEqual(flattened.visionConfiguration.numLayers, 7)
        XCTAssertEqual(flattened.visionConfiguration.numHeads, 5)
    }
}
