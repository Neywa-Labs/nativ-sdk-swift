import Foundation
import XCTest

@testable import NativSDK

final class ModelExecutionErrorTests: XCTestCase {
    func testLocalizedDescriptionsSurviveNSErrorBridging() throws {
        let errors: [ModelExecutionError] = [
            .unsupportedOperation("audio output"), .foreignPreparedInput, .sessionClosed,
            .multipleConsumers, .outputBufferOverflow, .missingCompletion,
            .invalidOptions("temperature must be finite and nonnegative"),
        ]
        for error in errors {
            let description = try XCTUnwrap(error.errorDescription)
            let recovery = try XCTUnwrap(error.recoverySuggestion)
            XCTAssertFalse(description.isEmpty)
            XCTAssertFalse(recovery.isEmpty)
            let bridged = error as NSError
            XCTAssertEqual(bridged.localizedDescription, description)
            XCTAssertEqual(bridged.localizedRecoverySuggestion, recovery)
            XCTAssertEqual(error.localizedDescription, description)
        }
    }

    func testDescriptionsRetainOperationAndValidationDetails() {
        XCTAssertTrue(
            ModelExecutionError.unsupportedOperation("audio output")
                .localizedDescription.contains("audio output"))
        XCTAssertTrue(
            ModelExecutionError.invalidOptions("temperature must be finite and nonnegative")
                .localizedDescription.contains("temperature must be finite and nonnegative"))
    }
}
