import NativSDK
import XCTest

final class HuggingFaceDownloaderTests: XCTestCase {

    func testInvalidRepositoryFailsBeforeDownload() async {
        do {
            _ = try await HuggingFaceDownloader().snapshot(
                id: "invalid/repository/id", revision: nil, matching: ["*.json"],
                progress: { _ in XCTFail("No download should start") })
            XCTFail("Invalid repository should be rejected")
        } catch HuggingFaceDownloader.DownloadError.invalidRepositoryID(let id) {
            XCTAssertEqual(id, "invalid/repository/id")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}
