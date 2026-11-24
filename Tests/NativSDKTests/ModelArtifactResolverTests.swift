import Foundation
import XCTest

@testable import NativSDK

private struct SnapshotCall: Equatable, Sendable {
    let id: String
    let revision: String?
    let patterns: [String]

}

private actor SnapshotRecorder: ModelSnapshotProvider {
    var calls = [SnapshotCall]()
    let failure: (any Error)?

    init(failure: (any Error)? = nil) { self.failure = failure }

    func snapshot(
        id: String, revision: String?, matching patterns: [String],
        progress: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        calls.append(.init(id: id, revision: revision, patterns: patterns))
        if let failure { throw failure }
        progress(Progress(totalUnitCount: 1))
        return URL(fileURLWithPath: "/snapshots/\(id)")
    }
}

final class ModelArtifactResolverTests: XCTestCase {
    func testObsoleteMoondreamGPT2TokenizerFallsBackButExplicitOverrideWins() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"moondream1"}"#.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        var vocabulary = Dictionary(uniqueKeysWithValues: (0 ..< 50_256).map { ("token\($0)", $0) })
        vocabulary["<|endoftext|>"] = 50_256
        try JSONSerialization.data(withJSONObject: ["model": ["vocab": vocabulary]])
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        let recorder = SnapshotRecorder()
        let result = try await ModelArtifactResolver.resolve(
            model: .directory(directory), provider: recorder)
        XCTAssertEqual(result.tokenizerSource, .repository("moondream/starmie-v1"))
        let explicit = try await ModelArtifactResolver.resolve(
            model: .directory(directory), tokenizer: .directory(directory), provider: recorder)
        XCTAssertEqual(explicit.tokenizerDirectory, directory)
        vocabulary["new-token"] = 50_257
        try JSONSerialization.data(withJSONObject: ["model": ["vocab": vocabulary]])
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        let bundled = try await ModelArtifactResolver.resolve(
            model: .directory(directory), provider: recorder)
        XCTAssertNil(bundled.tokenizerSource)
    }

    func testMoondreamTokenizerFallbackAndExplicitOverrides() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for type in ["moondream1", "moondream2", "moondream3"] {
            try Data("{\"model_type\":\"\(type)\"}".utf8)
                .write(to: directory.appendingPathComponent("config.json"))
            let recorder = SnapshotRecorder()
            let result = try await ModelArtifactResolver.resolve(
                model: .directory(directory), provider: recorder)
            XCTAssertEqual(result.tokenizerSource, .repository("moondream/starmie-v1"))
            let calls = await recorder.calls
            XCTAssertEqual(calls.map(\.id), ["moondream/starmie-v1"])
            XCTAssertEqual(calls.first?.patterns, ModelArtifactResolver.tokenizerPatterns)
            let explicit = try await ModelArtifactResolver.resolve(
                model: .directory(directory), tokenizer: .directory(directory))
            XCTAssertEqual(explicit.tokenizerDirectory, directory)

        }
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        let bundled = try await ModelArtifactResolver.resolve(model: .directory(directory))
        XCTAssertNil(bundled.tokenizerSource)
        XCTAssertEqual(bundled.tokenizerDirectory, directory)
    }

    func testIndependentRemoteSourcesAndProgressThroughDownloaderAdapter() async throws {
        let recorder = SnapshotRecorder()
        let progress = expectation(description: "model progress only")
        progress.assertForOverFulfill = true
        let model = ModelSource.repository("model", revision: "commit-model")
        let tokenizer = ModelSource.repository("tokenizer", revision: "commit-tokenizer")
        let result = try await ModelArtifactResolver.resolve(
            model: model, tokenizer: tokenizer, provider: recorder,
            progress: { _ in progress.fulfill() })
        XCTAssertEqual(result.source, model)
        XCTAssertEqual(result.tokenizerSource, tokenizer)
        XCTAssertEqual(result.modelDirectory.path, "/snapshots/model")
        XCTAssertEqual(result.tokenizerDirectory.path, "/snapshots/tokenizer")
        let calls = await recorder.calls
        XCTAssertEqual(
            calls,
            [
                SnapshotCall(
                    id: "model", revision: "commit-model",
                    patterns: ["*.safetensors", "*.json", "*.jinja"]),
                SnapshotCall(
                    id: "tokenizer", revision: "commit-tokenizer",
                    patterns: ["*.json", "*.jinja"]),
            ])
        await fulfillment(of: [progress], timeout: 1)
    }

    func testLocalAndDefaultTokenizerLocationsDoNotDownload() async throws {
        let recorder = SnapshotRecorder()
        let model = URL(fileURLWithPath: "/local/model")
        let tokenizer = URL(fileURLWithPath: "/local/tokenizer")
        let result = try await ModelArtifactResolver.resolve(
            model: .directory(model), tokenizer: .directory(tokenizer),
            provider: recorder)
        XCTAssertEqual(result.tokenizerDirectory, tokenizer)
        let calls = await recorder.calls
        XCTAssertTrue(calls.isEmpty)
        let defaultResult = try await ModelArtifactResolver.resolve(model: .directory(model))
        XCTAssertEqual(defaultResult.modelDirectory, defaultResult.tokenizerDirectory)
        XCTAssertNil(defaultResult.tokenizerSource)
    }

    func testRemoteModelCanUseLocalTokenizerAndCustomPatterns() async throws {
        let recorder = SnapshotRecorder()
        let tokenizer = URL(fileURLWithPath: "/local/tokenizer")
        let result = try await ModelArtifactResolver.resolve(
            model: .repository("pipeline", revision: nil), tokenizer: .directory(tokenizer),
            provider: recorder,
            modelPatterns: ["transformer/*.safetensors", "tokenizer/**"])
        XCTAssertEqual(result.tokenizerDirectory, tokenizer)
        let calls = await recorder.calls
        XCTAssertEqual(
            calls,
            [
                .init(
                    id: "pipeline", revision: nil,
                    patterns: ["transformer/*.safetensors", "tokenizer/**"])
            ])
    }

    func testProviderErrorsAndCancellationPropagate() async throws {
        struct DownloadFailure: Error {}
        for failure in [DownloadFailure() as any Error, CancellationError()] {
            let recorder = SnapshotRecorder(failure: failure)
            do {
                _ = try await ModelArtifactResolver.resolve(
                    model: .repository("model"), tokenizer: .repository("tokenizer"),
                    provider: recorder)
                XCTFail("Expected provider failure")
            } catch {
                XCTAssertEqual(
                    String(reflecting: type(of: error)), String(reflecting: type(of: failure)))
            }
            let calls = await recorder.calls
            XCTAssertEqual(calls.count, 1)
        }
    }

    func testCancelledResolutionDoesNotInvokeProvider() async {
        let recorder = SnapshotRecorder()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await ModelArtifactResolver.resolve(
                model: .repository("model"), provider: recorder)
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        let calls = await recorder.calls
        XCTAssertTrue(calls.isEmpty)
    }
}
