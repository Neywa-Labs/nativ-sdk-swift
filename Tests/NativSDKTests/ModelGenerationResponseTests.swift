import Foundation
import XCTest

@testable import NativSDK

private final class ResponseCleanupProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func finish() { lock.withLock { value = true } }
    var finished: Bool { lock.withLock { value } }
}

final class ModelGenerationResponseTests: XCTestCase {
    private func summary() -> ModelGenerationSummary {
        .init(
            promptTokenCount: 2, generatedTokenCount: 3, promptSeconds: 0,
            generationSeconds: 0, stopReason: .stop)
    }

    private func request(events: [ModelGenerationEvent]) -> ModelGeneration {
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream()
        let summary = summary()
        let task = Task<ModelGenerationSummary, Error> {
            for event in events { continuation.yield(event) }
            continuation.yield(.completion(summary))
            continuation.finish()
            return summary
        }
        return ModelGeneration(stream: stream, task: task)
    }

    func testResponseJoinsOnlyResponseChunksAndMetricsRemainAvailable() async throws {
        let call = ToolCall(function: .init(name: "weather", arguments: [String: JSONValue]()))
        let generation = request(events: [
            .reasoning("private reasoning"), .chunk("Hello "), .toolCall(call),
            .protocolError("diagnostic"), .chunk("🌎!"),
        ])
        let text = try await generation.response()
        XCTAssertEqual(text, "Hello 🌎!")
        let metrics = try await generation.metrics()
        XCTAssertEqual(metrics.generatedTokenCount, 3)
        let repeated = try await generation.metrics()
        XCTAssertEqual(repeated.stopReason, .stop)
    }

    func testEmptyResponseAndMetricsDoNotClaimTheStream() async throws {
        let empty = request(events: [.reasoning("thinking")])
        _ = try await empty.metrics()
        let text = try await empty.response()
        XCTAssertEqual(text, "")
    }

    func testCollectionCannotBeRepeatedOrCombinedWithIteration() async throws {
        let collected = request(events: [.chunk("answer")])
        _ = try await collected.response()
        do {
            _ = try await collected.response()
            XCTFail("Expected single-consumer rejection")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .multipleConsumers) }

        let streamed = request(events: [.chunk("answer")])
        var iterator = streamed.makeAsyncIterator()
        do {
            _ = try await streamed.response()
            XCTFail("Expected single-consumer rejection")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .multipleConsumers) }
        guard case .chunk(let text) = try await iterator.next() else {
            return XCTFail("Rejected collection must not cancel the existing consumer")
        }
        XCTAssertEqual(text, "answer")
        while try await iterator.next() != nil {}
    }

    func testRawTokensCannotBeCollectedAsResponseText() async throws {
        let generation = request(events: [.token(42)])
        do {
            _ = try await generation.response()
            XCTFail("Expected raw-token rejection")
        } catch {
            XCTAssertEqual(
                error as? ModelExecutionError,
                .unsupportedOperation("response requires text output"))
        }
    }

    func testStreamFailureThrowsInsteadOfReturningPartialTextAndJoinsWorker() async throws {
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream()
        let probe = ResponseCleanupProbe()
        let started = expectation(description: "worker started")
        let summary = summary()
        let task = Task {
            defer { probe.finish() }
            continuation.yield(.chunk("partial"))
            continuation.finish(throwing: ModelExecutionError.outputBufferOverflow)
            started.fulfill()
            try await Task.sleep(for: .seconds(30))
            return summary
        }
        let generation = ModelGeneration(stream: stream, task: task)
        await fulfillment(of: [started], timeout: 3)
        do {
            _ = try await generation.response()
            XCTFail("Expected stream failure")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .outputBufferOverflow) }
        XCTAssertTrue(probe.finished, "Collection must wait for worker cleanup on failure")
    }

    func testCancellingCollectionCancelsAndJoinsTheWorker() async throws {
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream()
        let probe = ResponseCleanupProbe()
        let started = expectation(description: "worker started")
        let summary = summary()
        let task = Task {
            defer { probe.finish() }
            continuation.yield(.chunk("partial"))
            started.fulfill()
            do {
                try await Task.sleep(for: .seconds(30))
                continuation.finish()
                return summary
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }
        let generation = ModelGeneration(stream: stream, task: task)
        let collector = Task { try await generation.response() }
        await fulfillment(of: [started], timeout: 3)
        collector.cancel()
        do {
            _ = try await collector.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertTrue(probe.finished, "Cancellation must join worker cleanup")
    }
}
