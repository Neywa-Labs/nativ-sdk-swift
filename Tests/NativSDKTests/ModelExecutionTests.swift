import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class ExecutionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var log: [String] = []
    private var cacheCount = 0
    private var peak = 0
    let onPrepare: (@Sendable () async throws -> Void)?
    let onPrefill: (@Sendable () throws -> Void)?
    let onForward: (@Sendable () -> Void)?
    init(
        onPrepare: (@Sendable () async throws -> Void)? = nil,
        onPrefill: (@Sendable () throws -> Void)? = nil,
        onForward: (@Sendable () -> Void)? = nil
    ) {
        self.onPrepare = onPrepare
        self.onPrefill = onPrefill
        self.onForward = onForward
    }
    func record(_ value: String) { lock.withLock { log.append(value) } }
    func createCache() {
        lock.withLock {
            cacheCount += 1
            peak = max(peak, cacheCount)
            log.append("cache+")
        }
    }
    func destroyCache() {
        lock.withLock {
            cacheCount -= 1
            log.append("cache-")
        }
    }
    var events: [String] { lock.withLock { log } }
    var liveCaches: Int { lock.withLock { cacheCount } }
    var peakCaches: Int { lock.withLock { peak } }
}

private final class ExecutionCache: KVCache {
    let probe: ExecutionProbe
    init(_ probe: ExecutionProbe) {
        self.probe = probe
        probe.createCache()
    }
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var state: [MLXArray] = []
    var metaState: [String] = []
    var isTrimmable: Bool { false }
    func innerState() -> [MLXArray] { state }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    func trim(_ n: Int) -> Int { 0 }
    func copy() -> any KVCache { ExecutionCache(probe) }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    { .none }
    deinit { probe.destroyCache() }
}

private final class ExecutionModel: Module, LanguageModel {
    let probe: ExecutionProbe
    init(_ probe: ExecutionProbe) { self.probe = probe }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [ExecutionCache(probe)] }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        probe.record("prefill")
        try probe.onPrefill?()
        try Task.checkCancellation()
        return .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        probe.onForward?()
        probe.record("forward")
        let logits = MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
        logits[.ellipsis, 3] = MLXArray(Float(10))
        return logits
    }
}

private struct ExecutionTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(repeating: "x", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
}

private struct ExecutionProcessor: UserInputProcessor {
    let probe: ExecutionProbe
    func prepare(input: UserInput) async throws -> LMInput {
        probe.record("prepare")
        try await probe.onPrepare?()
        return LMInput(tokens: MLXArray([1, 2]))
    }
}

private actor PreparationBarrier {
    var waiters: [CheckedContinuation<Void, Never>] = []
    var open = false
    func wait() async {
        if !open { await withCheckedContinuation { waiters.append($0) } }
    }
    func release() {
        open = true
        for waiter in waiters { waiter.resume() }
        waiters = []
    }
}

private final class ForwardBarrier: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    let reached: XCTestExpectation
    private let semaphore = DispatchSemaphore(value: 0)
    let blockAt: Int
    init(reached: XCTestExpectation, blockAt: Int = 3) {
        self.reached = reached
        self.blockAt = blockAt
    }
    func step() {
        let shouldBlock = lock.withLock {
            calls += 1
            return calls == blockAt
        }
        if shouldBlock {
            reached.fulfill()
            // A timeout prevents a failed assertion from hanging the suite.
            _ = semaphore.wait(timeout: .now() + 5)
        }
    }
    func release() { semaphore.signal() }
}

final class ModelExecutionTests: XCTestCase {
    private func resources(_ probe: ExecutionProbe = .init()) -> LoadedModel {
        let directory = URL(fileURLWithPath: "/tmp/execution-fixture")
        return LoadedModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(), model: ExecutionModel(probe),
            processor: ExecutionProcessor(probe: probe), tokenizer: ExecutionTokenizer())
    }
    private func options(_ count: Int = 5) -> ModelGenerationOptions {
        .init(sampling: .init(temperature: 0), output: .init(maxTokens: count))
    }
    private func consume(_ run: ModelGeneration) async throws -> (String, Int) {
        var text = ""
        var completions = 0
        for try await event in run {
            switch event {
            case .chunk(let chunk): text += chunk
            case .completion: completions += 1
            default: break
            }
        }
        return (text, completions)
    }
    private func assertCancelled(_ run: ModelGeneration) async {
        do {
            _ = try await run.metrics()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
    }

    private func waitUntilQueued(_ loadedModel: NativModel) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while await loadedModel.queuedRequestCount == 0 {
            guard ContinuousClock.now < deadline else {
                XCTFail("Expected a request waiting for exclusive model access")
                return
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    func testScopedModelAccessHoldsAdmissionAcrossSuspensionAndReleasesAfterFailure() async throws {
        let entered = expectation(description: "exclusive access")
        let barrier = PreparationBarrier()
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let inspection = Task {
            try await loadedModel.withModel { _, _ in
                entered.fulfill()
                await barrier.wait()
                throw ModelExecutionError.invalidOptions("fixture failure")
            }
        }
        await fulfillment(of: [entered], timeout: 3)
        let request = loadedModel.generate(input: UserInput(prompt: "queued"), options: options())
        try await waitUntilQueued(loadedModel)
        XCTAssertFalse(probe.events.contains("prepare"))
        await barrier.release()
        do {
            _ = try await inspection.value
            XCTFail("Expected inspection failure")
        } catch {
            XCTAssertEqual(error as? ModelExecutionError, .invalidOptions("fixture failure"))
        }
        let result = try await consume(request)
        XCTAssertEqual(result.1, 1)
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testBatchRowsMatchIndividualRequestsAndIsolateCancellation() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let batch = loadedModel.generateBatch(
            inputs: [UserInput(prompt: "a"), UserInput(prompt: "b"), UserInput(prompt: "c")],
            options: options())
        batch.requests[1].cancel()
        let first = try await consume(batch.requests[0])
        let third = try await consume(batch.requests[2])
        let individual = try await consume(
            loadedModel.generate(input: UserInput(prompt: "a"), options: options()))
        XCTAssertEqual(first.0, individual.0)
        XCTAssertEqual(third.0, individual.0)
        let results = await batch.metrics()
        XCTAssertEqual(results.count, 3)
        XCTAssertNoThrow(try results[0].get())
        XCTAssertThrowsError(try results[1].get())
        XCTAssertNoThrow(try results[2].get())
        XCTAssertEqual(probe.liveCaches, 0)
        XCTAssertEqual(probe.peakCaches, 1)
    }

    func testSeededBatchMatchesIndividualSampling() async throws {
        let loadedModel = resources().intoNativModel()
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 10, seed: 42),
            output: .init(maxTokens: 24, format: .tokens))
        func tokens(_ request: ModelGeneration) async throws -> [Int] {
            var values: [Int] = []
            for try await event in request {
                if case .token(let token) = event { values.append(token) }
            }
            return values
        }
        let expected = try await tokens(
            loadedModel.generate(input: UserInput(prompt: "a"), options: options))
        XCTAssertGreaterThan(Set(expected).count, 1)
        let batch = loadedModel.generateBatch(
            inputs: (0 ..< 3).map { UserInput(prompt: "\($0)") }, options: options)
        for row in batch.requests {
            let actual = try await tokens(row)
            XCTAssertEqual(actual, expected)
        }
    }

    func testBatchCancellationJoinsEveryRequest() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let batch = loadedModel.generateBatch(
            inputs: (0 ..< 4).map { UserInput(prompt: "\($0)") }, options: options(10000))
        batch.cancel()
        let results = await batch.metrics()
        XCTAssertEqual(results.count, 4)
        for result in results { XCTAssertThrowsError(try result.get()) }
        XCTAssertEqual(probe.liveCaches, 0)
        let empty = loadedModel.generateBatch(inputs: [])
        let emptyResults = await empty.metrics()
        XCTAssertTrue(emptyResults.isEmpty)
    }

    func testBatchResponsesDrainOutputsLargerThanTheEventBuffer() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let batch = loadedModel.generateBatch(
            inputs: (0 ..< 3).map { UserInput(prompt: "\($0)") }, options: options(1024))
        let responses = await batch.responses()
        XCTAssertEqual(responses.count, 3)
        for response in responses {
            XCTAssertEqual(try response.get(), String(repeating: "x", count: 1024))
        }
        let metrics = await batch.metrics()
        for result in metrics {
            XCTAssertEqual(try result.get().generatedTokenCount, 1024)
        }
        XCTAssertEqual(probe.liveCaches, 0)
        XCTAssertEqual(probe.peakCaches, 1)
    }

    func testBatchResponsesPreserveInputOrderAndIsolateRowFailures() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let batch = ModelGenerationBatch(requests: [
            loadedModel.generate(input: UserInput(prompt: "first"), options: options(7)),
            loadedModel.generate(input: UserInput(prompt: "invalid"), options: options(-1)),
            loadedModel.generate(input: UserInput(prompt: "last"), options: options(3)),
        ])
        let responses = await batch.responses()
        XCTAssertEqual(responses.count, 3)
        XCTAssertEqual(try responses[0].get(), "xxxxxxx")
        XCTAssertThrowsError(try responses[1].get()) {
            XCTAssertEqual(
                $0 as? ModelExecutionError, .invalidOptions("maxTokens cannot be negative"))
        }
        XCTAssertEqual(try responses[2].get(), "xxx")
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testBatchResponsesConsumeLaterRowsWhileTheFirstIsBlocked() async throws {
        let barrier = PreparationBarrier()
        let firstModel = resources(ExecutionProbe(onPrepare: { await barrier.wait() }))
            .intoNativModel()
        let secondModel = resources().intoNativModel()
        let first = firstModel.generate(input: UserInput(prompt: "first"), options: options(2))
        let second = secondModel.generate(
            input: UserInput(prompt: "second"), options: options(1024))
        let batch = ModelGenerationBatch(requests: [first, second])
        // The first row cannot run until the second finishes. A sequential collector
        // would leave the second unread, forcing it to overflow its 256-event buffer.
        let release = Task {
            _ = try? await second.metrics()
            await barrier.release()
        }
        let responses = await batch.responses()
        await release.value
        XCTAssertEqual(try responses[0].get(), "xx")
        XCTAssertEqual(try responses[1].get(), String(repeating: "x", count: 1024))
    }

    func testCancellingBatchResponsesJoinsActiveAndQueuedWorkers() async throws {
        let reached = expectation(description: "generation started")
        let barrier = ForwardBarrier(reached: reached)
        defer { barrier.release() }
        let probe = ExecutionProbe(onForward: { barrier.step() })
        let loadedModel = resources(probe).intoNativModel()
        let batch = loadedModel.generateBatch(
            inputs: (0 ..< 3).map { UserInput(prompt: "\($0)") }, options: options(10000))
        let collector = Task { await batch.responses() }
        await fulfillment(of: [reached], timeout: 3)
        collector.cancel()
        barrier.release()
        let results = await collector.value
        XCTAssertEqual(results.count, 3)
        for result in results {
            XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is CancellationError) }
        }
        XCTAssertEqual(probe.liveCaches, 0)
        let next = try await loadedModel.generate(
            input: UserInput(prompt: "after cancellation"), options: options(2)
        ).response()
        XCTAssertEqual(next, "xx")
    }

    func testBatchResponsesHandleEmptyBatchesAndRejectRepeatedConsumption() async throws {
        let loadedModel = resources().intoNativModel()
        let empty = await loadedModel.generateBatch(inputs: []).responses()
        XCTAssertTrue(empty.isEmpty)
        let batch = loadedModel.generateBatch(
            inputs: [UserInput(prompt: "test")], options: options(2))
        let first = await batch.responses()
        XCTAssertEqual(try first[0].get(), "xx")
        let second = await batch.responses()
        XCTAssertThrowsError(try second[0].get()) {
            XCTAssertEqual($0 as? ModelExecutionError, .multipleConsumers)
        }
    }

    func testPreparedAndUnpreparedGenerationMatchAndCompleteAfterCleanup() async throws {
        let reference = resources().intoNativModel().generate(
            input: UserInput(prompt: "test"), options: options())
        let expected = try await consume(reference).0
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let prepared = try await loadedModel.prepare(input: UserInput(prompt: "test"))
        XCTAssertEqual(prepared.tokenCount, 2)
        let run = loadedModel.generate(prepared: prepared, options: options())
        let result = try await consume(run)
        let summary = try await run.metrics()
        XCTAssertEqual(result.0, expected)
        XCTAssertEqual(result.1, 1)
        XCTAssertEqual(summary.generatedTokenCount, 5)
        XCTAssertEqual(summary.stopReason, .length)
        XCTAssertEqual(probe.liveCaches, 0)
        XCTAssertEqual(probe.events.filter { $0 == "prepare" }.count, 1)
    }

    func testConcurrentRequestsHoldAdmissionThroughWorkerCleanup() async throws {
        let reached = expectation(description: "decoder active")
        let barrier = ForwardBarrier(reached: reached)
        let probe = ExecutionProbe(onForward: { barrier.step() })
        let loadedModel = resources(probe).intoNativModel()
        let first = loadedModel.generate(input: UserInput(prompt: "first"), options: options())
        await fulfillment(of: [reached], timeout: 3)
        let second = loadedModel.generate(input: UserInput(prompt: "second"), options: options())
        try await waitUntilQueued(loadedModel)
        first.cancel()
        // Cancellation must not release model access while the worker is still in a forward.
        XCTAssertEqual(probe.events.filter { $0 == "prepare" }.count, 1)
        barrier.release()
        await assertCancelled(first)
        _ = try await consume(second)
        XCTAssertEqual(probe.peakCaches, 1)
        XCTAssertEqual(probe.liveCaches, 0)
        let events = probe.events
        let secondPrepare = try XCTUnwrap(events.lastIndex(of: "prepare"))
        let cleanup = try XCTUnwrap(events.firstIndex(of: "cache-"))
        XCTAssertLessThan(cleanup, secondPrepare)
    }

    func testQueuedCancellationDoesNotPrepareOrWaitForActiveRequest() async throws {
        let reached = expectation(description: "preparing")
        let barrier = PreparationBarrier()
        let probe = ExecutionProbe(onPrepare: {
            reached.fulfill()
            await barrier.wait()
        })
        let loadedModel = resources(probe).intoNativModel()
        let first = loadedModel.generate(input: UserInput(prompt: "first"), options: options())
        await fulfillment(of: [reached], timeout: 3)
        let queued = loadedModel.generate(input: UserInput(prompt: "queued"), options: options())
        try await waitUntilQueued(loadedModel)
        queued.cancel()
        await assertCancelled(queued)
        XCTAssertEqual(probe.events, ["prepare"])
        first.cancel()
        await barrier.release()
        await assertCancelled(first)
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testCancellationDuringPrefillReleasesCachesBeforeNextRequest() async throws {
        let reached = expectation(description: "prefill active")
        let barrier = ForwardBarrier(reached: reached, blockAt: 1)
        let probe = ExecutionProbe(onPrefill: {
            barrier.step()
            try Task.checkCancellation()
        })
        let loadedModel = resources(probe).intoNativModel()
        let run = loadedModel.generate(input: UserInput(prompt: "first"), options: options())
        await fulfillment(of: [reached], timeout: 3)
        run.cancel()
        barrier.release()
        await assertCancelled(run)
        XCTAssertEqual(probe.liveCaches, 0)
        let next = loadedModel.generate(input: UserInput(prompt: "next"), options: options())
        _ = try await consume(next)
        XCTAssertEqual(probe.peakCaches, 1)
    }

    func testBreakingStreamCancelsEvenWhenRequestIsRetained() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let run = loadedModel.generate(input: UserInput(prompt: "test"), options: options(100_000))
        for try await event in run {
            if case .chunk = event { break }
        }
        await assertCancelled(run)
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testDroppingUnconsumedRequestCancelsItsWork() async throws {
        let reached = expectation(description: "preparing")
        let barrier = PreparationBarrier()
        let probe = ExecutionProbe(onPrepare: {
            reached.fulfill()
            await barrier.wait()
        })
        let loadedModel = resources(probe).intoNativModel()
        var run: ModelGeneration? = loadedModel.generate(
            input: UserInput(prompt: "test"), options: options())
        let task = try XCTUnwrap(run).task
        await fulfillment(of: [reached], timeout: 3)
        run = nil
        await barrier.release()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testSessionsHaveIndependentDefaultsAndClosingOneDoesNotCloseOthers() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        let first = ModelSession(model: loadedModel, options: options(3))
        let second = ModelSession(model: loadedModel, options: options(7))
        let a = try await first.generate(input: UserInput(prompt: "a"))
        let b = try await second.generate(input: UserInput(prompt: "b"))
        let aResult = try await consume(a)
        let bResult = try await consume(b)
        XCTAssertEqual(aResult.0, "xxx")
        XCTAssertEqual(bResult.0, "xxxxxxx")
        await first.close()
        await first.close()
        do {
            _ = try await first.generate(input: UserInput(prompt: "closed"))
            XCTFail()
        } catch { XCTAssertEqual(error as? ModelExecutionError, .sessionClosed) }
        let next = try await second.generate(input: UserInput(prompt: "still open"))
        _ = try await consume(next)
        await second.close()
        XCTAssertEqual(probe.peakCaches, 2)
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testPreparationFailureAndForeignInputDoNotPoisonExecutor() async throws {
        struct PreparationFailure: Error {}
        let failing = resources(ExecutionProbe(onPrepare: { throw PreparationFailure() }))
            .intoNativModel()
        let run = failing.generate(input: UserInput(prompt: "test"))
        do {
            _ = try await consume(run)
            XCTFail()
        } catch { XCTAssertTrue(error is PreparationFailure) }
        let first = resources().intoNativModel()
        let second = resources().intoNativModel()
        let input = try await first.prepare(input: UserInput(prompt: "test"))
        let foreign = second.generate(prepared: input, options: options())
        do {
            _ = try await foreign.metrics()
            XCTFail()
        } catch { XCTAssertEqual(error as? ModelExecutionError, .foreignPreparedInput) }
        _ = try await consume(second.generate(input: UserInput(prompt: "next"), options: options()))
    }

    func testBufferOverflowIsAnErrorAndJoinsWorker() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        var parameters = options(1000)
        parameters.output.bufferCapacity = 1
        let run = loadedModel.generate(input: UserInput(prompt: "test"), options: parameters)
        do {
            _ = try await run.metrics()
            XCTFail()
        } catch { XCTAssertEqual(error as? ModelExecutionError, .outputBufferOverflow) }
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testCancellingConsumerTaskJoinsWorker() async throws {
        let reached = expectation(description: "decoder active")
        let barrier = ForwardBarrier(reached: reached)
        let probe = ExecutionProbe(onForward: { barrier.step() })
        let loadedModel = resources(probe).intoNativModel()
        let run = loadedModel.generate(input: UserInput(prompt: "test"), options: options(1000))
        let consumer = Task {
            for try await _ in run {}
        }
        await fulfillment(of: [reached], timeout: 3)
        consumer.cancel()
        barrier.release()
        do {
            try await consumer.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testSessionCloseCancelsAndJoinsActiveAndQueuedWork() async throws {
        let reached = expectation(description: "preparing")
        let barrier = PreparationBarrier()
        let probe = ExecutionProbe(onPrepare: {
            reached.fulfill()
            await barrier.wait()
        })
        let loadedModel = resources(probe).intoNativModel()
        let session = ModelSession(model: loadedModel, options: options())
        let active = try await session.generate(input: UserInput(prompt: "active"))
        await fulfillment(of: [reached], timeout: 3)
        let queued = try await session.generate(input: UserInput(prompt: "queued"))
        while await session.queuedRequestCount == 0 { await Task.yield() }
        let closing = Task { await session.close() }
        // A queued cancellation completes even while active preparation is suspended.
        await assertCancelled(queued)
        await barrier.release()
        await closing.value
        await assertCancelled(active)
        XCTAssertEqual(probe.events, ["prepare"])
    }

    func testSecondConsumerIsRejectedWithoutCancellingFirst() async throws {
        let loadedModel = resources().intoNativModel()
        let run = loadedModel.generate(input: UserInput(prompt: "test"), options: options())
        var first = run.makeAsyncIterator()
        do {
            var second = run.makeAsyncIterator()
            _ = try await second.next()
            XCTFail("Expected single-consumer enforcement")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .multipleConsumers) }
        var completions = 0
        while let event = try await first.next() {
            if case .completion = event { completions += 1 }
        }
        XCTAssertEqual(completions, 1)
    }

    func testEngineInitializationFailureDoesNotKeepAdmission() async throws {
        let probe = ExecutionProbe()
        let loadedModel = resources(probe).intoNativModel()
        var invalid = options()
        invalid.output.bufferCapacity = 0
        let run = loadedModel.generate(input: UserInput(prompt: "invalid"), options: invalid)
        do {
            _ = try await consume(run)
            XCTFail("Expected cache policy validation failure")
        } catch { XCTAssertFalse(error is CancellationError) }
        _ = try await consume(
            loadedModel.generate(input: UserInput(prompt: "valid"), options: options()))
        XCTAssertEqual(probe.liveCaches, 0)
    }

    func testInvalidGenerationOptionsAreRejected() throws {
        var options = ModelGenerationOptions()
        options.output.bufferCapacity = 0
        XCTAssertThrowsError(try GenerationDriver.validate(options))
        options.output.bufferCapacity = 8
        options.prefill.stepSize = 0
        XCTAssertThrowsError(try GenerationDriver.validate(options))
        options.prefill.stepSize = 17
        options.output.maxTokens = -1
        XCTAssertThrowsError(try GenerationDriver.validate(options))
    }
}

extension ModelExecutionTests {
    func testDroppingSessionCancelsRetainedRequest() async throws {
        let reached = expectation(description: "session preparing")
        let barrier = PreparationBarrier()
        let probe = ExecutionProbe(onPrepare: {
            reached.fulfill()
            await barrier.wait()
        })
        var session: ModelSession? = ModelSession(model: resources(probe).intoNativModel())
        let request = try await session!.respond(to: "test")
        await fulfillment(of: [reached], timeout: 3)
        session = nil
        await barrier.release()
        await assertCancelled(request)
        XCTAssertEqual(probe.liveCaches, 0)
    }
}
