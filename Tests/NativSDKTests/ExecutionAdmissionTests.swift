import XCTest

@testable import NativSDK

private actor AdmissionTestGate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        open = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }
}

final class ExecutionAdmissionTests: XCTestCase {
    func testReservedOrderSurvivesReverseTaskStart() async throws {
        let admission = ExecutionAdmission()
        let first = admission.reserve()
        let second = admission.reserve()
        let started = expectation(description: "second task started first")
        let later = Task {
            started.fulfill()
            try await admission.acquire(second)
            defer { admission.release() }
            return 2
        }
        await fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(admission.queuedCount, 1)
        try await admission.acquire(first)
        XCTAssertEqual(admission.queuedCount, 1)
        admission.release()
        let result = try await later.value
        XCTAssertEqual(result, 2)
        XCTAssertEqual(admission.queuedCount, 0)
    }

    func testCancellationBeforeAcquiringReservedOwnerPromotesNext() async throws {
        let admission = ExecutionAdmission()
        let first = admission.reserve()
        let second = admission.reserve()
        let gate = AdmissionTestGate()
        let cancelled = Task {
            await gate.wait()
            try await admission.acquire(first)
            defer { admission.release() }
            try Task.checkCancellation()
        }
        cancelled.cancel()
        let admitted = expectation(description: "next reservation admitted")
        let next = Task {
            try await admission.acquire(second)
            defer { admission.release() }
            admitted.fulfill()
        }
        await gate.release()
        await fulfillment(of: [admitted], timeout: 3)
        do {
            try await cancelled.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        // Cancel also permits cleanup after a failed expectation instead of leaving a waiter.
        next.cancel()
        _ = await next.result
        XCTAssertEqual(admission.queuedCount, 0)
        try await admission.acquire()
        admission.release()
    }
}
