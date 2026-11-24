import Dispatch
import Foundation
import MLX

/// Resource policy retained through numerical generation and worker cleanup.
public struct ModelExecutionPolicy: Sendable {
    public var wiredMemoryTicket: WiredMemoryTicket?
    public init(wiredMemoryTicket: WiredMemoryTicket? = nil) {
        self.wiredMemoryTicket = wiredMemoryTicket
    }
}

/// A cancellable operation whose result is available only after worker cleanup.
/// Releasing the handle cancels outstanding work. Results can be shared across tasks.
public final class ModelOperation<Result: Sendable>: Sendable {
    private let task: Task<Result, Error>
    init(_ task: Task<Result, Error>) { self.task = task }
    deinit { task.cancel() }
    public func cancel() { task.cancel() }
    public func result() async throws -> Result {
        try await withTaskCancellationHandler {
            let value = try await task.value
            try Task.checkCancellation()
            return value
        } onCancel: {
            task.cancel()
        }
    }
}

/// Shared FIFO admission for complete operations, including synchronous cleanup.
final class ExecutionAdmission: @unchecked Sendable {
    /// A slot reserved synchronously, before an operation task can be scheduled.
    final class Ticket: @unchecked Sendable {
        fileprivate let admission: ObjectIdentifier
        // All mutable fields are protected by the admission's lock.
        fileprivate var acquired = false
        fileprivate var cancelled = false
        fileprivate var continuation: CheckedContinuation<Void, Error>?
        fileprivate init(admission: ObjectIdentifier) { self.admission = admission }
    }

    private let lock = NSLock()
    private var owner: Ticket?
    private var waiters: [Ticket] = []
    var queuedCount: Int { lock.withLock { waiters.count } }

    func reserve() -> Ticket {
        lock.withLock {
            let ticket = Ticket(admission: ObjectIdentifier(self))
            if owner == nil { owner = ticket } else { waiters.append(ticket) }
            return ticket
        }
    }

    func acquire() async throws { try await acquire(reserve()) }

    func acquire(_ ticket: Ticket) async throws {
        precondition(ticket.admission == ObjectIdentifier(self))
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, Error>) in
                lock.withLock {
                    if Task.isCancelled { cancelLocked(ticket) }
                    if ticket.cancelled {
                        continuation.resume(throwing: CancellationError())
                    } else if owner === ticket {
                        ticket.acquired = true
                        continuation.resume()
                    } else {
                        ticket.continuation = continuation
                    }
                }
            }
        } onCancel: {
            self.lock.withLock { self.cancelLocked(ticket) }
        }
    }

    private func cancelLocked(_ ticket: Ticket) {
        // Once acquired, the caller owns cleanup and must release the slot.
        guard !ticket.acquired, !ticket.cancelled else { return }
        ticket.cancelled = true
        let continuation = ticket.continuation
        ticket.continuation = nil
        if owner === ticket {
            advanceLocked()
        } else if let index = waiters.firstIndex(where: { $0 === ticket }) {
            waiters.remove(at: index)
        }
        continuation?.resume(throwing: CancellationError())
    }

    private func advanceLocked() {
        if waiters.isEmpty {
            owner = nil
        } else {
            let next = waiters.removeFirst()
            owner = next
            if let continuation = next.continuation {
                next.continuation = nil
                next.acquired = true
                continuation.resume()
            }
        }
    }

    func release() { lock.withLock { advanceLocked() } }
}

/// The public API consumes resources/inputs, but upstream MLX reference types do
/// not express transfer in their contracts. This box moves each value exactly once
/// into the receiving model or worker. It is never used to expose concurrent access to that value.
final class ExecutionTransfer<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: T?
    init(_ value: consuming T) { self.value = value }
    func take() -> T {
        lock.lock()
        defer { lock.unlock() }
        guard let value else { preconditionFailure("Execution value already transferred") }
        self.value = nil
        return value
    }
}

/// Blocking MLX work runs on a dedicated serial executor while retaining Swift
/// task cancellation and task-local values. The caller awaits the entire lifetime.
actor ModelWorker {
    private nonisolated let queue = DispatchSerialQueue(label: "NativSDK.generation")
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }
    func run<R: Sendable>(_ body: @Sendable () throws -> R) rethrows -> R { try body() }
}

func runModelWorker<Result: Sendable>(
    policy: ModelExecutionPolicy, _ body: @escaping @Sendable () throws -> Result
) async throws -> Result {
    let worker = ModelWorker()
    let operation: @Sendable () async throws -> Result = {
        try await worker.run {
            defer { Stream.defaultStream.synchronize() }
            try Task.checkCancellation()
            let result = try body()
            try Task.checkCancellation()
            return result
        }
    }
    if let ticket = policy.wiredMemoryTicket {
        return try await WiredMemoryTicket.withWiredLimit(ticket, operation)
    }
    return try await operation()
}
