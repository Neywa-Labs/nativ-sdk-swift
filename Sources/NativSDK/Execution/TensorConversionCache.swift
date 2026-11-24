import os

/// Shares a lazily converted representation across copies of an immutable result.
struct TensorConversionCache<Value: Sendable>: Sendable {
    private let storage = OSAllocatedUnfairLock<Value?>(initialState: nil)

    func value(_ convert: @Sendable () -> Value) -> Value {
        storage.withLock { cached in
            if let cached { return cached }
            let value = convert()
            cached = value
            return value
        }
    }
}
