import Foundation

/// NumPy's `SeedSequence` entropy mixer.
///
/// Reproduced rather than approximated because a model whose weights are indexed by
/// `numpy.random.default_rng(seed)` draws — DeepSeek-V4.1's Engram hash multipliers are
/// the case in hand — reads different rows under any other stream. The constants and the
/// mixing order are NumPy's; every step is `UInt32` arithmetic that wraps.
public struct NumPySeedSequence {
    private static let initA: UInt32 = 0x43B0_D7E5
    private static let multA: UInt32 = 0x931E_8875
    private static let initB: UInt32 = 0x8B51_F9DD
    private static let multB: UInt32 = 0x58F3_8DED
    private static let mixMultLeft: UInt32 = 0xCA01_F9DD
    private static let mixMultRight: UInt32 = 0x4973_F715
    private static let xshift: UInt32 = 16
    private static let poolSize = 4

    private var pool: [UInt32]

    public init(entropy: UInt64) {
        var words: [UInt32] = []
        var remaining = entropy
        repeat {
            words.append(UInt32(truncatingIfNeeded: remaining))
            remaining >>= 32
        } while remaining != 0
        self.pool = NumPySeedSequence.mix(entropy: words)
    }

    private static func mix(entropy words: [UInt32]) -> [UInt32] {
        var hashConst = initA
        func hashmix(_ value: UInt32) -> UInt32 {
            var value = value ^ hashConst
            hashConst = hashConst &* multA
            value = value &* hashConst
            return value ^ (value >> xshift)
        }
        func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
            let result = (mixMultLeft &* x) &- (mixMultRight &* y)
            return result ^ (result >> xshift)
        }

        var pool = (0 ..< poolSize).map { hashmix($0 < words.count ? words[$0] : 0) }
        for source in 0 ..< poolSize {
            for destination in 0 ..< poolSize where source != destination {
                pool[destination] = mix(pool[destination], hashmix(pool[source]))
            }
        }
        for source in stride(from: poolSize, to: words.count, by: 1) {
            for destination in 0 ..< poolSize {
                pool[destination] = mix(pool[destination], hashmix(words[source]))
            }
        }
        return pool
    }

    public func generateState(words count: Int) -> [UInt32] {
        var hashConst = NumPySeedSequence.initB
        return (0 ..< count).map { index in
            var value = pool[index % NumPySeedSequence.poolSize]
            value ^= hashConst
            hashConst = hashConst &* NumPySeedSequence.multB
            value = value &* hashConst
            return value ^ (value >> NumPySeedSequence.xshift)
        }
    }

    /// The 32-bit words viewed as little-endian 64-bit ones, which is how NumPy's
    /// `generate_state(n, dtype=uint64)` reads them.
    public func generateState64(words count: Int) -> [UInt64] {
        let words32 = generateState(words: count * 2)
        return (0 ..< count).map {
            UInt64(words32[2 * $0]) | (UInt64(words32[2 * $0 + 1]) << 32)
        }
    }
}

/// The 128 bits the PCG64 state needs, as a pair of 64-bit halves. `UInt128` would do
/// the same but is only available from macOS 15, and this package targets lower.
struct Wide128 {
    var high: UInt64
    var low: UInt64

    init(high: UInt64, low: UInt64) {
        self.high = high
        self.low = low
    }

    static func + (lhs: Wide128, rhs: Wide128) -> Wide128 {
        let (low, carry) = lhs.low.addingReportingOverflow(rhs.low)
        return Wide128(high: lhs.high &+ rhs.high &+ (carry ? 1 : 0), low: low)
    }

    static func * (lhs: Wide128, rhs: Wide128) -> Wide128 {
        let (high, low) = lhs.low.multipliedFullWidth(by: rhs.low)
        return Wide128(
            high: high &+ (lhs.low &* rhs.high) &+ (lhs.high &* rhs.low), low: low)
    }

    var shiftedLeftOne: Wide128 {
        Wide128(high: (high << 1) | (low >> 63), low: low << 1)
    }
}

/// NumPy's PCG64 (XSL-RR output over a 128-bit LCG), with its bounded-integer draw.
public struct NumPyPCG64 {
    private static let multiplier = Wide128(
        high: 0x2360_ED05_1FC6_5DA4, low: 0x4385_DF64_9FCC_F645)

    var state: Wide128
    var increment: Wide128

    /// The state and increment as NumPy reports them in `bit_generator.state`.
    public var stateWords: (high: UInt64, low: UInt64) { (state.high, state.low) }
    public var incrementWords: (high: UInt64, low: UInt64) { (increment.high, increment.low) }

    public init(seed: UInt64) {
        let words = NumPySeedSequence(entropy: seed).generateState64(words: 4)
        let initialState = Wide128(high: words[0], low: words[1])
        let initialSequence = Wide128(high: words[2], low: words[3])
        self.state = Wide128(high: 0, low: 0)
        self.increment = initialSequence.shiftedLeftOne
        self.increment.low |= 1
        step()
        self.state = state + initialState
        step()
    }

    private mutating func step() {
        state = state * NumPyPCG64.multiplier + increment
    }

    public mutating func next() -> UInt64 {
        step()
        let rotation = state.high >> 58
        let value = state.high ^ state.low
        return (value >> rotation) | (value << ((64 - rotation) & 63))
    }

    /// One draw from `[0, bound)`, by Lemire's method with NumPy's rejection threshold.
    /// The Generator API takes this path; the masked one belongs to legacy RandomState.
    public mutating func nextBounded(_ bound: UInt64) -> UInt64 {
        precondition(bound > 0, "bound must be positive")
        if bound == 1 { return 0 }
        var (high, low) = next().multipliedFullWidth(by: bound)
        if low < bound {
            let threshold = (UInt64.max - bound + 1) % bound
            while low < threshold {
                (high, low) = next().multipliedFullWidth(by: bound)
            }
        }
        return high
    }

    /// `Generator.integers(low, high)`, half-open. The span is taken on the bit pattern
    /// so a range wider than `Int64.max` does not overflow on the way in.
    public mutating func integers(low: Int64, high: Int64, count: Int) -> [Int64] {
        precondition(high > low, "high must exceed low")
        let bound = UInt64(bitPattern: high) &- UInt64(bitPattern: low)
        return (0 ..< count).map { _ in
            low &+ Int64(bitPattern: nextBounded(bound))
        }
    }
}
