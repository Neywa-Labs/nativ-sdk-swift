import MLX
import XCTest

@testable import NativSDK

final class SpeculativeSamplingTests: XCTestCase {
    func testEveryRNGCommitPrefixMatchesLegacySequence() {
        for seed: UInt64 in [0, 42, UInt64.max] {
            for retained in 0 ... 10 {
                let legacy = MLXRandom.RandomState(seed: seed)
                var cursor = SpeculativeRandomCursor(seed: seed)
                let block = cursor.prepare(count: 10)
                for index in 0 ..< retained {
                    XCTAssertEqual(
                        block.keys[index].asArray(UInt32.self), legacy.next().asArray(UInt32.self))
                }
                cursor.commit(block, retaining: retained)
                let next = cursor.prepare(count: 1)
                XCTAssertEqual(
                    next.keys[0].asArray(UInt32.self), legacy.next().asArray(UInt32.self))
                // Preparing a discarded block never advances target randomness.
                XCTAssertEqual(
                    cursor.prepare(count: 3).keys[0].asArray(UInt32.self),
                    next.keys[0].asArray(UInt32.self))
            }
        }
    }

    func testExplicitKeysAndBlockRowsMatchOrdinarySamplerAndLegacyCategorical() {
        for dtype in [DType.float32, .bfloat16] {
            let logits = MLXArray([Float(-3), -1, 0, 1, 2, 3, 4, 5], [1, 8]).asType(dtype)
            for sampling in [
                ModelSamplingOptions(temperature: 0.7, seed: 42),
                .init(temperature: 0.7, topP: 0.9, topK: 3, minP: 0.05, seed: 42),
            ] {
                let ordinary = sampling.makeSampler()
                let explicit = sampling.makeKeyedSampler()
                var cursor = SpeculativeRandomCursor(seed: 42)
                for retained in [1, 3, 8, 2, 5] {
                    let block = cursor.prepare(count: 10)
                    let actual = sampleSpeculativeBlock(
                        broadcast(logits, to: [10, 8]), sampler: explicit, keys: block.keys
                    )
                    .asArray(Int.self)
                    for index in 0 ..< retained {
                        XCTAssertEqual(
                            actual[index], ordinary.sample(logits: logits).item(Int.self))
                    }
                    cursor.commit(block, retaining: retained)
                }
            }
            let legacy = MLXRandom.RandomState(seed: 42)
            let sampler = CategoricalSampler(temperature: 0.7, seed: 42)
            for _ in 0 ..< 32 {
                let expected = withRandomState(legacy) {
                    categorical(logits * (1 / MLXArray(Float(0.7))))
                }
                XCTAssertEqual(
                    sampler.sample(logits: logits).item(Int.self), expected.item(Int.self))
            }
        }
    }

    func testPrefixDecisionIncludesCorrectionOrBonus() {
        let predictions = MLXArray([Int32(2), 3, 4, 5])
        for (proposals, retained) in [
            ([9, 3, 4], 1), ([2, 9, 4], 2), ([2, 3, 9], 3), ([2, 3, 4], 4), ([], 1),
        ] {
            let targets = predictions[..<(proposals.count + 1)]
            let decision = speculativeSamplingDecision(
                proposals: MLXArray(proposals).asType(.int32), predictions: targets
            )
            .asArray(Int32.self).map(Int.init)
            XCTAssertEqual(decision[0] + 1, retained)
            XCTAssertEqual(
                Array(decision[1 ..< (retained + 1)]), [2, 3, 4, 5].prefix(retained).map { $0 })
        }
    }

    func testSampledAcceptancePreservesKnownTargetDistribution() {
        let probabilities: [Float] = [0.1, 0.3, 0.6]
        let logits = broadcast(log(MLXArray(probabilities)), to: [4, 3])
        let sampler = ModelSamplingOptions(temperature: 1).makeKeyedSampler()
        var cursor = SpeculativeRandomCursor(seed: 99)
        var histogram = [Int](repeating: 0, count: 3)
        var total = 0
        // Deliberately biased proposals exercise rejection and accepted prefixes.
        for _ in 0 ..< 1024 {
            let block = cursor.prepare(count: 4)
            let predictions = sampleSpeculativeBlock(logits, sampler: sampler, keys: block.keys)
            let decision = speculativeSamplingDecision(
                proposals: MLXArray([Int32(2), 2, 2]), predictions: predictions
            )
            .asArray(Int32.self).map(Int.init)
            let retained = decision[0] + 1
            cursor.commit(block, retaining: retained)
            for token in decision[1 ..< (retained + 1)] {
                histogram[token] += 1
                total += 1
            }
        }
        for (token, probability) in probabilities.enumerated() {
            XCTAssertEqual(Float(histogram[token]) / Float(total), probability, accuracy: 0.035)
        }
    }

    func testEntropyAndProposalRNGAreIndependentOfTargetCursor() {
        var target = SpeculativeRandomCursor(seed: 7)
        let expected = MLXRandom.RandomState(seed: 7)
        let proposal = CategoricalSampler(temperature: 1, seed: 7 ^ 0x0DFA_5202)
        let logits = MLXArray([Float(0), 1, 2])
        for _ in 0 ..< 16 {
            eval(proposal.sample(logits: logits))
            let block = target.prepare(count: 1)
            XCTAssertEqual(block.keys[0].asArray(UInt32.self), expected.next().asArray(UInt32.self))
            target.commit(block, retaining: 1)
        }
        let a = SpeculativeRandomCursor(seed: nil).prepare(count: 1)
        let b = SpeculativeRandomCursor(seed: nil).prepare(count: 1)
        XCTAssertNotEqual(a.keys[0].asArray(UInt32.self), b.keys[0].asArray(UInt32.self))
    }
}
