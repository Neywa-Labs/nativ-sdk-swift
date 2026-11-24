import NativSDK
import XCTest

final class PromptReusePlannerTests: XCTestCase {
    func testPlanColdOnZeroCommonPrefix() {
        let plan = PromptReusePlanner.plan(history: [9, 8, 7], prompt: [1, 2, 3])
        XCTAssertEqual(plan, PromptReusePlan(kind: .cold, commonPrefix: 0, keep: 0, rewind: 3))
    }

    func testPlanPromptIsPrefixOfHistory() {
        // Undo/edit case: history has extra tokens beyond the new prompt.
        let plan = PromptReusePlanner.plan(history: [1, 2, 3, 4, 5], prompt: [1, 2, 3])
        XCTAssertEqual(plan, PromptReusePlan(kind: .exact, commonPrefix: 3, keep: 2, rewind: 3))
    }

    func testPlanSingleTokenExactKeepsSuffixNonEmpty() {
        let plan = PromptReusePlanner.plan(history: [7], prompt: [7])
        XCTAssertEqual(plan, PromptReusePlan(kind: .exact, commonPrefix: 1, keep: 0, rewind: 1))
    }

    func testKeepIsAlwaysBelowPromptCount() {
        let histories: [[Int]] = [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [9, 9]]
        let prompts: [[Int]] = [[1], [1, 2], [1, 2, 3], [1, 9], [2]]
        for history in histories {
            for prompt in prompts {
                let plan = PromptReusePlanner.plan(history: history, prompt: prompt)
                XCTAssertLessThan(
                    plan.keep, prompt.count,
                    "keep must leave at least one token to feed (history: \(history), prompt: \(prompt))",
                )
                XCTAssertEqual(plan.rewind, history.count - plan.keep)
            }
        }
    }
}
