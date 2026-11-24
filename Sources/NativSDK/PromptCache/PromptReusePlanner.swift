/// How a new prompt relates to the token history already resident in a prompt cache.
public enum PromptReuseKind: String, Sendable, Codable {
    /// No usable prefix: prefill the whole prompt into a fresh cache.
    case cold
    /// Prompt equals the history or is a prefix of it: rewind and re-feed the last token.
    case exact
    /// Prompt strictly extends the history: feed only the new suffix.
    case extend
    /// Prompt shares a partial prefix with the history: rewind to it, feed the rest.
    case diverge
    /// Reuse skipped entirely (multimodal input, KV quantization, or reuse disabled).
    case bypass
}

/// Result of comparing a new prompt against cached token history.
public struct PromptReusePlan: Sendable, Equatable {
    public let kind: PromptReuseKind
    /// Longest common prefix between history and prompt.
    public let commonPrefix: Int
    /// Cache tokens to keep. Always < prompt.count: at least one token must be fed to
    /// prime generation, because prefill consumes the entire input it is given.
    public let keep: Int
    /// Tokens to rewind from the live cache (history.count - keep).
    public let rewind: Int

    public init(kind: PromptReuseKind, commonPrefix: Int, keep: Int, rewind: Int) {
        self.kind = kind
        self.commonPrefix = commonPrefix
        self.keep = keep
        self.rewind = rewind
    }
}

/// Pure token-array reuse planning (EXACT / EXTEND / DIVERGE / COLD).
public enum PromptReusePlanner {
    public static func longestCommonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var index = 0
        while index < limit, a[index] == b[index] {
            index += 1
        }
        return index
    }

    /// - Parameters:
    ///   - history: tokens actually resident in the cache (fed prompt + generated tokens).
    ///   - prompt: the full tokenization of the new request. Must be non-empty.
    public static func plan(history: [Int], prompt: [Int]) -> PromptReusePlan {
        precondition(!prompt.isEmpty, "prompt must contain at least one token")
        let lcp = longestCommonPrefix(history, prompt)
        let kind: PromptReuseKind
        let keep: Int
        if lcp == 0 {
            kind = .cold
            keep = 0
        } else if lcp == history.count, prompt.count > history.count {
            kind = .extend
            keep = history.count
        } else if lcp == prompt.count {
            kind = .exact
            keep = prompt.count - 1
        } else {
            kind = .diverge
            keep = lcp
        }
        return PromptReusePlan(
            kind: kind, commonPrefix: lcp, keep: keep, rewind: history.count - keep)
    }
}
