// Copyright © 2026 Apple Inc.

/// Applies the requested cache strategy as layers become eligible.
struct GenerationCachePolicy {
    let configuration: KVCacheConfiguration?
    private var terminal = false

    init(_ configuration: KVCacheConfiguration?) {
        self.configuration = configuration
    }

    func validate(_ cache: [KVCache]) throws {
        if let configuration {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
    }

    mutating func apply(_ cache: inout [KVCache]) throws {
        guard !terminal else { return }
        guard let configuration else { return }
        let result = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
        terminal = result.pendingLayerCount == 0
    }
}
