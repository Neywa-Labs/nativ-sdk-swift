import MLX

/// Architecture-independent contract for fixed-width, target-bound drafters.
protocol DFlashDraftConfiguration {
    var strategy: SpeculativeGenerationStatistics.Strategy { get }
    var targetLayers: [Int] { get }
    var proposalCount: Int { get }
    var defaultProposalCount: Int { get }
    func validate(target: any DFlashTarget) throws
}

protocol DFlashDrafter: BaseLanguageModel {
    associatedtype Configuration: DFlashDraftConfiguration
    var config: Configuration { get }
    func newCache() -> [KVCache]
    func propose(
        seed: MLXArray, features: MLXArray, count: Int,
        target: any DFlashTarget, cache: [KVCache], sampler: any LogitSampler
    ) throws -> MLXArray
}
