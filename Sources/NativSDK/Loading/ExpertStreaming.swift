import Darwin
import Foundation
import MLX
import MLXNN

/// Streams routed expert weights from local safetensors on demand.
/// Allow memory beyond the expert cache budget for the dense model, activations, and KV cache.
/// Prefetching schedules reads. The model's router selects every expert that runs.
public struct ExpertStreamingOptions: Sendable {
    public let cacheSizeBytes: Int
    public let prefetch: Bool
    /// Reuse per-layer expert buffers shared by CPU readers and the GPU. Requires
    /// enough cache for a complete batch per layer; otherwise uses the host cache.
    public let sharedExpertBuffers: Bool
    /// MiniCPM and Gemma 4 keep vision weights lazy until their first image input.
    public let deferVisionWeights: Bool
    /// Maximum experts evaluated together. Packed weights are additionally bounded to
    /// the smaller of the cache budget and 16 MiB (or one expert if it is larger).
    /// Single-token whole-expert decode can use all available shared-buffer slots,
    /// without allocating another packed weight batch.
    /// Single-token projection groups additionally cap their combined live weights at
    /// the smaller of the cache budget and 32 MiB.
    public let expertBatchSize: Int

    public init(
        cacheSizeBytes: Int = 64 * 1024 * 1024, prefetch: Bool = true, expertBatchSize: Int = 8,
        sharedExpertBuffers: Bool = false, deferVisionWeights: Bool = false
    ) {
        self.cacheSizeBytes = cacheSizeBytes
        self.prefetch = prefetch
        self.sharedExpertBuffers = sharedExpertBuffers
        self.deferVisionWeights = deferVisionWeights
        self.expertBatchSize = expertBatchSize
    }
}

/// Nonthrowing model forwards report storage failures before generation emits a token.
protocol InferenceErrorProviding {
    func checkInferenceError() throws
}

/// Models own their expert layout, activation and module replacement policy.
protocol ExpertStreamingModel: InferenceErrorProviding {
    func configureExpertStreaming(
        urls: [URL], options: ExpertStreamingOptions,
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws -> Set<String>
}

struct ExpertStreamingLayout {
    enum Activation { case silu, gelu }
    let layerCount: Int
    let expertCount: Int
    let expertsPerToken: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let expertModulePaths: [String]
    let quantizedModulePath: String
    let activation: Activation
    let canonicalWeightKey: (String) -> String?
}

/// One speculative reader overlaps inference and demand reads. The cache is lock guarded,
/// pread never changes shared file offsets. A demand batch uses up to four readers,
/// in addition to the single speculative reader.
/// MLX arrays stay on the inference thread. Keeping Data here avoids sharing
/// mutable MLX arrays across streams.
final class ExpertWeightStore: @unchecked Sendable {
    struct Key: Hashable, Sendable {
        let layer: Int
        let expert: Int
    }

    struct Tensor: Sendable {
        struct Location: Sendable {
            let url: URL
            let offset: UInt64
        }

        enum Storage: Sendable {
            case stacked(Location)
            case separate([Location])
        }

        let storage: Storage
        let shape: [Int]
        let dtype: DType
        let bytesPerExpert: Int

        init(url: URL, offset: UInt64, shape: [Int], dtype: DType, bytesPerExpert: Int) {
            self.init(
                storage: .stacked(.init(url: url, offset: offset)), shape: shape,
                dtype: dtype, bytesPerExpert: bytesPerExpert)
        }

        init(storage: Storage, shape: [Int], dtype: DType, bytesPerExpert: Int) {
            self.storage = storage
            self.shape = shape
            self.dtype = dtype
            self.bytesPerExpert = bytesPerExpert
        }

        var urls: Set<URL> {
            switch storage {
            case .stacked(let location): [location.url]
            case .separate(let locations): Set(locations.map(\.url))
            }
        }

        func location(for expert: Int) -> Location {
            switch storage {
            case .stacked(let location):
                .init(url: location.url, offset: location.offset + UInt64(expert * bytesPerExpert))
            case .separate(let locations): locations[expert]
            }
        }
    }

    struct Layer: Sendable {
        let tensors: [String: Tensor]
        let quantization: [String: BaseConfiguration.Quantization]
    }

    private struct Entry {
        let data: [String: Data]
        let bytes: Int
        var age: UInt64
    }

    let options: ExpertStreamingOptions
    let excludedWeightNames: Set<String>
    let layers: [Layer]
    private let activation: ExpertStreamingLayout.Activation
    private let readWorkers = min(4, ProcessInfo.processInfo.activeProcessorCount)
    private var layerSlotCaches: [Int: ExpertSlotCache] = [:]
    private var pendingExpertOutputs: [Int: MLXArray] = [:]
    private var sharedBuffersEligible: Bool?
    private func slots(layer: Int) throws -> ExpertSlotCache? {
        guard options.sharedExpertBuffers else { return nil }
        if let ready = layerSlotCaches[layer] { return ready }
        let budget = options.cacheSizeBytes / layers.count
        if sharedBuffersEligible == nil {
            // Do not combine full-budget host caching with partially enabled layer
            // pools. Either every layer fits its share, or all use the host cache.
            sharedBuffersEligible = layers.indices.allSatisfy { index in
                let bytes = layers[index].tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
                let padding = layers[index].tensors.count * Int(getpagesize())
                return budget > padding && (budget - padding) / bytes >= batchSize(layer: index)
            }
        }
        guard sharedBuffersEligible == true else { return nil }
        let pool = try ExpertSlotCache(layers: [layers[layer]], budget: budget)
        layerSlotCaches[layer] = pool
        return pool
    }
    private let queue = DispatchQueue(label: "NativSDK.expert-reads", qos: .userInitiated)
    private var handles: [URL: FileHandle] = [:]
    private var cache: [Key: Entry] = [:]
    private var cachedBytes = 0
    private var age: UInt64 = 0
    private let lock = NSLock()
    private var prefetchGeneration: UInt64 = 0
    // Only the inference thread accesses failure. Speculative read errors are retried on demand.
    private(set) var failure: (any Error)?

    /// Projection-based adapters share the same bounded cache and reader as the
    /// model-specific, whole-expert execution path.
    init(projections: [Layer], excluded: Set<String>, options: ExpertStreamingOptions) throws {
        guard options.cacheSizeBytes > 0, options.expertBatchSize > 0 else {
            throw Self.invalid("cache size and expert batch size must be positive")
        }
        self.options = options
        self.activation = .silu
        self.layers = projections
        self.excludedWeightNames = excluded
        for projection in projections {
            let bytes = projection.tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
            guard bytes <= options.cacheSizeBytes else {
                throw Self.invalid("cache must fit one expert projection (\(bytes) bytes)")
            }
            for url in Set(projection.tensors.values.flatMap(\.urls)) where handles[url] == nil {
                handles[url] = try FileHandle(forReadingFrom: url)
            }
        }
    }

    static func invalid(_ message: String) -> ModelExecutionError {
        .invalidOptions("Expert streaming: " + message)
    }

    convenience init(
        urls: [URL], configuration: MiniCPMVConfiguration,
        options: ExpertStreamingOptions,
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        let text = configuration.textConfiguration
        try self.init(
            urls: urls,
            layout: ExpertStreamingLayout(
                layerCount: text.hiddenLayers, expertCount: text.numExperts,
                expertsPerToken: text.numExpertsPerToken, hiddenSize: text.hiddenSize,
                intermediateSize: text.moeIntermediateSize,
                expertModulePaths: ["mlp.switch_mlp", "mlp.experts"],
                quantizedModulePath: "mlp.switch_mlp", activation: .silu,
                canonicalWeightKey: MiniCPMV.canonicalWeightKey),
            options: options, quantization: quantization,
            perLayerQuantization: perLayerQuantization)
    }

    init(
        urls: [URL], layout: ExpertStreamingLayout, options: ExpertStreamingOptions,
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        guard options.cacheSizeBytes > 0 else { throw Self.invalid("cache size must be positive") }
        guard options.expertBatchSize > 0 else {
            throw Self.invalid("expert batch size must be positive")
        }
        self.options = options
        self.activation = layout.activation
        let count = layout.expertCount
        guard count > 0 else { throw Self.invalid("the model has no routed experts") }
        guard (1 ... count).contains(layout.expertsPerToken) else {
            throw Self.invalid("invalid number of experts per token")
        }
        var tensors = Array(
            repeating: [String: Tensor](), count: layout.layerCount)
        var excluded = Set<String>()
        var separate = Array(
            repeating: [String: [Int: ExpertTensorIndex.Tensor]](), count: layout.layerCount)
        let index = try ExpertTensorIndex(urls: urls)
        for (name, tensor) in index.tensors {
            guard let canonical = layout.canonicalWeightKey(name),
                canonical.hasPrefix("language_model.model.layers.")
            else { continue }
            let components = canonical.split(separator: ".")
            guard components.count >= 6, let layer = Int(components[3]),
                tensors.indices.contains(layer)
            else { continue }
            let relative = components.dropFirst(4).joined(separator: ".")
            guard
                let modulePath = layout.expertModulePaths.first(where: {
                    relative.hasPrefix($0 + ".")
                })
            else { continue }
            let suffix = String(relative.dropFirst(modulePath.count + 1))
            let parts = suffix.split(separator: ".", maxSplits: 1)
            if parts.count == 2, let expert = Int(parts[0]) {
                guard (0 ..< count).contains(expert) else {
                    throw Self.invalid("invalid expert index in \(name)")
                }
                let projection = String(parts[1])
                guard separate[layer][projection]?[expert] == nil else {
                    throw Self.invalid("duplicate tensor \(name)")
                }
                separate[layer][projection, default: [:]][expert] = tensor
            } else {
                guard tensors[layer][suffix] == nil else {
                    throw Self.invalid("duplicate tensor \(name)")
                }
                tensors[layer][suffix] = try tensor.experts(count: count)
            }
            excluded.insert(name)
            if handles[tensor.url] == nil {
                handles[tensor.url] = try FileHandle(forReadingFrom: tensor.url)
            }
        }
        for layer in separate.indices {
            for (projection, experts) in separate[layer] {
                guard tensors[layer][projection] == nil else {
                    throw Self.invalid("both stacked and separate expert tensors at layer \(layer)")
                }
                guard experts.count == count, let first = experts[0], first.byteCount > 0,
                    experts.values.allSatisfy({
                        $0.shape == first.shape && $0.dtype == first.dtype
                            && $0.byteCount == first.byteCount
                    })
                else {
                    throw Self.invalid("missing or inconsistent separate experts at layer \(layer)")
                }
                tensors[layer][projection] = Tensor(
                    storage: .separate(
                        (0 ..< count).map {
                            .init(url: experts[$0]!.url, offset: experts[$0]!.offset)
                        }),
                    shape: first.shape, dtype: first.dtype, bytesPerExpert: first.byteCount)
            }
        }
        var layerSpecs: [Layer] = []
        for (index, layer) in tensors.enumerated() {
            let fused = layer["gate_up_proj"] != nil
            let projections =
                fused ? ["gate_up_proj", "down_proj"] : ["gate_proj", "up_proj", "down_proj"]
            var quantizers: [String: BaseConfiguration.Quantization] = [:]
            for projection in projections {
                let name = fused ? projection : projection + ".weight"
                guard let weight = layer[name], weight.shape.count == 2 else {
                    throw Self.invalid("missing expert projection at layer \(index): \(name)")
                }
                if layer[projection + ".scales"] != nil {
                    let path =
                        "language_model.model.layers.\(index).\(layout.quantizedModulePath).\(projection)"
                    let selected =
                        perLayerQuantization.map { $0.quantization(layer: path) } ?? quantization
                    guard let q = selected,
                        q.groupSize > 0, [2, 3, 4, 5, 6, 8].contains(q.bits),
                        weight.dtype == .uint32
                    else { throw Self.invalid("missing or invalid quantization for \(path)") }
                    if q.mode == .affine, layer[projection + ".biases"] == nil {
                        throw Self.invalid("missing affine biases for \(path)")
                    }
                    quantizers[projection] = q
                } else if ![DType.float32, .float16, .bfloat16].contains(weight.dtype) {
                    throw Self.invalid("packed expert weights require quantization metadata")
                }
            }
            let hidden = layout.hiddenSize
            let intermediate = layout.intermediateSize
            var expectedNames = Set<String>()
            for projection in projections {
                let name = fused ? projection : projection + ".weight"
                expectedNames.insert(name)
                let rows =
                    projection == "down_proj"
                    ? hidden : (projection == "gate_up_proj" ? 2 * intermediate : intermediate)
                let columns = projection == "down_proj" ? intermediate : hidden
                if let q = quantizers[projection] {
                    guard columns % q.groupSize == 0, columns * q.bits % 32 == 0,
                        layer[name]?.shape == [rows, columns * q.bits / 32],
                        layer[projection + ".scales"]?.shape == [rows, columns / q.groupSize]
                    else {
                        throw Self.invalid(
                            "incorrect packed projection dimensions at layer \(index)")
                    }
                    expectedNames.insert(projection + ".scales")
                    if let bias = layer[projection + ".biases"] {
                        guard bias.shape == [rows, columns / q.groupSize] else {
                            throw Self.invalid(
                                "incorrect quantization bias dimensions at layer \(index)")
                        }
                        expectedNames.insert(projection + ".biases")
                    }
                } else if layer[name]?.shape != [rows, columns] {
                    throw Self.invalid("incorrect expert dimensions at layer \(index)")
                }
            }
            guard Set(layer.keys) == expectedNames else {
                throw Self.invalid("unsupported expert tensor layout at layer \(index)")
            }
            let expertBytes = layer.values.reduce(0) { $0 + $1.bytesPerExpert }
            guard expertBytes <= options.cacheSizeBytes else {
                throw Self.invalid("cache must fit at least one expert (\(expertBytes) bytes)")
            }
            layerSpecs.append(Layer(tensors: layer, quantization: quantizers))
        }
        layers = layerSpecs
        excludedWeightNames = excluded
    }

    deinit { for handle in handles.values { try? handle.close() } }

    var residentBytes: Int {
        layerSlotCaches.values.reduce(0, { $0 + $1.byteCount }) + lock.withLock { cachedBytes }
    }

    func prefetch(layer: Int, experts: [Int]) {
        guard options.prefetch else { return }
        let direct: ExpertSlotCache?
        do { direct = try slots(layer: layer) } catch {
            failure = error
            return
        }
        let generation = lock.withLock {
            prefetchGeneration &+= 1
            return prefetchGeneration
        }
        queue.async { [self] in
            for expert in Set(experts).sorted() {
                guard lock.withLock({ prefetchGeneration == generation }) else { return }
                if let direct {
                    _ = try? direct.read(
                        Key(layer: layer, expert: expert), tensors: layers[layer].tensors,
                        handles: handles, speculative: true)
                } else {
                    _ = try? read(Key(layer: layer, expert: expert))
                }
            }
        }
    }

    /// Snapshot the ready set without waiting for speculative I/O.
    func readyFirst(layer: Int, experts: [Int]) -> [Int] {
        if let direct = layerSlotCaches[layer] {
            let ready = direct.ready(experts.map { Key(layer: layer, expert: $0) })
            return experts.filter { ready.contains(Key(layer: layer, expert: $0)) }
                + experts.filter { !ready.contains(Key(layer: layer, expert: $0)) }
        }
        return lock.withLock {
            experts.filter { cache[Key(layer: layer, expert: $0)] != nil }
                + experts.filter { cache[Key(layer: layer, expert: $0)] == nil }
        }
    }

    private func read(_ key: Key) throws -> [String: Data] {
        if let hit = lock.withLock({ () -> [String: Data]? in
            age &+= 1
            guard var entry = cache[key] else { return nil }
            entry.age = age
            cache[key] = entry
            return entry.data
        }) {
            return hit
        }
        let specs = layers[key.layer].tensors
        let bytes = specs.values.reduce(0) { $0 + $1.bytesPerExpert }
        var result: [String: Data] = [:]
        for (name, tensor) in specs.sorted(by: {
            $0.value.location(for: key.expert).offset < $1.value.location(for: key.expert).offset
        }) {
            let location = tensor.location(for: key.expert)
            let descriptor = handles[location.url]!.fileDescriptor
            let offset = location.offset
            var data = Data(count: tensor.bytesPerExpert)
            try data.withUnsafeMutableBytes { buffer in
                var completed = 0
                while completed < buffer.count {
                    let count = pread(
                        descriptor, buffer.baseAddress! + completed, buffer.count - completed,
                        off_t(offset) + off_t(completed))
                    if count < 0 && errno == EINTR { continue }
                    guard count > 0 else {
                        throw Self.invalid(
                            "failed read for layer \(key.layer), expert \(key.expert)")
                    }
                    completed += count
                }
            }
            if name == "weight_global_scale" || name == "weight_scale_2" {
                let value: Float = data.withUnsafeBytes { bytes in
                    switch tensor.dtype {
                    case .float32:
                        Float(
                            bitPattern: UInt32(littleEndian: bytes.loadUnaligned(as: UInt32.self)))
                    case .float16:
                        Float(
                            Float16(
                                bitPattern: UInt16(
                                    littleEndian: bytes.loadUnaligned(as: UInt16.self))))
                    case .bfloat16:
                        Float(
                            bitPattern: UInt32(
                                UInt16(littleEndian: bytes.loadUnaligned(as: UInt16.self))) << 16)
                    default: .nan
                    }
                }
                guard value.isFinite, value > 0 else {
                    throw Self.invalid(
                        "invalid NVFP4 global scale at layer \(key.layer), expert \(key.expert)")
                }
            }
            result[name] = data
        }
        return lock.withLock {
            // A demand read may race the one speculative reader; prefer the already cached copy.
            if let entry = cache[key] { return entry.data }
            while cachedBytes + bytes > options.cacheSizeBytes,
                let oldest = cache.min(by: { $0.value.age < $1.value.age })
            {
                cachedBytes -= oldest.value.bytes
                cache[oldest.key] = nil
            }
            age &+= 1
            cache[key] = Entry(data: result, bytes: bytes, age: age)
            cachedBytes += bytes
            return result
        }
    }

    /// Workers only stage CPU bytes. MLX graph construction stays on the
    /// inference thread; concurrentPerform joins every worker before collection.
    private final class BatchReads: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Result<[String: Data], any Error>?]

        init(count: Int) { values = Array(repeating: nil, count: count) }
        func set(_ value: Result<[String: Data], any Error>, at index: Int) {
            lock.withLock { values[index] = value }
        }
        func collect() throws -> [[String: Data]] {
            try values.map { try $0!.get() }
        }
    }

    /// Parallelize staging within the existing batch bound. Small batches retain
    /// serial reads to avoid dispatch overhead; results retain requested expert order.
    private func read(layer: Int, experts: [Int]) throws -> [[String: Data]] {
        let bytes = layers[layer].tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
        guard readWorkers > 1, experts.count > 1, bytes * experts.count >= 1024 * 1024 else {
            return try experts.map { try read(Key(layer: layer, expert: $0)) }
        }
        let results = BatchReads(count: experts.count)
        let workers = min(readWorkers, experts.count)
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            for index in stride(from: worker, to: experts.count, by: workers) {
                results.set(
                    Result { try read(Key(layer: layer, expert: experts[index])) }, at: index)
            }
        }
        return try results.collect()
    }

    /// Bound the live weight graph independently of the number of routed prompt experts.
    func batchSize(layer: Int, singleToken: Bool = false) -> Int {
        if singleToken, let slots = layerSlotCaches[layer] {
            return min(options.expertBatchSize, slots.count)
        }
        let bytes = layers[layer].tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
        let budget = min(options.cacheSizeBytes, 16 * 1024 * 1024)
        return min(options.expertBatchSize, max(1, budget / bytes))
    }

    func apply(_ x: MLXArray, layer: Int, expert: Int) throws -> MLXArray {
        try apply(
            x, layer: layer, experts: [expert],
            expertIndices: Array(repeating: 0, count: x.dim(0)))
    }

    /// Returns a lazy result. The caller must evaluate its scatter before loading another batch.
    /// expertIndices maps each input row to a local index in experts, not a checkpoint expert ID.
    func apply(
        _ x: MLXArray, layer: Int, experts: [Int], expertIndices: [Int],
        singleToken: Bool = false
    ) throws -> MLXArray {
        let direct = try slots(layer: layer)
        guard !experts.isEmpty, experts.count <= batchSize(layer: layer, singleToken: singleToken),
            expertIndices.count == x.dim(0),
            expertIndices.allSatisfy({ experts.indices.contains($0) })
        else { throw Self.invalid("invalid expert execution batch") }
        let slices: [[String: Data]]
        let mappedIndices: [Int]
        if let direct {
            let keys = experts.map { Key(layer: layer, expert: $0) }
            try direct.begin(keys)
            let results = BatchReads(count: experts.count)
            DispatchQueue.concurrentPerform(iterations: min(readWorkers, experts.count)) { worker in
                for i in stride(
                    from: worker, to: experts.count, by: min(readWorkers, experts.count))
                {
                    results.set(
                        Result {
                            _ = try direct.read(
                                keys[i], tensors: layers[layer].tensors, handles: handles,
                                speculative: false)
                            return [String: Data]()
                        }, at: i)
                }
            }
            _ = try results.collect()
            let slots = try direct.selected(keys)
            mappedIndices = expertIndices.map { slots[$0] }
            slices = []
        } else {
            slices = try read(layer: layer, experts: experts)
            mappedIndices = expertIndices
        }
        let spec = layers[layer]
        let indices = MLXArray(mappedIndices.map(UInt32.init))
        func array(_ name: String) -> MLXArray? {
            if let direct { return direct.arrays[name] }
            guard let tensor = spec.tensors[name] else { return nil }
            if slices.count == 1 {
                return MLXArray(slices[0][name]!, [1] + tensor.shape, dtype: tensor.dtype)
            }
            // Pack on the inference stream (GPU by default), avoiding the additional
            // zero-filled host buffer and CPU copy. The caller's existing scatter eval
            // realizes this concatenate and releases its inputs within the batch bound.
            let parts = slices.map {
                MLXArray($0[name]!, [1] + tensor.shape, dtype: tensor.dtype)
            }
            return concatenated(parts, axis: 0)
        }
        func floatingProject(_ input: MLXArray, _ weight: MLXArray) -> MLXArray {
            gatherMM(
                input.expandedDimensions(axis: -2), weight.swappedAxes(-1, -2),
                rhsIndices: indices
            ).squeezed(axis: -2)
        }
        func project(_ input: MLXArray, _ name: String) -> MLXArray {
            let weight = array(name + ".weight")!
            if let q = spec.quantization[name] {
                return gatherQuantizedMM(
                    input.expandedDimensions(axis: -2), weight,
                    scales: array(name + ".scales")!, biases: array(name + ".biases"),
                    rhsIndices: indices,
                    transpose: true, groupSize: q.groupSize, bits: q.bits, mode: q.mode
                ).squeezed(axis: -2)
            }
            return floatingProject(input, weight)
        }
        if let gateUp = array("gate_up_proj"), let down = array("down_proj") {
            if experts.count == 1 && direct == nil {
                // Preserve the original fused checkpoint path when the byte cap permits only
                // one expert; matrix and gathered-vector kernels can round differently.
                let parts = matmul(x, gateUp[0].T).split(parts: 2, axis: -1)
                let gate = activation == .silu ? silu(parts[0]) : geluApproximate(parts[0])
                return matmul(gate * parts[1], down[0].T)
            }
            let parts = floatingProject(x, gateUp).split(parts: 2, axis: -1)
            let gate = activation == .silu ? silu(parts[0]) : geluApproximate(parts[0])
            return floatingProject(gate * parts[1], down)
        }
        let gate = project(x, "gate_proj")
        let up = project(x, "up_proj")
        let activated =
            activation == .silu ? compiledSiluProduct(gate, up) : geluApproximate(gate) * up
        return project(activated, "down_proj")
    }

    /// A gathered projection with the same broadcasting contract as SwitchLinear.
    /// Activations, router weights, expert scales and input transforms stay in the model.
    func project(
        _ x: MLXArray, indices: MLXArray, projection: Int, inputSize: Int, outputSize: Int,
        expertCount: Int, sortedIndices: Bool, assignments: [Int]? = nil,
        evaluation: ExpertEvaluationBatch? = nil
    ) -> MLXArray {
        let spec = layers[projection]
        var dtype = x.dtype
        let types: [DType]
        if let q = spec.quantization["weight"] {
            types =
                q.mode == .affine
                ? [spec.tensors["scales"]!.dtype, spec.tensors["biases"]!.dtype] : []
        } else {
            types = [spec.tensors["weight"]!.dtype]
        }
        for type in types + (spec.tensors["bias"].map { [$0.dtype] } ?? []) {
            // Let MLX apply its promotion rules without allocating a weight matrix.
            dtype = (MLXArray.zeros([0], dtype: dtype) + MLXArray.zeros([0], dtype: type)).dtype
        }
        let shape = indices.shape + [x.dim(-2), outputSize]
        guard failure == nil else { return .zeros(shape, dtype: dtype) }
        let assignments = assignments ?? indices.asArray(Int.self)
        guard assignments.allSatisfy({ (0 ..< expertCount).contains($0) }) else {
            record(Self.invalid("router selected an invalid expert"))
            return .zeros(shape, dtype: dtype)
        }
        guard !assignments.isEmpty else { return .zeros(shape, dtype: dtype) }
        let rowsPerAssignment = x.dim(-2)
        // Match MLX GatherQMM's matrix/vector dispatch using the original expert
        // count. Packing changes B/E; switching kernels changes low-precision
        // dequantization rounding, even with identical weights and activations.
        let quantizedMatrix =
            sortedIndices && rowsPerAssignment == 1
            && assignments.count >= 16 && assignments.count / expertCount >= 4
        let input = broadcast(x, to: indices.shape + [rowsPerAssignment, inputSize])
            .reshaped(-1, rowsPerAssignment, inputSize)
        let output = MLXArray.zeros(
            [assignments.count, rowsPerAssignment, outputSize], dtype: dtype)
        var slots: [Int: [Int]] = [:]
        for (slot, expert) in assignments.enumerated() { slots[expert, default: []].append(slot) }
        let experts = readyFirst(layer: projection, experts: slots.keys.sorted())
        do {
            let direct = try self.slots(layer: projection)
            let size = batchSize(layer: projection, singleToken: evaluation != nil)
            let batches = stride(from: 0, to: experts.count, by: size).map { start in
                Array(experts[start ..< min(start + size, experts.count)])
            }
            for batch in batches {
                try autoreleasepool {
                    // Direct decode weights already belong to the bounded cache;
                    // they do not allocate another packed weight graph.
                    let weightBytes =
                        direct != nil && !quantizedMatrix
                        ? 0 : batch.count * spec.tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
                    evaluation?.prepare(weightBytes: weightBytes)
                    defer { if evaluation == nil { direct?.releasePins() } }
                    let slices: [[String: Data]]
                    let mapped: [Int]
                    if let direct {
                        evaluation?.hold(direct)
                        let keys = batch.map { Key(layer: projection, expert: $0) }
                        try direct.begin(keys)
                        let results = BatchReads(count: keys.count)
                        let workers = min(readWorkers, keys.count)
                        DispatchQueue.concurrentPerform(iterations: workers) { worker in
                            for i in stride(from: worker, to: keys.count, by: workers) {
                                results.set(
                                    Result {
                                        _ = try direct.read(
                                            keys[i], tensors: spec.tensors,
                                            handles: handles, speculative: false)
                                        return [String: Data]()
                                    }, at: i)
                            }
                        }
                        _ = try results.collect()
                        mapped = try direct.selected(keys)
                        slices = []
                    } else {
                        slices = try read(layer: projection, experts: batch)
                        mapped = Array(batch.indices)
                    }
                    // During grouped decode the evaluation owns the pins. Prefill
                    // evaluates each batch below before releasing its weight slots.
                    let packedSlots = direct != nil && quantizedMatrix ? MLXArray(mapped) : nil
                    func tensor(_ name: String) -> MLXArray? {
                        if let direct {
                            guard let array = direct.arrays[name] else { return nil }
                            // Preserve GatherQMM's matrix dispatch during prefill.
                            return packedSlots.map { array[$0] } ?? array
                        }
                        guard let metadata = spec.tensors[name] else { return nil }
                        return concatenated(
                            slices.map {
                                MLXArray($0[name]!, [1] + metadata.shape, dtype: metadata.dtype)
                            },
                            axis: 0)
                    }
                    var positions: [Int] = []
                    var local: [UInt32] = []
                    for (index, expert) in batch.enumerated() {
                        let selected = slots[expert]!
                        positions += selected
                        local += Array(
                            repeating: UInt32(packedSlots == nil ? mapped[index] : index),
                            count: selected.count)
                    }
                    let rows = MLXArray(positions)
                    var selectedInput = input[rows]
                    if spec.quantization["weight"] != nil, quantizedMatrix {
                        // Rare experts can leave a packed batch below GatherQMM's
                        // matrix threshold. Duplicate its last row, then discard the
                        // padding; this keeps the kernel without loading more experts.
                        let padding = max(16, 4 * batch.count) - local.count
                        if padding > 0 {
                            selectedInput = concatenated(
                                [
                                    selectedInput,
                                    broadcast(selectedInput[(-1)...], to: [padding, 1, inputSize]),
                                ], axis: 0)
                            local += Array(repeating: local.last!, count: padding)
                        }
                    }
                    let selected = MLXArray(local)
                    let weight = tensor("weight")!
                    let inverseGlobal = tensor("weight_global_scale")
                    let global = inverseGlobal ?? tensor("weight_scale_2")
                    var result: MLXArray
                    if let q = spec.quantization["weight"] {
                        result = gatherQuantizedMM(
                            global == nil ? selectedInput : selectedInput.asType(.float32),
                            weight, scales: tensor("scales")!,
                            biases: tensor("biases"),
                            rhsIndices: selected, transpose: true,
                            groupSize: q.groupSize, bits: q.bits, mode: q.mode,
                            sortedIndices: quantizedMatrix)
                    } else {
                        result = gatherMM(
                            input[rows], weight.swappedAxes(-1, -2), rhsIndices: selected,
                            sortedIndices: sortedIndices)
                    }
                    if let global {
                        let scale = global[selected].asType(.float32)
                            .expandedDimensions(axes: [-1, -2])
                        result = (inverseGlobal != nil ? result / scale : result * scale).asType(
                            x.dtype)
                    }
                    if let bias = tensor("bias") {
                        result = result + bias[selected].expandedDimensions(axis: -2)
                    }
                    output[rows] = result[..<positions.count]
                    if let evaluation {
                        evaluation.append(
                            output, weightBytes: weightBytes, pipelined: batches.count > 1)
                    } else {
                        // Bound live weights during prefill, including all-expert prompts.
                        eval(output)
                    }
                }
            }
        } catch {
            evaluation?.flush()
            record(error)
        }
        return output.reshaped(shape)
    }

    /// Evaluate selected experts in bounded batches, preserving the router slot order.
    func selectedExperts(_ x: MLXArray, indices: MLXArray, layer: Int) -> MLXArray {
        let topK = indices.dim(-1)
        let assignments = indices.asArray(Int.self)
        // Router readback normally completes these dependent outputs already.
        // Evaluate explicitly for callers with independent routing inputs too,
        // then release pins instead of reserving them until the next token.
        if !pendingExpertOutputs.isEmpty {
            eval(Array(pendingExpertOutputs.values))
            for pendingLayer in pendingExpertOutputs.keys {
                layerSlotCaches[pendingLayer]?.releasePins()
            }
            pendingExpertOutputs.removeAll(keepingCapacity: true)
        }
        let flat = x.reshaped(-1, x.dim(-1))
        var slots: [Int: [Int]] = [:]
        for (slot, expert) in assignments.enumerated() {
            slots[expert, default: []].append(slot)
        }
        let output = MLXArray.zeros([assignments.count, x.dim(-1)], dtype: x.dtype)
        do {
            let singleToken = flat.dim(0) == 1
            if singleToken { _ = try self.slots(layer: layer) }
            let experts = readyFirst(layer: layer, experts: slots.keys.sorted())
            let batchSize = batchSize(layer: layer, singleToken: singleToken)
            for start in stride(from: 0, to: experts.count, by: batchSize) {
                let batch = Array(experts[start ..< min(start + batchSize, experts.count)])
                var positions: [Int] = []
                var expertIndices: [Int] = []
                for (index, expert) in batch.enumerated() {
                    let selected = slots[expert]!
                    positions.append(contentsOf: selected)
                    expertIndices.append(
                        contentsOf: repeatElement(index, count: selected.count))
                }
                let rows = MLXArray(positions.map { $0 / topK })
                output[MLXArray(positions)] = try apply(
                    flat[rows], layer: layer, experts: batch, expertIndices: expertIndices,
                    singleToken: singleToken)
                // A single decode batch can remain pinned until the next layer's
                // router readback completes its dependent hidden-state graph. Submit
                // it now while the CPU constructs the rest of the layer. Prefill and
                // multi-batch calls retain the explicit reuse fence.
                if layerSlotCaches[layer] != nil, flat.dim(0) == 1, experts.count <= batchSize,
                    options.sharedExpertBuffers
                {
                    asyncEval(output)
                    pendingExpertOutputs[layer] = output
                } else {
                    eval(output)
                    layerSlotCaches[layer]?.releasePins()
                }
            }
        } catch {
            record(error)
        }
        return output.reshaped(x.shape.dropLast() + [topK, x.dim(-1)])
    }

    func record(_ error: any Error) { if failure == nil { failure = error } }
    func checkError() throws { if let failure { throw failure } }
}
