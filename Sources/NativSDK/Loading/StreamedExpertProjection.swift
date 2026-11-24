import Foundation
import MLX
import MLXNN

/// Models describe architecture-specific expert aliases using normalized tensor names.
protocol SeparateExpertWeightsProviding {
    func separateExpertWeightNames(for path: String, expert: Int) -> [String]
}

/// Installs storage-backed projections without replacing the model's expert computation.
/// Native MLX keys are the default; models may also declare separate expert aliases.
enum StreamedExpertProjection {
    static func install(
        model: Module, urls: [URL], options: ExpertStreamingOptions,
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?,
        transformation: CheckpointTransformation = .none
    ) throws -> Set<String> {
        let modules = model.namedModules().compactMap { path, module -> (String, SwitchLinear)? in
            guard let projection = module as? SwitchLinear else { return nil }
            return (path, projection)
        }.sorted { $0.0 < $1.0 }
        guard !modules.isEmpty else {
            throw ExpertWeightStore.invalid("the model has no routed expert projections")
        }
        let index = try ExpertTensorIndex(urls: urls)
        let separate = model as? any SeparateExpertWeightsProviding
        var layers: [ExpertWeightStore.Layer] = []
        var excluded = Set<String>()
        for (path, module) in modules {
            var tensors: [String: ExpertWeightStore.Tensor] = [:]
            for name in [
                "weight", "scales", "biases", "bias", "weight_global_scale", "weight_scale_2",
            ] {
                let key = path + "." + name
                if let resolved = try index.projection(
                    transformation.sourceTensorName(for: key), count: module.numExperts,
                    separateNames: {
                        (separate?.separateExpertWeightNames(for: key, expert: $0) ?? [])
                            .map { transformation.sourceTensorName(for: $0) }
                    })
                {
                    tensors[name] = try transformation.prepareExpertTensor(
                        resolved.tensor, for: key)
                    excluded.formUnion(resolved.names)
                }
            }
            guard let weight = tensors["weight"] else {
                throw ExpertWeightStore.invalid(
                    "missing expert tensor \(path).weight; use a supported separate-expert layout or convert to native MLX"
                )
            }
            var quantizers: [String: BaseConfiguration.Quantization] = [:]
            if tensors["scales"] != nil {
                let selected =
                    transformation.expertQuantization
                    ?? (perLayerQuantization.map { $0.quantization(layer: path) } ?? quantization)
                guard let q = selected else {
                    throw ExpertWeightStore.invalid("missing quantization metadata for \(path)")
                }
                try q.validateForConversion()
                guard module.inputDims.isMultiple(of: q.groupSize),
                    (module.inputDims * q.bits).isMultiple(of: 32), weight.dtype == .uint32,
                    weight.shape == [
                        module.outputDims, module.inputDims * q.bits / 32,
                    ],
                    tensors["scales"]?.shape == [
                        module.outputDims, module.inputDims / q.groupSize,
                    ],
                    q.mode != .affine || tensors["biases"] != nil
                else {
                    throw ExpertWeightStore.invalid("invalid packed expert projection \(path)")
                }
                let floating: [DType] = [.float32, .float16, .bfloat16]
                guard
                    q.mode == .affine
                        ? floating.contains(tensors["scales"]!.dtype)
                            && tensors["biases"].map({ floating.contains($0.dtype) }) == true
                        : tensors["scales"]!.dtype == .uint8 && tensors["biases"] == nil
                else {
                    throw ExpertWeightStore.invalid("invalid quantization types for \(path)")
                }
                if let biases = tensors["biases"], biases.shape != tensors["scales"]?.shape {
                    throw ExpertWeightStore.invalid("invalid quantization biases for \(path)")
                }
                quantizers["weight"] = q
            } else {
                guard [.float32, .float16, .bfloat16].contains(weight.dtype),
                    weight.shape == [module.outputDims, module.inputDims],
                    tensors["biases"] == nil
                else {
                    throw ExpertWeightStore.invalid("invalid floating expert projection \(path)")
                }
            }
            guard tensors["weight_global_scale"] == nil || tensors["weight_scale_2"] == nil else {
                throw ExpertWeightStore.invalid("conflicting NVFP4 global scales for \(path)")
            }
            if let global = tensors["weight_global_scale"] ?? tensors["weight_scale_2"] {
                guard quantizers["weight"]?.mode == .nvfp4, global.shape.isEmpty,
                    [.float32, .float16, .bfloat16].contains(global.dtype)
                else {
                    throw ExpertWeightStore.invalid("invalid NVFP4 global scale for \(path)")
                }
            } else if transformation.expertQuantization?.mode == .nvfp4 {
                throw ExpertWeightStore.invalid("missing NVFP4 global scale for \(path)")
            }
            guard (module.bias != nil) == (tensors["bias"] != nil),
                tensors["bias"].map({
                    $0.shape == [module.outputDims]
                        && [.float32, .float16, .bfloat16].contains($0.dtype)
                }) ?? true
            else { throw ExpertWeightStore.invalid("invalid expert bias for \(path)") }
            layers.append(.init(tensors: tensors, quantization: quantizers))
        }
        // All descriptors are checked before mutating the model or materializing weights.
        let store = try ExpertWeightStore(
            projections: layers, excluded: excluded, options: options)
        let schedule = ExpertProjectionSchedule(
            store: store, paths: modules.map(\.0), expertCounts: modules.map { $0.1.numExperts })
        let replacements: [(String, Module)] = modules.enumerated().map { index, entry in
            let (path, module) = entry
            return (path, StreamingSwitchLinear(module, projection: index, schedule: schedule))
        }
        try replaceByParent(
            model: model, replacements: Dictionary(uniqueKeysWithValues: replacements))
        model.invalidateCompiledTraces()
        return excluded
    }
}

/// Learns the model's execution order. Same-block projections reuse the actual routed
/// IDs; transitions to another block use its previous observed routing. Prediction only
/// warms storage: the actual indices passed to each projection always decide execution.
final class ExpertProjectionSchedule {
    let store: ExpertWeightStore
    private let groups: [String]
    private let expertCounts: [Int]
    private var successor: [Int: Int] = [:]
    private var recentExperts: [Int: [Int]] = [:]
    private var previous: Int?
    let evaluation: ExpertEvaluationBatch
    private let groupMembers: [String: Set<Int>]
    private var currentGroup: String?
    private var visited = Set<Int>()
    private var routedIndices: [ObjectIdentifier: (MLXArray, [Int])] = [:]

    init(store: ExpertWeightStore, paths: [String], expertCounts: [Int]) {
        self.store = store
        self.expertCounts = expertCounts
        let groups = paths.map { path in
            path.lastIndex(of: ".").map { String(path[..<$0]) } ?? ""
        }
        self.groups = groups
        self.groupMembers = Dictionary(grouping: groups.indices, by: { groups[$0] })
            .mapValues { Set($0) }
        self.evaluation = ExpertEvaluationBatch(
            byteLimit: min(store.options.cacheSizeBytes, 32 * 1024 * 1024))
    }

    /// Group only single-token expert calls. All activations, biases, transforms and
    /// reductions still execute in their original parent modules. Projection batches
    /// keep their existing size; their combined lazy graph has a separate byte cap.
    func project(
        _ x: MLXArray, indices: MLXArray, projection: Int, inputSize: Int,
        outputSize: Int, expertCount: Int, sortedIndices: Bool
    ) -> MLXArray {
        let group = groups[projection]
        let grouped =
            !sortedIndices && x.dim(-2) == 1 && indices.ndim > 0
            && indices.size == indices.dim(-1) && (groupMembers[group]?.count ?? 0) > 1
        if !grouped || currentGroup != group || visited.contains(projection) {
            if !store.options.sharedExpertBuffers { evaluation.flush() }
            clearGroup()
        }
        var assignments: [Int]?
        if grouped {
            currentGroup = group
            let identity = ObjectIdentifier(indices)
            if let cached = routedIndices[identity] {
                assignments = cached.1
            } else {
                let values = indices.asArray(Int.self)
                routedIndices[identity] = (indices, values)
                assignments = values
            }
        }
        // The next router normally waits on the prior block already. Keep an
        // explicit completion check for independent routing inputs as well.
        if store.options.sharedExpertBuffers && visited.isEmpty { evaluation.flush() }
        schedule(after: projection, indices: indices, assignments: assignments)
        let result = store.project(
            x, indices: indices, projection: projection, inputSize: inputSize,
            outputSize: outputSize, expertCount: expertCount, sortedIndices: sortedIndices,
            assignments: assignments, evaluation: grouped ? evaluation : nil)
        if grouped {
            visited.insert(projection)
            if visited == groupMembers[group] {
                if store.options.sharedExpertBuffers {
                    evaluation.submit()
                } else {
                    evaluation.flush()
                }
                clearGroup()
            }
        }
        return result
    }

    private func clearGroup() {
        currentGroup = nil
        visited.removeAll(keepingCapacity: true)
        routedIndices.removeAll(keepingCapacity: true)
    }

    func schedule(after projection: Int, indices: MLXArray, assignments: [Int]? = nil) {
        guard store.options.prefetch else { return }
        if let previous, previous != projection { successor[previous] = projection }
        previous = projection
        // A bounded tail also works during prefill, without fetching every prompt expert.
        let count = min(indices.size, store.options.expertBatchSize)
        guard count > 0 else { return }
        let routed =
            assignments.map { Array($0.suffix(count)) }
            ?? indices.flattened()[(-count)...].asArray(Int.self)
        recentExperts[projection] = routed
        guard let next = successor[projection] else { return }
        let predicted = groups[next] == groups[projection] ? routed : recentExperts[next] ?? []
        store.prefetch(
            layer: next, experts: predicted.filter { $0 >= 0 && $0 < expertCounts[next] })
    }
}

/// Bounds newly packed weights and retains shared slots until GPU work completes.
/// Shared-buffer blocks can submit asynchronously and flush at the next router;
/// packed weights flush before exceeding the cap. An oversized single projection
/// batch is evaluated immediately, matching the existing one-expert minimum.
final class ExpertEvaluationBatch {
    let byteLimit: Int
    private var outputs: [MLXArray] = []
    private var pinnedCaches: [ExpertSlotCache] = []
    private(set) var pendingWeightBytes = 0
    private(set) var peakPendingWeightBytes = 0
    private(set) var evaluationCount = 0

    init(byteLimit: Int) { self.byteLimit = byteLimit }

    func prepare(weightBytes: Int) {
        if weightBytes > byteLimit - pendingWeightBytes { flush() }
    }

    /// A pool can be reused only after its earlier projection has completed.
    func hold(_ cache: ExpertSlotCache) {
        if pinnedCaches.contains(where: { $0 === cache }) { flush() }
        pinnedCaches.append(cache)
    }

    func append(_ output: MLXArray, weightBytes: Int, pipelined: Bool) {
        // Pipeline shared buffers and partial projections while the CPU stages
        // sibling or remaining batches. Complete packed projections stay lazy.
        // Capture this scatter's graph separately from later output[rows] updates.
        let submitted = output.reshaped(output.shape)
        if (pipelined || (weightBytes == 0 && !pinnedCaches.isEmpty))
            && weightBytes < byteLimit - pendingWeightBytes
        {
            asyncEval(submitted)
        }
        outputs.append(submitted)
        pendingWeightBytes += weightBytes
        peakPendingWeightBytes = max(peakPendingWeightBytes, pendingWeightBytes)
        if pendingWeightBytes >= byteLimit { flush() }
    }

    func submit() {
        if !outputs.isEmpty { asyncEval(outputs) }
    }

    func flush() {
        defer {
            for cache in pinnedCaches { cache.releasePins() }
            pinnedCaches.removeAll(keepingCapacity: true)
        }
        guard !outputs.isEmpty else { return }
        eval(outputs)
        evaluationCount += 1
        outputs.removeAll(keepingCapacity: true)
        pendingWeightBytes = 0
    }
}

final class StreamingSwitchLinear: SwitchLinear, InferenceErrorProviding {
    let projection: Int
    let schedule: ExpertProjectionSchedule

    init(_ module: SwitchLinear, projection: Int, schedule: ExpertProjectionSchedule) {
        self.projection = projection
        self.schedule = schedule
        super.init(
            inputDims: module.inputDims, outputDims: module.outputDims,
            numExperts: module.numExperts,
            weight: MLXArray.zeros([0]))
        freeze()
    }

    // Streamed tensors belong to the store, not the resident parameter tree. Keeping the
    // original SwitchLinear type preserves typed @ModuleInfo properties in every model.
    override func items() -> ModuleItems { ModuleItems() }

    override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        return schedule.project(
            x, indices: indices, projection: projection, inputSize: inputDims,
            outputSize: outputDims, expertCount: numExperts, sortedIndices: sortedIndices)
    }

    func checkInferenceError() throws { try schedule.store.checkError() }
}

/// Capture once per inference operation, including nested multimodal/auxiliary experts.
struct InferenceErrorCheck {
    private let providers: [any InferenceErrorProviding]
    init(_ model: Module) {
        var stores = Set<ObjectIdentifier>()
        providers = model.modules().compactMap { module in
            if let projection = module as? StreamingSwitchLinear,
                !stores.insert(ObjectIdentifier(projection.schedule.store)).inserted
            {
                return nil
            }
            return module as? any InferenceErrorProviding
        }
    }
    func check() throws { for provider in providers { try provider.checkInferenceError() } }
}

/// Header-only tensor descriptors. All offsets, dimensions and byte counts are checked
/// before any expert data is read; source files remain unchanged.
struct ExpertTensorIndex {
    struct Tensor {
        let url: URL
        let offset: UInt64
        let shape: [Int]
        let dtype: DType
        let byteCount: Int

        func experts(count: Int) throws -> ExpertWeightStore.Tensor {
            guard count > 0, shape.first == count, byteCount.isMultiple(of: count) else {
                throw ExpertWeightStore.invalid("invalid expert dimension")
            }
            return .init(
                url: url, offset: offset, shape: Array(shape.dropFirst()), dtype: dtype,
                bytesPerExpert: byteCount / count)
        }
    }
    let tensors: [String: Tensor]

    /// Resolve every expert from headers before loading weights. A separate projection
    /// may span shards, but every matrix must have the same shape and precision.
    func projection(
        _ path: String, count: Int, separateNames: (Int) -> [String]
    ) throws -> (tensor: ExpertWeightStore.Tensor, names: Set<String>)? {
        guard count > 0 else {
            throw ExpertWeightStore.invalid("invalid expert count for \(path)")
        }
        var sources: [(String, Tensor)] = []
        for expert in 0 ..< count {
            let matches = separateNames(expert).compactMap { name in
                tensors[name].map { (name, $0) }
            }
            guard matches.count <= 1 else {
                throw ExpertWeightStore.invalid(
                    "duplicate expert aliases for \(path), expert \(expert)")
            }
            if let match = matches.first { sources.append(match) }
        }
        if let native = tensors[path] {
            guard sources.isEmpty else {
                throw ExpertWeightStore.invalid(
                    "both stacked and separate expert tensors for \(path)")
            }
            return (try native.experts(count: count), [path])
        }
        guard let first = sources.first?.1 else { return nil }
        guard sources.count == count else {
            throw ExpertWeightStore.invalid("missing separate expert tensors for \(path)")
        }
        guard first.byteCount > 0,
            sources.allSatisfy({ $0.1.shape == first.shape && $0.1.dtype == first.dtype })
        else {
            throw ExpertWeightStore.invalid("inconsistent separate expert tensors for \(path)")
        }
        return (
            .init(
                storage: .separate(sources.map { .init(url: $0.1.url, offset: $0.1.offset) }),
                shape: first.shape, dtype: first.dtype, bytesPerExpert: first.byteCount),
            Set(sources.map(\.0))
        )
    }

    init(urls: [URL]) throws {
        let dtypes: [String: DType] = [
            "F32": .float32, "F16": .float16, "BF16": .bfloat16, "U32": .uint32,
            "U8": .uint8, "U16": .uint16, "U64": .uint64,
            "I8": .int8, "I16": .int16, "I32": .int32, "I64": .int64, "BOOL": .bool,
            "F64": .float64, "F8_E4M3": .uint8,
        ]
        var tensors: [String: Tensor] = [:]
        for url in urls {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            guard let length = try handle.read(upToCount: 8), length.count == 8 else {
                throw ExpertWeightStore.invalid("truncated safetensors header")
            }
            let headerSize = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
                .littleEndian
            guard headerSize > 0, headerSize <= 64 * 1024 * 1024,
                let data = try handle.read(upToCount: Int(headerSize)),
                data.count == Int(headerSize),
                let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw ExpertWeightStore.invalid("invalid safetensors header") }
            let fileSize = try handle.seekToEnd()
            for (name, value) in header where name != "__metadata__" {
                guard tensors[name] == nil else {
                    throw ExpertWeightStore.invalid("duplicate tensor \(name)")
                }
                guard let entry = value as? [String: Any],
                    let shape = entry["shape"] as? [Int], shape.allSatisfy({ $0 >= 0 }),
                    let offsets = entry["data_offsets"] as? [UInt64], offsets.count == 2,
                    offsets[1] >= offsets[0], offsets[1] <= fileSize,
                    headerSize + 8 <= fileSize - offsets[1],
                    let type = entry["dtype"] as? String, let dtype = dtypes[type]
                else { throw ExpertWeightStore.invalid("invalid tensor \(name)") }
                var bytes = dtype.size
                for dimension in shape {
                    let (value, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                    guard !overflow else {
                        throw ExpertWeightStore.invalid("oversized tensor \(name)")
                    }
                    bytes = value
                }
                guard UInt64(bytes) == offsets[1] - offsets[0] else {
                    throw ExpertWeightStore.invalid("incorrect byte count for \(name)")
                }
                tensors[name] = .init(
                    url: url, offset: 8 + headerSize + offsets[0], shape: shape, dtype: dtype,
                    byteCount: bytes)
            }
        }
        self.tensors = tensors
    }
}
