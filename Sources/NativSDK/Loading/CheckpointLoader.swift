import Foundation
import MLX
import MLXNN
import OSLog

struct InferencePreparationReport: Sendable {
    let modelType: String
    let optimizationError: (any Error)?
}

/// Checkpoint loading has no dependency on a decoding engine or generation state.
enum CheckpointLoader {
    private static let logger = Logger(subsystem: "NativSDK", category: "model-loading")

    @discardableResult
    static func load(
        modelDirectory: URL, model: any BaseLanguageModel,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
        auxiliaryWeightDirectories: [String] = [],
        expertStreaming: ExpertStreamingOptions? = nil
    ) throws -> InferencePreparationReport {
        let mistralLarge4 =
            (model as? MistralLarge4)?.config ?? (model as? MistralLarge4TextModel)?.config
        let transformation: CheckpointTransformation =
            if mistralLarge4 == nil {
                try CheckpointTransformation(modelDirectory: modelDirectory)
            } else { .none }
        var perLayerQuantization =
            if perLayerQuantization == nil, quantization == nil, model is Inkling2 {
                try Inkling2Checkpoint.quantization(from: modelDirectory)
            } else {
                perLayerQuantization
            }
        let urls = try safetensorWeightURLs(
            in: modelDirectory,
            additionalFiles: (model as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
                ?? [])
        var excluded = Set<String>()
        if let options = expertStreaming {
            if let streamingModel = model as? any ExpertStreamingModel {
                excluded = try streamingModel.configureExpertStreaming(
                    urls: urls, options: options, quantization: quantization,
                    perLayerQuantization: perLayerQuantization)
            } else {
                excluded = try StreamedExpertProjection.install(
                    model: model, urls: urls, options: options, quantization: quantization,
                    perLayerQuantization: perLayerQuantization, transformation: transformation)
            }
        }
        if mistralLarge4 != nil {
            for url in urls {
                let names = try safetensorSpansInFileOrder(url: url).map(\.name)
                excluded.formUnion(names.filter(MistralLarge4Checkpoint.isDraftWeight))
            }
        }
        let visionPrefixes: [String]
        let canonicalVisionKey: (String) -> String?
        if model is MiniCPMV {
            visionPrefixes = ["vision_tower.", "vit_merger.", "merger."]
            canonicalVisionKey = MiniCPMV.canonicalWeightKey
        } else if model is Gemma4 {
            visionPrefixes = ["vision_tower.", "embed_vision."]
            canonicalVisionKey = Gemma4TextLanguageModel.canonicalWeightKey
        } else {
            visionPrefixes = []
            canonicalVisionKey = { $0 }
        }
        let deferVision =
            !visionPrefixes.isEmpty && expertStreaming?.deferVisionWeights == true
        var deferred =
            deferVision
            ? Set(
                try ExpertTensorIndex(urls: urls).tensors.keys.filter { name in
                    guard let canonical = canonicalVisionKey(name) else { return false }
                    return visionPrefixes.contains(where: canonical.hasPrefix)
                }) : Set<String>()
        let streamPerception = model is NemotronTranscribe && expertStreaming != nil
        if let transcribe = model as? NemotronTranscribe, streamPerception {
            deferred.formUnion(try transcribe.configurePerceptionStreaming(urls: urls))
        }
        var loaded = try loadWeightArrays(urls: urls, excluding: excluded, deferred: deferred)
        for directory in auxiliaryWeightDirectories {
            let auxiliary = try loadWeightArrays(
                urls: topLevelSafetensorURLs(in: modelDirectory.appendingPathComponent(directory)))
            for (key, value) in auxiliary.weights {
                let name = directory + "." + key
                guard loaded.weights[name] == nil else {
                    throw ModelExecutionError.invalidOptions("Duplicate embedding weight: \(name)")
                }
                loaded.weights[name] = value
            }
        }
        loaded.weights = try transformation.prepare(loaded.weights)
        if let ernie = model as? Ernie55Model, perLayerQuantization == nil, quantization == nil,
            try Ernie55Checkpoint.prepareSource(&loaded.weights, config: ernie.config)
        {
            perLayerQuantization = .init(
                quantization: Ernie55Checkpoint.sourceQuantization, perLayerQuantization: [:])
        }
        let weights: [String: MLXArray]
        if let mistralLarge4 {
            weights = try MistralLarge4Checkpoint.normalize(
                loaded.weights, config: mistralLarge4, vision: model as? MistralLarge4)
        } else {
            weights = model.sanitize(weights: loaded.weights, metadata: loaded.metadata)
        }
        // Once sanitized weights own their dependencies, release the source handles.
        // Otherwise stacking separate experts keeps both copies alive during eval.
        loaded.weights.removeAll()
        try transformation.install(in: model, weights: weights)
        if quantization != nil || perLayerQuantization != nil {
            try quantizeByParent(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                }
                return quantization?.asTuple
            }
        }
        // A module can have numeric child names (for example Omni's audio decoder).
        // Reuse its tree when keys match; unflattening would turn those modules into arrays.
        // Missing/extra keys still take the ordinary strict validation path.
        let template = model.parameters()
        let parameters =
            Set(template.flattened().map(\.0)) == Set(weights.keys)
            ? template.mapValues { key, _ in weights[key]! }
            : ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])

        let modelType = String(reflecting: type(of: model))
        var optimizationError: (any Error)?
        if let languageModel = model as? any LanguageModel {
            do {
                try languageModel.prepare()
            } catch {
                optimizationError = error
                logger.error(
                    "Failed to prepare inference state for \(modelType): \(String(describing: error))"
                )
            }
        }
        if deferVision || streamPerception {
            eval(
                model.parameters().flattened().filter { name, _ in
                    !visionPrefixes.contains(where: name.hasPrefix)
                        && !(streamPerception && name.hasPrefix("perception."))
                }.map(\.1))
        } else {
            eval(model)
        }
        return InferencePreparationReport(
            modelType: modelType, optimizationError: optimizationError)
    }

    /// Loading runs on a dispatch queue because file I/O and concurrent evaluations
    /// block threads. The caller must retain exclusive model access until this returns.
    @discardableResult
    static func load(
        modelDirectory: URL, model: any BaseLanguageModel,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
        auxiliaryWeightDirectories: [String] = [],
        expertStreaming: ExpertStreamingOptions? = nil
    ) async throws -> InferencePreparationReport {
        let transfer = LoadingTransfer(model)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(
                    with: Result {
                        try load(
                            modelDirectory: modelDirectory, model: transfer.take(),
                            quantization: quantization, perLayerQuantization: perLayerQuantization,
                            auxiliaryWeightDirectories: auxiliaryWeightDirectories,
                            expertStreaming: expertStreaming)
                    })
            }
        }
    }
}

/// A single handoff to the loading queue, never shared with inference work.
/// Only that queue reads or clears the value; its completion resumes the caller.
private final class LoadingTransfer<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: consuming T) { self.value = consume value }

    func take() -> T {
        guard let value else { preconditionFailure("Loading value already transferred") }
        self.value = nil
        return value
    }
}

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// The safetensors files in `modelDirectory` that hold the model's weights.
///
/// Only the top level of the directory is considered. Checkpoints keep auxiliary weights that
/// belong to a different module in subdirectories (for example `mlx-community/Qwen3.5-4B-OptiQ-4bit`
/// and its `optiq/mtp.safetensors`), and a nested Hugging Face snapshot cache under a local
/// checkpoint directory would otherwise be pulled in as well.
///
/// Files are chosen in this order:
///
/// 1. The files named by `model.safetensors.index.json`, when it exists and every file it names
///    exists. The index is precise about which of several weight files belong to the model, which
///    matters for a repo that ships both a consolidated file and shards.
/// 2. The conventional `model*.safetensors` names, matching `mlx_lm.utils.load_model`. Uploads
///    regularly ship an index carried over from an unquantized source repo that names shards the
///    repo does not contain, and the convention is what those repos actually follow.
/// 3. `weight*.safetensors`, then every safetensors file present, so a directory that follows no
///    convention at all still loads.
///
/// `additionalFiles` names files the model requires that no rule above selects, for example the
/// Jina reranker's `projector.safetensors`. They are appended, so a file the index already names
/// is not loaded twice, and names that are not present are ignored.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     selected ones. See ``AdditionalWeightFilesProviding/additionalWeightFiles``.
func safetensorWeightURLs(
    in modelDirectory: URL,
    additionalFiles: [String] = []
) throws -> [URL] {
    let present = topLevelSafetensorURLs(in: modelDirectory)

    let selected = try indexedWeightURLs(in: modelDirectory) ?? conventionalWeightURLs(in: present)

    var seen = Set(selected.map(\.standardizedFileURL.path))
    var urls = selected
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }
    return urls
}

/// The files named by `model.safetensors.index.json`, or `nil` when there is no index or it
/// names a file the directory does not contain.
///
/// Existence is checked against the file system rather than the top-level listing: an index may
/// legitimately map weights into a subdirectory, and that is a deliberate statement about where
/// this model's weights live rather than an unrelated file that happens to be nearby.
private func indexedWeightURLs(in modelDirectory: URL) throws -> [URL]? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return nil
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let urls = Set(index.weightMap.values)
        .sorted()
        .map { modelDirectory.appendingPathComponent($0) }

    guard !urls.isEmpty,
        urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return nil
    }
    return urls
}

/// The conventionally named weight files among `present`, matching `mlx_lm.utils.load_model`'s
/// `model*.safetensors` glob, with `weight*.safetensors` and then everything as fallbacks.
private func conventionalWeightURLs(in present: [URL]) -> [URL] {
    for prefix in ["model", "weight"] {
        let matches = present.filter { $0.lastPathComponent.hasPrefix(prefix) }
        if !matches.isEmpty {
            return matches
        }
    }
    return present
}

private func topLevelSafetensorURLs(in modelDirectory: URL) -> [URL] {
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    return
        contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// Safetensor arrays load lazily. Evaluating contiguous byte ranges concurrently
// overlaps file I/O and allocation while preserving deterministic file precedence.

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// The tensors of the safetensors file at `url`, ordered by their position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorSpansInFileOrder(url: URL) throws -> [SafetensorSpan] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    // a header bigger than this is not a header
    guard headerLength > 0, headerLength <= 512 * 1024 * 1024 else { throw Malformed() }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, byteCount: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = (offsets[0] as? NSNumber)?.int64Value,
            let end = (offsets[1] as? NSNumber)?.int64Value,
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end - begin))
    }
    spans.sort { $0.begin < $1.begin }
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.byteCount) }
}

/// Contiguous index ranges of `byteCounts` whose byte totals are balanced around
/// `total / groupCount`, preserving order.
func contiguousLoadGroups(byteCounts: [Int64], groupCount: Int) -> [Range<Int>] {
    guard !byteCounts.isEmpty else { return [] }
    let total = byteCounts.reduce(0, +)
    guard groupCount > 1, total > 0 else { return [0 ..< byteCounts.count] }

    let groups = Int64(groupCount)
    var ranges = [Range<Int>]()
    var start = 0
    var cumulative: Int64 = 0
    var boundary: Int64 = 1
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= total * boundary / groups {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// How many concurrent evaluations to spread a model's weight loading across.
///
/// Throughput rises with concurrent readers until the disk (cold) or the memory system (warm)
/// saturates -- around 8-16 in-flight readers on Apple silicon. More workers than cores only
/// adds contention.
func weightLoadConcurrency(processorCount: Int = ProcessInfo.processInfo.activeProcessorCount)
    -> Int
{
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Lock-guarded shared state for the concurrent load.
private final class ConcurrentLoadState: @unchecked Sendable {
    private let lock = NSLock()
    private var perFile: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var firstError: Error?

    init(fileCount: Int) {
        perFile = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
    }

    func merge(file: Int, weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        perFile[file].merge(weights) { _, new in new }
        perFileMetadata[file] = metadata
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    /// Weights merged in file order (a later file overwrites a duplicate name, matching the
    /// serial loader) and the first nonempty file metadata.
    func result() throws -> (weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        var weights = [String: MLXArray]()
        for fileWeights in perFile {
            weights.merge(fileWeights) { _, new in new }
        }
        let metadata = perFileMetadata.first { !$0.isEmpty } ?? [:]
        return (weights, metadata)
    }
}

/// Load and materialize the weights of every file in `urls`, evaluating contiguous byte
/// ranges of each file concurrently.
///
/// Each work item lazily opens its file, evaluates only its assigned tensors (forcing that
/// range's I/O inside the work item), and the results are merged in file order. A file whose
/// header cannot be parsed is loaded whole by one work item, which is exactly the serial
/// loader's behavior for that file.
func loadWeightArrays(urls: [URL], excluding excluded: Set<String> = [], deferred: Set<String> = [])
    throws -> (
        weights: [String: MLXArray], metadata: [String: String]
    )
{
    struct WorkItem {
        let file: Int
        let url: URL
        /// tensors this item evaluates; nil evaluates the whole file
        let names: [String]?
    }

    let items: [WorkItem] = {
        var spansPerFile = [[SafetensorSpan]?]()
        var totalBytes: Int64 = 0
        for url in urls {
            let spans = try? safetensorSpansInFileOrder(url: url).filter {
                !excluded.contains($0.name)
            }
            spansPerFile.append(spans)
            totalBytes += spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        }

        let concurrency = weightLoadConcurrency()
        let groupBytes = max(minimumBytesPerLoadGroup, totalBytes / Int64(concurrency))
        var items = [WorkItem]()
        for (file, url) in urls.enumerated() {
            if let spans = spansPerFile[file], !spans.isEmpty {
                let bytes = spans.reduce(0) { $0 + $1.byteCount }
                let groupCount = max(1, Int(bytes / groupBytes))
                for range in contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount)
                {
                    items.append(
                        WorkItem(file: file, url: url, names: spans[range].map(\.name)))
                }
            } else {
                items.append(WorkItem(file: file, url: url, names: nil))
            }
        }
        return items
    }()

    let state = ConcurrentLoadState(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: items.count) { index in
        let item = items[index]
        do {
            let (all, metadata) = try loadArraysAndMetadata(url: item.url, stream: .cpu)

            var selected = [String: MLXArray]()
            if let names = item.names {
                for name in names {
                    if let array = all[name] { selected[name] = array }
                }
            } else {
                selected = all.filter { !excluded.contains($0.key) }
            }

            // force this range's I/O here, on this stream, in file-offset order
            if !selected.isEmpty {
                eval(selected.filter { !deferred.contains($0.key) }.map(\.value))
            }
            state.merge(file: item.file, weights: selected, metadata: metadata)
        } catch {
            state.record(error: error)
        }
    }
    return try state.result()
}
