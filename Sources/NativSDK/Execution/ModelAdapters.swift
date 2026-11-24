import Foundation
import MLX
import MLXNN

extension NativModel {
    /// Loads one native MLX LoRA/DoRA or PEFT LoRA adapter. Unload or fuse it before
    /// loading another. Model updates wait for active generation to finish.
    public func loadAdapter(
        from source: ModelSource,
        downloader: HuggingFaceDownloader = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        guard installedAdapter == nil else { throw ModelAdapterError.alreadyLoaded }
        guard context.model is any LoRAModel, let model = context.model as? any LanguageModel else {
            throw ModelAdapterError.unsupportedModel
        }
        let directory: URL
        switch source {
        case .directory(let url): directory = url
        case .repository(let id, let revision):
            directory = try await downloader.snapshot(
                id: id, revision: revision,
                matching: [
                    "adapter_config.json", "adapters.safetensors", "adapter_model.safetensors",
                ],
                progress: progress)
        }
        try Task.checkCancellation()
        let adapter = try AdapterCheckpoint(directory: directory).prepare(for: model)
        try Task.checkCancellation()
        try updateAdapterModules(adapter.adapted, restoring: adapter.originals, model: model)
        installedAdapter = adapter
    }

    /// Restores the exact base modules. Session transcripts remain intact; their
    /// prompt caches are rebuilt on the next turn after any adapter change.
    public func unloadAdapter() async throws {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        guard let adapter = installedAdapter else { throw ModelAdapterError.notLoaded }
        let model = try context.languageModel
        try updateAdapterModules(adapter.originals, restoring: adapter.adapted, model: model)
        installedAdapter = nil
    }

    /// Permanently merges the active adapter into the base model and removes its
    /// wrappers. Quantized base layers are requantized, which can change numerics.
    /// Reload the original checkpoint to undo fusion.
    public func fuseAdapter() async throws {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        guard let adapter = installedAdapter else { throw ModelAdapterError.notLoaded }
        let fused = adapter.adapted.mapValues { module in
            let result = (module as! any LoRALayer).fused()
            result.train(false)
            eval(result)
            return result
        }
        try Task.checkCancellation()
        try updateAdapterModules(fused, restoring: adapter.adapted, model: context.languageModel)
        installedAdapter = nil
    }

    private func updateAdapterModules(
        _ replacements: [String: Module], restoring originals: [String: Module],
        model: any LanguageModel
    ) throws {
        // Even a rolled-back update may rebuild derived state; never reuse earlier KV tensors.
        defer { modelRevision &+= 1 }
        do {
            try replaceByParent(model: model, replacements: replacements)
            model.invalidateCompiledTraces()
            try model.prepare()
            eval(model)
        } catch {
            let installationError = error
            do {
                let current = Dictionary(
                    model.namedModules(), uniquingKeysWith: { first, _ in first })
                let changed = originals.filter { path, original in current[path] !== original }
                try replaceByParent(model: model, replacements: changed)
                model.invalidateCompiledTraces()
                try model.prepare()
                eval(model)
            } catch {
                throw ModelAdapterError.restorationFailed(
                    "\(installationError); restoration: \(error)")
            }
            throw installationError
        }
    }
}
