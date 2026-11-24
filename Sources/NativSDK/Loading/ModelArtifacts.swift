import Foundation
import HuggingFace

/// An explicit artifact source. Local files are used directly; repositories are
/// downloaded through Hugging Face.
public enum ModelSource: Sendable, Equatable {
    case repository(String, revision: String? = "main")
    case directory(URL)

    public var name: String {
        switch self {
        case .repository(let id, _): id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }
}

public struct ResolvedModelAssets: Sendable, Equatable {
    public let source: ModelSource
    public let tokenizerSource: ModelSource?
    public let modelDirectory: URL
    public let tokenizerDirectory: URL

    public let name: String

    public init(
        source: ModelSource, tokenizerSource: ModelSource? = nil,
        modelDirectory: URL, tokenizerDirectory: URL, name: String? = nil
    ) {
        self.name = name ?? source.name
        self.source = source
        self.tokenizerSource = tokenizerSource
        self.modelDirectory = modelDirectory
        self.tokenizerDirectory = tokenizerDirectory
    }
}

/// A snapshot provider returns local files; it never constructs or runs a model.
protocol ModelSnapshotProvider: Sendable {
    func snapshot(
        id: String, revision: String?, matching patterns: [String],
        progress: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL
}

public enum ModelArtifactResolver {
    public static let modelPatterns = ["*.safetensors", "*.json", "*.jinja"]
    public static let tokenizerPatterns = ["*.json", "*.jinja"]

    public static func resolve(
        model: ModelSource, tokenizer: ModelSource? = nil,
        downloader: HuggingFaceDownloader = .init(),

        modelPatterns: [String] = modelPatterns,
        tokenizerPatterns: [String] = tokenizerPatterns,
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ResolvedModelAssets {
        try await resolve(
            model: model, tokenizer: tokenizer, provider: downloader,
            modelPatterns: modelPatterns,
            tokenizerPatterns: tokenizerPatterns, progress: progress)
    }

    static func resolve(
        model: ModelSource, tokenizer: ModelSource? = nil,
        provider: any ModelSnapshotProvider,

        modelPatterns: [String] = modelPatterns,
        tokenizerPatterns: [String] = tokenizerPatterns,
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ResolvedModelAssets {
        let directory = try await resolve(
            model, provider: provider, patterns: modelPatterns,
            progress: progress)
        let tokenizer = tokenizer ?? fallbackTokenizer(in: directory)
        let tokenizerDirectory: URL
        if let tokenizer {
            // Preserve the separate tokenizer download's existing progress policy.
            tokenizerDirectory = try await resolve(
                tokenizer, provider: provider, patterns: tokenizerPatterns,
                progress: { _ in })
        } else {
            tokenizerDirectory = directory
        }
        return ResolvedModelAssets(
            source: model, tokenizerSource: tokenizer, modelDirectory: directory,
            tokenizerDirectory: tokenizerDirectory)
    }

    /// Prefer tokenizer files shipped with a checkpoint. Moondream checkpoints may
    /// omit them or bundle the obsolete GPT-2 vocabulary; use Starmie in those cases.
    static func fallbackTokenizer(in directory: URL) -> ModelSource? {
        guard
            let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
            let configuration = try? JSONDecoder.json5().decode(BaseConfiguration.self, from: data),
            ["moondream1", "moondream2", "moondream3"].contains(configuration.modelType)
        else { return nil }
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            // Early Moondream2 releases bundle GPT-2's 50,257-token vocabulary instead
            // of the Starmie vocabulary used by their trained embedding/output weights.
            guard let data = try? Data(contentsOf: tokenizerURL),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let model = json["model"] as? [String: Any],
                let vocabulary = model["vocab"] as? [String: Int],
                vocabulary.count == 50_257, vocabulary["<|endoftext|>"] == 50_256
            else { return nil }
        }
        return .repository("moondream/starmie-v1")
    }

    private static func resolve(
        _ source: ModelSource, provider: any ModelSnapshotProvider, patterns: [String],
        progress: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()
        switch source {
        case .directory(let directory): return directory
        case .repository(let id, let revision):
            let directory = try await provider.snapshot(
                id: id, revision: revision, matching: patterns,
                progress: progress)
            try Task.checkCancellation()
            return directory
        }
    }
}

/// Downloads model snapshots using the Hugging Face Hub cache.
/// Named revisions are checked remotely; cached file content is reused by hash.
public struct HuggingFaceDownloader: ModelSnapshotProvider {
    public let client: HubClient

    public init(client: HubClient = HubClient()) {
        self.client = client
    }

    public enum DownloadError: LocalizedError {
        case invalidRepositoryID(String)

        public var errorDescription: String? {
            switch self {
            case .invalidRepositoryID(let id): "Invalid Hugging Face repository ID: \(id)"
            }
        }
    }

    public func snapshot(
        id: String, revision: String?, matching patterns: [String],
        progress: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let components = id.split(separator: "/", omittingEmptySubsequences: false)
        guard (1 ... 2).contains(components.count), components.allSatisfy({ !$0.isEmpty }),
            let repo = Repo.ID(rawValue: id)
        else {
            throw DownloadError.invalidRepositoryID(id)
        }
        // HubClient checks repository metadata and reuses cached content by hash.
        return try await client.downloadSnapshot(
            of: repo, revision: revision ?? "main", matching: patterns,
            progressHandler: { value in progress(value) })
    }
}
