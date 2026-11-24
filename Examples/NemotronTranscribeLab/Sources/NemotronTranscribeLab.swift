import AVFoundation
import Foundation
import MLX
import NativSDK
import Observation
import SwiftUI
import UniformTypeIdentifiers

private enum AudioTask: String, CaseIterable, Identifiable {
    case transcription, translation, speakers, question
    var id: String { rawValue }
    var title: String {
        switch self {
        case .transcription: "Transcription"
        case .translation: "Translation"
        case .speakers: "Speaker attribution"
        case .question: "Audio Q&A"
        }
    }
    var tag: String {
        switch self {
        case .transcription: "asr"
        case .translation: "translate"
        case .speakers: "saa"
        case .question: "qa"
        }
    }
    var symbol: String {
        switch self {
        case .transcription: "text.alignleft"
        case .translation: "character.bubble"
        case .speakers: "person.2.wave.2"
        case .question: "questionmark.bubble"
        }
    }
}

private enum AudioCheckpoint: String, CaseIterable, Identifiable {
    case fourBit = "NemotronTranscribe4bit"
    case threeBitExperts = "NemotronTranscribe3bitExperts"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .fourBit: "4-bit language weights"
        case .threeBitExperts: "Mixed 3/4-bit experts"
        }
    }
}

private struct RunResult: Identifiable, Codable {
    var id = UUID()
    let task: String
    let audio: String
    let text: String
    let promptTokens: Int
    let generatedTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let totalSeconds: Double
    let firstChunkSeconds: Double?
    let streamChunks: Int
    let peakProcessBytes: UInt64
    let thermalStart: Int
    let thermalEnd: Int
    let stopReason: String
    var checkpoint: String?
    var lifetimePeakProcessBytes: UInt64?
    var expertCacheMiB: Int?
    var decodeRate: Double { decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0 }
}

@MainActor @Observable private final class AudioLab {
    var task = AudioTask.transcription
    var language = "German"
    var question = "What time is the design review, where is it, and what should people bring?"
    var selectedAudio: URL?
    var status = "Checking model files…"
    var error: String?
    var output = ""
    var busy = false
    var recording = false
    var ready = false
    var elapsed = 0.0
    var currentMemory: UInt64 = 0
    var peakMemory: UInt64 = 0
    var lifetimePeakMemory: UInt64 = 0
    var results: [RunResult] = []
    var showImporter = false
    @ObservationIgnored private var model: NativModel?
    @ObservationIgnored private var request: ModelGeneration?
    @ObservationIgnored private var recorder: AVAudioRecorder?
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var started = false
    @ObservationIgnored private var suiteRunning = false
    @ObservationIgnored private var stopping = false
    @ObservationIgnored private var begin: Date?
    @ObservationIgnored private var lastControl = ""
    let documents = URL.documentsDirectory
    let cacheMiB = 1024
    var checkpoint =
        AudioCheckpoint(
            rawValue: UserDefaults.standard.string(forKey: "checkpoint") ?? "") ?? .fourBit
    var modelURL: URL {
        documents.appendingPathComponent(checkpoint.rawValue, isDirectory: true)
    }
    var installed: Bool {
        FileManager.default.fileExists(
            atPath: modelURL.appendingPathComponent("model.safetensors.index.json").path)
    }
    var canRun: Bool { ready && !busy && !recording && selectedAudio != nil }

    func sample(_ name: String) {
        selectedAudio = Bundle.main.url(forResource: name, withExtension: "wav")
    }

    func start() async {
        guard !started else { return }
        started = true
        Memory.cacheLimit = 16 * 1024 * 1024
        sample("announcement")
        if let data = try? Data(contentsOf: documents.appendingPathComponent("results.json")),
            let saved = try? JSONDecoder().decode([RunResult].self, from: data)
        {
            results = saved
        }
        if let data = try? Data(contentsOf: documents.appendingPathComponent("control.json")),
            let command = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        {
            lastControl = command["id"] ?? ""
        }
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.sampleMemory()
                if let begin = self.begin { self.elapsed = Date().timeIntervalSince(begin) }
                self.saveStatus()
                if !self.busy { await self.readControl() }
            }
        }
        if CommandLine.arguments.contains("--transfer") {
            UIApplication.shared.isIdleTimerDisabled = true
            status = "Ready for model transfer"
        } else if installed {
            await load()
            if CommandLine.arguments.contains("--validate") { await runAll() }
        } else {
            status = "Copy the model into Documents/\(checkpoint.rawValue), then load it."
        }
        saveStatus()
    }

    func load() async {
        guard !busy, model == nil else { return }
        guard installed else {
            error = "Model files are not installed."
            return
        }
        busy = true
        error = nil
        status = "Loading speech model…"
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            busy = false
            saveStatus()
        }
        do {
            model = try await NativModel.loadModel(
                from: .directory(modelURL),
                expertStreaming: .init(
                    cacheSizeBytes: cacheMiB * 1024 * 1024, sharedExpertBuffers: true))
            ready = true
            status = "Ready · all processing stays on this device"
        } catch {
            self.error = error.localizedDescription
            status = "Couldn’t load model"
        }
    }

    func unload() {
        guard !busy else { return }
        model = nil
        ready = false
        Memory.clearCache()
        status = "Model unloaded"
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func selectCheckpoint(_ selected: AudioCheckpoint) {
        guard !busy, !recording, !suiteRunning, selected != checkpoint else { return }
        unload()
        checkpoint = selected
        UserDefaults.standard.set(selected.rawValue, forKey: "checkpoint")
        error = nil
        output = ""
        status = installed ? "Model selected · tap Load model" : "Selected model is not installed"
        saveStatus()
    }

    func run() async {
        guard canRun, let model, let selectedAudio else { return }
        busy = true
        error = nil
        output = ""
        stopping = false
        player?.stop()
        status = "Encoding audio…"
        begin = Date()
        elapsed = 0
        sampleMemory()
        peakMemory = currentMemory
        let startedAt = begin!
        let thermalStart = ProcessInfo.processInfo.thermalState.rawValue
        UIApplication.shared.isIdleTimerDisabled = true
        var chunks = 0
        var firstChunk: Double?
        let selectedTask = task
        defer {
            request = nil
            begin = nil
            busy = false
            UIApplication.shared.isIdleTimerDisabled = suiteRunning
            sampleMemory()
            saveStatus()
        }
        do {
            let prompt = selectedTask == .question ? question : ""
            if selectedTask == .translation
                && language.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                throw NSError(
                    domain: "AudioLab", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Enter a target language."])
            }
            let generation = model.generate(
                input: UserInput(
                    prompt: prompt, audios: [.url(selectedAudio)],
                    additionalContext: ["task": selectedTask.tag, "language": language]),
                options: .init(
                    prefill: .init(stepSize: 256), sampling: .init(temperature: 0),
                    output: .init(maxTokens: 512)))
            request = generation
            for try await event in generation {
                switch event {
                case .chunk(let chunk):
                    if !chunk.isEmpty {
                        output += chunk
                        chunks += 1
                        if firstChunk == nil { firstChunk = Date().timeIntervalSince(startedAt) }
                        status = "Generating…"
                    }
                case .protocolError(let detail):
                    throw NSError(
                        domain: "AudioLab", code: 2, userInfo: [NSLocalizedDescriptionKey: detail])
                default: break
                }
            }
            let metrics = try await generation.metrics()
            sampleMemory()
            elapsed = Date().timeIntervalSince(startedAt)
            results.append(
                RunResult(
                    task: selectedTask.rawValue, audio: selectedAudio.lastPathComponent,
                    text: output,
                    promptTokens: metrics.promptTokenCount,
                    generatedTokens: metrics.generatedTokenCount,
                    prefillSeconds: metrics.promptSeconds, decodeSeconds: metrics.generationSeconds,
                    totalSeconds: elapsed, firstChunkSeconds: firstChunk, streamChunks: chunks,
                    peakProcessBytes: peakMemory, thermalStart: thermalStart,
                    thermalEnd: ProcessInfo.processInfo.thermalState.rawValue,
                    stopReason: String(describing: metrics.stopReason),
                    checkpoint: checkpoint.rawValue, lifetimePeakProcessBytes: lifetimePeakMemory,
                    expertCacheMiB: cacheMiB))
            status = stopping ? "Stopped" : "Finished"
            if let data = try? JSONEncoder().encode(results) {
                try? data.write(
                    to: documents.appendingPathComponent("results.json"), options: .atomic)
            }
        } catch {
            request?.cancel()
            _ = try? await request?.metrics()
            self.error = stopping ? nil : error.localizedDescription
            status = stopping ? "Stopped" : "Couldn’t finish request"
        }
    }

    func runAll() async {
        guard ready, !busy, !suiteRunning else { return }
        suiteRunning = true
        stopping = false
        defer {
            suiteRunning = false
            UIApplication.shared.isIdleTimerDisabled = false
            saveStatus()
        }
        for mode in AudioTask.allCases {
            if stopping { break }
            task = mode
            sample(mode == .speakers ? "conversation" : "announcement")
            language = "German"
            question = "What time is the design review, where is it, and what should people bring?"
            await run()
            if error != nil { break }
        }
    }

    func stop() {
        stopping = true
        request?.cancel()
        status = "Stopping…"
    }

    func play() {
        guard let selectedAudio else { return }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: selectedAudio)
            player?.play()
        } catch { self.error = error.localizedDescription }
    }

    func toggleRecording() async {
        guard !busy else { return }
        if recording {
            recorder?.stop()
            recording = false
            selectedAudio = recorder?.url
            recorder = nil
            status = "Recording ready"
            UIApplication.shared.isIdleTimerDisabled = false
            return
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            error =
                "Microphone access is disabled. Enable it in Settings to record, or import an audio file."
            return
        }
        do {
            player?.stop()
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
            let url = documents.appendingPathComponent("Recording-\(UUID().uuidString).wav")
            recorder = try AVAudioRecorder(
                url: url,
                settings: [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                    AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                ])
            guard recorder!.record() else { throw CocoaError(.fileWriteUnknown) }
            recording = true
            status = "Recording… tap Stop recording when finished"
            UIApplication.shared.isIdleTimerDisabled = true
        } catch { self.error = error.localizedDescription }
    }

    func importAudio(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let target = documents.appendingPathComponent(
                UUID().uuidString + "-" + url.lastPathComponent)
            try FileManager.default.copyItem(at: url, to: target)
            selectedAudio = target
            status = "Audio imported"
        } catch { self.error = error.localizedDescription }
    }

    private func sampleMemory() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            currentMemory = info.phys_footprint
            lifetimePeakMemory = UInt64(max(0, info.ledger_phys_footprint_peak))
        }
        peakMemory = max(peakMemory, currentMemory)
    }

    private func saveStatus() {
        var data: [String: Any] = [
            "status": status, "ready": ready, "busy": busy, "suite_running": suiteRunning,
            "model_installed": installed, "response": output, "task": task.rawValue,
            "elapsed_seconds": elapsed,
            "current_process_bytes": currentMemory, "peak_process_bytes": peakMemory,
            "lifetime_peak_process_bytes": lifetimePeakMemory,
            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "mlx_active_bytes": Memory.activeMemory, "mlx_cache_bytes": Memory.cacheMemory,
            "completed_runs": results.count, "expert_cache_mib": cacheMiB,
            "checkpoint": checkpoint.rawValue,
            "date": Date().ISO8601Format(),
        ]
        if let error { data["error"] = error }
        if let encoded = try? JSONSerialization.data(
            withJSONObject: data, options: [.prettyPrinted, .sortedKeys])
        {
            try? encoded.write(
                to: documents.appendingPathComponent("status.json"), options: .atomic)
        }
    }

    // Optional local diagnostic control, using the exact same paths as the buttons.
    private func readControl() async {
        guard let data = try? Data(contentsOf: documents.appendingPathComponent("control.json")),
            let command = try? JSONSerialization.jsonObject(with: data) as? [String: String],
            let id = command["id"], id != lastControl
        else { return }
        lastControl = id
        if let name = command["checkpoint"], let selected = AudioCheckpoint(rawValue: name) {
            selectCheckpoint(selected)
        }
        switch command["action"] {
        case "load": await load()
        case "validate":
            if !ready { await load() }
            await runAll()
        case "benchmark":
            if !ready { await load() }
            task = AudioTask(rawValue: command["task"] ?? "transcription") ?? .transcription
            if let audio = command["audio"] {
                selectedAudio = documents.appendingPathComponent(audio)
            } else {
                sample("announcement")
            }
            if let prompt = command["question"] { question = prompt }
            await run()
        case "unload": unload()
        default: break
        }
    }
}

@main struct NemotronTranscribeLabApp: App {
    var body: some Scene { WindowGroup { AudioLabView() } }
}

private struct AudioLabView: View {
    @State private var lab = AudioLab()
    var body: some View {
        NavigationStack {
            List {
                Section {
                    Label("Nemotron 3.5 Transcribe", systemImage: "waveform")
                        .font(.headline)
                    Text("30B A4B · Expert streaming · speech encoders in source precision")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker(
                        "Weights",
                        selection: Binding(
                            get: { lab.checkpoint }, set: { lab.selectCheckpoint($0) })
                    ) {
                        ForEach(AudioCheckpoint.allCases) { checkpoint in
                            Text(checkpoint.title).tag(checkpoint)
                        }
                    }.disabled(lab.busy || lab.recording)
                    Text(lab.status).font(.subheadline)
                    if let error = lab.error { Text(error).foregroundStyle(.red) }
                    if !lab.ready {
                        Button("Load model") { Task { await lab.load() } }.disabled(lab.busy)
                    }
                }
                Section("Task") {
                    Picker("Mode", selection: $lab.task) {
                        ForEach(AudioTask.allCases) { task in
                            Label(task.title, systemImage: task.symbol).tag(task)
                        }
                    }.disabled(lab.busy || lab.recording)
                    if lab.task == .translation {
                        TextField("Target language", text: $lab.language).disabled(lab.busy)
                    }
                    if lab.task == .question {
                        TextField(
                            "Question about the audio (optional)", text: $lab.question,
                            axis: .vertical
                        )
                        .lineLimit(2 ... 4).disabled(lab.busy)
                    }
                    if lab.task == .speakers {
                        Text(
                            "Speaker numbers identify voices within this recording; they are not real-world identities or timestamps."
                        )
                        .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("Audio") {
                    Text(lab.selectedAudio?.lastPathComponent ?? "Choose a recording").lineLimit(2)
                    Menu("Bundled samples") {
                        Button("Design review announcement") { lab.sample("announcement") }
                        Button("Two-speaker conversation") { lab.sample("conversation") }
                        Button("Spoken question") {
                            lab.sample("spoken_request")
                            lab.task = .question
                            lab.question = ""
                        }
                    }.disabled(lab.busy || lab.recording)
                    HStack {
                        Button("Import", systemImage: "square.and.arrow.down") {
                            lab.showImporter = true
                        }
                        Spacer()
                        Button("Play", systemImage: "play.fill") { lab.play() }
                    }.disabled(lab.busy || lab.recording)
                    Button(
                        lab.recording ? "Stop recording" : "Record microphone",
                        systemImage: lab.recording ? "stop.circle.fill" : "mic.fill"
                    ) {
                        Task { await lab.toggleRecording() }
                    }.disabled(lab.busy)
                    Text("Bundled audio is synthetic speech generated with Phonon.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    if lab.busy {
                        HStack {
                            ProgressView()
                            Text(String(format: "%.1f seconds", lab.elapsed))
                            Spacer()
                            Button("Stop") { lab.stop() }
                        }
                    } else {
                        Button("Run \(lab.task.title)", systemImage: "play.circle.fill") {
                            Task { await lab.run() }
                        }.disabled(!lab.canRun)
                        Button("Test all four tasks") { Task { await lab.runAll() } }.disabled(
                            !lab.canRun)
                    }
                }
                if !lab.output.isEmpty {
                    Section("Result") {
                        Text(lab.output).textSelection(.enabled).font(.body.monospaced())
                    }
                }
                if let result = lab.results.last {
                    Section("Last run") {
                        LabeledContent(
                            "Audio + prompt processing",
                            value: String(format: "%.2f s", result.prefillSeconds))
                        LabeledContent(
                            "Decode", value: String(format: "%.2f tok/s", result.decodeRate))
                        LabeledContent("Generated tokens", value: "\(result.generatedTokens)")
                        LabeledContent(
                            "Total time", value: String(format: "%.2f s", result.totalSeconds))
                        LabeledContent(
                            "Peak process memory",
                            value: String(format: "%.2f GB", Double(result.peakProcessBytes) / 1e9))
                    }
                }
                if !lab.results.isEmpty {
                    Section("Completed runs") {
                        ForEach(lab.results.reversed()) { result in
                            DisclosureGroup(
                                "\(result.task.capitalized) · \(String(format: "%.1f s", result.totalSeconds))"
                            ) {
                                Text(result.text).textSelection(.enabled)
                                Text(
                                    String(
                                        format: "%.2f tok/s · %.2f GB sampled peak",
                                        result.decodeRate,
                                        Double(result.peakProcessBytes) / 1e9)
                                )
                                .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Section {
                    LabeledContent(
                        "Current process memory",
                        value: String(format: "%.2f GB", Double(lab.currentMemory) / 1e9))
                    LabeledContent(
                        "Peak since app launch",
                        value: String(format: "%.2f GB", Double(lab.lifetimePeakMemory) / 1e9))
                    Button("Unload model") { lab.unload() }.disabled(lab.busy || !lab.ready)
                }
            }
            .navigationTitle("Nemotron Audio")
            .fileImporter(
                isPresented: $lab.showImporter, allowedContentTypes: [.audio],
                allowsMultipleSelection: false, onCompletion: lab.importAudio
            )
            .task { await lab.start() }
        }
    }
}
