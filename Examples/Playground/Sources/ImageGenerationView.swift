import SwiftUI
import UniformTypeIdentifiers

struct ImageGenerationView: View {
    @Bindable var model: ImageGenerationModel
    @Binding var exportsImage: Bool
    @Binding var showsSettings: Bool

    var body: some View {
        ImageCanvasView(model: model)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 0) {
                    if let notice = model.notice {
                        WorkspaceNotice(message: notice, dismiss: { model.notice = nil })
                            .frame(maxWidth: 760)
                            .padding(.horizontal, 24)
                            .padding(.bottom, 8)
                    }
                    ImageComposerView(model: model, showsOptions: $showsSettings)
                }
                .frame(maxWidth: .infinity)
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Save PNG…", systemImage: "square.and.arrow.down") {
                        exportsImage = true
                    }
                    .disabled(model.output == nil || model.isBusy)
                    .help("Save image (⌘S)")
                }
            }
            .fileExporter(
                isPresented: $exportsImage,
                document: model.output.map { PNGDocument(data: $0.png) },
                contentType: .png, defaultFilename: "qwen-image"
            ) { result in
                if case .failure(let error) = result {
                    model.notice = "Couldn’t save the image. \(error.localizedDescription)"
                }
            }
            .onDisappear { model.stop() }
    }
}

private struct ImageCanvasView: View {
    var model: ImageGenerationModel

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 24) {
                    if let output = model.output {
                        Image(decorative: output.image, scale: 1)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: 640, maxHeight: max(160, geometry.size.height - 160))
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .accessibilityLabel("Generated image: \(output.prompt)")
                            .accessibilityIdentifier("GeneratedImage")
                        VStack(spacing: 8) {
                            Text(output.prompt)
                                .font(.body)
                                .textSelection(.enabled)
                            Text(
                                "\(output.size) × \(output.size) · \(output.steps) steps · Seed \(output.seed)"
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        }
                        .frame(maxWidth: 560)
                    } else {
                        VStack(spacing: 14) {
                            Image(systemName: "photo.badge.plus")
                                .font(.system(size: 36, weight: .regular))
                                .foregroundStyle(.tint)
                                .frame(width: 76, height: 76)
                                .background(
                                    .tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 22)
                                )
                                .accessibilityHidden(true)
                            Text("An idea becomes an image")
                                .font(.largeTitle.weight(.semibold))
                            Text(
                                "Describe a subject, a setting, and a style.\nQwen-Image-2.1 brings them together on your device."
                            )
                            .font(.body)
                            .foregroundStyle(.secondary)
                        }
                        Text(
                            "The model downloads on your first generation and is saved for next time."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: 360)
                    }
                    if model.isBusy { ImageGenerationStatusView(model: model) }
                }
                .multilineTextAlignment(.center)
                .padding(32)
                .frame(maxWidth: .infinity)
                .frame(minHeight: geometry.size.height)
            }
        }
    }
}

private struct ImageGenerationStatusView: View {
    var model: ImageGenerationModel

    var body: some View {
        VStack(spacing: 8) {
            if model.phase == .loading, model.downloadProgress > 0, model.downloadProgress < 1 {
                ProgressView(value: model.downloadProgress) {
                    Text(
                        "Downloading… \(model.downloadProgress, format: .percent.precision(.fractionLength(0)))"
                    )
                }
                .frame(maxWidth: 280)
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(
                        model.phase == .stopping
                            ? "Stopping…"
                            : model.phase == .loading ? "Loading model…" : "Creating your image…")
                }
            }
            if let start = model.startedAt {
                Text(start, style: .timer).monospacedDigit()
            }
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
    }
}

private struct ImageComposerView: View {
    @Bindable var model: ImageGenerationModel
    @Binding var showsOptions: Bool
    @FocusState private var focused: Bool

    var body: some View {
        ComposerSurface {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .bottom, spacing: 16) {
                    TextField("Describe an image…", text: $model.prompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.body)
                        .lineLimit(2 ... 5)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 6)
                        .focused($focused)
                        .disabled(model.isBusy)
                        .accessibilityLabel("Image prompt")
                        .accessibilityIdentifier("ImagePrompt")
                    if model.isBusy {
                        Button(
                            model.phase == .stopping ? "Stopping…" : "Stop",
                            systemImage: "stop.fill"
                        ) {
                            model.stop()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(model.phase == .stopping)
                        .keyboardShortcut(.escape, modifiers: [])
                    } else {
                        Button("Generate", systemImage: "sparkles") { model.generate() }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)
                            .disabled(!model.canGenerate)
                            .keyboardShortcut(.return, modifiers: .command)
                            .accessibilityIdentifier("GenerateImage")
                    }
                }
                HStack(spacing: 12) {
                    Button("Image Options", systemImage: "slider.horizontal.3") {
                        showsOptions.toggle()
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Image size, steps, and seed")
                    .popover(isPresented: $showsOptions, arrowEdge: .top) {
                        ImageOptionsView(model: model)
                            .presentationCompactAdaptation(.popover)
                    }
                    Text("\(model.size) × \(model.size) · \(model.steps) steps")
                    Spacer(minLength: 8)
                    #if os(macOS)
                        Text("⌘ Return to generate")
                    #endif
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        #if os(macOS)
            .onAppear { focused = true }
            .onChange(of: model.isBusy) { _, busy in
                if !busy { focused = true }
            }
        #endif
    }
}

private struct ImageOptionsView: View {
    @Bindable var model: ImageGenerationModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Image Options").font(.headline)
            Form {
                Picker("Size", selection: $model.size) {
                    ForEach([512, 768, 1024], id: \.self) { size in
                        Text("\(size) × \(size)").tag(size)
                    }
                }
                .pickerStyle(.menu)
                Stepper("Steps: \(model.steps)", value: $model.steps, in: 1 ... 50)
                LabeledContent("Seed") {
                    TextField("Seed", value: $model.seed, format: .number.grouping(.never))
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 100)
                }
            }
            .disabled(model.isBusy)
            Text("More steps take longer. Reuse a seed and prompt to reproduce an image.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(width: 320)
    }
}

private struct PNGDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }
    var data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw DemoError("Couldn’t read the PNG.")
        }
        self.data = data
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
