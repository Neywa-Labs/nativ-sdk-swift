enum PlaygroundRuntime {
    static var unavailableReason: String? {
        #if targetEnvironment(simulator)
            "Model inference requires a physical device. Run Playground on an iPhone, iPad, or Mac to try this feature."
        #else
            nil
        #endif
    }
}
