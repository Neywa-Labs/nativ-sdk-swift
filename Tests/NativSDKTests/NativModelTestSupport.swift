import Foundation

@testable import NativSDK

func makeTestModel(context: sending ModelContext) -> NativModel {
    let directory = URL(fileURLWithPath: "/tmp/nativ-model-fixture")
    return NativModel(
        assets: .init(
            source: .directory(directory), modelDirectory: directory, tokenizerDirectory: directory),
        metadata: context.metadata, context: ExecutionTransfer(context), policy: .init())
}
