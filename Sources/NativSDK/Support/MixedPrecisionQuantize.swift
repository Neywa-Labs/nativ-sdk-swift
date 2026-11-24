import Foundation
import MLX
import MLXNN

/// Replaces selected projections at their nearest module parent. Updating from the
/// root cannot represent sparse replacements in an array whose first layer is unchanged.
func quantizeByParent(
    model: Module,
    groupSize: Int = 64,
    bits: Int = 4,
    mode: QuantizationMode = .affine,
    filter: (String, Module) -> Bool = { _, _ in true },
    apply: (Module, Int, Int, QuantizationMode) -> Module? = quantizeSingle(
        layer:groupSize:bits:mode:)
) throws {
    try replaceByParent(model: model) { path, leaf in
        guard filter(path, leaf) else { return nil }
        return apply(leaf, groupSize, bits, mode)
    }
}

/// Groups leaf replacements by parent to preserve unmodified layers and tuple members.
func replaceByParent(
    model: Module,
    replacement: (String, Module) throws -> Module?
) throws {
    var replacements: [String: Module] = [:]
    for (path, leaf) in model.leafModules().flattened() {
        if let replacement = try replacement(path, leaf) { replacements[path] = replacement }
    }
    try replaceByParent(model: model, replacements: replacements)
}

/// Applies explicit replacements, including modules that contain child modules.
func replaceByParent(model: Module, replacements: [String: Module]) throws {
    var replacementsByParent: [String: [(String, Module)]] = [:]

    for (path, replacement) in replacements {
        guard let separator = path.lastIndex(of: ".") else {
            // A leaf directly on the root has no array between it and the root.
            replacementsByParent["", default: []].append((path, replacement))
            continue
        }
        let parent = String(path[path.startIndex ..< separator])
        let name = String(path[path.index(after: separator)...])
        replacementsByParent[parent, default: []].append((name, replacement))
    }

    guard !replacementsByParent.isEmpty else { return }

    let modulesByPath = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
    for (parentPath, replacements) in replacementsByParent {
        try applyReplacements(
            replacements, under: parentPath, root: model, modulesByPath: modulesByPath)
    }
}

/// Tuples and arrays are not named modules, so resolve their nearest module ancestor.
private func applyReplacements(
    _ replacements: [(String, Module)],
    under parentPath: String,
    root: Module,
    modulesByPath: [String: Module]
) throws {
    if let parent = parentPath.isEmpty ? root : modulesByPath[parentPath] {
        try parent.update(modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
        return
    }

    // MLX chooses array replacement/recurse behavior from the first entry. Keep
    // unchanged leading and trailing modules when replacing selected array slots.
    var completeReplacements = replacements
    if replacements.allSatisfy({ Int($0.0) != nil }) {
        let replaced = Set(replacements.map(\.0))
        var index = 0
        while let module = modulesByPath["\(parentPath).\(index)"] {
            let name = String(index)
            if !replaced.contains(name) { completeReplacements.append((name, module)) }
            index += 1
        }
    }

    var ancestorPath = parentPath
    var trailing: [String] = []
    while let separator = ancestorPath.lastIndex(of: ".") {
        trailing.insert(String(ancestorPath[ancestorPath.index(after: separator)...]), at: 0)
        ancestorPath = String(ancestorPath[ancestorPath.startIndex ..< separator])
        if let ancestor = modulesByPath[ancestorPath] {
            try ancestor.update(
                modules: ModuleChildren.unflattened(
                    completeReplacements.map {
                        ("\(trailing.joined(separator: ".")).\($0.0)", $0.1)
                    }),
                verify: .noUnusedKeys)
            return
        }
    }

    // Nothing above it resolved either, so address the whole path from the root.
    try root.update(
        modules: ModuleChildren.unflattened(
            completeReplacements.map { ("\(parentPath).\($0.0)", $0.1) }), verify: .noUnusedKeys)
}

/// Per-layer variant matching `MLXNN.quantize(model:filter:apply:)`, where the filter returns
/// each layer's own quantization parameters.
func quantizeByParent(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?,
    apply: (Module, Int, Int, QuantizationMode) -> Module? = quantizeSingle(
        layer:groupSize:bits:mode:)
) throws {
    try replaceByParent(model: model) { path, leaf in
        guard let (groupSize, bits, mode) = filter(path, leaf) else { return nil }
        return apply(leaf, groupSize, bits, mode)
    }
}
